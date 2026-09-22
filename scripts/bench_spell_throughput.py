#!/usr/bin/env python3
"""Steady-state throughput benchmark for an OpenAI-compatible spelling server.

This measures *serving throughput* for the deployed workload -- one sentence
with a `<TYPO>...</TYPO>` span in, one word out, `max_tokens=5`, greedy -- and
nothing else. Accuracy is not scored here, and cannot be: the prompts are
synthesised from `data/wikipedia_misspellings.txt` (vendored, CC BY-SA), never
from BEA-60K, because BEA-60K is a locked benchmark and a hill-climbing loop is
exactly the repeated-measurement surface it has to stay out of.

Shape of the measurement:

* closed loop -- `--concurrency` workers, each with its own keep-alive HTTP
  connection, every one issuing the next request the moment its last returns.
  At 5 output tokens per request a new TCP connection per call would be a
  double-digit percentage of the cost, so connection reuse is not an optional
  detail;
* a `--warmup` window whose requests are issued but discarded, then a fixed
  `--duration` measurement window. Throughput is counted over the window's wall
  clock, so queueing, HTTP and scheduler overhead are all inside the number;
* vLLM's own `/metrics` is scraped before and after the window when it is
  reachable, so the prefix-cache hit rate and the engine's view of the run are
  recorded next to the client's view.

Usage:

    python scripts/bench_spell_throughput.py --base-url http://127.0.0.1:8000 \\
        --model spell --concurrency 64 --duration 60 --api chat
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import re
import statistics
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PROMPT_TEMPLATE_PATH = ROOT / "artifacts/spell_slm_m7_q4/direct_correct_v1.txt"
MISSPELLINGS_PATH = ROOT / "data/wikipedia_misspellings.txt"

#: Used when the vendored list is missing (a bare checkout, a partial clone).
FALLBACK_PAIRS = [
    ("recieve", "receive"),
    ("occurence", "occurrence"),
    ("seperate", "separate"),
    ("definately", "definitely"),
    ("neccessary", "necessary"),
    ("accomodate", "accommodate"),
    ("begining", "beginning"),
    ("existance", "existence"),
    ("independant", "independent"),
    ("maintainance", "maintenance"),
]

#: Carrier sentences. Lengths are spread on purpose: a serving benchmark run
#: entirely on one prompt length measures one point of the prefill curve and
#: flatters any engine with a prefix cache.
CARRIERS = [
    "I think we should {W} the proposal before the meeting on Friday.",
    "The committee will {W} the new guidelines once the review has finished.",
    "She told me that the package would {W} sometime next week, but it never did.",
    "There is no reason to {W} the results of a study that has not been published yet.",
    "After the storm the council had to {W} every road on the east side of town, "
    "which took the better part of a month and most of the emergency budget.",
    "Our team will {W} the migration plan, run it past the architects, and then "
    "schedule the cutover for a weekend when traffic is low.",
    "Could you {W} that?",
    "The report says the vendor failed to {W} the agreed service levels for three "
    "consecutive quarters, and the contract allows us to terminate without penalty "
    "if that happens again in the current year.",
]


def load_misspelling_pairs(path: Path = MISSPELLINGS_PATH, limit: int = 4000) -> list[tuple[str, str]]:
    """Parse `typo->gold` lines out of the vendored Wikipedia list."""
    if not Path(path).is_file():
        return list(FALLBACK_PAIRS)
    pairs: list[tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        typo, sep, gold = line.partition("->")
        if not sep:
            continue
        typo, gold = typo.strip(), gold.split(",")[0].strip()
        if not typo.isalpha() or not gold.isalpha():
            continue
        pairs.append((typo, gold))
        if len(pairs) >= limit:
            break
    return pairs or list(FALLBACK_PAIRS)


def build_sentences(count: int, seed: int = 20260922) -> list[str]:
    """`count` distinct sentences, each with exactly one <TYPO> span."""
    rng = random.Random(seed)
    pairs = load_misspelling_pairs()
    out: list[str] = []
    for i in range(count):
        typo, _gold = pairs[i % len(pairs)]
        carrier = CARRIERS[rng.randrange(len(CARRIERS))]
        out.append(carrier.replace("{W}", f"<TYPO>{typo}</TYPO>"))
    return out


def load_prompt_template() -> str:
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def build_user_prompt(sentence: str) -> str:
    return load_prompt_template().replace("{{SENTENCE}}", sentence)


def render_raw_prompts(user_prompts: list[str], model_path: str) -> list[str]:
    """Apply the model's chat template client-side, for `--api completions`.

    Moving template rendering off the server is one of the levers this
    benchmark exists to measure, so it has to produce *exactly* what the chat
    endpoint would produce. The template is applied once around a sentinel and
    the resulting prefix/suffix reused, which also keeps the tokenizer import
    off the per-request path.
    """
    from transformers import AutoTokenizer  # imported lazily: only this mode needs it

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sentinel = "\u0000SENTINEL\u0000"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if sentinel not in rendered:
        raise SystemExit("chat template did not echo the sentinel; cannot split prefix/suffix")
    prefix, _, suffix = rendered.partition(sentinel)
    return [prefix + p + suffix for p in user_prompts]


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class Endpoint:
    """One keep-alive connection to the server, reconnecting on failure."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        parsed = urlparse(base_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.https = parsed.scheme == "https"
        self.timeout = timeout
        self.conn: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        if self.conn is None:
            cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
            self.conn = cls(self.host, self.port, timeout=self.timeout)
        return self.conn

    def post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        for attempt in (0, 1):
            conn = self._connect()
            try:
                conn.request("POST", path, body=body, headers=headers)
                response = conn.getresponse()
                data = response.read()
                if response.status >= 400:
                    raise RuntimeError(f"{response.status}: {data[:300].decode('utf-8', 'replace')}")
                return json.loads(data)
            except (http.client.HTTPException, OSError):
                self.close()
                if attempt:
                    raise
        raise AssertionError("unreachable")

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None


def get_text(base_url: str, path: str, timeout: float = 10.0) -> str | None:
    try:
        endpoint = Endpoint(base_url, timeout=timeout)
        conn = endpoint._connect()
        conn.request("GET", path)
        response = conn.getresponse()
        data = response.read()
        endpoint.close()
        if response.status >= 400:
            return None
        return data.decode("utf-8", "replace")
    except Exception:
        return None


def scrape_metrics(base_url: str) -> dict:
    """A few vLLM prometheus counters, by name, as floats."""
    text = get_text(base_url, "/metrics")
    if not text:
        return {}
    wanted = (
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:gpu_prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_hits_total",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:iteration_tokens_total_sum",
        "vllm:iteration_tokens_total_count",
    )
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name, _, rest = line.partition("{")
        if not rest:
            name, _, value = line.partition(" ")
        else:
            _, _, value = rest.partition("} ")
        name = name.strip()
        if name in wanted:
            try:
                out[name] = out.get(name, 0.0) + float(value.strip())
            except ValueError:
                continue
    return out


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


class Result:
    __slots__ = ("latency", "prompt_tokens", "completion_tokens", "text", "index")

    def __init__(self, latency, prompt_tokens, completion_tokens, text, index):
        self.latency = latency
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.text = text
        self.index = index


def run_benchmark(args) -> dict:
    sentences = build_sentences(args.num_prompts)
    user_prompts = [build_user_prompt(s) for s in sentences]
    if args.api == "completions":
        payload_prompts = render_raw_prompts(user_prompts, args.tokenizer or args.model)
        path = "/v1/completions"
    else:
        payload_prompts = user_prompts
        path = "/v1/chat/completions"

    def make_payload(prompt: str) -> dict:
        common = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        if args.api == "completions":
            return {**common, "prompt": prompt}
        body = {**common, "messages": [{"role": "user", "content": prompt}]}
        if not args.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    stop = threading.Event()
    measuring = threading.Event()
    results: list[Result] = []
    errors: list[str] = []
    lock = threading.Lock()
    counter = threading.Lock()
    next_index = [0]

    def worker(worker_id: int) -> None:
        endpoint = Endpoint(args.base_url)
        local: list[Result] = []
        local_errors: list[str] = []
        try:
            while not stop.is_set():
                with counter:
                    idx = next_index[0]
                    next_index[0] += 1
                prompt = payload_prompts[idx % len(payload_prompts)]
                started = time.perf_counter()
                try:
                    response = endpoint.post(path, make_payload(prompt))
                except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                    if not stop.is_set():
                        local_errors.append(str(exc)[:200])
                    continue
                elapsed = time.perf_counter() - started
                if not measuring.is_set():
                    continue
                usage = response.get("usage") or {}
                choice = (response.get("choices") or [{}])[0]
                text = choice.get("text")
                if text is None:
                    text = (choice.get("message") or {}).get("content") or ""
                local.append(
                    Result(
                        elapsed,
                        int(usage.get("prompt_tokens") or 0),
                        int(usage.get("completion_tokens") or 0),
                        text,
                        idx % len(payload_prompts),
                    )
                )
        finally:
            endpoint.close()
            with lock:
                results.extend(local)
                errors.extend(local_errors)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.concurrency)]
    t_start = time.time()
    for thread in threads:
        thread.start()
    time.sleep(args.warmup)
    metrics_before = scrape_metrics(args.base_url)
    window_start = time.perf_counter()
    measuring.set()
    time.sleep(args.duration)
    measuring.clear()
    window_end = time.perf_counter()
    metrics_after = scrape_metrics(args.base_url)
    stop.set()
    for thread in threads:
        thread.join(timeout=180)

    window = window_end - window_start
    latencies = sorted(r.latency for r in results)
    completion_tokens = sum(r.completion_tokens for r in results)
    prompt_tokens = sum(r.prompt_tokens for r in results)
    empty = sum(1 for r in results if not (r.text or "").strip())

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        k = min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))
        return latencies[k]

    delta = {k: metrics_after.get(k, 0.0) - metrics_before.get(k, 0.0) for k in metrics_after}
    queries = delta.get("vllm:gpu_prefix_cache_queries_total", delta.get("vllm:prefix_cache_queries_total", 0.0))
    hits = delta.get("vllm:gpu_prefix_cache_hits_total", delta.get("vllm:prefix_cache_hits_total", 0.0))

    return {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "api": args.api,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "duration_s": args.duration,
            "warmup_s": args.warmup,
            "num_prompts": args.num_prompts,
            "thinking": args.thinking,
        },
        "window_s": round(window, 3),
        "requests": len(results),
        "errors": len(errors),
        "error_samples": errors[:5],
        "empty_answers": empty,
        "sample_answers": [r.text.strip()[:40] for r in results[:5]],
        "throughput_rps": round(len(results) / window, 2) if window else 0.0,
        "output_tokens_per_s": round(completion_tokens / window, 1) if window else 0.0,
        "prompt_tokens_per_s": round(prompt_tokens / window, 1) if window else 0.0,
        "total_tokens_per_s": round((completion_tokens + prompt_tokens) / window, 1) if window else 0.0,
        "mean_prompt_tokens": round(prompt_tokens / len(results), 1) if results else 0.0,
        "mean_completion_tokens": round(completion_tokens / len(results), 2) if results else 0.0,
        "latency_s": {
            "mean": round(statistics.fmean(latencies), 4) if latencies else 0.0,
            "p50": round(pct(0.50), 4),
            "p90": round(pct(0.90), 4),
            "p99": round(pct(0.99), 4),
            "max": round(latencies[-1], 4) if latencies else 0.0,
        },
        "prefix_cache": {
            "queries": queries,
            "hits": hits,
            "hit_rate": round(hits / queries, 4) if queries else None,
        },
        "started_at": t_start,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="spell")
    parser.add_argument("--tokenizer", default=None, help="local path/HF id for --api completions rendering")
    parser.add_argument("--api", choices=("chat", "completions"), default="chat")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--duration", type=float, default=60.0, help="measurement window, seconds")
    parser.add_argument("--warmup", type=float, default=15.0, help="discarded window before measuring")
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--num-prompts", type=int, default=512)
    parser.add_argument("--thinking", action="store_true", help="leave the model's <think> block enabled")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_benchmark(args)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
