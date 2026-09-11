#!/usr/bin/env python3
"""Throughput/latency benchmark for a running llama.cpp `llama-server`.

Why this exists separately from scripts/llm_judge_bea60k_cpu.py: that script
measures *accuracy* on the locked BEA-60K benchmark and happens to report
latency as a side effect, one request at a time. This one measures nothing but
performance, and measures the things a serving decision actually turns on --
prefill vs decode throughput, the latency distribution under load, and how
total throughput scales with concurrency against the server's `--parallel`
slots -- so that a cost per 1,000 corrections can be computed honestly.

Two independent clocks are reported for every number:

* llama-server's own `timings` block (`prompt_per_second`, `predicted_per_second`),
  which is the server's view and excludes queueing and HTTP overhead;
* wall clock measured by this client, which includes all of it.

They disagree under concurrency -- that disagreement *is* the queueing cost,
and hiding it behind one number would make a batched server look like a
single-stream one.

Benchmark prompts are synthesised from `data/wikipedia_misspellings.txt`
(vendored, CC BY-SA) wrapped in the same message builders the real judge uses,
never from BEA-60K: BEA-60K is a locked evaluation benchmark and a throughput
loop is exactly the kind of repeated-measurement tuning surface it must stay
out of.

Usage (against an already-running server):

    python scripts/benchmark_llama_server.py \\
        --base-url http://127.0.0.1:8080 --model gemma-4-e2b-q4 \\
        --output reports/gpu_llama/benchmark \\
        --server-log /workspace/out/llama_server.log \\
        --runinfo /workspace/out/RUNINFO.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_prompt_builders():
    """Import the judge's message builders without importing the package.

    `spelling_reranker/__init__.py` imports torch, and this benchmark needs
    nothing but the standard library -- it has to run on a bare pod, and on a
    GPU pod the torch import would also be several seconds of nothing. Loading
    the one module by path keeps the prompts identical to production without
    dragging the training stack in behind them.
    """
    path = ROOT / "spelling_reranker" / "llm_judge_cpu.py"
    spec = importlib.util.spec_from_file_location("_llm_judge_cpu_prompts", path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise ImportError(f"cannot load prompt builders from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_messages, module.build_sentence_messages


build_messages, build_sentence_messages = _load_prompt_builders()

#: Community-cloud on-demand USD/hr, queried 2026-09-11. Only a fallback: the
#: launcher records the price the API actually returned for the pod it got into
#: RUNINFO.json, and that wins when present, because these move.
GPU_PRICE_USD_HR = {
    "NVIDIA GeForce RTX 3070": 0.13,
    "NVIDIA RTX A5000": 0.16,
    "NVIDIA GeForce RTX 3080 Ti": 0.18,
    "NVIDIA RTX 4000 SFF Ada Generation": 0.18,
    "NVIDIA GeForce RTX 4070 Ti": 0.19,
    "NVIDIA RTX A4500": 0.19,
    "Tesla V100-PCIE-16GB": 0.19,
    "NVIDIA RTX 4000 Ada Generation": 0.20,
    "NVIDIA GeForce RTX 3090": 0.22,
    "NVIDIA GeForce RTX 3090 Ti": 0.27,
    "NVIDIA GeForce RTX 4080 SUPER": 0.28,
    "NVIDIA RTX A6000": 0.33,
    "NVIDIA GeForce RTX 4090": 0.34,
    "NVIDIA A40": 0.35,
}

#: Carrier sentences the synthetic typos are dropped into. Deliberately
#: generic English of realistic length -- the point is prompt *shape* (token
#: count), not content, and none of it comes from the locked benchmark.
CARRIER_SENTENCES = [
    ("The committee agreed that the ", " was the main reason for the delay."),
    ("She wrote in her notebook that the ", " had already been discussed twice."),
    ("According to the report, every ", " must be reviewed before the deadline."),
    ("I think the ", " should be explained more clearly in the introduction."),
    ("Most of the students said the ", " was harder than they had expected."),
    ("He asked whether the ", " would be available again next year."),
    ("The article claims that a ", " is not always the best solution."),
    ("Before leaving, please make sure the ", " has been signed and dated."),
]

#: Used when data/wikipedia_misspellings.txt is absent (e.g. a bare checkout).
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

#: The two shapes real traffic has: a one-number answer (index mode) and a
#: whole-sentence rewrite. Everything downstream is reported per workload,
#: because their cost per correction differs by more than an order of magnitude.
WORKLOADS = {
    "short_answer": {"max_tokens": 8, "builder": "index"},
    "sentence_rewrite": {"max_tokens": 64, "builder": "sentence"},
}

DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16, 32)


# --------------------------------------------------------------------------
# pure helpers (unit-tested; no network)
# --------------------------------------------------------------------------


def load_misspelling_pairs(path: Path, limit: int = 2000) -> list[tuple[str, str]]:
    """Parse `typo->gold` lines out of the vendored Wikipedia list.

    The file has a prose header and one `typo->gold` pair per line after it;
    anything without the arrow, and any multi-word entry, is skipped.
    """
    if not Path(path).is_file():
        return list(FALLBACK_PAIRS)
    pairs: list[tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        typo, sep, gold = line.strip().partition("->")
        if not sep:
            continue
        gold = gold.split(",")[0].strip()
        typo = typo.strip()
        if not typo or not gold or " " in typo or " " in gold:
            continue
        pairs.append((typo, gold))
        if len(pairs) >= limit:
            break
    return pairs or list(FALLBACK_PAIRS)


def build_workload_prompts(
    pairs: list[tuple[str, str]], workload: str, n: int, *, seed: int = 1337, n_candidates: int = 8
) -> list[dict]:
    """Deterministic synthetic prompts in the same shape the judge sends.

    Reusing the judge's own message builders means the measured prefill length
    is the length the production prompt actually has, rather than a round
    number chosen for the benchmark.
    """
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}; known: {sorted(WORKLOADS)}")
    rng = random.Random(seed)
    builder = build_messages if WORKLOADS[workload]["builder"] == "index" else build_sentence_messages
    golds = [gold for _, gold in pairs]
    prompts: list[dict] = []
    for i in range(n):
        typo, gold = pairs[i % len(pairs)]
        before, after = CARRIER_SENTENCES[i % len(CARRIER_SENTENCES)]
        candidates = [gold]
        while len(candidates) < n_candidates and len(golds) > len(candidates):
            pick = golds[rng.randrange(len(golds))]
            if pick not in candidates:
                candidates.append(pick)
        rng.shuffle(candidates)
        prompts.append(
            {
                "messages": builder(before, typo, after, candidates),
                "max_tokens": WORKLOADS[workload]["max_tokens"],
                "workload": workload,
            }
        )
    return prompts


def parse_timings(response: dict) -> dict:
    """Normalise llama-server's `timings` block.

    Returns an empty dict when the server did not send one (older builds, or
    `timings_per_token` unsupported), so callers fall back to wall clock rather
    than reporting a fabricated zero.
    """
    timings = response.get("timings") or {}
    if not isinstance(timings, dict) or not timings:
        return {}
    out = {
        "prompt_tokens": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_tok_per_s": timings.get("prompt_per_second"),
        "predicted_tokens": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_tok_per_s": timings.get("predicted_per_second"),
    }
    # Older builds send the ms but not the per-second fields; derive rather
    # than drop, since prefill tok/s is one of the two headline numbers.
    if out["prompt_tok_per_s"] is None and out["prompt_ms"]:
        out["prompt_tok_per_s"] = 1000.0 * (out["prompt_tokens"] or 0) / out["prompt_ms"]
    if out["predicted_tok_per_s"] is None and out["predicted_ms"]:
        out["predicted_tok_per_s"] = 1000.0 * (out["predicted_tokens"] or 0) / out["predicted_ms"]
    return out


def _pct(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile. Exact on small n, where interpolation lies."""
    if not sorted_values:
        raise ValueError("no values")
    rank = max(1, min(len(sorted_values), int(-(-q * len(sorted_values) // 1))))
    return sorted_values[rank - 1]


def latency_summary(latencies_s: list[float]) -> dict:
    """mean/p50/p90/p99/max in milliseconds, plus n."""
    if not latencies_s:
        return {"n": 0}
    ms = sorted(v * 1000.0 for v in latencies_s)
    return {
        "n": len(ms),
        "mean_ms": statistics.fmean(ms),
        "p50_ms": _pct(ms, 0.50),
        "p90_ms": _pct(ms, 0.90),
        "p99_ms": _pct(ms, 0.99),
        "min_ms": ms[0],
        "max_ms": ms[-1],
    }


_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")
_DEVICE_RE = re.compile(r"Device\s+\d+:\s+([^,]+),\s+compute capability\s+([\d.]+)")
_CUDA_BUFFER_RE = re.compile(r"(CUDA\d+)[^=]*buffer size\s*=\s*([\d.]+)\s*MiB")
_LISTENING_RE = re.compile(r"server is listening")


def parse_server_log(text: str) -> dict:
    """What llama-server said about GPU offload, from its own stdout log.

    /props does not report offload, so the log is the only place the "did we
    actually get all the layers onto the card" question is answered -- and a
    silent CPU fallback is the single most likely way a GPU run quietly
    produces CPU numbers.
    """
    info: dict = {
        "layers_offloaded": None,
        "layers_total": None,
        "gpu_name": None,
        "compute_capability": None,
        "cuda_buffers_mib": {},
        "cuda_buffer_total_mib": None,
        "server_listening": bool(_LISTENING_RE.search(text)),
    }
    match = _OFFLOAD_RE.search(text)
    if match:
        info["layers_offloaded"] = int(match.group(1))
        info["layers_total"] = int(match.group(2))
    match = _DEVICE_RE.search(text)
    if match:
        info["gpu_name"] = match.group(1).strip()
        info["compute_capability"] = match.group(2)
    buffers: dict[str, float] = {}
    for device, mib in _CUDA_BUFFER_RE.findall(text):
        buffers[device] = buffers.get(device, 0.0) + float(mib)
    if buffers:
        info["cuda_buffers_mib"] = buffers
        info["cuda_buffer_total_mib"] = sum(buffers.values())
    info["full_offload"] = (
        info["layers_offloaded"] is not None and info["layers_offloaded"] == info["layers_total"]
    )
    return info


def cost_per_1000_corrections(requests_per_second: float | None, usd_per_hr: float | None) -> float | None:
    """USD to serve 1,000 corrections at a measured request rate."""
    if not requests_per_second or usd_per_hr is None:
        return None
    return usd_per_hr / (requests_per_second * 3600.0) * 1000.0


def resolve_gpu_price(runinfo: dict | None, gpu_name: str | None, override: float | None) -> tuple[str | None, float | None, str]:
    """Price the run: explicit flag > what the API charged > the static table."""
    if override is not None:
        return gpu_name, override, "--gpu-price-usd-hr"
    if runinfo:
        name = runinfo.get("chosen_gpu") or gpu_name
        price = runinfo.get("cost_per_hr")
        if price is not None:
            return name, float(price), "RUNINFO.json (price charged by the API)"
        gpu_name = name or gpu_name
    if gpu_name and gpu_name in GPU_PRICE_USD_HR:
        return gpu_name, GPU_PRICE_USD_HR[gpu_name], "static table in benchmark_llama_server.py"
    return gpu_name, None, "unknown"


def render_markdown(report: dict) -> str:
    """Small human-readable companion to report.json."""
    env = report.get("environment", {})
    lines = [
        "# llama-server GPU benchmark",
        "",
        f"- model: `{report.get('model')}`",
        f"- GPU: {env.get('gpu_name') or 'unknown'} "
        f"(${env.get('usd_per_hr')}/hr, source: {env.get('price_source')})",
        f"- offload: {env.get('layers_offloaded')}/{env.get('layers_total')} layers"
        f" (full_offload={env.get('full_offload')}), "
        f"CUDA buffers {env.get('cuda_buffer_total_mib')} MiB",
        f"- server startup / model load: {env.get('startup_seconds')} s",
        "",
        "## Single-request latency",
        "",
        "| workload | n | mean ms | p50 | p90 | p99 | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, block in (report.get("latency") or {}).items():
        s = block.get("latency", {})
        lines.append(
            f"| {name} | {s.get('n')} | {_f(s.get('mean_ms'))} | {_f(s.get('p50_ms'))} | "
            f"{_f(s.get('p90_ms'))} | {_f(s.get('p99_ms'))} | {_f(s.get('max_ms'))} |"
        )
    lines += [
        "",
        "## Throughput (server timings vs wall clock)",
        "",
        "| workload | prefill tok/s (server) | prefill tok/s (wall) | decode tok/s (server) | decode tok/s (wall) |",
        "|---|---|---|---|---|",
    ]
    for name, block in (report.get("latency") or {}).items():
        lines.append(
            f"| {name} | {_f(block.get('prefill_tok_per_s_server'))} | "
            f"{_f(block.get('prefill_tok_per_s_wall'))} | "
            f"{_f(block.get('decode_tok_per_s_server'))} | {_f(block.get('decode_tok_per_s_wall'))} |"
        )
    lines += [
        "",
        "## Concurrency sweep",
        "",
        "| workload | concurrency | req/s | generated tok/s | p50 ms | p99 ms | $/1k corrections |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, levels in (report.get("concurrency") or {}).items():
        for level in levels:
            lines.append(
                f"| {name} | {level.get('concurrency')} | {_f(level.get('requests_per_second'))} | "
                f"{_f(level.get('generated_tok_per_s'))} | {_f((level.get('latency') or {}).get('p50_ms'))} | "
                f"{_f((level.get('latency') or {}).get('p99_ms'))} | "
                f"{_f(level.get('usd_per_1000_corrections'), '.4f')} |"
            )
    lines += ["", "## Cost", "", "| workload | best req/s | at concurrency | $/1k corrections |", "|---|---|---|---|"]
    for name, block in (report.get("cost") or {}).items():
        lines.append(
            f"| {name} | {_f(block.get('best_requests_per_second'))} | {block.get('at_concurrency')} | "
            f"{_f(block.get('usd_per_1000_corrections'), '.4f')} |"
        )
    return "\n".join(lines) + "\n"


def _f(value, spec: str = ".1f") -> str:
    return "n/a" if value is None else format(value, spec)


# --------------------------------------------------------------------------
# HTTP (stdlib only: this script must run on a bare pod without pip installs)
# --------------------------------------------------------------------------


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-supplied URL
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: float = 15.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def one_request(base_url: str, model: str, prompt: dict, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": prompt["messages"],
        "max_tokens": prompt["max_tokens"],
        "temperature": 0.0,
        "top_k": 1,
        "seed": 0,
        "stream": False,
        # Asks llama-server to attach its own `timings` block to the
        # OpenAI-shaped response; without it the only clock is the client's.
        "timings_per_token": True,
    }
    t0 = time.perf_counter()
    try:
        response = post_json(f"{base_url}/v1/chat/completions", payload, timeout)
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc), "wall_s": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    timings = parse_timings(response)
    usage = response.get("usage") or {}
    return {
        "ok": True,
        "wall_s": wall,
        "timings": timings,
        "completion_tokens": timings.get("predicted_tokens") or usage.get("completion_tokens"),
        "prompt_tokens": timings.get("prompt_tokens") or usage.get("prompt_tokens"),
    }


def run_serial(base_url: str, model: str, prompts: list[dict], timeout: float) -> dict:
    """Concurrency-1 pass: the latency distribution without queueing."""
    results = [one_request(base_url, model, p, timeout) for p in prompts]
    return summarize_results(results)


def run_concurrent(base_url: str, model: str, prompts: list[dict], concurrency: int, timeout: float) -> dict:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda p: one_request(base_url, model, p, timeout), prompts))
    elapsed = time.perf_counter() - t0
    summary = summarize_results(results)
    ok = summary["n_ok"]
    summary.update(
        {
            "concurrency": concurrency,
            "elapsed_seconds": elapsed,
            "requests_per_second": (ok / elapsed) if elapsed > 0 else None,
            "generated_tok_per_s": (summary["total_completion_tokens"] / elapsed) if elapsed > 0 else None,
        }
    )
    return summary


def summarize_results(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("ok")]
    latencies = [r["wall_s"] for r in ok]
    completion = sum(r.get("completion_tokens") or 0 for r in ok)
    prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in ok)
    prefill = [t for r in ok if (t := (r.get("timings") or {}).get("prompt_tok_per_s"))]
    decode = [t for r in ok if (t := (r.get("timings") or {}).get("predicted_tok_per_s"))]
    wall_decode = None
    wall_prefill = None
    if latencies:
        total_latency = sum(latencies)
        # The wall-clock rates charge each phase with the *whole* round trip --
        # a non-streaming request cannot separate prefill from decode from the
        # outside. They are therefore both pessimistic by construction, and the
        # gap between them and the server's own timings is the HTTP + queueing
        # overhead a caller of the endpoint actually pays.
        if completion:
            wall_decode = completion / total_latency
        if prompt_tokens:
            wall_prefill = prompt_tokens / total_latency
    return {
        "n": len(results),
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "errors": sorted({r.get("error") for r in results if not r.get("ok")} - {None})[:5],
        "total_completion_tokens": completion,
        "total_prompt_tokens": prompt_tokens,
        "latency": latency_summary(latencies),
        "prefill_tok_per_s_server": statistics.fmean(prefill) if prefill else None,
        "decode_tok_per_s_server": statistics.fmean(decode) if decode else None,
        "decode_tok_per_s_wall": wall_decode,
        "prefill_tok_per_s_wall": wall_prefill,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-server root (not /v1)")
    parser.add_argument("--model", default="gemma-4-e2b-q4", help="served model alias")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, default=None, help="llama-server stdout log, for offload info")
    parser.add_argument("--runinfo", type=Path, default=None, help="RUNINFO.json written by launch_gpu_llama.py")
    parser.add_argument("--gpu-price-usd-hr", type=float, default=None, help="override the $/hr used for cost math")
    parser.add_argument("--startup-seconds", type=float, default=None, help="measured server start + model load")
    parser.add_argument("--latency-requests", type=int, default=32, help="serial requests per workload")
    parser.add_argument("--sweep-requests", type=int, default=64, help="requests per concurrency level")
    parser.add_argument(
        "--concurrency",
        type=int,
        action="append",
        default=None,
        help=f"concurrency level, repeatable (default: {list(DEFAULT_CONCURRENCIES)})",
    )
    parser.add_argument("--workload", action="append", choices=sorted(WORKLOADS), default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--misspellings",
        type=Path,
        default=ROOT / "data" / "wikipedia_misspellings.txt",
        help="vendored Wikipedia common-misspellings list used to synthesise prompts",
    )
    args = parser.parse_args()

    concurrencies = args.concurrency or list(DEFAULT_CONCURRENCIES)
    workloads = args.workload or sorted(WORKLOADS)
    args.output.mkdir(parents=True, exist_ok=True)

    pairs = load_misspelling_pairs(args.misspellings)
    print(f"{len(pairs)} synthetic typo/gold pairs from {args.misspellings}", flush=True)

    props: dict = {}
    try:
        props = get_json(f"{args.base_url}/props")
    except Exception as exc:  # noqa: BLE001 - /props is informational
        print(f"warning: /props unavailable: {exc}", file=sys.stderr)

    log_info: dict = {}
    if args.server_log and args.server_log.is_file():
        log_info = parse_server_log(args.server_log.read_text(encoding="utf-8", errors="replace"))
        if not log_info.get("full_offload"):
            print(
                "warning: server log does not show a full GPU offload "
                f"({log_info.get('layers_offloaded')}/{log_info.get('layers_total')})",
                file=sys.stderr,
            )

    runinfo = None
    if args.runinfo and args.runinfo.is_file():
        try:
            runinfo = json.loads(args.runinfo.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001
            print(f"warning: cannot read {args.runinfo}: {exc}", file=sys.stderr)
    gpu_name, usd_per_hr, price_source = resolve_gpu_price(
        runinfo, log_info.get("gpu_name"), args.gpu_price_usd_hr
    )

    # Warm up: the first request after load pays for CUDA graph capture and
    # KV-cache allocation and would otherwise land in the p99.
    warm = build_workload_prompts(pairs, workloads[0], 1, seed=args.seed)
    print("warmup ...", flush=True)
    print(f"  warmup ok={one_request(args.base_url, args.model, warm[0], args.timeout)['ok']}", flush=True)

    latency: dict = {}
    for workload in workloads:
        prompts = build_workload_prompts(pairs, workload, args.latency_requests, seed=args.seed)
        print(f"latency: {workload} x{len(prompts)} at concurrency 1 ...", flush=True)
        block = run_serial(args.base_url, args.model, prompts, args.timeout)
        latency[workload] = block
        print(
            f"  p50={_f(block['latency'].get('p50_ms'))}ms "
            f"decode={_f(block.get('decode_tok_per_s_server'))} tok/s (server)",
            flush=True,
        )

    concurrency: dict = {}
    for workload in workloads:
        levels = []
        for level in concurrencies:
            prompts = build_workload_prompts(pairs, workload, args.sweep_requests, seed=args.seed + level)
            print(f"sweep: {workload} concurrency={level} n={len(prompts)} ...", flush=True)
            block = run_concurrent(args.base_url, args.model, prompts, level, args.timeout)
            block["usd_per_1000_corrections"] = cost_per_1000_corrections(
                block.get("requests_per_second"), usd_per_hr
            )
            levels.append(block)
            print(
                f"  req/s={_f(block.get('requests_per_second'), '.2f')} "
                f"tok/s={_f(block.get('generated_tok_per_s'))} "
                f"p99={_f((block.get('latency') or {}).get('p99_ms'))}ms "
                f"failed={block.get('n_failed')}",
                flush=True,
            )
        concurrency[workload] = levels

    cost: dict = {}
    for workload, levels in concurrency.items():
        usable = [lv for lv in levels if lv.get("requests_per_second")]
        best = max(usable, key=lambda lv: lv["requests_per_second"], default=None)
        cost[workload] = {
            "best_requests_per_second": best.get("requests_per_second") if best else None,
            "at_concurrency": best.get("concurrency") if best else None,
            "usd_per_hr": usd_per_hr,
            "usd_per_1000_corrections": cost_per_1000_corrections(
                best.get("requests_per_second") if best else None, usd_per_hr
            ),
        }

    report = {
        "model": args.model,
        "base_url": args.base_url,
        "environment": {
            "gpu_name": gpu_name,
            "usd_per_hr": usd_per_hr,
            "price_source": price_source,
            "startup_seconds": args.startup_seconds
            or (runinfo or {}).get("llama_server_startup_seconds"),
            **{k: v for k, v in log_info.items() if k != "server_listening"},
            "props": {
                "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
                "model_path": props.get("model_path"),
                "chat_template_present": bool(props.get("chat_template")),
                "total_slots": props.get("total_slots"),
            },
        },
        "config": {
            "concurrencies": concurrencies,
            "workloads": workloads,
            "latency_requests": args.latency_requests,
            "sweep_requests": args.sweep_requests,
            "seed": args.seed,
            "prompt_source": str(args.misspellings),
        },
        "latency": latency,
        "concurrency": concurrency,
        "cost": cost,
        "runinfo": runinfo,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
