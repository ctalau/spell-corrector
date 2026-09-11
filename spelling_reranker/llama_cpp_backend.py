"""llama.cpp (GGUF) backend for the LLM-judge CPU track.

Why this exists: the transformers backend (`spelling_reranker/llm_judge_cpu.py`)
runs the bf16 checkpoint through PyTorch's CPU path, where decoding is
bandwidth-bound and costs ~215ms per generated token on a 4-vCPU box. A q4_0
GGUF is ~3.4GB instead of ~10GB and llama.cpp's quantized kernels are written
for exactly this case, so the same prompts decode substantially faster.

The trade this makes, stated plainly: **quantization changes what the model
outputs.** Unlike prompt-lookup speculative decoding (which is greedy-identical
by construction), q4_0 answers may differ from the bf16 answers, so accuracy
has to be re-measured on the benchmark rather than assumed to carry over. Use
Google's QAT ("quantization-aware training") GGUF where available, since it is
trained for this quantization rather than rounded into it after the fact.

Transport is llama.cpp's own HTTP server (`llama-server`), talked to over its
OpenAI-compatible `/v1/chat/completions` endpoint:

- the GGUF carries its own chat template, so the prompt is built by the same
  code that serves it, rather than being reconstructed here;
- `/tokenize` gives real token counts for the per-example generation budget,
  instead of borrowing a different tokenizer's idea of length;
- one server process holds the weights, so repeated runs skip model load.

Greedy decoding is requested explicitly (`temperature=0`, `top_k=1`), matching
the transformers backend's `do_sample=False`.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


class LlamaServerError(RuntimeError):
    pass


def _post(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed localhost URL
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class LlamaCppModel:
    """Mirrors the parts of `llm_judge_cpu.LoadedModel` the harness touches."""

    model_id: str
    base_url: str
    request_timeout: float = 600.0
    process: subprocess.Popen | None = None
    server_metadata: dict = field(default_factory=dict)

    # The harness records these for the transformers backend; keep the same
    # keys so results.json stays one shape across backends.
    load_class: str = "llama.cpp/llama-server"
    supports_system_role: bool = True
    supports_enable_thinking: bool = False

    @property
    def device(self):  # noqa: D401 - mirrors torch.device duck-typing in the harness
        return _CpuDevice()

    def n_tokens(self, text: str) -> int:
        """Token count from the served model's own tokenizer."""
        out = _post(f"{self.base_url}/tokenize", {"content": text}, self.request_timeout)
        return len(out.get("tokens", []))

    def generate(self, messages: list[dict], *, max_new_tokens: int) -> tuple[str, float]:
        payload = {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 0,
            "stream": False,
        }
        t0 = time.perf_counter()
        out = _post(f"{self.base_url}/v1/chat/completions", payload, self.request_timeout)
        latency = time.perf_counter() - t0
        try:
            message = out["choices"][0]["message"]
        except (KeyError, IndexError) as exc:  # noqa: BLE001
            raise LlamaServerError(f"unexpected llama-server response: {out}") from exc
        text = message.get("content") or ""
        if not text and message.get("reasoning_content"):
            # The model answered in thinking mode: llama.cpp put the chain of
            # thought in reasoning_content and left content empty. Scoring the
            # reasoning text as if it were the answer would quietly produce
            # nonsense, so this is raised as the configuration error it is --
            # the server needs `--reasoning off`, the equivalent of the
            # transformers path's enable_thinking=False.
            raise LlamaServerError(
                "llama-server returned only reasoning_content (thinking mode is on); "
                "start the server with --reasoning off"
            )
        return text, latency

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()


@dataclass
class _CpuDevice:
    type: str = "cpu"

    def __str__(self) -> str:
        return "cpu"


def wait_for_server(base_url: str, *, timeout_s: float = 300.0, process: subprocess.Popen | None = None) -> dict:
    """Block until /health reports ready, surfacing an early server exit."""
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise LlamaServerError(f"llama-server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    try:
                        with urllib.request.urlopen(f"{base_url}/props", timeout=10) as props:  # noqa: S310
                            return json.loads(props.read().decode("utf-8"))
                    except Exception:  # noqa: BLE001 - /props is informational only
                        return {}
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
            time.sleep(1.0)
    raise LlamaServerError(f"llama-server not ready within {timeout_s:.0f}s: {last_error}")


def load_llama_cpp(
    gguf_path: Path,
    *,
    server_binary: Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    n_threads: int | None = None,
    n_ctx: int = 4096,
    extra_args: list[str] | None = None,
    startup_timeout_s: float = 300.0,
    log_path: Path | None = None,
) -> LlamaCppModel:
    """Spawn `llama-server` on the given GGUF and wait until it answers.

    Server output goes to `log_path` (or is discarded), never to a pipe:
    llama-server logs every request, and an undrained `subprocess.PIPE` blocks
    the server as soon as the 64KB pipe buffer fills -- which looks exactly
    like the model hanging mid-generation.
    """
    if not Path(gguf_path).is_file():
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")
    if not Path(server_binary).is_file():
        raise FileNotFoundError(f"llama-server binary not found: {server_binary}")

    cmd = [
        str(server_binary),
        "-m", str(gguf_path),
        "--host", host,
        "--port", str(port),
        "-c", str(n_ctx),
        # Thinking off, matching the transformers path's enable_thinking=False.
        # Left on, gemma-4 spends the whole generation budget on a chain of
        # thought and returns an empty answer.
        "--reasoning", "off",
        # One request at a time: the harness measures single-request latency,
        # not batched throughput, exactly as the transformers backend does.
        "-np", "1",
    ]
    if n_threads:
        cmd += ["-t", str(n_threads)]
    if extra_args:
        cmd += extra_args

    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - closed with the process
    else:
        log_handle = subprocess.DEVNULL
    process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    base_url = f"http://{host}:{port}"
    try:
        props = wait_for_server(base_url, timeout_s=startup_timeout_s, process=process)
    except Exception:
        process.terminate()
        raise
    return LlamaCppModel(
        model_id=str(gguf_path),
        base_url=base_url,
        process=process,
        server_metadata={
            "command": cmd,
            "model_path": str(gguf_path),
            "n_ctx": props.get("default_generation_settings", {}).get("n_ctx", n_ctx),
            "chat_template_from_gguf": bool(props.get("chat_template")),
            "reasoning": "off",
        },
    )
