#!/usr/bin/env python3
"""Job server for the 3090 throughput hill-climb. Runs *on the pod*.

The controlling sandbox has outbound HTTPS only and no SSH, so a hill-climb --
which by definition decides step N+1 from step N's number -- cannot be a script
baked into the pod's entrypoint. This is the missing half: a small HTTP control
plane, exposed through Runpod's port proxy, that accepts one benchmark step at
a time, runs it asynchronously, and keeps every result.

    POST /job      one step: which engine, which flags, which client load,
                   or {"action": "quantize", "params": {...}} to re-prepare a
                   quantized copy of the model without redeploying the pod,
                   or {"action": "eval", "params": {...}} to score a held-out
                   BEA-60K split on the running engine
    GET  /status   what it is doing, and the last result
    GET  /results  results.jsonl, every step ever run
    GET  /file?p=  a file under the output directory (server logs, bench JSON)
    POST /shutdown stop the inference server, keep the control plane up

Every route requires the launcher's CONTROL_TOKEN, as `?token=` or an
`X-Control-Token` header: the Runpod port proxy is public.

A job never carries a command line. The engine is one of a fixed set, the model
is one of a fixed set of prepared directories, and the tuning knobs arrive as
`{"--flag": "value"}` which this server renders into that engine's argv. The
climb is over serving flags, so that is expressive enough to be interesting
without turning a public URL into a shell.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

OUT = Path(os.environ.get("OUT_DIR", "/workspace/out"))
REPO = Path(os.environ.get("REPO_DIR", "/workspace/spell-corrector"))
MODELS = Path(os.environ.get("MODEL_DIR", "/workspace/models"))
RESULTS = OUT / "results.jsonl"
CONTROL_PORT = int(os.environ.get("CONTROL_PORT", "8001"))
CONTROL_TOKEN = os.environ.get("CONTROL_TOKEN", "")
SERVE_PORT = int(os.environ.get("SERVE_PORT", "8080"))
#: The quantizer runs under its own interpreter: llmcompressor needs a
#: compressed-tensors the engine is not built against, so the bootstrap puts it
#: in a --system-site-packages venv and names it here.
QUANT_PYTHON = os.environ.get("QUANT_PYTHON", sys.executable)

#: Model directories the pod prepared. A job names one of these keys; it can
#: never name a path.
MODEL_KEYS = {
    "w4a16": MODELS / "w4a16",
    "w8a8": MODELS / "w8a8",
    "fp16": MODELS / "fp16",
}

#: Flags that are structural rather than tunable: a job cannot set them,
#: because the control plane owns the port, the model and the served name.
RESERVED_FLAGS = {"--model", "--port", "--served-model-name", "--host"}

STATE = {
    "state": "idle",
    "current": None,
    "queued": 0,
    "steps_done": 0,
    "server": None,
    "last": None,
    "error": None,
}
JOBS: "queue.Queue[dict]" = queue.Queue()

SERVER = {"spec": None, "proc": None, "log": None}


def log(message: str) -> None:
    line = f"[control {time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with (OUT / "control.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# --------------------------------------------------------------------------
# argv construction (pure; the only place a job's flags become a command)
# --------------------------------------------------------------------------


def flatten_flags(flags: dict | None) -> list[str]:
    """`{"--max-num-seqs": 512, "--no-enable-prefix-caching": true}` -> argv.

    A boolean true renders as a bare switch, false drops the flag entirely, and
    a list renders as repeated `--flag value` pairs. Reserved flags are refused
    rather than silently ignored, so a job that tries to move the model or the
    port fails loudly.
    """
    argv: list[str] = []
    for key, value in (flags or {}).items():
        if not isinstance(key, str) or not key.startswith("--"):
            raise ValueError(f"flag {key!r} must be a string starting with --")
        if key in RESERVED_FLAGS:
            raise ValueError(f"flag {key} is owned by the control plane")
        if value is True:
            argv.append(key)
        elif value is False or value is None:
            continue
        elif isinstance(value, (list, tuple)):
            for item in value:
                argv += [key, str(item)]
        else:
            argv += [key, str(value)]
    return argv


def build_argv(engine: str, model_dir: Path, flags: dict | None, port: int) -> list[str]:
    if engine == "vllm":
        return [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_dir),
            "--served-model-name", "spell",
            "--host", "0.0.0.0",
            "--port", str(port),
        ] + flatten_flags(flags)
    if engine == "llamacpp":
        binary = shutil.which("llama-server") or "/workspace/llama.cpp/build/bin/llama-server"
        gguf = MODELS / "gguf" / "model.gguf"
        return [binary, "-m", str(gguf), "--host", "0.0.0.0", "--port", str(port)] + flatten_flags(flags)
    raise ValueError(f"unknown engine {engine!r}")


# --------------------------------------------------------------------------
# inference server lifecycle
# --------------------------------------------------------------------------


def stop_server() -> None:
    proc = SERVER.get("proc")
    if proc is None:
        return
    log("stopping inference server")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except Exception:
        pass
    for _ in range(60):
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            pass
    SERVER["proc"] = None
    SERVER["spec"] = None
    STATE["server"] = None
    time.sleep(5)  # the GPU takes a moment to actually come free


def wait_ready(port: int, path: str, timeout: float) -> bool:
    url = f"http://127.0.0.1:{port}{path}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        proc = SERVER.get("proc")
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode} before becoming ready")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 400:
                    return True
        except Exception:
            time.sleep(2)
    return False


def start_server(spec: dict, name: str) -> None:
    """(Re)start the inference server unless an identical one is already up.

    An engine restart costs ~a minute of a ten-minute step, so a job that only
    changes the client load reuses the running server.
    """
    if SERVER.get("spec") == spec and SERVER.get("proc") and SERVER["proc"].poll() is None:
        log("reusing the running server (spec unchanged)")
        return
    stop_server()
    engine = spec.get("engine", "vllm")
    model_key = spec.get("model", "w4a16")
    if model_key not in MODEL_KEYS:
        raise ValueError(f"unknown model key {model_key!r}; have {sorted(MODEL_KEYS)}")
    model_dir = MODEL_KEYS[model_key]
    if engine == "vllm" and not model_dir.is_dir():
        raise RuntimeError(f"model {model_key} was not prepared at {model_dir}")
    argv = build_argv(engine, model_dir, spec.get("flags"), SERVE_PORT)
    env = {**os.environ}
    for key, value in (spec.get("env") or {}).items():
        if not isinstance(key, str) or not key.replace("_", "").isalnum():
            raise ValueError(f"bad env key {key!r}")
        env[key] = str(value)
    log_path = OUT / f"server_{name}.log"
    handle = log_path.open("w", encoding="utf-8")
    log(f"starting {engine}/{model_key}: {' '.join(argv[3:])[:400]}")
    proc = subprocess.Popen(
        argv, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=str(REPO),
        start_new_session=True,  # its own process group, so the stop reaches the children
    )
    SERVER["proc"] = proc
    SERVER["log"] = str(log_path)
    started = time.time()
    ready_path = "/health" if engine == "vllm" else "/health"
    if not wait_ready(SERVE_PORT, ready_path, float(spec.get("ready_timeout", 900))):
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        stop_server()
        raise RuntimeError(f"server not ready in time; log tail:\n{tail}")
    SERVER["spec"] = spec
    STATE["server"] = {
        "engine": engine, "model": model_key, "flags": spec.get("flags"),
        "env": spec.get("env"), "startup_s": round(time.time() - started, 1),
        "log": log_path.name,
    }
    log(f"server ready in {time.time() - started:.1f}s")


# --------------------------------------------------------------------------
# one benchmark step
# --------------------------------------------------------------------------


def run_client(port: int, client: dict, name: str, tag: str) -> dict:
    cmd = [
        sys.executable,
        str(REPO / "scripts/bench_spell_throughput.py"),
        "--base-url", f"http://127.0.0.1:{port}",
        "--model", "spell",
        "--api", "completions" if client.get("api") == "completions" else "chat",
        "--concurrency", str(int(client["concurrency"])),
        "--duration", str(float(client.get("duration", 60))),
        "--warmup", str(float(client.get("warmup", 15))),
        "--max-tokens", str(int(client.get("max_tokens", 5))),
        "--num-prompts", str(int(client.get("num_prompts", 512))),
        "--output", str(OUT / f"bench_{name}_{tag}.json"),
    ]
    if client.get("api") == "completions":
        cmd += ["--tokenizer", str(MODEL_KEYS.get(client.get("tokenizer", "fp16"), MODEL_KEYS["fp16"]))]
    if client.get("thinking"):
        cmd += ["--thinking"]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if completed.returncode != 0:
        raise RuntimeError(f"client failed rc={completed.returncode}: {completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


#: The only preparation a job may trigger, and the only arguments it may set.
#: Re-quantizing is the one setup step likely to need a second attempt (the
#: scheme, the algorithm and the ignore list are all judgement calls), and
#: redeploying a pod to change one of them costs more than the step it feeds.
QUANT_PARAMS = {
    "scheme": str, "algorithm": str, "samples": int, "max_seq_len": int, "ignore": list,
    "model_key": str,
}
QUANT_ALGORITHMS = {"gptq", "awq", "rtn"}

#: The only splits a job may score. `train` and `val` were trained on by the M7
#: student, so scoring them would measure memorisation and produce a number
#: that looks like accuracy; they are refused here rather than trusted to
#: whoever writes the job.
EVAL_SPLITS = {"test", "frozen_100", "dev"}
EVAL_PARAMS = {"split": str, "limit": int, "concurrency": int, "label": str, "quantization": str}
SPLIT_DIR = Path(os.environ.get("SPLIT_DIR", str(REPO / "data" / "distill")))


def run_eval(params: dict) -> dict:
    params = params or {}
    unknown = set(params) - set(EVAL_PARAMS)
    if unknown:
        raise ValueError(f"unknown eval params: {sorted(unknown)}")
    split = str(params.get("split", "test"))
    if split not in EVAL_SPLITS:
        raise ValueError(f"split must be one of {sorted(EVAL_SPLITS)} (train/val were trained on)")
    split_path = SPLIT_DIR / f"{split}.jsonl"
    if not split_path.is_file():
        raise RuntimeError(f"{split_path} does not exist; the pod did not build the BEA splits")
    label = str(params.get("label") or f"{split}")
    safe = "".join(c for c in label if c.isalnum() or c in "-_")
    argv = [
        sys.executable, str(REPO / "scripts/distill/eval_openai_chat.py"),
        "--split", str(split_path),
        "--base-url", f"http://127.0.0.1:{SERVE_PORT}",
        "--model", "spell",
        "--concurrency", str(int(params.get("concurrency", 64))),
        "--limit", str(int(params.get("limit", 0))),
        "--label", label,
        "--quantization", str(params.get("quantization", "")),
        "--out-metrics", str(OUT / f"metrics_{safe}.json"),
        "--out-predictions", str(OUT / f"predictions_{safe}.jsonl"),
    ]
    log(f"scoring {split} as {label}")
    completed = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO), timeout=7200)
    if completed.returncode != 0:
        raise RuntimeError(f"eval rc={completed.returncode}: {completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def run_quantize(params: dict) -> dict:
    params = params or {}
    unknown = set(params) - set(QUANT_PARAMS)
    if unknown:
        raise ValueError(f"unknown quantize params: {sorted(unknown)}")
    algorithm = str(params.get("algorithm", "gptq"))
    if algorithm not in QUANT_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {sorted(QUANT_ALGORITHMS)}")
    scheme = str(params.get("scheme", "W4A16"))
    if not scheme.replace("A", "").replace("W", "").isdigit():
        raise ValueError("scheme looks wrong; expected something like W4A16")
    model_key = str(params.get("model_key", "w4a16"))
    if model_key not in MODEL_KEYS or model_key == "fp16":
        raise ValueError(f"model_key must be a quantized key, one of {sorted(set(MODEL_KEYS) - {'fp16'})}")
    argv = [
        QUANT_PYTHON, str(REPO / "scripts/distill/quantize_w4a16.py"),
        "--model", str(MODEL_KEYS["fp16"]),
        "--output", str(MODEL_KEYS[model_key]),
        "--scheme", scheme,
        "--algorithm", algorithm,
        "--samples", str(int(params.get("samples", 256))),
        "--max-seq-len", str(int(params.get("max_seq_len", 512))),
        "--dump-modules", str(OUT / "linear_modules.json"),
    ]
    for pattern in params.get("ignore") or []:
        if not isinstance(pattern, str):
            raise ValueError("ignore patterns must be strings")
        argv += ["--ignore", pattern]
    stop_server()  # quantization wants the whole card
    log(f"quantizing -> {model_key} ({algorithm} {scheme})")
    log_path = OUT / f"quantize_{model_key}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, cwd=str(REPO), timeout=5400)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
    if completed.returncode != 0:
        raise RuntimeError(f"quantize rc={completed.returncode}; log tail:\n{tail}")
    return {"model_key": model_key, "algorithm": algorithm, "scheme": scheme,
            "log": log_path.name, "tail": tail}


def gpu_snapshot() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()[0]
        used, total, util, power = [p.strip() for p in out.split(",")]
        return {"mem_used_mib": float(used), "mem_total_mib": float(total),
                "gpu_util_pct": float(util), "power_w": float(power)}
    except Exception:
        return {}


def run_job(job: dict) -> dict:
    name = job.get("name") or f"step{STATE['steps_done'] + 1}"
    STATE["current"] = name
    record = {"name": name, "note": job.get("note"), "started_at": time.time(), "runs": []}
    if job.get("action") == "quantize":
        STATE["state"] = "quantizing"
        record["quantize"] = run_quantize(job.get("params"))
        record["finished_at"] = time.time()
        with RESULTS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return record
    if job.get("action") not in (None, "bench", "eval"):
        raise ValueError(f"unknown action {job.get('action')!r}")
    STATE["state"] = "starting-server"
    server_spec = job.get("server")
    if server_spec:
        start_server(server_spec, name)
    record["server"] = STATE["server"]
    if job.get("action") == "eval":
        STATE["state"] = "evaluating"
        record["eval"] = run_eval(job.get("params"))
        record["finished_at"] = time.time()
        STATE["last"] = {"name": name, "split": record["eval"]["split"],
                         "acc_casefold": record["eval"]["acc@1_casefold"],
                         "n": record["eval"]["n"]}
        with RESULTS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        STATE["steps_done"] += 1
        return record
    client = dict(job.get("client") or {})
    sweep = client.pop("sweep", None) or [client.get("concurrency", 64)]
    STATE["state"] = "benchmarking"
    for concurrency in sweep:
        this = {**client, "concurrency": concurrency}
        log(f"{name}: benchmarking at concurrency={concurrency}")
        report = run_client(SERVE_PORT, this, name, f"c{concurrency}")
        report["gpu"] = gpu_snapshot()
        record["runs"].append(report)
        STATE["last"] = {"name": name, "concurrency": concurrency,
                         "rps": report["throughput_rps"], "p50": report["latency_s"]["p50"],
                         "errors": report["errors"], "empty": report["empty_answers"]}
    best = max(record["runs"], key=lambda r: r["throughput_rps"]) if record["runs"] else None
    record["best_rps"] = best["throughput_rps"] if best else 0.0
    record["best_concurrency"] = best["config"]["concurrency"] if best else None
    record["finished_at"] = time.time()
    with RESULTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    STATE["steps_done"] += 1
    return record


def worker() -> None:
    while True:
        job = JOBS.get()
        STATE["queued"] = JOBS.qsize()
        try:
            record = run_job(job)
            STATE["state"] = "idle"
            STATE["error"] = None
            if "best_rps" in record:
                log(f"{record['name']}: best {record['best_rps']} req/s at c={record['best_concurrency']}")
            else:  # a preparation job (quantize) produces no throughput number
                log(f"{record['name']}: done")
        except Exception as exc:  # noqa: BLE001 - reporting it is the point
            STATE["state"] = "failed"
            STATE["error"] = f"{exc}\n{traceback.format_exc()[-1500:]}"
            log(f"job failed: {exc}")
            with RESULTS.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"name": job.get("name"), "error": str(exc)[:4000],
                                         "finished_at": time.time()}) + "\n")
        finally:
            STATE["current"] = None


# --------------------------------------------------------------------------
# HTTP control plane
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102 - silence the default stderr spam
        return

    def _authorized(self) -> bool:
        if not CONTROL_TOKEN:
            return True
        supplied = self.headers.get("X-Control-Token") or (
            parse_qs(urlparse(self.path).query).get("token") or [""]
        )[0]
        return supplied == CONTROL_TOKEN

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if not self._authorized():
            self._send(403, b'{"error":"bad or missing control token"}')
            return
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/status"):
            payload = {**STATE, "queued": JOBS.qsize(), "gpu": gpu_snapshot(),
                       "models": {k: v.is_dir() for k, v in MODEL_KEYS.items()}, "now": time.time()}
            self._send(200, json.dumps(payload, indent=2).encode())
        elif parsed.path == "/results":
            text = RESULTS.read_text(encoding="utf-8") if RESULTS.is_file() else ""
            self._send(200, text.encode(), "text/plain; charset=utf-8")
        elif parsed.path == "/file":
            query = parse_qs(parsed.query)
            rel = (query.get("p") or [""])[0]
            target = (OUT / rel).resolve()
            if not str(target).startswith(str(OUT.resolve())) or not target.is_file():
                self._send(404, b'{"error":"no such file"}')
                return
            text = target.read_text(encoding="utf-8", errors="replace")
            tail = int((query.get("tail") or [0])[0])
            if tail:
                text = text[-tail:]
            self._send(200, text.encode(), "text/plain; charset=utf-8")
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):  # noqa: N802
        if not self._authorized():
            self._send(403, b'{"error":"bad or missing control token"}')
            return
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if parsed.path == "/shutdown":
            threading.Thread(target=stop_server, daemon=True).start()
            self._send(200, b'{"ok":true,"stopping":"inference server"}')
            return
        if parsed.path != "/job":
            self._send(404, b'{"error":"not found"}')
            return
        try:
            job = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send(400, json.dumps({"error": f"bad json: {exc}"}).encode())
            return
        try:  # reject a malformed job now, not sixty seconds into the queue
            if job.get("action") == "quantize":
                run_quantize.__doc__  # noqa: B018 - params are validated in the worker
            spec = job.get("server")
            if spec:
                build_argv(spec.get("engine", "vllm"), Path("/dev/null"), spec.get("flags"), SERVE_PORT)
        except ValueError as exc:
            self._send(400, json.dumps({"error": str(exc)}).encode())
            return
        JOBS.put(job)
        STATE["queued"] = JOBS.qsize()
        self._send(202, json.dumps({"queued": JOBS.qsize(), "name": job.get("name")}).encode())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", CONTROL_PORT), Handler)
    log(f"control plane listening on {CONTROL_PORT}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
