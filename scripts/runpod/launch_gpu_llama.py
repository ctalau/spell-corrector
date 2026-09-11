#!/usr/bin/env python3
"""Create a GPU Runpod pod that benchmarks gemma-4-E2B q4_0 under llama.cpp/CUDA.

Same self-driving shape as scripts/runpod/launch.py -- the controlling sandbox
has outbound HTTPS only, so the pod runs `scripts/runpod/bootstrap_gpu_llama.sh`
as its *entrypoint* (never `dockerStartCmd`: the base image's init swallows
that) and reports over Runpod's HTTP proxy:

    https://<pod-id>-8000.proxy.runpod.net/run.log
    https://<pod-id>-8000.proxy.runpod.net/STATUS
    https://<pod-id>-8000.proxy.runpod.net/DONE        (appears when finished)
    https://<pod-id>-8000.proxy.runpod.net/RUNINFO.json
    https://<pod-id>-8000.proxy.runpod.net/artifacts/...

Why a cheapest-first *walk* rather than one GPU type: the q4_0 weights are
3.35GB, so every card from an 8GB RTX 3070 up holds the model with room for an
8k context -- the only thing that varies is price. But on the community cloud
almost every cheap type sits at "Low" stock, so the cheapest type that is
*actually free right now* is not knowable in advance. The launcher therefore
tries one type at a time in price order and takes the first that accepts a pod,
recording every attempt (including the failures and their errors) in
RUNINFO.json so a write-up can state which GPU was obtainable at that moment
rather than which was theoretically cheapest.

Reads the API key from RUNPOD_KEY and never prints it. Terminate with
scripts/runpod/terminate.py -- pods bill for as long as they exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REST = "https://rest.runpod.io/v1"

# Runpod sits behind Cloudflare, which rejects the default urllib user agent
# with a 403 (error 1010). requests also picks up proxy environment variables.
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "spell-corrector-runpod/1.0",
}

#: (gpu type id, USD/hr, VRAM GB). Exact `id` strings the REST pod-create call
#: expects, from a live GraphQL query on 2026-09-11 -- do not "tidy" these, the
#: API matches them verbatim.
#:
#: Ordered cheapest-first, with two deliberate exceptions:
#:  * Tesla V100 ($0.19) is ranked last despite its price: sm_70, no bf16 and an
#:    old CUDA generation, so it is a fallback rather than a preference.
#:  * RTX A5000 ($0.16) quoted a price but returned no stock entry on a
#:    follow-up query, so it is kept in the walk but never relied on alone.
GPU_CANDIDATES: list[tuple[str, float, int]] = [
    ("NVIDIA GeForce RTX 3070", 0.13, 8),
    ("NVIDIA RTX A5000", 0.16, 24),
    ("NVIDIA GeForce RTX 3080 Ti", 0.18, 12),
    ("NVIDIA RTX 4000 SFF Ada Generation", 0.18, 20),
    ("NVIDIA GeForce RTX 4070 Ti", 0.19, 12),
    ("NVIDIA RTX A4500", 0.19, 20),
    ("NVIDIA RTX 4000 Ada Generation", 0.20, 20),
    ("NVIDIA GeForce RTX 3090", 0.22, 24),
    ("NVIDIA GeForce RTX 3090 Ti", 0.27, 24),
    ("NVIDIA GeForce RTX 4080 SUPER", 0.28, 16),
    ("NVIDIA RTX A6000", 0.33, 48),
    ("NVIDIA GeForce RTX 4090", 0.34, 24),
    ("NVIDIA A40", 0.35, 48),
    # Last resort, see above.
    ("Tesla V100-PCIE-16GB", 0.19, 16),
]

#: CUDA compute capability per family, used to pick -DCMAKE_CUDA_ARCHITECTURES.
#: Building for exactly the card we got keeps the llama.cpp build to a few
#: minutes instead of compiling every architecture.
GPU_CUDA_ARCH = {
    "NVIDIA GeForce RTX 3070": "86",
    "NVIDIA RTX A5000": "86",
    "NVIDIA GeForce RTX 3080 Ti": "86",
    "NVIDIA GeForce RTX 3090": "86",
    "NVIDIA GeForce RTX 3090 Ti": "86",
    "NVIDIA RTX A4500": "86",
    "NVIDIA RTX A6000": "86",
    "NVIDIA A40": "86",
    "NVIDIA GeForce RTX 4070 Ti": "89",
    "NVIDIA GeForce RTX 4080 SUPER": "89",
    "NVIDIA GeForce RTX 4090": "89",
    "NVIDIA RTX 4000 Ada Generation": "89",
    "NVIDIA RTX 4000 SFF Ada Generation": "89",
    "Tesla V100-PCIE-16GB": "70",
}
#: Used when the chosen type is not in the table above (a new card, or a --gpu
#: override): Ampere + Ada covers everything this walk can land on.
DEFAULT_CUDA_ARCH = "86;89"

#: CUDA *devel* image: `-DGGML_CUDA=ON` needs nvcc, which the runtime images do
#: not ship. Python 3.11 (hunspell==0.5.5 builds there with the setuptools<60
#: trick; the py3.12 images need apt python3-hunspell instead -- the bootstrap
#: handles both, but 3.11 is the path this repo has already validated).
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

#: Pinned so a run is reproducible; bumping it is a deliberate act, not a side
#: effect of "whatever master was that morning". If the tag does not exist the
#: bootstrap degrades to the default branch rather than failing, and records the
#: commit it actually built in RUNINFO.json -- so a stale pin costs
#: reproducibility, never the run. Override with --llama-cpp-ref.
DEFAULT_LLAMA_CPP_REF = "b6390"

DEFAULT_GGUF_REPO = "google/gemma-4-E2B-it-qat-q4_0-gguf"
DEFAULT_GGUF_FILE = "gemma-4-E2B_q4_0-it.gguf"


def api(path: str, method: str = "GET", payload: dict | None = None, *, raise_on_error: bool = True):
    key = os.environ.get("RUNPOD_KEY")
    if not key:
        raise SystemExit("RUNPOD_KEY is not set")
    response = requests.request(
        method,
        f"{REST}{path}",
        json=payload,
        headers={**HEADERS, "Authorization": f"Bearer {key}"},
        timeout=120,
    )
    if response.status_code >= 400:
        message = f"runpod {method} {path} failed {response.status_code}: {response.text[:800]}"
        if raise_on_error:
            raise SystemExit(message)
        return {"_error": message, "_status": response.status_code}
    return response.json() if response.text.strip() else {}


def gpu_order(overrides: list[str] | None, max_price: float | None) -> list[tuple[str, float | None]]:
    """The (gpu id, expected USD/hr) walk, cheapest-first.

    `overrides` (--gpu, repeatable) replaces the list entirely and keeps the
    caller's order, so a human can force a specific card without editing code.
    """
    if overrides:
        known = {gpu: price for gpu, price, _ in GPU_CANDIDATES}
        return [(gpu, known.get(gpu)) for gpu in overrides]
    return [(gpu, price) for gpu, price, _ in GPU_CANDIDATES if max_price is None or price <= max_price]


def cuda_arch_for(gpu: str) -> str:
    return GPU_CUDA_ARCH.get(gpu, DEFAULT_CUDA_ARCH)


def create_cheapest_first(
    candidates: list[tuple[str, float | None]],
    build_payload,
    create,
) -> tuple[dict, str, float | None, list[dict]]:
    """Try each GPU type in order; return the first pod that is actually created.

    `build_payload(gpu, price, attempts)` produces the create payload for one
    attempt (it embeds the attempt log so far into the pod's env, so RUNINFO.json
    on the pod can report the failures too). `create(payload)` returns either the
    created pod dict or a dict carrying `_error`.

    Separated from main() and taking its collaborators as arguments so the walk
    itself is unit-testable without touching the Runpod API.
    """
    attempts: list[dict] = []
    for gpu, price in candidates:
        payload = build_payload(gpu, price, attempts)
        result = create(payload)
        if isinstance(result, dict) and result.get("id"):
            attempts.append({"gpu": gpu, "expected_usd_per_hr": price, "result": "created"})
            return result, gpu, price, attempts
        attempts.append(
            {
                "gpu": gpu,
                "expected_usd_per_hr": price,
                "result": "unavailable",
                "error": (result or {}).get("_error", "no pod id in response")[:300],
            }
        )
    raise SystemExit(
        "no GPU type had capacity; attempts:\n"
        + "\n".join(f"  {a['gpu']}: {a.get('error')}" for a in attempts)
    )


def resolve_commit(repo_url: str, branch: str, commit: str | None) -> str:
    """Pin an exact SHA: raw.githubusercontent caches branch paths for minutes,
    so a freshly pushed fix is not necessarily what the pod would fetch."""
    if commit:
        return commit
    slug = repo_url.rstrip("/").removeprefix("https://github.com/")
    ref = requests.get(
        f"https://api.github.com/repos/{slug}/commits/{branch}",
        headers={"Accept": "application/vnd.github.sha", **HEADERS},
        timeout=60,
    )
    if ref.status_code >= 400:
        raise SystemExit(f"cannot resolve {branch}: {ref.status_code} {ref.text[:200]}")
    return ref.text.strip()


def entrypoint_command(bootstrap_url: str) -> list[str]:
    """Small, defensive, and it never exits.

    It starts the progress server itself and retries the bootstrap fetch: on a
    cold container the entrypoint can run before networking is up, and a single
    failed curl would leave a pod idling forever with nothing listening.
    """
    entry = (
        "mkdir -p /workspace/out; "
        "(nohup python3 -m http.server 8000 --directory /workspace/out "
        ">/dev/null 2>&1 &); "
        "for i in $(seq 1 30); do "
        f"curl -fsSL {bootstrap_url} -o /workspace/out/bootstrap.sh && break; "
        'echo "entrypoint: fetch attempt $i failed" >> /workspace/out/run.log; '
        "sleep 10; done; "
        "if [ -s /workspace/out/bootstrap.sh ]; then "
        "bash -x /workspace/out/bootstrap.sh 2>>/workspace/out/boot.err; "
        'echo "entrypoint: bootstrap exited rc=$?" >> /workspace/out/run.log; '
        "else echo 'entrypoint: could not fetch bootstrap' >> /workspace/out/run.log; fi; "
        "sleep infinity"
    )
    return ["/bin/bash", "-lc", entry]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="spell-corrector-gpu-llama")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--disk-gb", type=int, default=40, help="3.35GB GGUF + a llama.cpp build tree fit easily in 40"
    )
    parser.add_argument(
        "--gpu",
        action="append",
        default=None,
        metavar="GPU_TYPE_ID",
        help="override the cheapest-first walk with these types, in this order (repeatable)",
    )
    parser.add_argument("--max-price", type=float, default=None, help="drop candidates above this USD/hr")
    parser.add_argument("--wait", type=int, default=600, help="seconds to wait for the pod to expose ports")
    parser.add_argument("--repo-url", default="https://github.com/ctalau/spell-corrector")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--commit", default=None, help="exact revision (default: branch head)")
    parser.add_argument("--llama-cpp-ref", default=DEFAULT_LLAMA_CPP_REF, help="llama.cpp tag/commit to build")
    parser.add_argument("--gguf-repo", default=DEFAULT_GGUF_REPO)
    parser.add_argument("--gguf-file", default=DEFAULT_GGUF_FILE)
    parser.add_argument(
        "--ctx-per-slot",
        type=int,
        default=1024,
        help=(
            "context tokens per serving slot. llama-server's -c is the TOTAL KV "
            "context shared by all --parallel slots, so the bootstrap passes "
            "-c (ctx-per-slot * parallel); setting -c directly with --parallel 32 "
            "is the classic way to end up with 256 usable tokens per request."
        ),
    )
    parser.add_argument("--parallel", type=int, default=32, help="llama-server --parallel (batched slots)")
    parser.add_argument("--n-samples", type=int, default=100, help="judge sample size per answer mode")
    parser.add_argument(
        "--bootstrap-path",
        default="scripts/runpod/bootstrap_gpu_llama.sh",
        help="repo-relative path to the entrypoint script the pod fetches and runs",
    )
    parser.add_argument("--idle", action="store_true", help="create the pod but do not run anything")
    parser.add_argument(
        "--runinfo-out",
        type=Path,
        default=Path("reports/gpu_llama/RUNINFO.launcher.json"),
        help=(
            "where to write the launcher-side run record. The pod's own "
            "RUNINFO.json can only carry the *expected* price (env is fixed at "
            "create time, and the pod deliberately has no API key), so the price "
            "the API actually charges is recorded here instead."
        ),
    )
    parser.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE", help="extra pod env var, repeatable"
    )
    args = parser.parse_args()

    extra_env = {}
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        extra_env[key] = value

    commit = resolve_commit(args.repo_url, args.branch, args.commit)
    raw_base = args.repo_url.replace("https://github.com/", "https://raw.githubusercontent.com/")
    bootstrap_url = f"{raw_base}/{commit}/{args.bootstrap_path}"
    candidates = gpu_order(args.gpu, args.max_price)
    if not candidates:
        raise SystemExit("no GPU candidates left after filtering; relax --max-price or pass --gpu")

    print(f"commit:    {commit}")
    print(f"bootstrap: {bootstrap_url}")
    print(f"image:     {args.image}")
    print(f"llama.cpp: {args.llama_cpp_ref}")
    print(f"gguf:      {args.gguf_repo}/{args.gguf_file}")
    print("gpu walk (cheapest first):")
    for gpu, price in candidates:
        print(f"  ${price if price is not None else '?':>5}  {gpu}")

    def build_payload(gpu: str, price: float | None, attempts: list[dict]) -> dict:
        payload = {
            "name": args.name,
            "imageName": args.image,
            "gpuTypeIds": [gpu],
            "gpuCount": 1,
            "cloudType": "COMMUNITY",
            "containerDiskInGb": args.disk_gb,
            "volumeInGb": 0,
            "ports": ["8000/http", "22/tcp"],
            "supportPublicIp": True,
            "interruptible": False,
            "env": {
                "REPO_URL": args.repo_url,
                "REPO_BRANCH": args.branch,
                "REPO_COMMIT": commit,
                "CHOSEN_GPU": gpu,
                "GPU_PRICE_USD_HR": "" if price is None else str(price),
                "GPU_ATTEMPT_LOG": json.dumps(attempts),
                "CUDA_ARCHS": cuda_arch_for(gpu),
                "LLAMA_CPP_REF": args.llama_cpp_ref,
                "GGUF_REPO": args.gguf_repo,
                "GGUF_FILE": args.gguf_file,
                "CTX_PER_SLOT": str(args.ctx_per_slot),
                "N_PARALLEL": str(args.parallel),
                "N_SAMPLES": str(args.n_samples),
                **extra_env,
            },
        }
        if not args.idle:
            # Override the entrypoint, not just the command: the base image
            # wraps CMD in its own init script, which swallowed a start command
            # passed through dockerStartCmd and left the container crash-looping.
            payload["dockerEntrypoint"] = entrypoint_command(bootstrap_url)
            payload["dockerStartCmd"] = []
        return payload

    pod, gpu, expected_price, attempts = create_cheapest_first(
        candidates,
        build_payload,
        lambda payload: api("/pods", "POST", payload, raise_on_error=False),
    )
    pod_id = pod["id"]
    print(f"\npod id:    {pod_id}")
    print(f"gpu:       {gpu} (expected ${expected_price}/hr)")
    skipped = [a for a in attempts if a["result"] != "created"]
    if skipped:
        print(f"skipped {len(skipped)} cheaper type(s) with no capacity: {[a['gpu'] for a in skipped]}")

    base = f"https://{pod_id}-8000.proxy.runpod.net"
    deadline = time.time() + args.wait
    while time.time() < deadline:
        info = api(f"/pods/{pod_id}")
        ports = info.get("portMappings") or {}
        ip = info.get("publicIp")
        if ip and ports.get("22"):
            record = {
                "id": pod_id,
                "gpu": gpu,
                "costPerHr": info.get("costPerHr", expected_price),
                "expectedCostPerHr": expected_price,
                "vcpu": info.get("vcpuCount"),
                "ramGb": info.get("memoryInGb"),
                "cudaArchs": cuda_arch_for(gpu),
                "commit": commit,
                "attempts": attempts,
                "ssh": f"ssh root@{ip} -p {ports['22']} -i ~/.ssh/id_ed25519",
                "log": f"{base}/run.log",
                "status": f"{base}/STATUS",
                "done": f"{base}/DONE",
                "runinfo": f"{base}/RUNINFO.json",
                "artifacts": f"{base}/artifacts/",
            }
            args.runinfo_out.parent.mkdir(parents=True, exist_ok=True)
            args.runinfo_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(record, indent=2))
            print(f"\nwrote {args.runinfo_out}")
            actual = info.get("costPerHr")
            if actual is not None and expected_price is not None and abs(float(actual) - expected_price) > 1e-9:
                print(
                    f"note: API charges ${actual}/hr, the pod's RUNINFO.json says ${expected_price}/hr. "
                    f"Re-run the cost math with --gpu-price-usd-hr {actual} if it matters.",
                )
            return 0
        print(f"  status={info.get('desiredStatus')} ip={ip} ...")
        time.sleep(10)
    print(f"pod {pod_id} did not expose ports within {args.wait}s; check {base}/run.log", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
