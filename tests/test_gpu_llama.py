"""GPU llama.cpp track: the pure pieces that can be checked without a GPU.

The local box is 4 vCPU with no CUDA, so nothing here talks to a pod, a
llama-server or the Runpod API. What it does cover is exactly the logic that
would otherwise only be exercised for the first time while a pod is billing:
the cheapest-first GPU walk, the cost arithmetic, the parsers that read
llama-server's timings and startup log, and the synthetic prompt builder (which
must never reach for the locked benchmark).

Modules are loaded from file so importing them pulls in neither torch,
transformers nor Hunspell.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load("benchmark_llama_server_test", "scripts/benchmark_llama_server.py")
launch = _load("launch_gpu_llama_test", "scripts/runpod/launch_gpu_llama.py")
backend = _load("llama_cpp_backend_test", "spelling_reranker/llama_cpp_backend.py")


# --------------------------------------------------------------- GPU walk


def test_gpu_order_is_cheapest_first_with_v100_last():
    order = launch.gpu_order(None, None)
    prices = [price for _, price in order]
    assert order[0][0] == "NVIDIA GeForce RTX 3070"
    # Tesla V100 is $0.19 but deliberately ranked last (sm_70, no bf16).
    assert order[-1][0] == "Tesla V100-PCIE-16GB"
    assert prices[:-1] == sorted(prices[:-1])


def test_gpu_order_respects_overrides_and_max_price():
    assert launch.gpu_order(["NVIDIA GeForce RTX 4090"], None) == [("NVIDIA GeForce RTX 4090", 0.34)]
    cheap = launch.gpu_order(None, 0.18)
    assert all(price <= 0.18 for _, price in cheap)
    assert "NVIDIA GeForce RTX 4090" not in [gpu for gpu, _ in cheap]


def test_gpu_ids_are_unique_and_priced():
    ids = [gpu for gpu, _, _ in launch.GPU_CANDIDATES]
    assert len(ids) == len(set(ids))
    assert all(isinstance(price, float) for _, price, _ in launch.GPU_CANDIDATES)


def test_every_candidate_fits_the_q4_0_weights():
    # 3.35GB of weights: the walk must not contain a card that cannot hold them.
    assert all(vram >= 8 for _, _, vram in launch.GPU_CANDIDATES)


def test_cuda_arch_per_gpu_and_broad_default():
    assert launch.cuda_arch_for("NVIDIA GeForce RTX 3090") == "86"
    assert launch.cuda_arch_for("NVIDIA GeForce RTX 4090") == "89"
    assert launch.cuda_arch_for("Tesla V100-PCIE-16GB") == "70"
    assert launch.cuda_arch_for("NVIDIA H999") == launch.DEFAULT_CUDA_ARCH


def test_create_cheapest_first_takes_the_first_type_with_capacity():
    candidates = launch.gpu_order(None, None)
    seen: list[str] = []

    def create(payload):
        gpu = payload["gpuTypeIds"][0]
        seen.append(gpu)
        if gpu != "NVIDIA GeForce RTX 4070 Ti":
            return {"_error": "no capacity", "_status": 400}
        return {"id": "pod123"}

    pod, gpu, price, attempts = launch.create_cheapest_first(
        candidates, lambda g, p, a: {"gpuTypeIds": [g], "env": {}}, create
    )
    assert pod["id"] == "pod123"
    assert gpu == "NVIDIA GeForce RTX 4070 Ti"
    assert price == 0.19
    # Every cheaper type was tried first, and the failures are recorded.
    assert seen[: len(seen) - 1] == [g for g, _ in candidates[: len(seen) - 1]]
    assert [a["result"] for a in attempts][-1] == "created"
    assert all(a["result"] == "unavailable" and a["error"] for a in attempts[:-1])


def test_create_cheapest_first_gives_up_with_a_readable_error():
    try:
        launch.create_cheapest_first(
            [("NVIDIA GeForce RTX 3070", 0.13)],
            lambda g, p, a: {"gpuTypeIds": [g]},
            lambda payload: {"_error": "no instances"},
        )
    except SystemExit as exc:
        assert "no GPU type had capacity" in str(exc)
        assert "NVIDIA GeForce RTX 3070" in str(exc)
    else:  # pragma: no cover - the walk must not silently succeed
        raise AssertionError("expected SystemExit")


def test_attempt_log_is_passed_into_the_payload_builder():
    """RUNINFO.json must be able to report which cheaper types were unavailable."""
    logs: list[int] = []

    def build(gpu, price, attempts):
        logs.append(len(attempts))
        return {"gpuTypeIds": [gpu], "env": {"GPU_ATTEMPT_LOG": json.dumps(attempts)}}

    launch.create_cheapest_first(
        [("a", 0.1), ("b", 0.2), ("c", 0.3)],
        build,
        lambda payload: {"id": "x"} if payload["gpuTypeIds"][0] == "c" else {"_error": "nope"},
    )
    assert logs == [0, 1, 2]


def test_entrypoint_overrides_entrypoint_not_start_cmd():
    cmd = launch.entrypoint_command("https://raw.example/abc/bootstrap.sh")
    assert cmd[:2] == ["/bin/bash", "-lc"]
    # The progress server must come up before the bootstrap is even fetched.
    assert "http.server 8000" in cmd[2]
    assert "sleep infinity" in cmd[2]
    assert "https://raw.example/abc/bootstrap.sh" in cmd[2]


# ------------------------------------------------------------ timings parser


def test_parse_timings_reads_server_block():
    out = bench.parse_timings(
        {
            "timings": {
                "prompt_n": 220,
                "prompt_ms": 20.0,
                "prompt_per_second": 11000.0,
                "predicted_n": 50,
                "predicted_ms": 500.0,
                "predicted_per_second": 100.0,
            }
        }
    )
    assert out["prompt_tokens"] == 220
    assert out["prompt_tok_per_s"] == 11000.0
    assert out["predicted_tok_per_s"] == 100.0


def test_parse_timings_derives_missing_rates():
    out = bench.parse_timings({"timings": {"prompt_n": 100, "prompt_ms": 50.0, "predicted_n": 10, "predicted_ms": 200.0}})
    assert round(out["prompt_tok_per_s"], 3) == 2000.0
    assert round(out["predicted_tok_per_s"], 3) == 50.0


def test_parse_timings_absent_is_empty_not_zero():
    assert bench.parse_timings({"choices": []}) == {}
    assert bench.parse_timings({"timings": {}}) == {}


# ----------------------------------------------------------- latency summary


def test_latency_summary_percentiles():
    stats = bench.latency_summary([i / 1000.0 for i in range(1, 101)])
    assert stats["n"] == 100
    assert stats["p50_ms"] == 50.0
    assert stats["p90_ms"] == 90.0
    assert stats["p99_ms"] == 99.0
    assert stats["max_ms"] == 100.0
    assert round(stats["mean_ms"], 1) == 50.5


def test_latency_summary_empty():
    assert bench.latency_summary([]) == {"n": 0}


# --------------------------------------------------------- server log parser


SERVER_LOG = """
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
load_tensors: offloading 30 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 31/31 layers to GPU
load_tensors:        CUDA0 model buffer size =  3200.00 MiB
load_tensors:   CPU_Mapped model buffer size =   525.00 MiB
llama_kv_cache_unified:      CUDA0 KV buffer size =   480.00 MiB
llama_context:      CUDA0 compute buffer size =   300.00 MiB
main: server is listening on http://127.0.0.1:8080 - starting the main loop
"""


def test_parse_server_log_reports_full_offload():
    info = bench.parse_server_log(SERVER_LOG)
    assert info["layers_offloaded"] == 31
    assert info["layers_total"] == 31
    assert info["full_offload"] is True
    assert info["gpu_name"] == "NVIDIA GeForce RTX 3090"
    assert info["compute_capability"] == "8.6"
    assert info["cuda_buffer_total_mib"] == 3980.0
    assert info["server_listening"] is True
    # The CPU_Mapped buffer is deliberately not counted as VRAM.
    assert set(info["cuda_buffers_mib"]) == {"CUDA0"}


def test_parse_server_log_flags_partial_offload():
    info = bench.parse_server_log("load_tensors: offloaded 12/31 layers to GPU\n")
    assert info["full_offload"] is False
    assert info["layers_offloaded"] == 12


def test_parse_server_log_of_a_cpu_only_run():
    info = bench.parse_server_log("main: server is listening\n")
    assert info["full_offload"] is False
    assert info["layers_offloaded"] is None
    assert info["cuda_buffer_total_mib"] is None


# ------------------------------------------------------------------- cost


def test_cost_per_1000_corrections():
    # 10 req/s on a $0.36/hr card => $0.0001 per request => $0.10 per 1,000.
    assert round(bench.cost_per_1000_corrections(10.0, 0.36), 6) == 0.01
    assert bench.cost_per_1000_corrections(0.0, 0.36) is None
    assert bench.cost_per_1000_corrections(10.0, None) is None


def test_cost_scales_inversely_with_throughput():
    cheap = bench.cost_per_1000_corrections(20.0, 0.22)
    dear = bench.cost_per_1000_corrections(10.0, 0.22)
    assert round(dear / cheap, 6) == 2.0


def test_resolve_gpu_price_prefers_runinfo_then_table():
    runinfo = {"chosen_gpu": "NVIDIA GeForce RTX 3090", "cost_per_hr": 0.25}
    name, price, source = bench.resolve_gpu_price(runinfo, None, None)
    assert (name, price) == ("NVIDIA GeForce RTX 3090", 0.25)
    assert "RUNINFO" in source

    name, price, source = bench.resolve_gpu_price(None, "NVIDIA GeForce RTX 4090", None)
    assert price == bench.GPU_PRICE_USD_HR["NVIDIA GeForce RTX 4090"]
    assert "static table" in source

    # An explicit flag beats everything, because prices move.
    assert bench.resolve_gpu_price(runinfo, None, 0.99)[1] == 0.99
    assert bench.resolve_gpu_price(None, "NVIDIA MADE-UP", None)[1] is None


def test_benchmark_price_table_agrees_with_the_launcher():
    for gpu, price, _ in launch.GPU_CANDIDATES:
        assert bench.GPU_PRICE_USD_HR[gpu] == price


# ---------------------------------------------------------------- prompts


def test_prompts_are_synthetic_and_deterministic():
    pairs = bench.load_misspelling_pairs(ROOT / "data" / "wikipedia_misspellings.txt")
    assert len(pairs) > 100
    a = bench.build_workload_prompts(pairs, "short_answer", 6, seed=7)
    b = bench.build_workload_prompts(pairs, "short_answer", 6, seed=7)
    assert a == b
    assert all(p["max_tokens"] == 8 for p in a)
    assert all(p["messages"][0]["role"] == "system" for p in a)
    assert "<TYPO>" in a[0]["messages"][1]["content"]


def test_sentence_workload_asks_for_more_tokens():
    pairs = bench.load_misspelling_pairs(ROOT / "data" / "wikipedia_misspellings.txt")
    prompts = bench.build_workload_prompts(pairs, "sentence_rewrite", 4, seed=7)
    assert all(p["max_tokens"] == 64 for p in prompts)
    assert "corrected_sentence" in prompts[0]["messages"][0]["content"]


def test_prompt_source_never_touches_the_locked_benchmark():
    """A throughput loop is repeated measurement; BEA-60K is evaluation only.

    The docstring explains that rule, so the check is on what the code *does*:
    it must not import the benchmark loader or point at its data directory.
    """
    text = (ROOT / "scripts" / "benchmark_llama_server.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ).lower()
    for banned in ("bea60k.py", "spelling_reranker.bea60k", "data/bea60k", "load_bea_pairs", "neuspell"):
        assert banned not in code, f"benchmark reaches for the locked benchmark via {banned!r}"
    # Positive half: prompts come from the vendored Wikipedia list.
    assert "wikipedia_misspellings.txt" in text


def test_load_misspelling_pairs_falls_back_when_absent(tmp_path):
    pairs = bench.load_misspelling_pairs(tmp_path / "nope.txt")
    assert pairs == bench.FALLBACK_PAIRS


# ------------------------------------------------------------- attach mode


def test_normalize_base_url_accepts_root_or_v1():
    assert backend.normalize_base_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert backend.normalize_base_url("http://127.0.0.1:8080/") == "http://127.0.0.1:8080"
    assert backend.normalize_base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"
    assert backend.normalize_base_url(" http://h:8080/v1/ ") == "http://h:8080"


def test_attached_model_never_kills_a_server_it_does_not_own():
    model = backend.LlamaCppModel(model_id="x", base_url="http://127.0.0.1:8080")
    assert model.process is None
    model.stop()  # must be a no-op, not an AttributeError


def test_render_markdown_survives_missing_numbers():
    md = bench.render_markdown(
        {
            "model": "gemma-4-e2b-q4",
            "environment": {"gpu_name": None, "usd_per_hr": None},
            "latency": {"short_answer": {"latency": {"n": 0}}},
            "concurrency": {"short_answer": [{"concurrency": 1, "latency": {}}]},
            "cost": {"short_answer": {}},
        }
    )
    assert "n/a" in md
    assert "llama-server GPU benchmark" in md


# ------------------------------------------------- judge CLI: attach vs spawn

_MISSING_GGUF = "requires --gguf"


def _load_judge_cli():
    """The judge script imports torch and Hunspell; skip cleanly without them."""
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("hunspell")
    return _load("llm_judge_bea60k_cpu_test", "scripts/llm_judge_bea60k_cpu.py")


def _run_judge_main(judge, argv: list[str]) -> str:
    """Run main() far enough to see its argument validation, and report why it stopped.

    main() cannot complete here -- it needs the locked benchmark on disk and a
    live server -- so the assertion is about *which* failure comes back.
    """
    import contextlib
    import io

    saved = sys.argv
    sys.argv = ["llm_judge_bea60k_cpu.py", *argv]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            judge.main()
        return ""
    except BaseException as exc:  # noqa: BLE001 - the message is the result
        return str(exc)
    finally:
        sys.argv = saved


class _Args:
    """Stand-in for the judge's parsed argparse namespace."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _judge_args(**overrides):
    base = dict(
        llama_base_url=None,
        gguf=None,
        llama_server_binary=None,
        llama_server_port=8080,
        llama_threads=None,
        model_id="google/gemma-4-E2B-it-qat-q4_0-gguf",
        output=Path("/tmp/unused"),
    )
    base.update(overrides)
    return _Args(**base)


def test_judge_attach_mode_does_not_demand_a_gguf(monkeypatch):
    judge = _load_judge_cli()
    seen = {}
    monkeypatch.setattr(
        judge, "attach_llama_cpp", lambda url, *, model_id: seen.update(url=url, model_id=model_id) or "MODEL"
    )
    monkeypatch.setattr(
        judge, "load_llama_cpp", lambda *a, **k: pytest_fail("spawned a server in attach mode")
    )
    loaded = judge.load_llama_backend(_judge_args(llama_base_url="http://127.0.0.1:8080/v1"))
    assert loaded == "MODEL"
    assert seen["url"] == "http://127.0.0.1:8080/v1"


def pytest_fail(msg):  # pragma: no cover - only reached on a regression
    raise AssertionError(msg)


def test_judge_spawn_mode_still_demands_a_gguf():
    """The default behaviour -- spawn a private server -- must be unchanged."""
    judge = _load_judge_cli()
    try:
        judge.load_llama_backend(_judge_args())
    except SystemExit as exc:
        assert _MISSING_GGUF in str(exc)
        assert "--llama-base-url" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit")


def test_judge_spawn_mode_is_still_the_default_path(monkeypatch, tmp_path):
    judge = _load_judge_cli()
    calls = {}
    monkeypatch.setattr(judge, "load_llama_cpp", lambda gguf, **kw: calls.update(gguf=gguf, **kw) or "SPAWNED")
    loaded = judge.load_llama_backend(
        _judge_args(gguf=tmp_path / "m.gguf", llama_server_binary=tmp_path / "llama-server", output=tmp_path)
    )
    assert loaded == "SPAWNED"
    assert calls["server_binary"] == tmp_path / "llama-server"
    assert calls["log_path"] == tmp_path / "llama_server.log"


def test_judge_rejects_beam_mode_on_llama_cpp(tmp_path):
    """index/open/sentence are portable to this backend; beam is not."""
    judge = _load_judge_cli()
    message = _run_judge_main(
        judge,
        [
            "--backend", "llama-cpp",
            "--llama-base-url", "http://127.0.0.1:9",
            "--answer-mode", "beam",
            "--model-id", "x",
            "--model-name", "y",
            "--output", str(tmp_path / "out"),
        ],
    )
    assert "beam" in message and "transformers" in message
