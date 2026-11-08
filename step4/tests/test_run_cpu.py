"""CPU-only integration tests: default Python needs numpy/pytest, never torch."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from step4 import run


@pytest.fixture
def cpu_dir():
    # Force every newly written artifact to live under step4/ (including tests).
    import tempfile
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".run-") as folder:
        yield Path(folder)


def invoke(folder, *, smoke="0", arms=",".join(run.ARMS), backend="fake", script=None, extra=None):
    env = dict(os.environ, S4_BACKEND=backend, S4_SMOKE=smoke, S4_ARMS=arms,
               S4_CYCLES="3", S4_STEPS="2", S4_OUT=str(folder / "out"),
               CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               S4_SOURCE_CACHE=str(folder / "source"))
    env.pop("S4_WORKER_CONFIG", None)
    if extra:
        env.update(extra)
    proc = subprocess.run([sys.executable, "-B", str(script or ROOT / "step4/run.py")],
                          env=env, cwd=folder, capture_output=True, text=True, timeout=90)
    if proc.returncode:
        proc.stderr += "\n".join(p.read_text(encoding="utf-8") for p in (folder / "out").glob("*/*.log"))
    paths = list((folder / "out").glob(("smoke-*" if smoke == "1" else "full-*") + "/results.json"))
    payload = json.loads(paths[0].read_text(encoding="utf-8")) if paths else None
    return proc, payload, paths[0].parent if paths else None


def test_fake_full_and_reuse(cpu_dir):
    proc, payload, out = invoke(cpu_dir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert set(payload["results"]) == set(run.ARMS)
    assert payload["gpu_hours"] == 0
    assert payload["comparison"]["status"] == "unavailable"
    for name, result in payload["results"].items():
        assert result["status"] == "complete"
        assert result["cuda_visible_devices"] == ""
        assert result["sealed_reads"] == 0
        assert not result["split_audit"]
        assert {s["cycle"] for s in result["scores"]} == {-1, 0, 1, 2}
        assert (out / name / "backend/backend.json").is_file()
        answers = [json.loads(line) for line in (out / name / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
        assert answers and all(row["arm"] == name and isinstance(row["prompt"], str)
                               and isinstance(row["retrieved"], list) and isinstance(row["answer"], str)
                               for row in answers)
        assert all({"task_id", "day", "family", "skill", "expert"} <= row.keys() for row in answers)
        # Every arm reports parse failures per cycle over the full hidden set.
        assert set(result["parse_by_cycle"]) == {"0", "1", "2"}
        assert all(t["answers"] == sum(1 for s in result["scores"]
                   if s["phase"] == "hidden" and s["cycle"] == int(c))
                   for c, t in result["parse_by_cycle"].items())
    # Fair budget everywhere except the starved row; ours is the first arm scheduled.
    assert run.ARMS[0] == "ours" and payload["config"]["context_tokens"] == 512
    assert payload["config"]["starved_tokens"] == 64 and payload["config"]["world"] == "step4"
    sleeps = payload["results"]["ours"]["sleeps"]
    assert [s["cycle"] for s in sleeps] == [0, 1, 2]
    assert all("routing" in s and "forced_train_accuracy" in s for s in sleeps)
    assert payload["results"]["no_memory"]["sleeps"][0].get("routing") is None
    assert payload["results"]["ours"]["training"]
    assert all(c["steps"] == 2 for c in payload["results"]["ours"]["training"])
    checkpoints = list((out / "ours/checkpoints").glob("*.json"))
    for path in checkpoints:
        state = json.loads(path.read_text(encoding="utf-8"))
        assert Path(state["backend_snapshot"]).is_relative_to(out)
        assert (Path(state["backend_snapshot"]) / "backend.json").is_file()
    mtimes = {p: p.stat().st_mtime_ns for p in out.glob("*/result.json")}
    proc2, payload2, out2 = invoke(cpu_dir)
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr
    assert out2 == out
    assert payload2["results"] == payload["results"]
    assert mtimes == {p: p.stat().st_mtime_ns for p in mtimes}
    assert (out / "RESULTS.md").is_file()
    assert (out / "timings.json").is_file()


@pytest.mark.parametrize("error", [OSError("disk full"), TypeError("bad JSON")])
def test_answer_log_failure_does_not_abort_scoring(cpu_dir, monkeypatch, error):
    monkeypatch.setenv("S4_BACKEND", "fake")
    monkeypatch.setenv("S4_CYCLES", "3")
    monkeypatch.setenv("S4_STEPS", "2")
    cfg, _, _, _ = run.settings(ROOT)
    original_open = Path.open
    original_dumps = json.dumps
    def failing_open(path, *args, **kwargs):
        if path.name == 'answers.jsonl' and isinstance(error, OSError):
            raise error
        return original_open(path, *args, **kwargs)
    def failing_dumps(value, *args, **kwargs):
        if isinstance(error, TypeError) and isinstance(value, dict) and 'task_id' in value and 'answer' in value:
            raise error
        return original_dumps(value, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', failing_open)
    monkeypatch.setattr(json, 'dumps', failing_dumps)
    assert run.worker('no_memory', cfg, cpu_dir, False) == 0
    result = json.loads((cpu_dir / 'no_memory' / 'result.json').read_text(encoding='utf-8'))
    assert result['status'] == 'complete' and result['answer_log_failures'] > 0
    assert result['scores'] and result['parse_by_cycle']


def test_salient_is_opt_in_and_runs(cpu_dir, monkeypatch):
    assert run.ARMS == ("ours", "expert_only", "hippo_only", "rag", "text_notes",
                        "no_memory", "rag_starved", "gold_context")
    monkeypatch.delenv("S4_ARMS", raising=False)
    assert run.settings(ROOT)[1] == list(run.ARMS)
    proc, payload, out = invoke(cpu_dir, arms="ours_salient")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert payload["results"]["ours_salient"]["status"] == "complete"
    assert "ours_salient" in (out / "RESULTS.md").read_text(encoding="utf-8")


def test_smoke_trains_only_longest_twice(cpu_dir):
    proc, payload, out = invoke(cpu_dir, smoke="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ours = payload["results"]["ours"]
    assert ours["smoke_only"]
    assert ours["longest_batch"]["optimizer_steps"] == 2
    call, = ours["training"]
    assert call["steps"] == 2 and len(call["examples"]) == 4
    assert all(e == call["examples"][0] for e in call["examples"])
    backend = run.make_backend(payload["config"], out)
    stream, _, _, _, prompt_for = run.import_world(payload["config"])
    n, prompt, target, tid = run.longest_example(stream, backend, prompt_for)
    assert tid == ours["longest_batch"]["task_id"]
    assert n == ours["longest_batch"]["encoded_tokens"]
    assert call["examples"][0] == [prompt, target]
    assert payload["comparison"]["status"] == "unavailable"


def test_failed_baseline_does_not_block_ours(cpu_dir, monkeypatch):
    # `ours` starts first and waits for nobody: a retained baseline failure only fails the run.
    monkeypatch.setenv("S4_BACKEND", "fake")
    monkeypatch.setenv("S4_STEPS", "2")
    monkeypatch.setenv("S4_CYCLES", "3")
    _, _, key, _ = run.settings(ROOT)
    out = cpu_dir / "out" / ("full-" + key)
    run.write_json(out / "rag/result.json", dict(arm="rag", status="failed",
                   error="CPU fixture simulates baseline crash", elapsed_seconds=0))
    proc, payload, _ = invoke(cpu_dir, arms="rag,ours,oracle")
    assert proc.returncode == 1
    assert payload["results"]["rag"]["status"] == "failed"
    assert payload["results"]["ours"]["status"] == "complete"
    assert payload["results"]["oracle"]["status"] == "complete"
    assert payload["comparison"]["status"] == "unavailable"


def test_interrupted_cycle_restores_snapshot(cpu_dir):
    proc, payload, out = invoke(cpu_dir, arms="ours")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    arm_dir = out / "ours"
    original_training = payload["results"]["ours"]["training"]
    records = [json.loads(line) for line in (arm_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    # Simulate a crash after scoring the last retention cycle, before cycle_done.
    records = [r for r in records if not (r["type"] == "run_summary" or
               (r["type"] == "cycle_done" and r["cycle"] == 2))]
    (arm_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (arm_dir / "result.json").unlink()
    resumed, payload2, _ = invoke(cpu_dir, arms="ours")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    result = payload2["results"]["ours"]
    # Completed cycles remain reused. Removing cycle_done makes that cycle
    # incomplete; a candidate rejected there may legitimately train again.
    assert result["training"][:len(original_training)] == original_training
    assert all(call["seed"] == 2 for call in result["training"][len(original_training):])
    def score_values(rows):
        return [(r["cycle"], r["task_id"], r["correct"]) for r in rows]
    assert score_values(result["scores"]) == score_values(payload["results"]["ours"]["scores"])
    for phase in ("hidden", "canary"):
        before = [r["correct"] for r in payload["results"]["ours"]["scores"] if r["cycle"] == 2 and r["phase"] == phase]
        after = [r["correct"] for r in result["scores"] if r["cycle"] == 2 and r["phase"] == phase]
        assert before == after  # incomplete attempt scores excluded, no double counting
    assert len(result["sleeps"]) == 3
    assert result["parse_by_cycle"] == payload["results"]["ours"]["parse_by_cycle"]  # no double count


def test_upload_script_is_standalone(cpu_dir):
    from step4.prepare import prepare
    target = prepare(target=cpu_dir / "prepared")
    # Execute only the uploaded script from a fresh directory. All original
    # imports must come from its frozen archive, not an accidental repository cwd.
    script = cpu_dir / "run.py"
    script.write_bytes((target / "run.py").read_bytes())
    proc, payload, _ = invoke(cpu_dir, smoke="1", script=script)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert all(r["status"] == "complete" for r in payload["results"].values())
    assert list((cpu_dir / "source").glob("*/scripts/smoke_sleep.py"))


def test_full_hf_requires_successful_smoke_before_setup(cpu_dir):
    # This must fail before torch import, pip installation or any network call.
    proc, payload, _ = invoke(cpu_dir, backend="hf", arms="ours")
    assert proc.returncode != 0
    assert "run matching S4_SMOKE=1 first" in proc.stderr
    assert payload is None


def test_random_tiny_hf_cpu(cpu_dir):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("optional: torch absent in default Python 3.13; fake path has no torch dependency")
    pytest.importorskip("transformers")
    proc, payload, _ = invoke(cpu_dir, smoke="1", arms="ours", backend="tiny")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ours = payload["results"]["ours"]
    assert ours["device"] == "cpu"
    call, = ours["training"]
    assert call["finite"] and call["optimizer_steps"] == 2 and call["skipped_steps"] == 0
    assert ours["peak_gpu_mib"] is None
    assert ours["kernel_environment"]["status"] == "verified"
    assert ours["kernel_environment"]["policy"] == "torch_reference_only"
    full, payload2, _ = invoke(cpu_dir, smoke="0", arms="ours", backend="tiny", extra={"S4_CYCLES": "3"})
    assert full.returncode == 0, full.stdout + full.stderr
    result = payload2["results"]["ours"]
    assert len(result["sleeps"]) == 3 and result["training"]
    assert all(c["finite"] and c["skipped_steps"] == 0 for c in result["training"])


def test_random_hybrid_targets_cpu(cpu_dir):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("optional: torch absent")
    import torch
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from step4.hf import make_hf_backend
    # No hub calls. The tiny hybrid keeps one four-layer repeat (3 linear, 1 full).
    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(vocab_size=256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        eos_token_id=2, pad_token_id=2, bos_token_id=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]})
    model = Qwen3_5ForCausalLM(config)
    # Bound synthetic decay rates for both releases. v5.3.0's random dt_bias=1
    # can overflow intermediate exp gradients in its reference chunk algorithm.
    # This is a test fixture, not a change to any pretrained model/production path.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".dt_bias"):
                parameter.fill_(-4)
    backend = make_hf_backend(dict(backend="tiny", lr=2e-4), cpu_dir, model=model)
    # This reference-dispatch/adapter check uses FP32 on CPU for BOTH releases.
    # The 5.3.0 reference l2norm normalizes tiny random Q/K in fp16 before
    # upcasting and can overflow RsqrtBackward. Production keeps fp16 and must
    # pass its own pretrained T4 smoke with zero skipped steps.
    backend.model.float()
    assert backend.kernel_environment["status"] == "verified"
    functions = backend.kernel_environment["deltanet_functions"]
    assert len(functions) == 3
    assert all(row["norm"].endswith(".Qwen3_5RMSNormGated") for row in functions.values())
    assert len(backend.layers) == 8
    assert sum("linear_attn" in n for n in backend.layers) == 6
    assert sum("self_attn" in n for n in backend.layers) == 2
    before = backend.frozen_fingerprint()
    handle = backend.train_expert([("Context:\n\n\nTask:\nApply code abcd.", "Final answer: DCBA")], 2, 0)
    assert backend.is_finite(handle)
    assert backend.train_calls[-1]["optimizer_steps"] == 2
    backend.commit_expert(handle)
    backend.save(cpu_dir / "saved")
    backend.load(cpu_dir / "saved")
    assert backend.frozen_fingerprint() == before
    assert isinstance(backend.generate("Context:\n\n\nTask:\nApply code efgh.", handle), str)
