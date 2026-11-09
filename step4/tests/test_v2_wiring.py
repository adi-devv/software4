"""Local CPU wiring checks; semantic inference uses an explicit test encoder."""
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from baselines import Rag
from baselines import semantic
from baselines.generic import GenericForcedNoRetArm, GenericForcedSleepArm
from hippocampus import HashingEmbedder
from step4 import run, v2_analyze
from step4.analyze import load_v1_run
from step4.prepare import prepare


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    for key, value in dict(S4_BACKEND="fake", S4_WORLD="step5", S4_WORLD_SEED="48",
                           S4_ARMS=",".join(run.V2_ARMS), S4_CYCLES="3", S4_STEPS="2").items():
        monkeypatch.setenv(key, value)
    return run.settings(ROOT)[0]


def inject_semantic_test_encoder(monkeypatch):
    cls = semantic.SemanticEmbedder
    monkeypatch.setattr(semantic, "SemanticEmbedder", lambda: cls(
        allow_unpinned=True, encoder=semantic.StubEncoder(dim=64)))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_arm_registration_and_unchanged_defaults(monkeypatch, cfg):
    assert run.ARMS == ("ours", "expert_only", "hippo_only", "rag", "text_notes",
                        "no_memory", "rag_starved", "gold_context")
    assert run.V2_ARMS == ("ours", "expert_only", "hippo_only", "rag_semantic", "rag_hashing",
                           "generic_forced_noret", "ours_generic_forced", "text_notes",
                           "no_memory", "gold_context")
    assert set(run.V2_ARMS) == set(v2_analyze.V2_ARMS)
    assert set(run.V2_ARMS) <= set(run.AVAILABLE_ARMS)
    assert len(set(run.AVAILABLE_ARMS)) == len(run.AVAILABLE_ARMS)
    assert run.settings(ROOT)[1] == list(run.V2_ARMS)
    monkeypatch.delenv("S4_ARMS")
    assert run.settings(ROOT)[1] == list(run.ARMS)
    stream, _, oracle, _, _ = run.import_world(cfg)
    backend = run.make_backend(cfg, None)
    for name, cls in (("ours_generic_forced", GenericForcedSleepArm),
                      ("generic_forced_noret", GenericForcedNoRetArm)):
        arm = run.make_arm(name, backend, cfg, oracle, stream)
        assert isinstance(arm, cls) and arm.name == name
        assert arm.cfg.steps == cfg["steps"] and arm.cfg.context_tokens == 512


def test_hashing_alias_has_identical_v1_behavior(cfg):
    from harness.tasks import arm_scope, grading_scope
    stream, _, oracle, _, _ = run.import_world(cfg)
    backend = run.make_backend(cfg, None)
    v1, v2 = [run.make_arm(name, backend, cfg, oracle, stream) for name in ("rag", "rag_hashing")]
    assert type(v1) is type(v2) is Rag
    assert type(v2.embedder) is HashingEmbedder and v2.name == "rag_hashing"
    assert run.retriever_config(v1) == run.retriever_config(v2)
    assert v1.k == v2.k == 5 and v1.context_tokens == v2.context_tokens == 512
    for item in stream.train_items(0)[:12]:
        with grading_scope():
            correction = stream.correction_for(item) if item.kind == "probe" else None
        with arm_scope():
            for arm in (v1, v2):
                arm.observe(item.task("day", 0), "Final answer: unknown", False, correction)
    for item in stream.hidden_items(0)[:4]:
        with arm_scope():
            assert v1.answer(item.task("test", 0)) == v2.answer(item.task("test", 0))
        assert v1.answer_trace == v2.answer_trace


def test_semantic_factory_requires_path_and_pins(cfg, monkeypatch, tmp_path):
    backend = run.make_backend(cfg, None)
    monkeypatch.delenv(semantic.MODEL_PATH_ENV, raising=False)
    with pytest.raises(ValueError, match="S4_SEMANTIC_MODEL_PATH"):
        run.make_arm("rag_semantic", backend, cfg, None)
    monkeypatch.setenv(semantic.MODEL_PATH_ENV, str(tmp_path))
    (tmp_path / semantic.WEIGHTS_FILENAME).write_bytes(b"not the registered weights")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run.make_arm("rag_semantic", backend, cfg, None)
    monkeypatch.setattr(semantic, "MODEL_REVISION", "TODO_PIN")
    with pytest.raises(ValueError, match="pins"):
        run.make_arm("rag_semantic", backend, cfg, None)


def test_semantic_factory_records_verified_local_identity(cfg, monkeypatch, tmp_path):
    # Constructor verification only; fake bytes are never loaded as a model.
    weights = b"local constructor fixture"
    (tmp_path / semantic.WEIGHTS_FILENAME).write_bytes(weights)
    monkeypatch.setattr(semantic, "WEIGHTS_SHA256", hashlib.sha256(weights).hexdigest())
    monkeypatch.setenv(semantic.MODEL_PATH_ENV, str(tmp_path))
    arm = run.make_arm("rag_semantic", run.make_backend(cfg, None), cfg, None)
    assert type(arm) is Rag and arm.name == "rag_semantic"
    info = run.retriever_config(arm)
    assert info["embedder"] == "SemanticEmbedder" and info["pin_verified"]
    assert info["model_name"] == semantic.MODEL_NAME
    assert info["model_revision"] == semantic.MODEL_REVISION
    assert info["weights_sha256"] == semantic.WEIGHTS_SHA256
    assert info["device"] == "cpu" and not info["test_encoder"]
    assert info["k"] == 5 and info["context_tokens"] == 512


@pytest.mark.parametrize("seed", [48, 49, 50, 51, 52])
def test_v2_seed_settings_and_report_dispatch(seed, cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("S4_WORLD_SEED", str(seed))
    config, _, _, _ = run.settings(ROOT)
    assert config["protocol"] == "v2" and run.is_v2_run(config)
    run.report(tmp_path, config, [], {}, False, 0, run.detect_hardware(config))
    assert v2_analyze.load_v2_run(tmp_path) == (seed, {})
    assert (tmp_path / "v2-verdict.json").is_file()
    assert not (tmp_path / "v1-verdict.json").exists()
    with pytest.raises(ValueError, match="v1 seeds"):
        load_v1_run(tmp_path)


@pytest.mark.parametrize("seed", [44, 45, 46])
def test_v1_report_dispatch_preserved(seed, cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("S4_WORLD_SEED", str(seed))
    monkeypatch.setenv("S4_ARMS", ",".join(run.ARMS))
    config, _, _, _ = run.settings(ROOT)
    assert not run.is_v2_run(config) and "protocol" not in config
    assert "semantic_retriever" not in config
    run.report(tmp_path, config, [], {}, False, 0, run.detect_hardware(config))
    assert load_v1_run(tmp_path) == (seed, {})
    assert (tmp_path / "v1-verdict.json").is_file()
    assert not (tmp_path / "v2-verdict.json").exists()
    with pytest.raises(ValueError, match="v2 seeds"):
        v2_analyze.load_v2_run(tmp_path)


def test_v2_cycles_and_model_pin_reuse_keys(cfg, monkeypatch):
    monkeypatch.setenv("S4_CYCLES", "4")
    with pytest.raises(ValueError, match="exactly 3 cycles"):
        run.settings(ROOT)
    monkeypatch.setenv("S4_CYCLES", "3")
    _, _, key, smoke_key = run.settings(ROOT)
    monkeypatch.setenv("S4_WORLD_SEED", "49")
    assert run.settings(ROOT)[2] != key and run.settings(ROOT)[3] == smoke_key
    monkeypatch.setattr(semantic, "MODEL_REVISION", "1" * 40)
    config, _, new_key, new_smoke = run.settings(ROOT)
    assert config["semantic_retriever"]["model_revision"] == "1" * 40
    assert new_key != key and new_smoke != smoke_key


@pytest.mark.parametrize("name", ["ours_generic_forced", "generic_forced_noret"])
def test_new_training_arm_smoke_trains_longest_batch(name, cfg, tmp_path):
    assert run.worker(name, cfg, tmp_path, True) == 0
    result = read_json(tmp_path / name / "result.json")
    assert result["smoke_only"] and result["status"] == "complete"
    assert result["longest_batch"]["optimizer_steps"] == 2
    call, = result["training"]
    assert call["steps"] == 2 and len(call["examples"]) == 4
    assert all(example == call["examples"][0] for example in call["examples"])
    stream, _, _, _, prompt_for = run.import_world(cfg)
    _, prompt, target, tid = run.longest_example(stream, run.make_backend(cfg, None), prompt_for)
    assert result["longest_batch"]["task_id"] == tid
    assert call["examples"][0] == [prompt, target]


def test_all_v2_workers_load_end_to_end(cfg, monkeypatch, tmp_path):
    inject_semantic_test_encoder(monkeypatch)
    results = {}
    for name in run.V2_ARMS:
        assert run.worker(name, cfg, tmp_path, False) == 0, name
        results[name] = read_json(tmp_path / name / "result.json")
        assert (tmp_path / name / "records.jsonl").is_file()
        assert (tmp_path / name / "answers.jsonl").is_file()
    payload = run.report(tmp_path, cfg, list(run.V2_ARMS), results, False, 0,
                         run.detect_hardware(cfg))
    seed, loaded = v2_analyze.load_v2_run(tmp_path)
    assert seed == 48 and set(loaded) == set(run.V2_ARMS)
    assert payload["gpu_hours"] == 0
    for name, result in loaded.items():
        assert result["status"] == "complete" and result["answer_log_failures"] == 0
        assert result["sealed_reads"] == 0 and not result["split_audit"]
        for endpoint in v2_analyze.ENDPOINTS:
            assert len(v2_analyze._vector(result, endpoint)) == (360 if endpoint == "pooled" else 120)
        hidden = [row for row in result["scores"] if row["phase"] == "hidden"]
        assert all(type(row["correct_exact"]) is bool and not row.get("evidence_error") for row in hidden)
        assert len(result["fact_answers"]) == 18
        assert all(type(n) is int and n >= 0 for n in result["context_tokens"].values())
    assert loaded["rag_semantic"]["retriever"]["test_encoder"]
    assert not loaded["rag_semantic"]["retriever"]["pin_verified"]
    assert loaded["rag_hashing"]["retriever"]["embedder"] == "HashingEmbedder"
    for name in ("ours", "expert_only", "ours_generic_forced", "generic_forced_noret"):
        assert len(loaded[name]["sleeps"]) == 3
        for night in loaded[name]["sleeps"]:
            assert type(night["examples"]) is int
            assert night["steps"] == (cfg["steps"] if night["examples"] else 0)
    for name in ("ours_generic_forced", "generic_forced_noret"):
        nights = loaded[name]["sleeps"]
        trained = [night for night in nights if "generic_train_accuracy" in night]
        assert trained and all(night["accepted"] and night["generic_train_examples"] > 0 for night in trained)
        assert all("checks" in night and "train_fit" in night for night in trained)
    assert not loaded["generic_forced_noret"]["retriever"]["enabled"]
    assert all(n == 1 for n in loaded["generic_forced_noret"]["context_tokens"].values())
    verdict = read_json(tmp_path / "v2-verdict.json")
    assert not all(verdict["claims"].values())  # one fake seed is never a registered pass
    assert "case-folded" in (tmp_path / "v2-verdict.md").read_text(encoding="utf-8")


def test_v2_retry_discards_only_unfinished_answer_cycles(cfg, tmp_path):
    name = "no_memory"
    assert run.worker(name, cfg, tmp_path, False) == 0
    out = tmp_path / name
    original_answers = (out / "answers.jsonl").read_bytes()
    records = [json.loads(line) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [row for row in records if row["type"] != "run_summary" and
               not (row["type"] == "cycle_done" and row["cycle"] == 2)]
    (out / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    assert run.worker(name, cfg, tmp_path, False) == 0
    assert (out / "answers.jsonl").read_bytes() == original_answers
    result = read_json(out / "result.json")
    run.report(tmp_path, cfg, [name], {name: result}, False, 0, run.detect_hardware(cfg))
    _, loaded = v2_analyze.load_v2_run(tmp_path)
    assert len(v2_analyze._vector(loaded[name], "pooled")) == 360
    # Genuine duplicates in a completed cycle must survive cleanup and invalidate evidence.
    with (out / "answers.jsonl").open("a", encoding="utf-8") as stream:
        row = next(json.loads(line) for line in original_answers.decode().splitlines()
                   if json.loads(line)["day"] == 2 and json.loads(line)["family"] == "rule")
        stream.write(json.dumps(row) + "\n")
    run.discard_unfinished_v2_answers(out)
    _, loaded = v2_analyze.load_v2_run(tmp_path)
    with pytest.raises(ValueError, match="evidence"):
        v2_analyze._vector(loaded[name], "pooled")


def test_v2_package_and_digest_cover_new_modules(cfg, tmp_path):
    target = prepare(world="step5", world_seed=48, arms=",".join(run.V2_ARMS), target=tmp_path / "prepared")
    tree = ast.parse((target / "run.py").read_text(encoding="utf-8"))
    packed = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "SOURCE_B64" for t in node.targets))
    unpacked = tmp_path / "source"
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(packed))) as archive:
        assert {"baselines/semantic.py", "step4/v2_analyze.py"} <= set(archive.namelist())
        archive.extractall(unpacked)
    digest = run.source_digest(unpacked)
    assert digest == run.source_digest(ROOT)
    for filename in ("baselines/semantic.py", "step4/v2_analyze.py"):
        path = unpacked / filename
        original = path.read_bytes()
        path.write_bytes(original + b"\n# digest coverage\n")
        assert run.source_digest(unpacked) != digest
        path.write_bytes(original)
    # Standalone prepared script: new worker choices, v2 seeds, report and packaged imports.
    env = dict(os.environ, S4_BACKEND="fake", S4_SMOKE="0", S4_ARMS="rag_hashing,generic_forced_noret",
               S4_OUT=str(tmp_path / "out"), S4_SOURCE_CACHE=str(tmp_path / "cache"),
               CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-I", "-B", str(target / "run.py")], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    output, = (tmp_path / "out").glob("full-*")
    seed, arms = v2_analyze.load_v2_run(output)
    assert seed == 48 and set(arms) == {"rag_hashing", "generic_forced_noret"}
    assert all(len(v2_analyze._vector(result, "pooled")) == 360 for result in arms.values())


def _run_setup(monkeypatch, tmp_path, cfg):
    """Drive setup_runtime on CPU with pip, the kernel check and the hub all stubbed."""
    import huggingface_hub
    import subprocess
    import step4.hf
    calls = []
    monkeypatch.setattr(run.Path, "is_dir", lambda self: True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(step4.hf, "require_reference_kernels", lambda run_dir: {})

    def fake_download(repo, revision=None, allow_patterns=None, **kw):
        calls.append((repo, revision, tuple(allow_patterns)))
        assert os.environ["HF_HUB_OFFLINE"] == "0"  # fetched before offline is forced
        return str(tmp_path / ("snap-" + repo.replace("/", "_")))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    verified = []
    monkeypatch.setattr(semantic, "SemanticEmbedder", lambda model_path=None: verified.append(model_path))
    monkeypatch.delenv(semantic.MODEL_PATH_ENV, raising=False)
    run.setup_runtime(tmp_path, cfg, tmp_path)
    return calls, verified


def test_v2_setup_fetches_pinned_minilm_and_sets_path(monkeypatch, tmp_path):
    cfg = dict(model="Qwen/x", revision="r1", world="step5", world_seed=48)
    calls, verified = _run_setup(monkeypatch, tmp_path, cfg)
    assert calls[0] == ("Qwen/x", "r1", ("*.json", "*.jinja", "*.txt", "*.model", "*.safetensors"))
    assert calls[1] == (semantic.MODEL_NAME, semantic.MODEL_REVISION, semantic.MODEL_FILES)
    expected = str(tmp_path / ("snap-" + semantic.MODEL_NAME.replace("/", "_")))
    assert os.environ[semantic.MODEL_PATH_ENV] == expected
    assert verified == [expected]  # SHA check ran against the downloaded snapshot
    monkeypatch.delenv(semantic.MODEL_PATH_ENV)


def test_v1_setup_does_not_touch_minilm(monkeypatch, tmp_path):
    cfg = dict(model="Qwen/x", revision="r1", world="step5", world_seed=46)
    calls, verified = _run_setup(monkeypatch, tmp_path, cfg)
    assert len(calls) == 1 and calls[0][0] == "Qwen/x"
    assert semantic.MODEL_PATH_ENV not in os.environ and verified == []
