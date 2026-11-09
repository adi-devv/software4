"""Offline CPU coverage of the V1 wrapper; never downloads weights."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

torch = pytest.importorskip("torch")
from baselines.sleep_hf import HFBackend
from baselines.sleep import SleepConfig
from step4.v1_capture import (ANALYSIS_FIELDS, CaptureBackend, read_cell_npz,
                              restore_stack, to_analysis_records, validate_capture,
                              write_cell_npz)


class Tokenizer:
    chat_template = ""
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 20 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + int(i)) for i in ids if not skip_special_tokens or int(i) > 2)


class TinyModel(torch.nn.Module):
    """Tiny causal stub, with genuine HFBackend adapter injection/save/load."""

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 2, bias=False)
        self.v_proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(2))
            self.v_proj.weight.copy_(torch.eye(2) * 2)
        self.generation_config = SimpleNamespace()
        self.generate_calls = []
        self.fail_forward = False

    def forward(self, input_ids, attention_mask=None, use_cache=False,
                output_hidden_states=False, return_dict=True):
        if self.fail_forward:
            raise RuntimeError("injected forward failure")
        x = input_ids.to(self.q_proj.base.weight.dtype if hasattr(self.q_proj, "base")
                         else self.q_proj.weight.dtype)
        h0 = torch.stack((x, x + 1), dim=-1)
        h1 = self.q_proj(h0)
        h2 = self.v_proj(h1)
        next_ids = (input_ids + 1) % 8 + 2
        logits = torch.zeros((*input_ids.shape, 32), dtype=h2.dtype)
        logits.scatter_(-1, next_ids[..., None], 4)
        return SimpleNamespace(logits=logits, hidden_states=(h0, h1, h2))

    def generate(self, **kwargs):
        self.generate_calls.append({k: v.clone() if torch.is_tensor(v) else v for k, v in kwargs.items()})
        ids = kwargs["input_ids"].clone()
        for _ in range(kwargs["max_new_tokens"]):
            nxt = self(input_ids=ids).logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat((ids, nxt), dim=1)
            if int(nxt.item()) == kwargs["eos_token_id"]:
                break
        self.last_output = ids.clone()
        return ids


def make_backend(path):
    backend = HFBackend(TinyModel(), Tokenizer(), device="cpu", directory=path)
    backend.infer_cap = 4
    backend.answer_prefix = "P:"
    return backend


@pytest.fixture
def backend(tmp_path):
    return make_backend(tmp_path / "working")


def test_capture_identical_generation_and_numerics_including_cache(backend):
    prompt = "long prompt for truncation"
    plain_text = backend.generate(prompt, max_new_tokens=8)
    plain_ids = backend.model.last_output.clone()
    plain_kwargs = backend.model.generate_calls[-1]
    cache_before = dict(backend._cache)
    capture = CaptureBackend(backend, layers=(0, -1), max_new_tokens=8)
    got = capture.capture(prompt, metadata={"task_id": "t"})
    assert got["answer"] == plain_text
    n_prompt = plain_kwargs["input_ids"].shape[1]
    assert n_prompt == 4 + len(backend.tokenizer.encode(backend.answer_prefix))
    np.testing.assert_array_equal(got["token_ids"], plain_ids[0, n_prompt:].numpy())
    assert got["token_ids"][-1] == 2  # emitted EOS retained, though decoding drops it
    assert backend._cache == cache_before
    assert len(backend.model.generate_calls) == 2  # cache was isolated
    captured_kwargs = backend.model.generate_calls[-1]
    assert captured_kwargs.keys() == plain_kwargs.keys()
    for key, value in plain_kwargs.items():
        if torch.is_tensor(value):
            assert torch.equal(value, captured_kwargs[key])
        else:
            assert value == captured_kwargs[key]
    assert not any(k in captured_kwargs for k in ("output_scores", "output_hidden_states"))
    assert got["hidden_last"].dtype == got["hidden_mean"].dtype == np.float16
    assert got["hidden_last"].shape == (2, 2)
    q = got["prompt_token_ids"].astype(float)
    np.testing.assert_array_equal(got["hidden_last"][0], [q[-1], q[-1] + 1])
    np.testing.assert_allclose(got["hidden_mean"][0], [q.mean(), q.mean() + 1], atol=0.01)
    np.testing.assert_allclose(got["hidden_mean"][1], got["hidden_mean"][0] * 2)
    np.testing.assert_array_equal(got["X"], got["hidden_mean"].reshape(-1))
    # Every emitted token has one logit=4 and 31 logits=0, computed independently.
    expected_lp = 4 - np.log(np.exp(4) + 31)
    np.testing.assert_allclose(got["token_logprobs"], expected_lp, atol=1e-6)
    assert got["mean_logprob"] == pytest.approx(expected_lp, abs=1e-6)
    assert got["min_logprob"] == pytest.approx(expected_lp, abs=1e-6)
    p = np.r_[np.exp(4), np.ones(31)] / (np.exp(4) + 31)
    assert got["first_token_entropy"] == pytest.approx(-(p * np.log(p)).sum(), abs=1e-6)
    assert all(layer.active is None for layer in backend.layers.values())


def test_active_expert_and_last_pooling_are_used_and_cleaned_up(backend):
    backend.experts["expert-0"] = {"finite": True, "committed": True}
    for layer in backend.layers.values():
        layer.add("expert-0")
        with torch.no_grad():
            layer.adapters["expert-0"]["a"].fill_(0.1)
            layer.adapters["expert-0"]["b"].fill_(0.1)
    captured = CaptureBackend(backend, layers=(-1,), pooling="last")
    plain = backend.generate("x", expert="expert-0")
    got = captured.capture("x", expert="expert-0")
    assert got["answer"] == plain
    base = captured.capture("x")
    assert not np.array_equal(got["hidden_last"], base["hidden_last"])
    np.testing.assert_array_equal(got["X"], got["hidden_last"].ravel())
    assert all(layer.active is None for layer in backend.layers.values())


def test_capture_restores_observer_cache_and_adapter_on_failure(backend):
    original = backend.model.generate
    cache = backend._cache
    backend.model.fail_forward = True
    with pytest.raises(RuntimeError, match="injected forward failure"):
        CaptureBackend(backend).capture("prompt")
    assert backend._cache is cache
    assert backend.model.generate == original
    assert all(layer.active is None for layer in backend.layers.values())
    backend.model.fail_forward = False
    # Inject failure only in the extra teacher-forced pass.
    original_generate = backend.model.generate

    def generate_then_fail(**kwargs):
        output = original_generate(**kwargs)
        backend.model.fail_forward = True
        return output

    backend.model.generate = generate_then_fail
    with pytest.raises(RuntimeError, match="injected forward failure"):
        CaptureBackend(backend).capture("prompt")
    assert backend._cache is cache
    assert all(layer.active is None for layer in backend.layers.values())


def attempt(task="t", **changes):
    result = dict(task_id=task, seed=44, cycle=0, split="eval", rule="A", code="abcd",
                  answer="AB", token_ids=[3, 4, 2], mean_logprob=-0.4,
                  token_logprobs=[-0.3, -0.4, -0.5], min_logprob=-0.5)
    result.update(changes)
    return result


def test_validate_one_changed_token_invalid_even_if_text_and_label_match():
    saved = [attempt()]
    got = [attempt(token_ids=[3, 5, 2])]
    report = validate_capture(got, saved, logprob_tol=100)
    assert not report["valid"]
    assert report["mismatches"][0]["mismatches"] == [
        {"field": "token_ids", "reason": "tokens differ", "positions": [1]}]


def test_validate_jitter_only_affects_logprobs_and_identity_is_order_independent():
    saved = [attempt(), attempt("u")]
    got = [attempt("u"), attempt(mean_logprob=-0.4001, token_logprobs=[-0.3001, -0.4, -0.5])]
    assert validate_capture(got, saved, 0.001)["valid"]
    assert not validate_capture(got, saved, 0.00001)["valid"]
    got[0]["answer"] += " "
    assert not validate_capture(got, saved, 100)["valid"]


@pytest.mark.parametrize("mutation", ["missing_tokens", "missing_item", "duplicate_item", "short_tokens"])
def test_validate_missing_evidence_and_duplicates_fail_closed(mutation):
    saved, got = [attempt()], [attempt()]
    if mutation == "missing_tokens":
        del saved[0]["token_ids"]
    elif mutation == "missing_item":
        got = []
    elif mutation == "duplicate_item":
        got.append(attempt())
    else:
        got[0]["token_ids"] = [3, 4]
    assert not validate_capture(got, saved, 0.01)["valid"]


def test_validate_nondeterministic_items_are_reported_and_native_day_alias_works():
    saved = [attempt(deterministic=False)]
    got = [attempt(token_ids=[9])]
    report = validate_capture(got, saved, 0)
    assert report["valid"] and report["mismatches"]
    del saved[0]["cycle"]
    del saved[0]["seed"]
    del saved[0]["split"]
    saved[0]["day"] = 0
    assert validate_capture([attempt()], saved, 0)["valid"]
    with pytest.raises(ValueError):
        validate_capture(got, saved, float("inf"))


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_validate_nonfinite_logprobs_never_match(bad):
    assert not validate_capture([attempt(mean_logprob=bad)], [attempt(mean_logprob=bad)], 0.1)["valid"]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def saved_stack(tmp_path, *, expert=True):
    root = tmp_path / "run"
    worker = root / "ours"
    seed, cycle = 44, 0
    backend = make_backend(tmp_path / "original-working")
    if expert:
        backend.experts["expert-0"] = {"finite": True, "committed": True}
        backend._next = 1
        for layer in backend.layers.values():
            layer.add("expert-0")
            with torch.no_grad():
                layer.adapters["expert-0"]["a"].fill_(0.01)
                layer.adapters["expert-0"]["b"].fill_(0.02)
    snapshot = worker / "snapshots" / "sleep-backend-original" / "version-0"
    backend.save(snapshot)
    state = dict(pending={}, events=[], tick=2, config=asdict(SleepConfig()),
                 active_experts=list(backend.experts),
                 centroids={h: [1.0] + [0.0] * 255 for h in backend.experts}, none_centroid=None,
                 accepted_ids=["m000001"] if expert else [], fingerprint=backend.frozen_fingerprint(),
                 backend_snapshot="/kaggle/working/old/ours/snapshots/sleep-backend-original/version-0")
    write_json(root / "run.json", {"config": {"backend": "tiny", "world": "step5", "world_seed": seed}})
    checkpoint = worker / "checkpoints" / "ours__step5-44__0.json"
    write_json(checkpoint, state)
    generation = dict(seed=seed, cycle=cycle, infer_cap=4, answer_prefix="P:", max_new_tokens=16,
                      do_sample=False, eos_token_id=2, pad_token_id=2,
                      model_generation_config={"eos_token_id": 2, "pad_token_id": 2, "do_sample": False})
    write_json(worker / "generation_settings" / "ours__step5-44__0.json", generation)
    (worker / "records.jsonl").write_text(json.dumps(dict(type="cycle_done", arm="ours", stream="step5-44",
                                                         cycle=0, info={"accepted": expert})) + "\n", encoding="utf-8")
    return root, backend, checkpoint


def test_restore_missing_cell_reports_components_without_constructing_model(tmp_path):
    calls = []
    report = restore_stack(tmp_path, 44, 0, backend_factory=lambda *args: calls.append(args))
    assert not report.valid and not calls
    for component in ("run configuration", "checkpoint", "versioned adapter set", "gate", "generation"):
        assert any(component in entry for entry in report.missing)


def test_restore_incomplete_cell_lists_all_missing_state_and_weights(tmp_path):
    root, _, checkpoint = saved_stack(tmp_path)
    state = json.loads(checkpoint.read_text())
    del state["centroids"]
    del state["accepted_ids"]
    write_json(checkpoint, state)
    (root / "ours/snapshots/sleep-backend-original/version-0/expert-0.pt").unlink()
    (root / "ours/generation_settings/ours__step5-44__0.json").unlink()
    report = restore_stack(root, 44, 0, backend_factory=lambda *args: pytest.fail("must not load"))
    assert not report.valid
    for value in ("checkpoint.centroids", "checkpoint.accepted_ids", "expert-0.pt", "generation settings"):
        assert any(value in entry for entry in report.missing)


@pytest.mark.parametrize("expert", [True, False])
def test_restore_exact_version_router_gate_and_generation(tmp_path, expert):
    pytest.importorskip("transformers")
    root, original, _ = saved_stack(tmp_path, expert=expert)
    snapshot_files = {p: p.read_bytes() for p in (root / "ours/snapshots").rglob("*") if p.is_file()}
    with restore_stack(root / "ours", 44, 0,
                       backend_factory=lambda cfg, out: make_backend(out / "adapters")) as report:
        assert report.valid, (report.missing, report.errors)
        assert report.stack_version
        assert report.arm.active_experts == list(original.experts)
        assert report.arm._accepted_ids == ({"m000001"} if expert else set())
        assert report.gate_state["decision"]["accepted"] == expert
        assert report.backend.infer_cap == 4 and report.backend.answer_prefix == "P:"
        wrapper = report.capture_backend(layers=(0, -1))
        assert wrapper.max_new_tokens == report.generation_settings["max_new_tokens"]
        handle = "expert-0" if expert else None
        assert report.backend.generate("same prompt", expert=handle) == original.generate("same prompt", expert=handle)
        if expert:
            np.testing.assert_array_equal(report.arm.centroids[handle], [1] + [0] * 255)
            for name in original.layers:
                for key in ("a", "b"):
                    assert torch.equal(report.backend._state(handle)[name][key], original._state(handle)[name][key])
        writable = report.backend.directory
        assert writable.is_dir()
    assert not writable.exists()
    with pytest.raises(ValueError, match="closed"):
        report.capture_backend()
    assert all(path.read_bytes() == value for path, value in snapshot_files.items())
    # No last-cycle fallback when a different cycle was requested.
    missing = restore_stack(root, 44, 1, backend_factory=lambda *a: pytest.fail("must not load"))
    assert not missing.valid and any("__1.json" in s for s in missing.missing)


def test_restore_rejects_wrong_seed_generation_and_core(tmp_path):
    root, _, _ = saved_stack(tmp_path)
    generation_path = root / "ours/generation_settings/ours__step5-44__0.json"
    generation = json.loads(generation_path.read_text())
    generation["seed"] = 45
    write_json(generation_path, generation)
    report = restore_stack(root, 44, 0)
    assert not report.valid and any("different seed/cycle" in s for s in report.errors)
    generation["seed"] = 44
    write_json(generation_path, generation)

    def wrong_core(cfg, out):
        backend = make_backend(out / "adapters")
        with torch.no_grad():
            backend.layers["q_proj"].base.weight.fill_(0)
        return backend

    report = restore_stack(root, 44, 0, backend_factory=wrong_core)
    assert not report.valid and any("fingerprint differs" in s for s in report.errors)


def test_restore_malformed_metadata_returns_report_without_loading(tmp_path):
    root, _, checkpoint = saved_stack(tmp_path)
    state = json.loads(checkpoint.read_text())
    state["config"] = None
    write_json(checkpoint, state)
    report = restore_stack(root, 44, 0, backend_factory=lambda *args: pytest.fail("must not load"))
    assert not report.valid and any("config has invalid type" in s for s in report.errors)


def test_npz_accepts_analysis_feature_records_without_hidden_arrays(tmp_path):
    record = dict(seed=44, cycle=0, rule="A", split="fit", code="abcd", y=1,
                  mean_logprob=-0.2, stack_version="s0", X=[1.0, 2.0])
    path = tmp_path / "analysis.npz"
    write_cell_npz(path, [record])
    data = to_analysis_records(path)
    assert data["X"].tolist() == [[1.0, 2.0]]
    with pytest.raises(ValueError, match="binary"):
        write_cell_npz(tmp_path / "bad-label.npz", [{**record, "y": 0.5}])


def test_npz_roundtrip_analysis_and_optional_retrieval(tmp_path, backend):
    backend.retrieval_features = {"top_score": 0.75, "hop_count": 2, "deep_search": True, "context_tokens": 20}
    wrapper = CaptureBackend(backend, layers=(0, 2), stack_version="snapshot-0")
    records = [wrapper.capture("prompt", metadata=dict(seed=44, cycle=0, rule="A", split=split,
                                                       code="abcd", y=y))
               for split, y in (("fit", 1), ("eval", 0))]
    del records[1]["top_score"]
    del records[1]["hop_count"]
    del records[1]["deep_search"]
    del records[1]["context_tokens"]
    records[1]["token_ids"] = records[1]["token_ids"][:1]
    records[1]["token_logprobs"] = records[1]["token_logprobs"][:1]
    path = tmp_path / "seed44-cycle0.npz"
    write_cell_npz(path, records)
    got = read_cell_npz(path)
    for field in ("X", "hidden_last", "hidden_mean"):
        np.testing.assert_array_equal(got[field], np.stack([r[field] for r in records]))
        assert got[field].dtype == np.float16
    assert got["stack_version"].tolist() == ["snapshot-0"] * 2
    assert got["top_score"][0] == 0.75 and np.isnan(got["top_score"][1])
    assert got["hop_count"].tolist() == [2, -1]
    assert got["deep_search"].tolist() == [1, -1]
    assert got["context_tokens"].tolist() == [20, -1]
    for i, record in enumerate(records):
        lo, hi = got["token_offsets"][i:i + 2]
        np.testing.assert_array_equal(got["token_ids"][lo:hi], record["token_ids"])
        np.testing.assert_array_equal(got["token_logprobs"][lo:hi], record["token_logprobs"])
        lo, hi = got["prompt_token_offsets"][i:i + 2]
        np.testing.assert_array_equal(got["prompt_token_ids"][lo:hi], record["prompt_token_ids"])
    assert all(a.dtype != object for a in got.values())
    analysis = to_analysis_records([got, path])
    assert set(analysis) == set(ANALYSIS_FIELDS)
    assert analysis["X"].shape == (4, 4)
    assert analysis["y"].tolist() == [1, 0, 1, 0]
    assert analysis["seed"].tolist() == [44] * 4
    with pytest.raises(ValueError, match="one seed-cycle"):
        write_cell_npz(tmp_path / "mixed.npz", [records[0], {**records[1], "cycle": 1}])


@pytest.mark.skip(reason="Requires original registered v1 model, versioned stacks and emitted-token reference logs")
def test_registered_real_model_capture_validation():
    """All nine real seed-cycle cells must be validated before a V1 PASS."""
