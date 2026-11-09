"""Separate V1 capture pass; nothing in the v1 driver imports this module.

Generation is delegated to HFBackend.generate with its original arguments. A
temporary observer records the *unmodified* model.generate inputs/output (even
EOS); no generation flags are added. The text cache is temporarily isolated,
since it stores no token IDs. A subsequent teacher-forced, use_cache=False
forward on exactly prompt+emitted tokens supplies raw model logprobs and core
hidden states. Thus forward-pass numeric differences cannot change emitted
tokens. Logprobs include emitted EOS, exclude the fixed answer_prefix prefill,
and are NOT probabilities after generation processors. Prompt pooling includes
the formatted/truncated prompt AND that prefill. Layers use HF hidden_states
indices (0 = embeddings, -1 = final state), in the caller's order.

Restoration accepts a native run directory (containing ours/) or its ours/
directory. It requires run.json, checkpoints/ours__step5-SEED__CYCLE.json,
the exact backend_snapshot named there (backend.json plus every expert .pt),
and a completed cycle in records.jsonl. Router state is centroids,
none_centroid, active_experts and config.threshold in the checkpoint; deployed
gate state is accepted_ids/config plus that cycle's recorded sleep decision.
No gate is reopened and no sleep is rerun. The original full memory event log
is restored too, so retrieval still uses the same stack.

Since v1 does not explicitly serialize generation settings, additionally require
ours/generation_settings/ours__step5-SEED__CYCLE.json with fields:
seed, cycle, infer_cap, answer_prefix, max_new_tokens, do_sample (false),
pad_token_id, eos_token_id, model_generation_config (full HF config dict).
These must come from the original run, never be inferred from today's defaults.
Missing evidence is INVALID, not permission to fabricate a sidecar. Absolute
Kaggle snapshot paths are relocated by their exact snapshots/ suffix only;
there is no search for a newer/other snapshot. The snapshot JSON digest is the
stack_version; backend.load verifies the frozen core fingerprint.

NPZ schema 1 (no pickle): per-row seed, cycle, rule, split, code, y,
mean_logprob, min_logprob, first_token_entropy, stack_version, answer,
top_score, hop_count, deep_search, context_tokens; X is [N,F] float16;
hidden_last/hidden_mean are [N,L,D] float16; layer_indices is [L] int64.
X defaults to hidden_mean flattened layer-major (or capture's chosen pooling).
Missing retrieval data: top_score=NaN, other fields=-1 (deep_search: 0/1/-1).
token_ids and token_logprobs are flat arrays indexed by token_offsets [N+1];
prompt_token_ids has its own prompt_token_offsets. schema_version is scalar.
read_cell_npz returns these arrays. to_analysis_records returns ONLY aligned
seed, cycle, rule, split, code, y, mean_logprob, X arrays, as v1_analysis expects.
Feature selection must be frozen before scored-seed capture.

Capture temporarily observes a shared model method/cache: do not concurrently
use the same backend. Restoration writes only a disposable working backend;
saved snapshots are never modified. Close the RestoreReport when done.
Use report.capture_backend(layers=...) to retain its saved token limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
from unittest.mock import patch

import numpy as np


ANALYSIS_FIELDS = ("seed", "cycle", "rule", "split", "code", "y", "mean_logprob", "X")
RETRIEVAL_FIELDS = ("top_score", "hop_count", "deep_search", "context_tokens")


class CaptureBackend:
    """Compose an existing HFBackend; capture() returns an attempt dict.

    generate() retains the backend's string-returning interface, allowing this
    wrapper to be used by an existing arm. Its latest record is last_capture.
    Metadata (seed/cycle/rule/split/code/y) can be supplied to capture().
    """

    def __init__(self, backend, layers=(-1,), *, pooling="mean", stack_version="",
                 max_new_tokens=16):
        if not layers or any(not isinstance(i, int) for i in layers):
            raise ValueError("layers must be a nonempty sequence of integer HF indices")
        if pooling not in ("mean", "last"):
            raise ValueError("pooling must be mean or last")
        self.backend, self.layers, self.pooling = backend, tuple(layers), pooling
        self.stack_version, self.max_new_tokens = stack_version, max_new_tokens
        self.last_capture = None

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def generate(self, prompt, expert=None, max_new_tokens=None):
        return self.capture(prompt, expert, max_new_tokens=max_new_tokens)["answer"]

    def capture(self, prompt, expert=None, *, max_new_tokens=None, metadata=None,
                retrieval_features=None):
        import torch

        backend = self.backend
        observed = []
        original_generate = backend.model.generate

        def observe(*args, **kwargs):
            inputs = kwargs["input_ids"].detach().clone()
            output = original_generate(*args, **kwargs)
            # HFBackend itself expects a tensor, not a GenerateOutput object.
            observed.append((inputs, output.detach().clone()))
            return output

        cache = backend._cache
        backend._cache = {}
        try:
            with patch.object(backend.model, "generate", observe):
                answer = backend.generate(prompt, expert=expert, max_new_tokens=(
                    self.max_new_tokens if max_new_tokens is None else max_new_tokens))
        finally:
            backend._cache = cache
        if len(observed) != 1:
            raise RuntimeError("capture requires exactly one uncached model.generate call")
        prompt_ids, full_ids = observed[0]
        n_prompt = prompt_ids.shape[1]
        if (prompt_ids.shape[0] != 1 or not n_prompt or full_ids.ndim != 2
                or full_ids.shape[0] != 1
                or not torch.equal(full_ids[:, :n_prompt], prompt_ids)):
            raise RuntimeError("expected one nonempty decoder-only prompt and its continuation")
        emitted = full_ids[:, n_prompt:]
        backend._activate(expert)
        try:
            with torch.inference_mode(), backend._autocast():
                result = backend.model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                                       use_cache=False, output_hidden_states=True, return_dict=True)
            hidden = result.hidden_states
            last = torch.stack([hidden[i][0, n_prompt - 1].float() for i in self.layers])
            mean = torch.stack([hidden[i][0, :n_prompt].float().mean(0) for i in self.layers])
            # Position p-1 predicts emitted token at p. Never score the prefill.
            logits = result.logits[0, n_prompt - 1:n_prompt + emitted.shape[1] - 1].float()
            log_probs = logits.log_softmax(-1)
            token_lp = log_probs.gather(-1, emitted[0, :, None]).squeeze(-1)
            first_lp = result.logits[0, n_prompt - 1].float().log_softmax(-1)
            entropy = -torch.special.xlogy(first_lp.exp(), first_lp.exp()).sum()
            record = dict(metadata or {})
            record.update(answer=answer, token_ids=emitted[0].cpu().numpy().astype(np.int64),
                          prompt_token_ids=prompt_ids[0].cpu().numpy().astype(np.int64),
                          layer_indices=np.asarray(self.layers, dtype=np.int64),
                          hidden_last=last.half().cpu().numpy(),
                          hidden_mean=mean.half().cpu().numpy(),
                          token_logprobs=token_lp.cpu().numpy(),
                          mean_logprob=float(token_lp.mean()) if token_lp.numel() else float("nan"),
                          min_logprob=float(token_lp.min()) if token_lp.numel() else float("nan"),
                          first_token_entropy=float(entropy), stack_version=self.stack_version)
            record["X"] = record["hidden_" + self.pooling].reshape(-1).copy()
            exposed = getattr(backend, "retrieval_features", {})
            exposed = exposed() if callable(exposed) else exposed
            features = {k: getattr(backend, k) for k in RETRIEVAL_FIELDS if hasattr(backend, k)}
            features.update(exposed or {})
            features.update(retrieval_features or {})
            record.update({k: features[k] for k in RETRIEVAL_FIELDS if k in features})
            self.last_capture = record
            return record
        finally:
            backend._activate(None)


@dataclass
class RestoreReport:
    seed: int
    cycle: int
    valid: bool = False
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stack_version: str | None = None
    backend: object = None
    arm: object = None
    router_state: dict = field(default_factory=dict)
    gate_state: dict = field(default_factory=dict)
    generation_settings: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    _scratch: object = field(default=None, repr=False)

    def capture_backend(self, layers=(-1,), *, pooling="mean"):
        if not self.valid or self._scratch is None:
            raise ValueError("cannot capture an invalid or closed restored stack")
        return CaptureBackend(self.backend, layers, pooling=pooling,
                              stack_version=self.stack_version,
                              max_new_tokens=self.generation_settings["max_new_tokens"])

    def close(self):
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def restore_stack(out_dir, seed, cycle, *, backend_factory=None):
    """Return RestoreReport; never initialize a model for an incomplete cell.

    backend_factory(cfg, writable_directory) defaults to step4.hf.make_hf_backend.
    It exists for offline tests/injected models, not as a stack fallback.
    Failures, including weight/fingerprint/load errors, leave valid=False.
    """
    report = RestoreReport(int(seed), int(cycle))
    root = Path(out_dir)
    worker = root if root.name == "ours" else root / "ours"
    run_root = worker.parent
    stem = f"ours__step5-{seed}__{cycle}"
    paths = {"run": run_root / "run.json", "checkpoint": worker / "checkpoints" / (stem + ".json"),
             "gate_state": worker / "records.jsonl",
             "generation_settings": worker / "generation_settings" / (stem + ".json")}
    report.paths = {k: str(p) for k, p in paths.items()}

    def read_json(path, component):
        if not path.is_file():
            report.missing.append(f"{component}: {path}")
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected JSON object")
            return value
        except (OSError, ValueError) as exc:
            report.errors.append(f"{component}: {exc}")
            return None

    run = read_json(paths["run"], "run configuration")
    state = read_json(paths["checkpoint"], "versioned checkpoint (router/gate/memory)")
    generation = read_json(paths["generation_settings"], "generation settings")
    decision = None
    if not paths["gate_state"].is_file():
        report.missing.append(f"gate state/completed cycle: {paths['gate_state']}")
    else:
        try:
            for line in paths["gate_state"].read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # same crash-tail convention as the runner
                if (row.get("type") == "cycle_done" and row.get("arm") == "ours"
                        and row.get("stream") == f"step5-{seed}" and row.get("cycle") == cycle):
                    decision = row.get("info")
            if not isinstance(decision, dict) or "accepted" not in decision:
                report.missing.append(f"gate decision/completed cycle: seed={seed}, cycle={cycle}")
        except OSError as exc:
            report.errors.append(f"gate state: {exc}")
    cfg = (run or {}).get("config", {})
    if not isinstance(cfg, dict):
        report.errors.append("run configuration must be an object")
        cfg = {}
    if run is not None and (cfg.get("world") != "step5" or cfg.get("world_seed") != seed):
        report.errors.append("run configuration does not identify the requested step5 seed")
    snapshot = None
    if state is not None:
        for name, expected_type in (("config", dict), ("centroids", dict), ("active_experts", list),
                                    ("accepted_ids", list), ("events", list)):
            if name in state and not isinstance(state[name], expected_type):
                report.errors.append(f"checkpoint.{name} has invalid type")
                state[name] = expected_type()  # continue auditing, never load this invalid cell
        required = ("backend_snapshot", "fingerprint", "active_experts", "centroids", "none_centroid",
                    "accepted_ids", "config", "events", "pending", "tick")
        report.missing.extend(f"checkpoint.{k}" for k in required if k not in state)
        gate_config = ("threshold", "min_trust", "replay_flip_fraction")
        report.missing.extend(f"checkpoint.config.{k}" for k in gate_config
                              if k not in state.get("config", {}))
        raw = state.get("backend_snapshot")
        if raw:
            # Use only the exact named version, relocated within this bundle.
            parts = PurePosixPath(str(raw).replace("\\", "/")).parts
            if "snapshots" in parts:
                snapshot = worker / "snapshots" / Path(*parts[parts.index("snapshots") + 1:])
            elif not Path(raw).is_absolute():
                snapshot = worker / "snapshots" / raw
            else:
                report.errors.append("backend_snapshot has no portable snapshots/ suffix")
        else:
            report.missing.append("versioned adapter set: checkpoint.backend_snapshot")
    else:
        report.missing.append("versioned adapter set: no checkpoint snapshot reference")
    if snapshot is not None:
        report.paths["adapter_snapshot"] = str(snapshot)
        backend_state = read_json(snapshot / "backend.json", "versioned adapter metadata")
        if backend_state is not None:
            for k in ("experts", "next", "fingerprint"):
                if k not in backend_state:
                    report.missing.append(f"backend.json.{k}")
            experts = backend_state.get("experts", {})
            if not isinstance(experts, dict):
                report.errors.append("backend.json.experts must be an object")
                experts = {}
            for handle in experts:
                if not (snapshot / (handle + ".pt")).is_file():
                    report.missing.append(f"adapter {handle}: {snapshot / (handle + '.pt')}")
            active = set(state.get("active_experts", []))
            if active != set(experts) or active != set(state.get("centroids", {})):
                report.errors.append("checkpoint router/deployed expert set differs from versioned adapter set")
            if state.get("fingerprint") != backend_state.get("fingerprint"):
                report.errors.append("checkpoint/backend frozen core fingerprints differ")
            report.stack_version = hashlib.sha256(json.dumps(
                {"checkpoint": state, "backend": backend_state, "generation": generation},
                sort_keys=True).encode()).hexdigest()
    if generation is not None:
        required = ("seed", "cycle", "infer_cap", "answer_prefix", "max_new_tokens", "do_sample",
                    "pad_token_id", "eos_token_id", "model_generation_config")
        report.missing.extend(f"generation_settings.{k}" for k in required if k not in generation)
        if generation.get("seed") != seed or generation.get("cycle") != cycle:
            report.errors.append("generation settings identify a different seed/cycle")
        if generation.get("do_sample") is not False:
            report.errors.append("HFBackend requires do_sample=False")
        for name in ("infer_cap", "max_new_tokens"):
            value = generation.get(name)
            if name in generation and (type(value) is not int or value < 1):
                report.errors.append(f"generation_settings.{name} must be a positive integer")
        if "answer_prefix" in generation and not isinstance(generation["answer_prefix"], str):
            report.errors.append("generation_settings.answer_prefix must be a string")
        if "model_generation_config" in generation and (
                not isinstance(generation["model_generation_config"], dict)
                or not generation["model_generation_config"]):
            report.errors.append("generation_settings.model_generation_config must be a nonempty full config")
    if report.missing or report.errors:
        return report
    try:
        from baselines.sleep import SleepArm, SleepConfig
        from transformers import GenerationConfig
        if backend_factory is None:
            from step4.hf import make_hf_backend
            backend_factory = make_hf_backend
        report._scratch = tempfile.TemporaryDirectory(prefix="v1-capture-")
        writable = Path(report._scratch.name)
        backend = backend_factory(cfg, writable)
        if (generation["eos_token_id"] != backend.tokenizer.eos_token_id
                or generation["pad_token_id"] != backend.tokenizer.eos_token_id):
            raise ValueError("saved generation EOS/pad IDs differ from HFBackend tokenizer")
        backend.infer_cap = generation["infer_cap"]
        backend.answer_prefix = generation["answer_prefix"]
        backend.model.generation_config = GenerationConfig.from_dict(generation["model_generation_config"])
        arm = SleepArm(backend, SleepConfig(**state["config"]), llm=backend,
                       snapshot_dir=writable / "snapshots")
        relocated = {**state, "backend_snapshot": str(snapshot.resolve())}
        arm.load_state(relocated)  # existing fingerprint + adapter loading implementation
        report.backend, report.arm = backend, arm
        report.router_state = {k: state[k] for k in ("centroids", "none_centroid", "active_experts")}
        report.gate_state = {"accepted_ids": state["accepted_ids"], "config": state["config"],
                             "decision": decision}
        report.generation_settings = generation
        report.valid = True
    except Exception as exc:
        report.errors.append(f"stack load failed: {type(exc).__name__}: {exc}")
        report.close()
    return report


def validate_capture(captured, saved_v1_answers, logprob_tol):
    """Exact token AND text validation, aligned by task identity, never by zip.

    Supply lists of attempt dicts with answer and token_ids. Legacy text-only
    v1 logs fail closed: re-encoding decoded text cannot prove emitted IDs.
    Identity is (seed, cycle/day, split, task_id) if task_id exists, otherwise
    (seed, cycle, split, rule, code). Fields absent from *all* saved rows (e.g.
    seed/split in native answer logs) are omitted from the identity on both sides.
    Logprobs are compared if saved (token_logprobs, mean_logprob, min_logprob),
    using absolute tolerance only. deterministic defaults to True. All item
    mismatches are returned, including explicitly nondeterministic items.
    """
    if not math.isfinite(logprob_tol) or logprob_tol < 0:
        raise ValueError("logprob_tol must be a finite nonnegative absolute tolerance")
    captured, saved = list(captured), list(saved_v1_answers)
    identity = ["seed", "cycle", "split"]
    identity += ["task_id"] if any("task_id" in r for r in saved) else ["rule", "code"]
    identity = [k for k in identity if any(k in r or k == "cycle" and "day" in r for r in saved)]
    if not saved or not identity or not any(k in identity for k in ("task_id", "code")):
        return {"valid": False, "items": [], "mismatches": [{"field": "identity", "reason": "no saved task identities"}]}

    def key(row):
        return tuple(row.get(k, row.get("day") if k == "cycle" else None) for k in identity)

    groups = {}
    for side, rows in (("captured", captured), ("saved", saved)):
        for row in rows:
            groups.setdefault(key(row), {"captured": [], "saved": []})[side].append(row)
    items = []
    for task_key, pair in groups.items():
        mismatch = []
        deterministic = True
        if len(pair["captured"]) != 1 or len(pair["saved"]) != 1:
            mismatch.append({"field": "identity", "reason": "missing or duplicate item",
                             "captured_count": len(pair["captured"]), "saved_count": len(pair["saved"])})
        else:
            got, expected = pair["captured"][0], pair["saved"][0]
            deterministic = expected.get("deterministic", True) is not False
            for name in ("answer", "token_ids"):
                if name not in got or name not in expected:
                    mismatch.append({"field": name, "reason": "missing evidence"})
                elif name == "answer":
                    if got[name] != expected[name]:
                        mismatch.append({"field": name, "reason": "text differs"})
                else:
                    a, b = np.asarray(got[name]), np.asarray(expected[name])
                    if (a.ndim != 1 or b.ndim != 1
                            or a.dtype.kind not in "iu" or b.dtype.kind not in "iu"
                            or not np.array_equal(a, b)):
                        positions = [i for i in range(max(a.size, b.size))
                                     if i >= a.size or i >= b.size or a.flat[i] != b.flat[i]]
                        mismatch.append({"field": name, "reason": "tokens differ", "positions": positions})
            for name in ("token_logprobs", "mean_logprob", "min_logprob"):
                if name in expected:
                    a = np.asarray(got.get(name, float("nan")), dtype=float)
                    b = np.asarray(expected[name], dtype=float)
                    if (a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
                            or not np.allclose(a, b, atol=logprob_tol, rtol=0, equal_nan=False)):
                        mismatch.append({"field": name, "reason": "logprob differs or is missing/nonfinite"})
        items.append({"key": task_key, "deterministic": deterministic, "mismatches": mismatch,
                      "valid": not mismatch or not deterministic})
    return {"valid": all(i["valid"] for i in items), "items": items,
            "mismatches": [i for i in items if i["mismatches"]]}


def write_cell_npz(path, records):
    """Write a nonempty single seed-cycle; exact array names are in module docs."""
    records = list(records)
    if not records:
        raise ValueError("a cell must contain records")
    if len({(r["seed"], r["cycle"]) for r in records}) != 1:
        raise ValueError("one seed-cycle per NPZ")
    arrays = {"schema_version": np.asarray(1, dtype=np.int64)}
    for k in ("seed", "cycle", "y"):
        if k == "y" and not np.isin([r[k] for r in records], (0, 1)).all():
            raise ValueError("y must be binary")
        arrays[k] = np.asarray([r[k] for r in records], dtype=np.int64)
    for k in ("rule", "split", "code", "stack_version"):
        arrays[k] = np.asarray([r[k] for r in records], dtype=str)
    if not np.isin(arrays["split"], ("fit", "eval")).all():
        raise ValueError("split must be fit or eval")
    arrays["answer"] = np.asarray([r.get("answer", "") for r in records], dtype=str)
    for k in ("mean_logprob", "min_logprob", "first_token_entropy", "top_score"):
        arrays[k] = np.asarray([r[k] if k == "mean_logprob" else r.get(k, np.nan)
                                for r in records], dtype=np.float32)
    for k in ("hop_count", "deep_search", "context_tokens"):
        arrays[k] = np.asarray([r.get(k, -1) for r in records], dtype=np.int64)
    layers = np.asarray(records[0].get("layer_indices", []), dtype=np.int64)
    if any(not np.array_equal(layers, r.get("layer_indices", [])) for r in records):
        raise ValueError("layer_indices must match across the cell")
    arrays["layer_indices"] = layers
    for k in ("hidden_last", "hidden_mean"):
        arrays[k] = np.stack([np.asarray(r.get(k, np.empty((0, 0))), dtype=np.float16) for r in records])
        if arrays[k].ndim != 3 or arrays[k].shape[1] != len(layers):
            raise ValueError(f"{k} must have shape [N,L,D]")
    arrays["X"] = np.stack([np.asarray(r["X"] if "X" in r else np.asarray(r["hidden_mean"]).reshape(-1),
                                      dtype=np.float16) for r in records])
    if arrays["X"].ndim != 2:
        raise ValueError("X must have shape [N,F]")
    for prefix in ("", "prompt_"):
        values = [np.asarray(r.get(prefix + "token_ids", []), dtype=np.int64) for r in records]
        if any(v.ndim != 1 for v in values):
            raise ValueError("token IDs must be one-dimensional")
        arrays[prefix + "token_offsets"] = np.r_[0, np.cumsum([len(v) for v in values])].astype(np.int64)
        arrays[prefix + "token_ids"] = np.concatenate(values)
    lp = [np.asarray(r.get("token_logprobs", np.full(len(r.get("token_ids", [])), np.nan)),
                     dtype=np.float32) for r in records]
    if any(v.ndim != 1 or len(v) != len(r.get("token_ids", [])) for v, r in zip(lp, records)):
        raise ValueError("token_logprobs must align with token_ids")
    arrays["token_logprobs"] = np.concatenate(lp)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def read_cell_npz(path):
    """Load schema-1 arrays without pickle; no model/GPU imports needed."""
    with np.load(path, allow_pickle=False) as cell:
        arrays = {k: cell[k].copy() for k in cell.files}
    if arrays.get("schema_version") != 1:
        raise ValueError("unsupported capture NPZ schema")
    _analysis_columns(arrays)
    return arrays


def _analysis_columns(cell):
    columns = {k: np.asarray(cell[k]) for k in ANALYSIS_FIELDS}
    n = len(columns["y"])
    if any(v.ndim != (2 if k == "X" else 1) or len(v) != n for k, v in columns.items()):
        raise ValueError("analysis columns must be aligned; X must be [N,F]")
    return columns


def to_analysis_records(cells):
    """Concatenate NPZ paths/loaded cells into v1_analysis's dict of arrays."""
    if isinstance(cells, (str, Path, dict)):
        cells = [cells]
    columns = [_analysis_columns(read_cell_npz(c) if isinstance(c, (str, Path)) else c) for c in cells]
    if not columns:
        raise ValueError("at least one cell required")
    return {k: np.concatenate([c[k] for c in columns], axis=0) for k in ANALYSIS_FIELDS}
