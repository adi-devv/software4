import dataclasses

import pytest

from harness.arms_fake import NoKnowledge
from harness.runner import run_arm, run_sealed_milestone
from harness.splits import SPLITS, assign_splits, audit, gate_batches
from harness.streams import build_stream, synthetic_stream


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_audit_clean_on_synthetic_streams(seed):
    assert synthetic_stream(seed).audit() == []


def test_assignment_is_stable_and_order_independent():
    keys = [f"k{i}" for i in range(200)]
    f = {"train": .3, "gate": .25, "hidden": .3, "sealed": .15}
    a = assign_splits(keys, 1, "sk", f)
    b = assign_splits(list(reversed(keys)), 1, "sk", f)
    assert a == b
    counts = {s: sum(v == s for v in a.values()) for s in SPLITS}
    assert counts == {"train": 60, "gate": 50, "hidden": 60, "sealed": 30}
    assert assign_splits(keys, 2, "sk", f) != a


def test_gate_items_used_in_one_cycle_only(stream):
    seen = {}
    for sk, batches in stream.gate_batches.items():
        for bi, batch in enumerate(batches):
            for it in batch:
                assert it.id not in seen
                seen[it.id] = bi
    assert seen
    per_cycle = [set(i.id for i in stream.gate_items(c)) for c in range(stream.n_cycles)]
    for i in range(len(per_cycle)):
        for j in range(i + 1, len(per_cycle)):
            assert not per_cycle[i] & per_cycle[j]


def test_splits_disjoint_in_stream(stream):
    train = {i.id for c in range(stream.n_cycles) for i in stream.train_items(c) if i.kind == "probe"}
    hidden = {i.id for i in stream.hidden_items(stream.n_cycles - 1)}
    gate = {i.id for c in range(stream.n_cycles) for i in stream.gate_items(c)}
    assert not train & hidden and not train & gate and not hidden & gate


def _audit_with(stream, fps):
    class S:
        fingerprints = fps
        gate_batches = stream.gate_batches
    return audit(S)


def test_audit_detects_planted_violations(stream):
    fps = list(stream.fingerprints)
    tr = next(f for f in fps if f.split == "train" and f.content)
    hi = next(f for f in fps if f.split == "hidden" and f.skill == tr.skill)

    def plant(**kw):
        return [dataclasses.replace(f, **kw) if f is hi else f for f in fps]

    assert any("input" in v for v in _audit_with(stream, plant(input_h=tr.input_h)))
    assert any("prompt" in v for v in _audit_with(stream, plant(prompt_h=tr.prompt_h)))
    assert any("item key" in v for v in _audit_with(stream, plant(key_h=tr.key_h)))
    assert any("near-duplicate" in v for v in _audit_with(stream, plant(content=tr.content)))
    assert any("required token" in v for v in _audit_with(stream, plant(missing_required=True)))


def test_audit_detects_gate_item_in_two_batches(stream):
    sk, batches = next((s, b) for s, b in stream.gate_batches.items()
                       if sum(1 for x in b if x) >= 1 and len(b) >= 2)
    src = next(i for i, x in enumerate(batches) if x)
    dst = (src + 1) % len(batches)
    bad = [list(x) for x in batches]
    bad[dst].append(bad[src][0])

    class S:
        fingerprints = stream.fingerprints
        gate_batches = {**stream.gate_batches, sk: bad}
    assert any("batches" in v for v in audit(S))


def test_gate_batches_helper_round_robin():
    gb = gate_batches([f"k{i}" for i in range(10)], 1, "s", 5)
    assert sorted(gb.values()) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_build_stream_rejects_cross_split_duplicates():
    class DuplicateSource:
        def raw(self):
            return ([{"name": "rule", "family": "rule", "intro_cycle": 0}],
                    [{"skill": "rule", "kind": "probe", "key": key, "input_key": "x=7",
                      "prompt": "What is f(7)?", "expected": "42", "split": split}
                     for key, split in (("a", "train"), ("b", "hidden"))])

    with pytest.raises(ValueError, match="input shared by splits train and hidden"):
        build_stream(DuplicateSource(), seed=0, n_cycles=1)


@pytest.mark.parametrize("sealed", [False, True])
def test_runner_rejects_audit_violations_before_arm_calls(tmp_path, sealed):
    stream = synthetic_stream(8)
    fps = stream.fingerprints
    tr = next(f for f in fps if f.split == "train")
    hi = next(f for f in fps if f.split == "hidden")
    stream.fingerprints = [dataclasses.replace(f, input_h=tr.input_h) if f is hi else f
                           for f in fps]
    calls = []

    class Spy(NoKnowledge):
        def reset(self):
            calls.append("reset")
            super().reset()

        def answer(self, task):
            calls.append("answer")
            return super().answer(task)

    arm = Spy()
    calls.clear()
    p = tmp_path / "r.jsonl"
    with pytest.raises(ValueError, match="input shared by splits"):
        if sealed:
            run_sealed_milestone(arm, stream, p, milestone="merge", reason="verify core merge")
        else:
            run_arm(arm, stream, [], p)
    assert calls == []
    assert not p.exists()
    assert stream.sealed.read_count == 0
