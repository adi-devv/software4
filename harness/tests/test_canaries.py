import json

import pytest

from harness.arms_fake import Oracle, build_key
from harness.canaries import JsonlCanarySet, ListCanarySet, SyntheticCanarySet
from harness.grading import grade_response
from harness.results import Results
from harness.runner import run_arm
from harness.stats import canaries_needed
from harness.tasks import grading_scope


def test_same_items_every_call_and_seeded():
    c = SyntheticCanarySet(seed=0, n=40)
    a, b = c.items(), c.items()
    assert [i.id for i in a] == [i.id for i in b]
    assert all(x is y for x, y in zip(a, b))
    assert [i.prompt for i in SyntheticCanarySet(seed=0, n=40).items()] == [i.prompt for i in a]
    assert [i.prompt for i in SyntheticCanarySet(seed=1, n=40).items()] != [i.prompt for i in a]


def test_runner_uses_same_ids_each_cycle(tmp_path, stream, canaries):
    p = tmp_path / "r.jsonl"
    run_arm(Oracle(build_key(stream, canaries)), stream, canaries, p)
    res = Results.from_file(p)
    ids = [set(res.canary_vector("oracle", stream.stream_id, c))
           for c in range(-1, stream.n_cycles)]
    assert len(ids[0]) == 120 and all(i == ids[0] for i in ids)


def test_jsonl_loader(tmp_path):
    p = tmp_path / "gsm.jsonl"
    rows = [{"id": "a", "prompt": "What is 2+3?", "expected": "5"},
            {"id": "b", "prompt": "What is 10-3?", "expected": 7}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    cs = JsonlCanarySet("gsm", p)
    assert len(cs.items()) == 2 and cs.items()[0].family == "canary"
    with grading_scope():
        assert cs.items()[1].expected_at(0) == "7"
    with pytest.raises(ValueError):
        ListCanarySet("dup", [("1", "p", "a"), ("1", "q", "b")])


def test_synthetic_answers_are_gradable():
    for it in SyntheticCanarySet(seed=0, n=40).items():
        with grading_scope():
            exp = it.expected_at(0)
        assert grade_response(exp, "Final answer: " + exp).correct


def test_canaries_needed_for_two_point_drop():
    paired = canaries_needed(0.02, discordance=0.04)
    unpaired = canaries_needed(0.02)
    assert paired < unpaired
    assert paired > 50
    assert canaries_needed(0.02, discordance=0.10) > paired
