"""Fast offline checks for data integrity, metric definitions, and native ranking."""
import json
from types import SimpleNamespace

import pytest

from baselines import Rag, TextNotes
from scripts import retrieval_check as rc


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_deterministic_generator_and_gold(seed):
    cases = rc.generate(seed, n=6, scales=(10, 30, 100))
    assert cases == rc.generate(seed, n=6, scales=(10, 30, 100))
    assert cases != rc.generate(seed + 1, n=6, scales=(10, 30, 100))
    for case in cases:
        by_id = {f.id: f for f in case.facts}
        assert len(by_id) == len(case.facts)
        assert len({q.id for q in case.queries}) == 6
        latest = {f.memory_id: f.id for f in case.facts}
        for q in case.queries:
            # The explicit multihop requirement overrides the single-gold rule.
            assert len(q.gold) == len(set(q.gold)) == (2 if case.category == "multihop" else 1)
            for gold in q.gold:
                fact = by_id[gold]
                assert fact.status == "trusted"
                assert latest[fact.memory_id] == gold
                if case.category == "revised":
                    assert fact.version == 2
        if case.category == "multihop":
            for q in case.queries:
                first, second = (by_id[g] for g in q.gold)
                ally = first.text.split(" is ")[1].rstrip(".")
                assert f" of {ally} is " in second.text
                assert ally not in q.text
    scales = cases[-3:]
    assert scales[0].queries == scales[1].queries == scales[2].queries
    assert scales[0].facts == scales[1].facts[:10] == scales[2].facts[:10]
    assert {f.status for f in cases[4].facts} == {"trusted", "quarantined", "untrusted", "tombstoned"}


@pytest.mark.parametrize("gold,ranked,k,expected", [
    (("a",), ["a"], 1, (1, 1, 1, 0)),
    (("a",), ["x", "a"], 1, (0, 0, 0, 0)),
    (("a",), ["x", "a"], 5, (1, .5, 1, 0)),
    (("a", "b"), ["a"], 1, (.5, 1, 0, 0)),
    (("a", "b"), ["a", "a", "b"], 5, (1, 1, 1, 0)),
    (("a", "b"), ["x", "b"], 5, (.5, .5, 0, 0)),
    (("a",), ["a", "bad"], 1, (1, 1, 1, 0)),
    (("a",), ["a", "bad"], 5, (0, 0, 0, 1)),
    (("a",), [], 5, (0, 0, 0, 0)),
    (("a:v2",), ["a:v1"], 5, (0, 0, 0, 0)),
])
def test_scoring(gold, ranked, k, expected):
    result = rc.score(gold, ranked, {"bad"}, k)
    assert tuple(result[m] for m in ("recall", "rr", "all_gold", "unsafe")) == expected


def test_empty_gold_rejected():
    with pytest.raises(ValueError):
        rc.score((), [])


def test_helpers_preserve_native_order_and_state():
    rag = Rag(llm=rc.NoLanguageModel(), k=2)
    rag.load_state({"pending": {}, "episodes": [{"text": t} for t in ("apple", "orange", "apple")]})
    notes = TextNotes(llm=rc.NoLanguageModel())
    notes.load_state({"pending": {}, "episodes": [], "notes": [
        {"text": "apple", "uses": 2, "successes": 0},
        {"text": "orange", "uses": 1, "successes": 0},
        {"text": "banana", "uses": 2, "successes": 0}]})
    for arm, expected in ((rag, [2, 0]), (notes, [2, 0, 1])):
        before = arm.save_state()
        task = SimpleNamespace(prompt="apple")
        assert arm.context(task) == arm.retrieve("apple")
        assert [i for i, _ in arm.retrieve("apple")] == expected
        assert arm.retrieve("apple", 1) == arm.retrieve("apple")[:1]
        assert arm.save_state() == before
    assert notes.retrieve("apple") == notes.retrieve("orange")
    with pytest.raises(AssertionError, match="forbidden"):
        notes.llm.generate("must not run")


def test_lifecycle_is_scored_by_version_and_safe_id():
    facts = (rc.Fact("a", "The leader of Alder is Bob."),
             rc.Fact("a", "The leader of Alder is Cora.", 2),
             rc.Fact("b", "The leader of Alder is Dana.", status="quarantined"),
             rc.Fact("c", "The leader of Alder is Eli.", status="tombstoned"))
    case = rc.Case("poison", facts, (rc.Query("poison-q0", "leader of Alder", ("a:v2",)),))
    retrievers = rc.prepare(case)
    assert retrievers["hippocampus"]("leader of Alder", 5) == ["a:v2"]
    assert set(retrievers["rag"]("leader of Alder", 5)) == {f.id for f in facts}
    assert retrievers["text_notes"]("leader of Alder", 5) == [f.id for f in reversed(facts)]


def test_summary_gate_and_report(tmp_path, monkeypatch):
    original = rc.generate
    monkeypatch.setattr(rc, "generate", lambda seed, n: original(seed, n, scales=(2, 4, 8)))
    assert rc.main(["--seed", "0", "1", "2", "--queries", "2", "--output-dir", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "retrieval_check.json").read_text())
    assert len(data["queries"]) == 3 * 8 * 2 * 3
    for summary in data["summary"]:
        for m in rc.METRICS:
            values = [s[m] for s in summary["per_seed"]]
            assert summary["metrics"][m] == pytest.approx({"mean": sum(values) / 3, "min": min(values), "max": max(values)})
    # A tie must fail: never relax the gate to manufacture a win.
    tied = [{"category": c, "system": s, "metrics": {m: {"mean": 1} for m in ("recall@1", "recall@5")},
             "per_seed": [{"recall@1": 1, "recall@5": 1}]} for c in rc.STRUCTURAL for s in ("hippocampus", "rag")]
    assert not rc.gate(tied)["passed"]
    assert "Gate: **" in (tmp_path / "RETRIEVAL_RESULTS.md").read_text()
    assert all(d["sha256"] for d in data["datasets"])
