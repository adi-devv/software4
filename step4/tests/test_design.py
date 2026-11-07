"""Design checks for the six plan-review changes: fair budgets, arms, fresh world, stop rule."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from step4 import run


def test_paired_bootstrap_detects_a_real_gap_and_is_reproducible():
    ids = [f"i{n}" for n in range(100)]
    ones, zeros = {i: 1 for i in ids}, {i: 0 for i in ids}
    gap = run.paired_bootstrap(ones, zeros, draws=500)
    assert gap["difference"] == 1 and gap["ci_low"] == 1 and gap["excludes_zero"]
    same = run.paired_bootstrap(ones, ones, draws=500)
    assert same["difference"] == 0 and not same["excludes_zero"]
    half = {i: int(n % 2) for n, i in enumerate(ids)}
    assert run.paired_bootstrap(ones, half, draws=500) == run.paired_bootstrap(ones, half, draws=500)


def test_paired_bootstrap_needs_identical_item_ids():
    with pytest.raises(ValueError):
        run.paired_bootstrap({"a": 1}, {"b": 1})
    with pytest.raises(ValueError):
        run.paired_bootstrap({}, {})


def _scores(accuracy_by_cycle_skill, n=100):
    rows = []
    for (cycle, skill), accuracy in accuracy_by_cycle_skill.items():
        for k in range(n):
            rows.append(dict(skill=skill, family="rule", task_id=f"{skill}-{k}", correct=int(k < accuracy * n),
                             cycle=cycle, phase="hidden"))
    return rows


def _arm(a0, a_last, b_last, accepted_b=True, fit=1.0, cycles=3):
    last = cycles - 1
    scores = _scores({(0, "rule:A"): a0, (last, "rule:A"): a_last, (last, "rule:B"): b_last})
    sleeps = [dict(cycle=0, accepted=True, forced_train_accuracy=fit),
              dict(cycle=1, accepted=accepted_b, forced_train_accuracy=fit)]
    return dict(scores=scores, sleeps=sleeps)


def _flat(accuracy):
    return _arm(accuracy, accuracy, accuracy)


def test_stop_rule_continues_only_when_every_clause_holds():
    results = dict(ours=_arm(0.9, 0.9, 0.9), rag=_flat(0.5), hippo_only=_flat(0.5))
    verdict = run.stop_rule(results, 2, draws=500)
    assert verdict["verdict"] == "continue" and verdict["continue"]
    assert all(v["passes"] and v["excludes_zero"] for v in verdict["comparisons"].values())
    assert verdict["rule_b_accepted"] and verdict["rule_a_not_dropped"]


def test_stop_rule_each_clause_can_fail():
    beat = dict(rag=_flat(0.5), hippo_only=_flat(0.5))
    # Only 10 points ahead of fair RAG: below the 15-point margin.
    close = dict(ours=_arm(0.6, 0.6, 0.6), rag=_flat(0.5), hippo_only=_flat(0.5))
    assert run.stop_rule(close, 2, draws=500)["verdict"] == "stop"
    # Ahead of both, but rule B was rejected on its night.
    rejected = run.stop_rule(dict(ours=_arm(0.9, 0.9, 0.9, accepted_b=False), **beat), 2, draws=500)
    assert not rejected["rule_b_accepted"] and rejected["verdict"] == "stop"
    # Ahead of both, but rule A dropped by more than the tolerance.
    dropped = run.stop_rule(dict(ours=_arm(0.95, 0.8, 0.9), **beat), 2, draws=500)
    assert not dropped["rule_a_not_dropped"] and dropped["verdict"] == "stop"
    # Beats RAG but not hippo_only: both comparisons are required.
    one = run.stop_rule(dict(ours=_arm(0.9, 0.9, 0.9), rag=_flat(0.5), hippo_only=_flat(0.9)), 2, draws=500)
    assert one["comparisons"]["rag"]["passes"] and not one["comparisons"]["hippo_only"]["passes"]
    assert one["verdict"] == "stop"


def test_unfit_expert_is_an_optimisation_failure_not_a_stop():
    results = dict(ours=_arm(0.6, 0.6, 0.6, fit=0.5), rag=_flat(0.5), hippo_only=_flat(0.5))
    verdict = run.stop_rule(results, 2, draws=500)
    assert verdict["optimisation_failure"] and not verdict["continue"]
    assert verdict["verdict"].startswith("inconclusive")
    # A fitted expert that still loses is a stop; a passing run is never relabelled.
    fitted = run.stop_rule(dict(results, ours=_arm(0.6, 0.6, 0.6, fit=1.0)), 2, draws=500)
    assert not fitted["optimisation_failure"] and fitted["verdict"] == "stop"
    passing = run.stop_rule(dict(results, ours=_arm(0.9, 0.9, 0.9, fit=0.5)), 2, draws=500)
    assert passing["verdict"] == "continue"


def test_parse_rate_sums_over_cycles():
    result = dict(parse_by_cycle={"0": dict(answers=10, parse_failures=1),
                                  "1": dict(answers=30, parse_failures=3)})
    assert run.parse_rate(result) == pytest.approx(0.1)
    assert run.parse_rate({}) is None


def test_fresh_world_has_no_seed0_codes_and_enough_hidden_items():
    from harness.step4_world import HIDDEN_PER_RULE, Step4Source, smoke_codes, step4_stream
    assert HIDDEN_PER_RULE >= 100
    _, items = Step4Source().raw()
    rules = [i for i in items if i["skill"].startswith("rule:")]
    assert not {i["input_key"] for i in rules} & smoke_codes()
    codes = [i["input_key"] for i in rules]
    assert len(codes) == len(set(codes))  # no code is reused across train, gate and hidden
    for rule in ("rule:A", "rule:B"):
        assert sum(i["split"] == "hidden" for i in rules if i["skill"] == rule) >= 100
    stream = step4_stream(3)
    stream.validate()
    for cycle in range(3):
        assert len([i for i in stream.hidden_items(cycle) if i.skill.startswith("rule:")]) >= (
            200 if cycle else 100)
    with pytest.raises(ValueError):
        Step4Source(hidden=99)


def cfg(**extra):
    return dict(backend="fake", steps=2, context_tokens=512, starved_tokens=64, **extra)


def test_every_arm_is_buildable_with_its_budget():
    from baselines import FakeBackend
    from scripts.smoke_sleep import Oracle
    assert run.ARMS[0] == "ours"  # ours starts first
    for name in run.ARMS:
        arm = run.make_arm(name, FakeBackend(), cfg(), Oracle)
        assert arm.name == name or name in ("rag_starved", "oracle")
    assert run.make_arm("rag", FakeBackend(), cfg(), Oracle).context_tokens == 512
    assert run.make_arm("rag_starved", FakeBackend(), cfg(), Oracle).context_tokens == 64
    assert run.make_arm("text_notes", FakeBackend(), cfg(), Oracle).context_tokens == 512


def test_fair_rag_keeps_lessons_the_starved_row_drops():
    from baselines.arms import build_prompt
    from baselines.llm import FakeLLM
    from harness.tasks import Task
    backend = FakeLLM()
    lessons = [f"Lesson {n}: the rule maps code {n:04d} onto its answer " + "word " * 12 for n in range(8)]
    task = Task(id="t", prompt="what is code 0003", family="rule", skill="rule:A", mode="test")
    fair, kept_fair = build_prompt(backend, task, lessons, 512)
    starved, kept_starved = build_prompt(backend, task, lessons, 64)
    assert len(kept_fair) > len(kept_starved)
    assert "Lesson 0:" in fair and "Lesson 7:" in fair
    assert backend.count_tokens(fair.split("\n\nTask:")[0]) <= 512
    assert backend.count_tokens(starved.split("\n\nTask:")[0]) <= 64


def test_backend_llm_caps_only_distillation_calls():
    seen = []

    class Backend:
        def count_tokens(self, text):
            return len(text.split())

        def generate(self, prompt, max_new_tokens=16):
            seen.append(max_new_tokens)
            return "ok"

    class Plain:
        def count_tokens(self, text):
            return 1

        def generate(self, prompt):
            return "ok"

    llm = run.BackendLLM(Backend())
    llm.generate("Distill lessons: x")
    llm.generate("Context:\n\nTask:\nx")
    assert seen == [run.BackendLLM.distill_tokens, 16]
    assert run.BackendLLM(Plain()).generate("Distill lessons: x") == "ok"  # no knob to turn
    assert llm.input_limit == 480 and llm.count_tokens("a b c") == 3
