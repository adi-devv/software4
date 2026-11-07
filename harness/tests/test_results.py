import json

import pytest

from harness.arms import PLAN_BASELINES
from harness.report import format_report
from harness.results import Results, success_check
from harness.stats import paired_stream_ci


def _records(scores, n_cycles=15):
    records = []
    for arm, values in scores.items():
        for stream, accuracy in enumerate(values):
            for cycle in range(-1, n_cycles):
                records.append(dict(type="cycle_done", arm=arm, stream=stream,
                                    cycle=cycle, attempt=0))
                for i in range(100):
                    records.append(dict(type="score", arm=arm, stream=stream,
                                        phase="hidden", cycle=cycle, attempt=0,
                                        task_id=f"hidden-{i}", correct=i < round(100 * accuracy),
                                        skill="rule", absorbed=False))
                if arm == "ours":
                    for i in range(400):
                        records.append(dict(type="score", arm=arm, stream=stream,
                                            phase="canary", cycle=cycle, attempt=0,
                                            task_id=f"canary-{i}", correct=True))
    return records


def _scores():
    return {"ours": [.9] * 3, **{b: [.4] * 3 for b in PLAN_BASELINES}}


def test_success_requires_positive_ci_against_every_baseline():
    scores = _scores()
    scores.update(hippo_only=[.8] * 3, text_notes=[1., 1., .3])
    assert paired_stream_ci(scores["ours"], scores["text_notes"]).lo < 0
    out = success_check(Results(_records(scores)))
    assert not out.passed
    assert out.best_baseline == "hippo_only"
    assert not out.criteria["baseline:text_notes"]["pass"]


def test_success_corrects_for_five_baseline_comparisons(tmp_path):
    scores = _scores()
    scores["text_notes"] = [.82, .80, .78]
    assert paired_stream_ci(scores["ours"], scores["text_notes"]).lo > 0
    records = _records(scores)
    out = success_check(Results(records))
    assert not out.passed
    comparison = out.criteria["baseline:text_notes"]
    assert comparison["lower"] < 0
    assert comparison["confidence"] == pytest.approx(.99)
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    report = format_report(path)
    assert "SUCCESS CHECK: FAIL" in report and "Bonferroni" in report
    assert "baseline:text_notes" in report


def test_success_requires_margin_against_every_baseline():
    scores = _scores()
    scores["text_notes"] = [.88] * 3
    out = success_check(Results(_records(scores)))
    assert not out.passed
    assert not out.criteria["baseline:text_notes"]["pass"]


def test_success_passes_when_every_baseline_is_beaten():
    out = success_check(Results(_records(_scores())))
    assert out.passed, out.summary()
    for baseline in PLAN_BASELINES:
        assert out.criteria[f"baseline:{baseline}"]["pass"]


@pytest.mark.parametrize("missing", PLAN_BASELINES)
@pytest.mark.parametrize("explicit_baselines", [False, True])
def test_success_requires_all_planned_baselines(missing, explicit_baselines):
    scores = _scores()
    del scores[missing]
    kw = {"baselines": PLAN_BASELINES} if explicit_baselines else {}
    out = success_check(Results(_records(scores)), **kw)
    assert not out.passed
    assert f"missing_required_baseline:{missing}" in out.criteria


@pytest.mark.parametrize("n_cycles", [1, 14])
def test_success_requires_fifteen_cycles(n_cycles):
    out = success_check(Results(_records(_scores(), n_cycles=n_cycles)))
    assert not out.passed
    assert "no_complete_streams" in out.criteria


@pytest.mark.parametrize("arm,cycle", [("ours", -1), ("no_memory", 7), ("hippo_only", 14)])
def test_success_rejects_gaps_in_completed_cycles(arm, cycle):
    records = [r for r in _records(_scores())
               if not (r["type"] == "cycle_done" and r["arm"] == arm and r["cycle"] == cycle)]
    out = success_check(Results(records))
    assert not out.passed
    assert "no_complete_streams" in out.criteria


@pytest.mark.parametrize("scores,n_cycles", [
    ({"ours": [.9] * 3, "hippo_only": [.8] * 3}, 1),
    (_scores(), 1),
    ({"ours": [.9] * 3, "hippo_only": [.8] * 3}, 15),
])
def test_default_report_refuses_missing_baselines_or_cycles(tmp_path, scores, n_cycles):
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _records(scores, n_cycles)))
    assert "SUCCESS CHECK: FAIL" in format_report(path)


def test_required_baselines_and_minimum_cycles_are_configurable(tmp_path):
    records = _records({"ours": [.9] * 3, "hippo_only": [.8] * 3}, n_cycles=1)
    kw = dict(baselines=("hippo_only",), require=("hippo_only",), n_cycles=1)
    out = success_check(Results(records), **kw)
    assert out.passed, out.summary()
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert "SUCCESS CHECK: PASS" in format_report(path, **kw)
