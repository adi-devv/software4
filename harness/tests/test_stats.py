import numpy as np
import pytest

from harness.canaries import SyntheticCanarySet
from harness.report import format_report
from harness.results import Results, success_check
from harness.stats import (CI, canaries_needed, design_effect, evaluate_success, mean_ci,
                           naive_task_ci, paired_stream_ci, paired_tasks_needed, plan_figure_check,
                           simulate_false_positive, streams_needed, t_crit, t_ppf)
from harness.tests.conftest import run_noisy_arms

BASE = {"no_memory": 0.55, "rag": 0.62, "long_context": 0.65, "text_notes": 0.66, "hippo_only": 0.70}


def test_t_critical_values():
    for df, want in [(1, 12.706), (2, 4.303), (4, 2.776), (9, 2.262), (30, 2.042), (120, 1.980)]:
        assert t_crit(df) == pytest.approx(want, abs=0.002)
    assert t_crit(1000) == pytest.approx(1.962, abs=0.002)
    assert t_ppf(0.5, 7) == pytest.approx(0.0, abs=1e-6)
    assert t_crit(5, 0.99) == pytest.approx(4.032, abs=0.003)


def test_mean_ci_and_paired():
    ci = mean_ci([0.1, 0.12, 0.08, 0.11, 0.09])
    assert ci.ok and ci.lo < ci.mean < ci.hi
    assert ci.hi - ci.mean == pytest.approx(2.776 * np.std([.1, .12, .08, .11, .09], ddof=1) / 5 ** .5,
                                            rel=1e-3)
    assert not mean_ci([0.1]).ok and not mean_ci([]).ok
    assert paired_stream_ci([3, 4, 5], [1, 1, 1]).mean == pytest.approx(3.0)
    with pytest.raises(ValueError):
        paired_stream_ci([1, 2], [1])


def test_naive_interval_is_narrower_than_stream_level():
    r = np.random.default_rng(0)
    streams = [r.normal(0.05, 0.03) + r.normal(0, 0.45, 300) * 0.0 + r.choice([-1, 0, 1], 300, p=[.1, .8, .1])
               for _ in range(4)]
    stream_means = [float(np.mean(s)) for s in streams]
    naive = naive_task_ci(np.concatenate(streams))
    proper = mean_ci(stream_means)
    assert (naive.hi - naive.lo) < (proper.hi - proper.lo)


def test_success_never_passes_with_too_few_streams():
    for n in (0, 1, 2):
        res = evaluate_success([0.2] * n, canary_worst_drops=[0.0] * n, n_canary_items=10 ** 6,
                               canary_needed=10)
        assert not res.passed and not res.criteria["streams_sufficient"]["pass"]
    res = evaluate_success([0.2, 0.19, 0.21], canary_worst_drops=[0.0] * 3,
                           n_canary_items=10 ** 6, canary_needed=10)
    assert res.passed


def test_success_rule_each_criterion():
    kw = dict(canary_worst_drops=[0.0] * 6, n_canary_items=1000, canary_needed=100)
    assert evaluate_success([0.05, 0.04, 0.06, 0.05, 0.05, 0.04], **kw).passed
    assert not evaluate_success([0.02] * 6, **kw).passed                       # gain < 3 points
    assert not evaluate_success([0.05, -0.05, 0.1, 0.0, 0.08, 0.04], **kw).passed   # CI crosses 0
    assert not evaluate_success([0.05] * 6, canary_worst_drops=[0.05] * 6, n_canary_items=1000,
                                canary_needed=100).passed                      # canary drop
    assert not evaluate_success([0.05] * 6, canary_worst_drops=[0.0] * 6, n_canary_items=10,
                                canary_needed=100).passed                      # too few canaries


def test_power_helpers_and_plan_figure():
    fig = plan_figure_check()
    assert 1650 <= fig["paired_two_sided_20pct_discordant"] <= 1800      # PLAN's "about 1.7k"
    assert 1650 <= fig["paired_one_sided_25pct_discordant"] <= 1800
    assert fig["paired_two_sided_10pct_discordant"] < 1000
    assert fig["paired_two_sided_30pct_discordant"] > 2400
    assert fig["unpaired_two_sided_p50_per_arm"] > 4000
    assert paired_tasks_needed(0.03, 0.2) == fig["paired_two_sided_20pct_discordant"]
    assert design_effect(300, 0.05) == pytest.approx(1 + 299 * 0.05)
    assert streams_needed(0.03, 0.02) < streams_needed(0.03, 0.04)
    assert canaries_needed(0.02, discordance=0.05) < canaries_needed(0.02, discordance=0.2)


@pytest.mark.parametrize("delta,q,two_sided,needed", [
    (0.03, 0.20, True, 1742),
    (0.03, 0.25, False, 1716),
    (0.02, 0.02, False, 308),
])
def test_paired_sample_size_uses_null_and_alternative_variance(delta, q, two_sided, needed):
    assert paired_tasks_needed(delta, q, two_sided=two_sided) == needed


def test_false_positive_rate_numpy_simulation():
    fp = simulate_false_positive(n_streams=6, true_gain=0.0, reps=3000, seed=1)
    assert fp["pass_rate"] < 0.05
    power = simulate_false_positive(n_streams=8, true_gain=0.10, reps=1000, seed=1)
    assert power["pass_rate"] > 0.8
    assert simulate_false_positive(n_streams=2, true_gain=0.5, reps=200)["pass_rate"] == 0.0


@pytest.fixture(scope="module")
def equal_runs(tmp_path_factory, canaries):
    p = tmp_path_factory.mktemp("eq") / "r.jsonl"
    ps = {"ours": 0.70, **{k: 0.70 for k in BASE}}
    streams = run_noisy_arms(p, range(100, 110), ps, canaries)
    return p, [s.stream_id for s in streams]


def _subset(by_stream, ids):
    return Results([r for i in ids for r in by_stream[i]])


def test_equal_skill_arms_rarely_pass_harness_level(equal_runs):
    p, ids = equal_runs
    by_stream = {i: [] for i in ids}
    for rec in Results.from_file(p).records:
        if rec.get("stream") in by_stream:
            by_stream[rec["stream"]].append(rec)
    r = np.random.default_rng(0)
    passes, trials = 0, 100
    for _ in range(trials):
        sub = list(r.choice(ids, size=6, replace=False))
        passes += success_check(_subset(by_stream, sub)).passed
    rate = passes / trials
    print(f"harness-level false-positive rate (equal arms, 6 streams): {rate:.3f}")
    assert rate <= 0.10


def test_better_arm_passes_given_enough_streams(tmp_path, canaries):
    p = tmp_path / "r.jsonl"
    ps = {"ours": 0.85, **BASE}
    big = [SyntheticCanarySet(seed=0, n=320)]       # normal approximation needs >= 308 canaries
    run_noisy_arms(p, range(200, 206), ps, big)
    res = Results.from_file(p)
    out = success_check(res)
    assert out.passed, out.summary()
    assert out.best_baseline == "hippo_only"
    assert out.ci.lo > 0
    assert out.naive_ci is not None
    rep = format_report(p)
    assert "SUCCESS CHECK: PASS" in rep and "Canary curve" in rep and "Sealed reads: 0" in rep


def test_better_arm_cannot_pass_with_two_streams(tmp_path, canaries):
    p = tmp_path / "r.jsonl"
    run_noisy_arms(p, range(300, 302), {"ours": 0.95, **BASE}, canaries)
    out = success_check(Results.from_file(p))
    assert not out.passed and not out.criteria["streams_sufficient"]["pass"]


def test_arm_10_cannot_be_skipped(tmp_path, canaries):
    p = tmp_path / "r.jsonl"
    ps = {"ours": 0.95, **{k: v for k, v in BASE.items() if k != "hippo_only"}}
    run_noisy_arms(p, range(400, 405), ps, canaries)
    out = success_check(Results.from_file(p))
    assert not out.passed
    assert "missing_required_baseline:hippo_only" in out.criteria


def test_forgetter_as_ours_fails_success_check(tmp_path, canaries):
    from harness.arms_fake import Forgetter, Noisy, build_key
    from harness.runner import run_arm
    from harness.tests.conftest import small_stream
    p = tmp_path / "r.jsonl"
    for sd in range(500, 505):
        st = small_stream(sd)
        key = build_key(st, canaries)
        run_arm(Forgetter(key, name="ours", keep=100), st, canaries, p)   # strong on stream, bleeds canaries
        for name, q in BASE.items():
            run_arm(Noisy(key, q * 0.5, name=name), st, canaries, p)
    out = success_check(Results.from_file(p))
    assert not out.criteria["canary_drop"]["pass"]
    assert not out.passed


def test_report_renders_on_partial_results(tmp_path, stream, canaries):
    from harness.arms_fake import Oracle, build_key
    from harness.runner import run_arm
    p = tmp_path / "r.jsonl"
    run_arm(Oracle(build_key(stream, canaries)), stream, canaries, p)
    rep = format_report(p)
    assert "oracle" in rep and "Arm 'ours' not in results" in rep
    assert format_report(tmp_path / "missing.jsonl").startswith("# Evaluation report")
