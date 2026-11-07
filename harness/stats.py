"""Statistics: Student-t intervals, power helpers, the PLAN.md success check on arrays.

The unit of analysis is the STREAM (or seed). Tasks inside a stream share a world, so they are not
independent; a task-level interval is offered only as `naive_task_ci`, to show how much too narrow
it is. With few streams the t-interval is wide. That is the honest result, not a bug.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

# ---------------------------------------------------------------- Student t (no scipy)


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(x: float, df: float) -> float:
    if x == 0.0:
        return 0.5
    p = 0.5 * betai(df / 2.0, 0.5, df / (df + x * x))
    return 1.0 - p if x > 0 else p


def t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t (bisection on the exact cdf; accurate to ~1e-12)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -1e6, 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t_crit(df: float, conf: float = 0.95) -> float:
    """Two-sided critical value, e.g. t_crit(2) = 4.303, t_crit(30) = 2.042."""
    return t_ppf(1.0 - (1.0 - conf) / 2.0, df)


# ---------------------------------------------------------------- intervals


@dataclass(frozen=True)
class CI:
    mean: float
    lo: float
    hi: float
    n: int
    se: float
    t: float
    ok: bool  # False when there are too few points to form an interval

    def __str__(self):
        if not self.ok:
            return f"{self.mean:+.4f} (n={self.n}: no interval)"
        return f"{self.mean:+.4f} [{self.lo:+.4f}, {self.hi:+.4f}] n={self.n}"


def mean_ci(values, conf: float = 0.95) -> CI:
    v = np.asarray(list(values), dtype=float)
    n = len(v)
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0, float("nan"), float("nan"), False)
    m = float(v.mean())
    if n < 2:
        return CI(m, float("nan"), float("nan"), n, float("nan"), float("nan"), False)
    se = float(v.std(ddof=1) / math.sqrt(n))
    t = t_crit(n - 1, conf)
    return CI(m, m - t * se, m + t * se, n, se, t, True)


def paired_stream_ci(ours, other, conf: float = 0.95) -> CI:
    """t-interval over paired stream-level differences (ours - other)."""
    a, b = np.asarray(ours, float), np.asarray(other, float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same length")
    return mean_ci(a - b, conf)


def naive_task_ci(task_diffs, conf: float = 0.95) -> CI:
    """Interval treating every paired task as independent. Shown only for comparison."""
    return mean_ci(task_diffs, conf)


# ---------------------------------------------------------------- power helpers


def paired_tasks_needed(delta: float, discordance: float, alpha: float = 0.05,
                        power: float = 0.8, two_sided: bool = True) -> int:
    """Independent paired tasks needed to detect a mean paired difference `delta`.

    `discordance` is the fraction of tasks where the two systems disagree (one right, one wrong).
    Variance of a paired 0/1 difference = discordance - delta^2.
    """
    nd = NormalDist()
    za = nd.inv_cdf(1 - alpha / (2 if two_sided else 1))
    zb = nd.inv_cdf(power)
    var = max(discordance - delta * delta, 1e-12)
    return int(math.ceil((za + zb) ** 2 * var / (delta * delta)))


def two_proportions_needed(delta: float, p: float = 0.5, alpha: float = 0.05,
                           power: float = 0.8, two_sided: bool = True) -> int:
    """Tasks per arm for UNPAIRED independent samples (different tasks for each arm)."""
    nd = NormalDist()
    za = nd.inv_cdf(1 - alpha / (2 if two_sided else 1))
    zb = nd.inv_cdf(power)
    return int(math.ceil((za + zb) ** 2 * 2 * p * (1 - p) / (delta * delta)))


def design_effect(tasks_per_stream: float, icc: float) -> float:
    """Variance inflation for clustered tasks: 1 + (m - 1) * icc."""
    return 1.0 + (tasks_per_stream - 1.0) * icc


def streams_needed(delta: float, sd_diff: float, alpha: float = 0.05, power: float = 0.8,
                   max_n: int = 10000) -> int:
    """Streams needed for a paired t-test on stream-level differences with sd `sd_diff`."""
    for n in range(3, max_n):
        need = (t_crit(n - 1, 1 - alpha) + t_ppf(power, n - 1)) * sd_diff / math.sqrt(n)
        if need <= delta:
            return n
    return max_n


def canaries_needed(drop: float = 0.02, discordance: float | None = None, p: float = 0.7,
                    alpha: float = 0.05, power: float = 0.8) -> int:
    """Canary items needed to detect a `drop` (default 2 points) in canary accuracy.

    Canaries are the SAME items before and after, so the test is paired (one-sided: we only care
    about drops). `discordance` = fraction of canaries whose right/wrong status flips. If it is
    not known, assume the unpaired worst case 2p(1-p) (the two measurements independent), which
    needs many more items; pass a measured value to get the honest paired number.
    """
    q = discordance if discordance is not None else 2 * p * (1 - p)
    q = max(q, drop)  # a net drop of `drop` needs at least that many flips
    return paired_tasks_needed(drop, q, alpha, power, two_sided=False)


def plan_figure_check() -> dict:
    """Does PLAN.md's 'about 1.7k independent paired tasks for 3 points' reproduce?"""
    out = {
        "paired_two_sided_20pct_discordant": paired_tasks_needed(0.03, 0.20),
        "paired_one_sided_25pct_discordant": paired_tasks_needed(0.03, 0.25, two_sided=False),
        "paired_two_sided_10pct_discordant": paired_tasks_needed(0.03, 0.10),
        "paired_two_sided_30pct_discordant": paired_tasks_needed(0.03, 0.30),
        "unpaired_two_sided_p50_per_arm": two_proportions_needed(0.03, 0.5),
    }
    out["note"] = ("About 1.7k reproduces for PAIRED tasks when ~20% of tasks are discordant "
                   "(two-sided, alpha .05, power .8) or ~25% (one-sided). With unpaired samples "
                   "it is about 4.4k per arm. With clustering the number rises by the design "
                   "effect 1+(m-1)*icc.")
    return out


# ---------------------------------------------------------------- success check on arrays


@dataclass
class SuccessResult:
    passed: bool
    criteria: dict = field(default_factory=dict)  # name -> {"pass": bool, ...numbers...}
    ci: CI | None = None
    naive_ci: CI | None = None
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"SUCCESS CHECK: {'PASS' if self.passed else 'FAIL'}"]
        for k, v in self.criteria.items():
            nums = ", ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}"
                             for a, b in v.items() if a != "pass")
            lines.append(f"  [{'x' if v['pass'] else ' '}] {k}: {nums}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def evaluate_success(diffs, *, canary_worst_drops, n_canary_items, canary_needed,
                     min_gain: float = 0.03, max_canary_drop: float = 0.02,
                     min_streams: int = 3, conf: float = 0.95,
                     naive_diffs=None, extra_failures=()) -> SuccessResult:
    """PLAN.md success rule on stream-level numbers.

    diffs: per-stream (ours - best baseline) primary score, 0-1 scale.
    canary_worst_drops: per-stream worst canary drop of ours over all cycles (0-1 scale).
    Criteria (all must pass):
      streams_sufficient: at least `min_streams` streams (else no interval can be formed)
      gain:               mean difference >= min_gain (3 points)
      ci_lower_positive:  95% t-interval lower bound > 0
      canary_drop:        mean worst canary drop <= max_canary_drop (2 points)
      canaries_sufficient: enough canary items to measure a 2-point drop
    """
    diffs = np.asarray(diffs, float)
    ci = mean_ci(diffs, conf)
    crit = {}
    n = len(diffs)
    crit["streams_sufficient"] = {"pass": n >= min_streams, "n_streams": n, "needed": min_streams}
    crit["gain"] = {"pass": bool(n > 0 and ci.mean >= min_gain), "mean_diff": ci.mean,
                    "required": min_gain}
    crit["ci_lower_positive"] = {"pass": bool(ci.ok and ci.lo > 0), "lower": ci.lo,
                                 "upper": ci.hi, "mean": ci.mean}
    d = np.asarray(canary_worst_drops, float)
    dm = float(d.mean()) if len(d) else float("nan")
    crit["canary_drop"] = {"pass": bool(len(d) > 0 and dm <= max_canary_drop),
                           "mean_worst_drop": dm, "allowed": max_canary_drop}
    crit["canaries_sufficient"] = {"pass": bool(n_canary_items >= canary_needed),
                                   "n_canaries": n_canary_items, "needed": canary_needed}
    notes = []
    if n < min_streams:
        notes.append(f"only {n} stream(s): too few to form a usable interval; cannot pass")
    elif ci.ok:
        notes.append(f"interval half-width {ci.t * ci.se:.4f} from {n} streams "
                     f"(t={ci.t:.3f}); few streams means a wide interval")
    for f in extra_failures:
        crit[f] = {"pass": False}
    passed = all(v["pass"] for v in crit.values())
    naive = naive_task_ci(naive_diffs, conf) if naive_diffs is not None else None
    if naive is not None and naive.ok and ci.ok and naive.se < ci.se:
        notes.append(f"naive task-level half-width {naive.t * naive.se:.4f} is narrower than the "
                     f"stream-level {ci.t * ci.se:.4f}: ignoring clustering overstates confidence")
    return SuccessResult(passed, crit, ci, naive, notes)


# ---------------------------------------------------------------- false-positive simulation


def simulate_false_positive(*, n_streams: int, true_gain: float = 0.0, n_baselines: int = 4,
                            base_score: float = 0.7, stream_sd: float = 0.03,
                            tasks_per_stream: int = 300, reps: int = 5000, seed: int = 0,
                            min_gain: float = 0.03, min_streams: int = 3) -> dict:
    """How often does the gain + CI rule pass? (canary criteria ignored: set to pass)

    Model: each arm's stream score = base + gain(if ours) + arm-by-stream effect N(0, stream_sd)
    + task sampling noise sqrt(p(1-p)/tasks). The best baseline is the one with the highest mean
    across streams (this selection makes the rule slightly harder for 'ours', as it should be).
    With true_gain=0 this is the false-positive rate; with true_gain>0 it is the power.
    """
    r = np.random.default_rng(seed)
    noise_sd = math.sqrt(base_score * (1 - base_score) / tasks_per_stream)
    sd = math.sqrt(stream_sd ** 2 + noise_sd ** 2)
    passes = 0
    for _ in range(reps):
        ours = base_score + true_gain + r.normal(0, sd, n_streams)
        base = base_score + r.normal(0, sd, (n_baselines, n_streams))
        best = base[int(np.argmax(base.mean(axis=1)))]
        res = evaluate_success(ours - best, canary_worst_drops=[0.0], n_canary_items=10 ** 9,
                               canary_needed=1, min_gain=min_gain, min_streams=min_streams)
        passes += res.passed
    return {"n_streams": n_streams, "true_gain": true_gain, "reps": reps,
            "pass_rate": passes / reps}
