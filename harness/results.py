"""Load results JSONL and compute the quantities the success check and report need."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .arms import OURS, PLAN_BASELINES
from .runner import read_lines
from .stats import (SuccessResult, canaries_needed, evaluate_success, mean_ci, naive_task_ci,
                    paired_stream_ci)


def load_records(path) -> list:
    """Logical records: score/info rows of an attempt that never reached cycle_done are dropped."""
    raw = read_lines(path)
    done = {}
    for r in raw:
        if r["type"] == "cycle_done":
            done[(r["arm"], r["stream"], r["cycle"])] = r["attempt"]
    out = []
    for r in raw:
        t = r["type"]
        if t == "score" and r["phase"] != "sealed":
            if done.get((r["arm"], r["stream"], r["cycle"])) != r["attempt"]:
                continue
        elif t == "cycle_start":
            continue
        elif t == "cycle_done":
            if done.get((r["arm"], r["stream"], r["cycle"])) != r["attempt"]:
                continue
        out.append(r)
    return out


class Results:
    def __init__(self, records):
        self.records = records
        self.scores = defaultdict(list)   # (arm, stream, phase, cycle) -> [score rec]
        self.done = defaultdict(set)      # (arm, stream) -> cycles done
        self.sealed_events = []
        self.violations = []
        self.summary_reads = 0
        for r in records:
            t = r["type"]
            if t == "score":
                self.scores[(r["arm"], r["stream"], r["phase"], r["cycle"])].append(r)
            elif t == "cycle_done":
                self.done[(r["arm"], r["stream"])].add(r["cycle"])
            elif t == "sealed_read":
                self.sealed_events.append(r)
            elif t == "violation":
                self.violations.append(r)
            elif t == "run_summary":
                self.summary_reads = max(self.summary_reads, r.get("sealed_reads", 0))

    @classmethod
    def from_file(cls, path):
        return cls(load_records(path))

    # ---- listings
    def arms(self):
        return sorted({a for a, _ in self.done})

    def streams(self, arm):
        return sorted(s for a, s in self.done if a == arm)

    def complete(self, arm, stream, n_cycles=None):
        c = self.done.get((arm, stream), set())
        if not c:
            return False
        return (n_cycles is None) or all(i in c for i in range(-1, n_cycles))

    def last_cycle(self, arm, stream):
        c = [x for x in self.done.get((arm, stream), set())]
        return max(c) if c else None

    @property
    def sealed_reads(self) -> int:
        return max(len(self.sealed_events), self.summary_reads)

    # ---- accuracy
    @staticmethod
    def _acc(recs):
        return float(np.mean([r["correct"] for r in recs])) if recs else float("nan")

    def hidden_acc(self, arm, stream, cycle):
        return self._acc(self.scores.get((arm, stream, "hidden", cycle), []))

    def canary_acc(self, arm, stream, cycle):
        return self._acc(self.scores.get((arm, stream, "canary", cycle), []))

    def primary(self, arm, stream, cycle: int, last_k: int = 1):
        """Mean hidden accuracy over the last_k cycles ending at `cycle`."""
        vals = [self.hidden_acc(arm, stream, c) for c in range(cycle - last_k + 1, cycle + 1)]
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float("nan")

    def task_vector(self, arm, stream, cycle):
        recs = self.scores.get((arm, stream, "hidden", cycle), [])
        return {r["task_id"]: r["correct"] for r in recs}

    def canary_vector(self, arm, stream, cycle):
        recs = self.scores.get((arm, stream, "canary", cycle), [])
        return {r["task_id"]: r["correct"] for r in recs}

    def canary_drop(self, arm, stream):
        """(worst drop over cycles, final drop, worst discordance, n_items). Drops are baseline - acc."""
        base = self.canary_acc(arm, stream, -1)
        bvec = self.canary_vector(arm, stream, -1)
        worst = final = 0.0
        q = 0.0
        cycles = sorted(c for c in self.done.get((arm, stream), set()) if c >= 0)
        for c in cycles:
            a = self.canary_acc(arm, stream, c)
            if a == a and base == base:
                worst = max(worst, base - a)
                final = base - a
            v = self.canary_vector(arm, stream, c)
            ks = set(v) & set(bvec)
            if ks:
                q = max(q, sum(v[k] != bvec[k] for k in ks) / len(ks))
        return worst, final, q, len(bvec)

    def poison(self, arm, stream, cycle):
        recs = [r for r in self.scores.get((arm, stream, "hidden", cycle), [])
                if r["skill"].startswith("poison:")]
        return sum(r["absorbed"] for r in recs), len(recs)


def common_streams(res: Results, arms, n_cycles=None):
    sets = [set(s for s in res.streams(a) if res.complete(a, s, n_cycles)) for a in arms]
    return sorted(set.intersection(*sets)) if sets else []


def success_check(res: Results, *, ours: str = OURS, baselines=None, require=PLAN_BASELINES,
                  n_cycles: int | None = 15, min_gain: float = 0.03,
                  max_canary_drop: float = 0.02, min_streams: int = 3, conf: float = 0.95,
                  last_k: int = 1) -> SuccessResult:
    """PLAN.md success rule: beat every baseline with Bonferroni-adjusted paired intervals.

    `require` defaults to arms 1-4 and 10, which must be selected and present in the results.
    Each common stream must complete the pre-learning baseline and at least `n_cycles` sleeps
    (default 15); pass None to disable the duration requirement for exploratory runs.
    """
    if baselines is None:
        baselines = PLAN_BASELINES
    baselines = list(baselines)
    missing = [r for r in require if r not in baselines or r not in res.arms()]
    arms = [ours] + baselines
    streams = common_streams(res, arms, n_cycles) if all(a in res.arms() for a in arms) else []
    failures = [f"missing_required_baseline:{m}" for m in missing]
    if not baselines:
        failures.append("no_baselines")
    if not streams or not baselines:
        return evaluate_success([], canary_worst_drops=[], n_canary_items=0, canary_needed=1,
                                min_gain=min_gain, max_canary_drop=max_canary_drop,
                                min_streams=min_streams, conf=conf,
                                extra_failures=failures + ["no_complete_streams"])
    final = {s: min(res.last_cycle(a, s) for a in arms) for s in streams}
    means = {b: np.mean([res.primary(b, s, final[s], last_k) for s in streams]) for b in baselines}
    best = max(means, key=lambda b: means[b])
    corrected_conf = 1 - (1 - conf) / len(baselines)  # Bonferroni simultaneous intervals
    diffs = [res.primary(ours, s, final[s], last_k) - res.primary(best, s, final[s], last_k)
             for s in streams]
    naive = []
    for s in streams:
        va, vb = res.task_vector(ours, s, final[s]), res.task_vector(best, s, final[s])
        naive += [va[k] - vb[k] for k in va if k in vb]
    drops = [res.canary_drop(ours, s) for s in streams]
    q = float(np.mean([d[2] for d in drops]))
    n_items = min(d[3] for d in drops)
    need = canaries_needed(max_canary_drop, discordance=q)
    out = evaluate_success(diffs, canary_worst_drops=[d[0] for d in drops],
                           n_canary_items=n_items, canary_needed=need, min_gain=min_gain,
                           max_canary_drop=max_canary_drop, min_streams=min_streams,
                           conf=corrected_conf,
                           naive_diffs=naive, extra_failures=failures)
    for b in baselines:
        ci = paired_stream_ci([res.primary(ours, s, final[s], last_k) for s in streams],
                              [res.primary(b, s, final[s], last_k) for s in streams],
                              conf=corrected_conf)
        out.criteria[f"baseline:{b}"] = {
            "pass": bool(ci.ok and ci.lo > 0 and ci.mean >= min_gain),
            "mean_diff": ci.mean, "required": min_gain,
            "lower": ci.lo, "upper": ci.hi, "confidence": corrected_conf,
        }
    out.passed = all(c["pass"] for c in out.criteria.values())
    out.notes.append(f"Bonferroni correction: {conf:.1%} family confidence across "
                     f"{len(baselines)} baselines; each paired CI uses {corrected_conf:.1%}")
    out.notes.insert(0, f"best baseline by mean primary score: {best} "
                        f"({means[best]:.4f}); ours: "
                        f"{np.mean([res.primary(ours, s, final[s], last_k) for s in streams]):.4f}")
    out.best_baseline = best
    out.streams = streams
    return out
