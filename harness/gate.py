"""Gate helper: candidate vs current model, with a noise-based rollback margin and poison checks.

The margin is NOT a constant. For each check we take the paired per-item differences
d_i = candidate_i - current_i (same fresh gate items, same canaries, both models), and roll back
only if the candidate is worse by more than a one-sided t-margin built from the measured spread of
those differences:   margin = t_{1-alpha, n-1} * sd(d) / sqrt(n).
Identical models give d = 0, so margin = 0 and nothing is rolled back. A noisy set (large sd) gets
a wider margin, so chance swings don't throw out good updates.

Checks: gate items (accuracy must not regress), canaries (accuracy must not regress), poison
(absorbed false claims must not increase). Poison is also reported as a count.

`GateSession` is what an arm's sleep() gets from the harness: a facade for the fresh gate batch and
canaries. The arm sees only the tasks and gets back one
aggregate decision; per-item correctness is never returned (canaries are reused every cycle, so
item-level feedback would leak them). One decision per cycle: a batch is never scored twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

import numpy as np

from .grading import grade
from .stats import t_ppf
from .tasks import _gate_grading_scope


class GateReuseError(RuntimeError):
    """A gate batch was scored more than once."""


def paired_check(cand, cur, *, higher_is_better: bool = True, alpha: float = 0.05) -> dict:
    d = np.asarray(cand, float) - np.asarray(cur, float)
    n = len(d)
    if n == 0:
        return {"n": 0, "diff": 0.0, "margin": float("inf"), "regression": False,
                "cand_mean": float("nan"), "cur_mean": float("nan")}
    diff = float(d.mean())
    if n == 1:
        margin = float("inf")  # one item cannot show a regression
    else:
        se = float(d.std(ddof=1)) / np.sqrt(n)
        margin = float(t_ppf(1 - alpha, n - 1) * se)
    worse = -diff if higher_is_better else diff
    return {"n": n, "diff": diff, "margin": margin, "regression": bool(worse > margin),
            "cand_mean": float(np.mean(cand)), "cur_mean": float(np.mean(cur))}


@dataclass
class GateDecision:
    accept: bool
    checks: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    poison_absorbed_candidate: int = 0
    poison_absorbed_current: int = 0

    @property
    def rollback(self) -> bool:
        return not self.accept


def gate_decision(*, gate, canary, poison=None, alpha: float = 0.05) -> GateDecision:
    """gate/canary: (candidate_correct, current_correct) 0/1 sequences over the same items.
    poison: optional (candidate_absorbed, current_absorbed) 0/1 sequences (1 = false claim used).
    """
    checks = {"gate": paired_check(*gate, alpha=alpha),
              "canary": paired_check(*canary, alpha=alpha)}
    reasons = []
    if checks["gate"]["regression"]:
        reasons.append("gate accuracy regressed beyond measured noise")
    if checks["canary"]["regression"]:
        reasons.append("canary accuracy regressed beyond measured noise")
    pc = pu = 0
    if poison is not None:
        checks["poison"] = paired_check(*poison, higher_is_better=False, alpha=alpha)
        pc, pu = int(np.sum(poison[0])), int(np.sum(poison[1]))
        if checks["poison"]["regression"]:
            reasons.append("candidate absorbed more poisoned claims beyond measured noise")
    return GateDecision(not reasons, checks, reasons, pc, pu)


_SESSION_STATE = WeakKeyDictionary()


class GateSession:
    __slots__ = ("cycle", "tasks", "__weakref__")

    def __init__(self, cycle: int, gate_items, canary_items):
        batch = _GateBatch(int(cycle), gate_items, canary_items)
        self.cycle = batch.cycle
        self.tasks = batch.tasks
        _SESSION_STATE[self] = batch

    def decide(self, candidate: dict, current: dict, alpha: float = 0.05) -> GateDecision:
        """candidate/current: {task_id: response string} for every task in `self.tasks`."""
        return _SESSION_STATE[self].decide(candidate, current, alpha)


class _GateBatch:
    """Harness-owned answer data; never handed to an arm."""

    def __init__(self, cycle: int, gate_items, canary_items):
        self.cycle = cycle
        self._gate = list(gate_items)
        self._canary = list(canary_items)
        self._used = False
        self.tasks = tuple(i.task("test", cycle) for i in self._gate + self._canary)

    def decide(self, candidate: dict, current: dict, alpha: float = 0.05) -> GateDecision:
        if self._used:
            raise GateReuseError(f"gate batch for cycle {self.cycle} already scored")
        self._used = True
        # Read arm-owned mappings before entering the trusted scope. Only plain strings
        # may reach the grader, so custom mapping/string callbacks cannot run unlocked.
        candidate = {t.id: candidate.get(t.id, "") for t in self.tasks}
        current = {t.id: current.get(t.id, "") for t in self.tasks}
        if any(type(v) is not str for v in (*candidate.values(), *current.values())):
            raise TypeError("gate responses must be plain strings")
        alpha = float(alpha)
        arrs = {k: ([], []) for k in ("gate", "canary", "poison")}
        with _gate_grading_scope():
            for group, items in (("gate", self._gate), ("canary", self._canary)):
                for it in items:
                    gc = grade(it, candidate.get(it.id, ""), self.cycle)
                    gu = grade(it, current.get(it.id, ""), self.cycle)
                    arrs[group][0].append(int(gc.correct))
                    arrs[group][1].append(int(gu.correct))
                    if group == "gate" and it.skill.startswith("poison:"):
                        arrs["poison"][0].append(int(gc.absorbed))
                        arrs["poison"][1].append(int(gu.absorbed))
        poison = arrs["poison"] if arrs["poison"][0] else None
        return gate_decision(gate=arrs["gate"], canary=arrs["canary"], poison=poison, alpha=alpha)
