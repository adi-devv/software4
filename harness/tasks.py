"""Tasks, items and the answer lock.

Arms only ever see a `Task` (no answer field). The runner and graders hold `Item`s, whose
expected answers are locked unless code runs inside `grading_scope()`. The runner wraps every
arm call in `arm_scope()`; while an arm runs, grading scopes and sealed reads are refused.

Scope of the guarantee: this stops accidental and naive leakage by arm code running in the same
process (reading answers off the stream, calling the grader, reading the sealed set). It cannot
stop code that deliberately rewrites module internals. Run untrusted arms in their own process.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass


class AnswerLocked(RuntimeError):
    """An expected answer was requested outside a grading scope."""


class SealedAccessError(RuntimeError):
    """The sealed set was accessed other than through a stated, logged milestone call."""


_STATE = {"arm": 0, "grading": 0}  # process-global on purpose, so threads don't escape it


@contextlib.contextmanager
def arm_scope():
    if _STATE["grading"]:
        raise RuntimeError("arm code must not run inside a grading scope")
    _STATE["arm"] += 1
    try:
        yield
    finally:
        _STATE["arm"] -= 1


@contextlib.contextmanager
def grading_scope():
    if _STATE["arm"]:
        raise AnswerLocked("grading scope requested while arm code is running")
    _STATE["grading"] += 1
    try:
        yield
    finally:
        _STATE["grading"] -= 1


def grading_active() -> bool:
    return _STATE["grading"] > 0 and _STATE["arm"] == 0


def arm_running() -> bool:
    return _STATE["arm"] > 0


@dataclass(frozen=True)
class Task:
    """What an arm sees. No expected answer.

    kind: "probe" (answer it) or "teach" (a statement to learn from; not answered or scored).
    mode: "day" (arm may learn from it afterwards via observe) or "test" (frozen; never learn).
    source: "trusted" | "untrusted" | "user".
    day: the cycle number this task is shown in (0-based; -1 = before the stream starts).
    """

    id: str
    prompt: str
    family: str
    skill: str
    kind: str = "probe"
    mode: str = "day"
    source: str = "user"
    day: int = 0


class Item:
    """Internal record of a task plus its locked answer data."""

    __slots__ = ("id", "skill", "family", "kind", "key", "input_key", "prompt", "split",
                 "intro_cycle", "cycle", "source", "required", "gate_batch", "lesson", "_secret")

    def __init__(self, *, id, skill, family, kind, key, input_key, prompt, split, intro_cycle=0,
                 cycle=None, source="user", required=(), gate_batch=None, expected=None,
                 history=(), prior=(), false_value=None, lesson=None):
        self.id = id
        self.skill = skill
        self.family = family
        self.kind = kind
        self.key = key
        self.input_key = input_key
        self.prompt = prompt
        self.split = split
        self.intro_cycle = intro_cycle
        self.cycle = cycle
        self.source = source
        self.required = tuple(required)
        self.gate_batch = gate_batch
        self.lesson = lesson  # shown to arms in corrections on train items only
        # (expected at intro, [(cycle_from, new_value)], values before intro, false claim)
        self._secret = (expected, tuple((int(c), v) for c, v in history), tuple(prior), false_value)

    def _lock(self):
        if not grading_active():
            raise AnswerLocked(f"answer for item {self.id} is locked outside a grading scope")

    def expected_at(self, cycle: int) -> str:
        self._lock()
        cur, hist, _, _ = self._secret
        for c, v in sorted(hist):
            if c <= cycle:
                cur = v
        return cur

    def stale_at(self, cycle: int) -> list:
        """Earlier values of this fact that are now out of date."""
        self._lock()
        cur = self.expected_at(cycle)
        exp, hist, prior, _ = self._secret
        seen = list(prior)
        seen.append(exp)
        seen += [v for c, v in hist if c <= cycle]
        return [v for v in dict.fromkeys(seen) if v is not None and v != cur]

    def false_value(self):
        self._lock()
        return self._secret[3]

    def task(self, mode: str = "day", day: int = 0) -> Task:
        return Task(id=self.id, prompt=self.prompt, family=self.family, skill=self.skill,
                    kind=self.kind, mode=mode, source=self.source, day=day)

    def __repr__(self):
        return f"Item({self.id}, {self.skill}, {self.split})"
