"""Clocks. Logic never reads the wall clock directly; it asks a clock or takes `now`."""
from __future__ import annotations

import time


class ManualClock:
    """A clock you move by hand. Used in tests and for deterministic replays."""

    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def set(self, t: float) -> None:
        self.t = float(t)

    def advance(self, dt: float) -> float:
        self.t += float(dt)
        return self.t


class SystemClock:
    """Wall clock in seconds. Only used if the caller injects nothing and passes no `now`."""

    def __call__(self) -> float:
        return time.time()
