"""Sealed test set guard.

Sealed items live inside a `SealedStore`. The only way to get them is `read(milestone=..., reason=...)`,
which needs both a milestone name and a real reason, refuses to run while arm code is running, and
appends a line to the access log. Iterating, indexing or listing the store raises. The number of
reads is `read_count` and is written into the results file by the runner.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .tasks import SealedAccessError, arm_running


class SealedStore:
    MIN_REASON = 10

    def __init__(self, items, log_path=None):
        self._items = list(items)
        self._log_path = Path(log_path) if log_path else None
        self.access_log = []  # in-memory copy of every successful read

    # ---- every ordinary way of getting at the items raises
    def __iter__(self):
        raise SealedAccessError("sealed set cannot be iterated; use read(milestone=..., reason=...)")

    def __getitem__(self, i):
        raise SealedAccessError("sealed set cannot be indexed; use read(milestone=..., reason=...)")

    def __len__(self):
        raise SealedAccessError("sealed set size is not exposed; use read(milestone=..., reason=...)")

    def __contains__(self, x):
        raise SealedAccessError("sealed set membership is not exposed")

    @property
    def items(self):
        raise SealedAccessError("sealed set cannot be listed; use read(milestone=..., reason=...)")

    def __repr__(self):
        return f"SealedStore(<sealed>, reads={self.read_count})"

    @property
    def read_count(self) -> int:
        return len(self.access_log)

    def read(self, *, milestone: str, reason: str, skill: str | None = None):
        """The one sanctioned access. Logs, then returns the sealed Items (answers still locked)."""
        if arm_running():
            raise SealedAccessError("sealed read refused: arm code is running")
        if not isinstance(milestone, str) or not milestone.strip():
            raise SealedAccessError("sealed read needs a milestone name")
        if not isinstance(reason, str) or len(reason.strip()) < self.MIN_REASON:
            raise SealedAccessError(f"sealed read needs a stated reason (>= {self.MIN_REASON} chars)")
        out = [i for i in self._items if skill is None or i.skill == skill]
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "milestone": milestone.strip(),
                 "reason": reason.strip(), "skill": skill, "n_items": len(out)}
        self.access_log.append(entry)
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        return out
