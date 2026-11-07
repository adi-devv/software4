"""Canaries: old-skill sets measured on every arm at every cycle, with exactly the same items.

`CanarySet` is the interface. `ListCanarySet` wraps (id, prompt, expected) rows. `JsonlCanarySet`
loads a prepared file of real items later (for GSM8K / ARC write one JSON object per line:
{"id", "prompt", "expected"}; GSM8K answers are numeric, ARC answers are the choice letter; the
prompt should end with the "Final answer:" instruction). `SyntheticCanarySet` is for tests.
`canaries_needed` lives in stats.py and is re-exported here.
"""
from __future__ import annotations

import json
from pathlib import Path

from .stats import canaries_needed  # noqa: F401  (re-export)
from .tasks import Item
from .util import rng, short_id
from .worlds import SUFFIX


class CanarySet:
    name = "canary"

    def items(self):
        raise NotImplementedError


class ListCanarySet(CanarySet):
    def __init__(self, name: str, rows):
        self.name = name
        self._items = []
        seen = set()
        for cid, prompt, expected in rows:
            if cid in seen:
                raise ValueError(f"duplicate canary id {cid}")
            seen.add(cid)
            self._items.append(Item(id=short_id("canary", name, cid), skill=f"canary:{name}",
                                    family="canary", kind="probe", key=f"canary|{name}|{cid}",
                                    input_key=f"canary|{name}|{cid}", prompt=prompt,
                                    split="canary", expected=str(expected)))

    def items(self):
        return list(self._items)  # same objects, same order, every call


class JsonlCanarySet(ListCanarySet):
    def __init__(self, name: str, path):
        rows = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    rows.append((d["id"], d["prompt"], d["expected"]))
        super().__init__(name, rows)


class SyntheticCanarySet(ListCanarySet):
    """Small deterministic 'old skills': arithmetic, comparison, word reversal, letter pick."""

    def __init__(self, seed: int = 0, n: int = 300, name: str = "synthetic"):
        r = rng("canaries", seed, name)
        rows = []
        for i in range(n):
            t = i % 4
            if t == 0:
                a, b = int(r.integers(10, 99)), int(r.integers(10, 99))
                rows.append((f"{i}", f"What is {a} + {b}?" + SUFFIX, a + b))
            elif t == 1:
                a, b = int(r.integers(100, 999)), int(r.integers(100, 999))
                while a == b:
                    b = int(r.integers(100, 999))
                rows.append((f"{i}", f"Which is larger, {a} or {b}? Give the number." + SUFFIX,
                             max(a, b)))
            elif t == 2:
                w = "".join(chr(97 + int(x)) for x in r.integers(0, 26, 6))
                rows.append((f"{i}", f"Spell '{w}' backwards." + SUFFIX, w[::-1]))
            else:
                w = "".join(chr(97 + int(x)) for x in r.integers(0, 26, 7))
                k = int(r.integers(1, 8))
                rows.append((f"{i}", f"What is letter number {k} of '{w}'?" + SUFFIX, w[k - 1]))
        super().__init__(name, rows)
