"""Plastic strengths: node (memory) strength and link (memory pair) strength.

Both live here, apart from the log's content records, and are rebuilt by replaying
verified-success outcomes at nap events. Strength is bounded in [0, smax] and
saturates (each success closes a fraction of the gap to smax). It fades on a
forgetting curve: v * 0.5 ** (age / half_life), age measured from the time it was
last strengthened. Nothing is deleted.
"""
from __future__ import annotations

from itertools import combinations


class Plasticity:
    def __init__(self, half_life: float, node_rate: float, link_rate: float,
                 smax: float = 1.0, max_link_group: int = 12):
        self.half_life, self.node_rate, self.link_rate = float(half_life), node_rate, link_rate
        self.smax, self.max_link_group = smax, max_link_group
        self.nodes: dict[str, tuple[float, float]] = {}   # id -> (value, t_last_strengthened)
        self.links: dict[tuple[str, str], tuple[float, float]] = {}

    def fade(self, v: float, t0: float, now: float) -> float:
        age = now - t0
        return v if age <= 0 else v * 0.5 ** (age / self.half_life)

    def node(self, mid: str, now: float) -> float:
        e = self.nodes.get(mid)
        return self.fade(e[0], e[1], now) if e else 0.0

    def link(self, a: str, b: str, now: float) -> float:
        e = self.links.get((a, b) if a < b else (b, a))
        return self.fade(e[0], e[1], now) if e else 0.0

    def _bump(self, table, key, t: float, rate: float) -> None:
        v, t0 = table.get(key, (0.0, t))
        e = self.fade(v, t0, t)
        table[key] = (e + rate * (self.smax - e), max(t, t0))

    def strengthen(self, ids, t: float) -> None:
        """A verified success used these memories together at time t."""
        ids = sorted(set(ids))
        for mid in ids:
            self._bump(self.nodes, mid, t, self.node_rate)
        for a, b in combinations(ids[: self.max_link_group], 2):
            self._bump(self.links, (a, b), t, self.link_rate)

    def state(self) -> dict:
        """Exact raw state, for rebuild-equivalence checks."""
        return {"nodes": dict(self.nodes), "links": dict(self.links)}
