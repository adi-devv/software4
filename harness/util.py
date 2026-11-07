"""Stable hashing and seeded randomness. Never use Python's hash() (varies per run)."""
from __future__ import annotations

import hashlib

import numpy as np


def stable_hash(*parts) -> int:
    """64-bit integer from the parts; identical across runs, machines and Python versions."""
    s = "\x1f".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def stable_unit(*parts) -> float:
    """Uniform number in [0, 1) derived from the parts."""
    return stable_hash(*parts) / 2.0**64


def short_id(*parts, n: int = 10) -> str:
    s = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def rng(*parts) -> np.random.Generator:
    """numpy Generator seeded from a stable hash of the parts."""
    return np.random.default_rng(stable_hash(*parts))
