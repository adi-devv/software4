"""Personalized PageRank by power iteration over a weighted edge list (numpy only)."""
from __future__ import annotations

import numpy as np


def personalized_pagerank(src: np.ndarray, dst: np.ndarray, w: np.ndarray, n: int,
                          seed: np.ndarray, damping: float = 0.5,
                          iters: int = 50, tol: float = 1e-9) -> np.ndarray:
    """Spreading activation. `damping` is the probability of following an edge;
    the rest teleports back to `seed` (sums to 1). Mass at dead ends returns to the seed."""
    out = np.bincount(src, weights=w, minlength=n)
    safe = np.where(out > 0, out, 1.0)
    wn = w / safe[src]
    dangling = out == 0
    x = seed.astype(np.float64).copy()
    for _ in range(iters):
        flow = np.bincount(dst, weights=x[src] * wn, minlength=n)
        xn = damping * (flow + x[dangling].sum() * seed) + (1.0 - damping) * seed
        done = np.abs(xn - x).sum() < tol
        x = xn
        if done:
            break
    return x
