"""Shared fixtures. Small configs keep the whole suite to about a minute."""
from __future__ import annotations

import pytest

from harness.arms_fake import Noisy, build_key
from harness.canaries import SyntheticCanarySet
from harness.runner import run_arm
from harness.streams import synthetic_stream
from harness.worlds import StreamConfig

SMALL = StreamConfig(n_cycles=16, entities_per_domain=18, items_per_rule=60)


@pytest.fixture(scope="session")
def stream():
    return synthetic_stream(1)


@pytest.fixture(scope="session")
def canaries():
    return [SyntheticCanarySet(seed=0, n=120)]


def small_stream(seed):
    return synthetic_stream(seed, SMALL)


def run_noisy_arms(path, seeds, ps, canaries, stream_sd=0.02):
    """ps: {arm_name: accuracy}. Runs every arm on every stream; returns the streams."""
    streams = [small_stream(s) for s in seeds]
    for st in streams:
        key = build_key(st, canaries)
        for name, p in ps.items():
            run_arm(Noisy(key, p, seed=0, name=name, stream_sd=stream_sd), st, canaries, path)
    return streams
