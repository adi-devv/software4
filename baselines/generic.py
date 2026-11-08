"""Matched generic-data control for deterministic dreams."""
import json
from pathlib import Path
import random

from .dream import DreamSleepArm


class GenericSleepArm(DreamSleepArm):
    name = 'ours_generic_data'

    def _extra_examples(self, selected, cycle):
        count = len(super()._extra_examples(selected, cycle))
        path = Path(__file__).resolve().parents[1] / 'data/generic/pool.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        rng = random.Random(self.seed + cycle)
        rng.shuffle(rows)
        return [(rows[i % len(rows)]['prompt'], 'Final answer: ' + rows[i % len(rows)]['answer'])
                for i in range(count)]
