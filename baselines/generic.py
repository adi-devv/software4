"""Generic control with one extra per selected episode within the cap.

The count is min(selected episodes, remaining capacity), independent of dream
verification; rejected dream variants are replaced by their admitted originals.
"""
import json
from pathlib import Path
import random

from .dream import DreamSleepArm, extra_count


class GenericSleepArm(DreamSleepArm):
    name = 'ours_generic_data'

    def _extra_examples(self, selected, cycle):
        count = extra_count(selected, self.cfg.max_examples)
        path = Path(__file__).resolve().parents[1] / 'data/generic/pool.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        rng = random.Random(self.seed + cycle)
        rng.shuffle(rows)
        return [(rows[i % len(rows)]['prompt'], 'Final answer: ' + rows[i % len(rows)]['answer'])
                for i in range(count)]
