"""Generic control with one extra per selected episode within the cap.

The count is min(selected episodes, remaining capacity), independent of dream
verification; rejected dream variants are replaced by their admitted originals.
"""
import json
from pathlib import Path
import random

from .dream import DreamSleepArm, extra_count
from .sleep import SleepArm


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


class GenericOnlySleepArm(SleepArm):
    """Train only on generic pool examples, with the standard episode selection count.

    Shuffle the pool with seed + cycle and cycle through it when selection exceeds
    its size. Episode admission, routing, and validation remain those of SleepArm.
    """
    name = 'ours_generic_only'

    def _expert_examples(self, selected, cycle):
        path = Path(__file__).resolve().parents[1] / 'data/generic/pool.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        if not rows:
            raise ValueError('generic pool is empty')
        random.Random(self.seed + cycle).shuffle(rows)
        return [(rows[i % len(rows)]['prompt'], 'Final answer: ' + rows[i % len(rows)]['answer'])
                for i in range(len(selected))]


# Separate v2 classes keep both the v1 generic control and SleepArm unchanged.
from .sleep import ExpertOnly, ForcedSleepArm


class GenericForcedSleepArm(GenericOnlySleepArm, ForcedSleepArm):
    """GenericOnly's pool/count/shuffle with diagnostic-only sleep gates."""
    name = 'ours_generic_forced'

    def sleep(self, cycle):
        info = super().sleep(cycle)
        if 'train_fit' in info:
            fit = info['train_fit']
            info.update(generic_train_accuracy=fit['accuracy'],
                        generic_train_examples=fit['examples'],
                        generic_train_correct=fit['correct'],
                        generic_train_exact_accuracy=fit['exact_accuracy'],
                        generic_train_exact_correct=fit['exact_correct'])
        return info


class GenericForcedNoRetArm(GenericForcedSleepArm):
    """Forced generic experts with ExpertOnly's empty context, including at gates."""
    name = 'generic_forced_noret'

    _prompt = ExpertOnly._prompt
