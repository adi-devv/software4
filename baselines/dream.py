"""Deterministic, target-preserving variants of admitted sleep episodes."""
import random

from .sleep import SleepArm


def split_budget(new, replay, cap):
    # Reserve equal space for at most one extra per real example.
    real_limit = max(1, (cap + 1) // 2)
    new = new[:real_limit]
    replay = replay[:max(0, real_limit - len(new))]
    return new, replay


def extra_count(selected, cap):
    return min(len(selected), cap - len(selected))


class DreamSleepArm(SleepArm):
    name = 'ours_dream'

    def _training_set(self, cycle):
        return split_budget(*super()._training_set(cycle), self.cfg.max_examples)

    def _variant(self, question, target, rng):
        prefix = rng.choice(('Please ', 'Kindly '))
        return prefix + question[0].lower() + question[1:], target

    def _extra_examples(self, selected, cycle):
        rng = random.Random(self.seed + cycle)
        extras = []
        for _, (question, target) in selected:
            variant, answer = self._variant(question, target, rng)
            # Backend has no forced-answer correctness hook; preserve the verified label.
            if variant and answer == target:
                extras.append((variant, answer))
        return extras[:extra_count(selected, self.cfg.max_examples)]
