"""Opt-in error-weighted replay over the standard trusted selection."""
from collections import Counter

from .sleep import SleepArm


class SalientSleepArm(SleepArm):
    name = 'ours_salient'

    def _training_set(self, cycle):
        new, replay = super()._training_set(cycle)
        # Keep the parent's admission and selection, including its replay quota.
        # There is no cheap correctness hook: generating again here would add
        # inference cost, so only the recorded corrected-failure signal is used.
        base = [(pair, False) for pair in new] + [(pair, True) for pair in replay]
        # Entries are (pair, old, extra). Extra copies of corrected failures are weighted in.
        entries = [(pair, old, False) for pair, old in base]
        entries += [(pair, old, True) for pair, old in base if pair[0].kind == 'corrected_failure']

        def skill(entry):
            return self.hippo.get(entry[0][0].episode_id)['meta'].get('skill')

        def keep_rank(entry):
            # Lower rank is dropped first: plain base entries, then extra copies, then failures.
            if entry[2]:
                return 1
            return 2 if entry[0][0].kind == 'corrected_failure' else 0

        # Trim to the budget and to the 50% per-skill rule together, always taking the
        # entry from the dominant skill that matters least, so failure weighting survives
        # at capacity instead of being cut off the tail.
        while entries:
            counts = Counter(skill(entry) for entry in entries)
            dominant, count = counts.most_common(1)[0]
            over = len(entries) > self.cfg.max_examples
            # The 50% rule only means something when another skill competes for the
            # budget; with a single skill it would empty the set (cycle 0 is all rule A).
            capped = len(counts) >= 2 and 2 * count > len(entries)
            if not over and not capped:
                break
            pool = [i for i, entry in enumerate(entries) if skill(entry) == dominant]
            entries.pop(min(pool, key=lambda i: (keep_rank(entries[i]), -i)))
        expanded = [(pair, old) for pair, old, _ in entries]
        return ([pair for pair, old in expanded if not old],
                [pair for pair, old in expanded if old])
