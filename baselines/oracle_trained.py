"""Ground-truth rule training ceiling, not a competitor."""
import re

from harness.smoke_world import rule_answer
from .sleep import SleepArm


class OracleTrainedSleepArm(SleepArm):
    """Train on the world's true rule answers: a ground-truth ceiling, not a competitor."""
    name = 'ours_oracle_trained'

    def _training_set(self, cycle):
        new, replay = super()._training_set(cycle)

        def gold(pairs):
            result = []
            for item, (question, _) in pairs:
                match = re.match(r'Apply the (Cobalt beacon|Amber orchard) rule to code ([a-z]{4})\.', question)
                if match:
                    rule = 'A' if match.group(1) == 'Cobalt beacon' else 'B'
                    result.append((item, (question, 'Final answer: ' + rule_answer(rule, match.group(2)))))
            return result
        return gold(new), gold(replay)
