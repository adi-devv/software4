"""Opt-in forced control: unchanged prompts with wrong, shape-preserving targets."""
import random

from .arms import parse_answer
from .sleep import ExpertOnly, ForcedSleepArm
from .v3 import shape_signature


WRONG_RULE_SEED = 0


def wrong_rule_mapping(answers):
    """Derange unique targets within signature groups using one fixed seed.

    Sorting makes this independent of example order. A singleton group cannot
    be shuffled; change its first letter/digit, preserving its signature.
    """
    groups = {}
    for answer in sorted(set(answers)):
        groups.setdefault(shape_signature(answer), {}).setdefault(answer.casefold(), []).append(answer)
    rng = random.Random(WRONG_RULE_SEED)
    mapping = {}
    for signature in sorted(groups):
        variants = groups[signature]
        values = sorted(variants)
        rng.shuffle(values)
        if len(values) > 1:
            for source, target in zip(values, values[1:] + values[:1]):
                for answer in variants[source]:
                    mapping[answer] = variants[target][0]
            continue
        for answer in variants[values[0]]:
            for i, char in enumerate(answer):
                if char.isalpha():
                    other = 'b' if char.casefold() == 'a' else 'a'
                elif char.isdigit():
                    other = '1' if char == '0' else '0'
                else:
                    continue
                mapping[answer] = answer[:i] + other + answer[i + 1:]
                break
            else:
                raise ValueError('wrong_rule target needs a letter or digit')
    return mapping


class WrongRuleSleepArm(ForcedSleepArm):
    """Install control adapters despite their deliberately wrong train targets.

    Uses ExpertOnly's empty-context prompts. True-target gate diagnostics remain
    logged by ForcedSleepArm. Facts keep their verified targets. No fallback is
    applied to this control by the runner.
    """
    name = 'wrong_rule'
    _prompt = ExpertOnly._prompt

    def __init__(self, backend, cfg=None, **kwargs):
        if cfg is None or not cfg.v3_wrong_rule:
            raise ValueError('wrong_rule requires v3_wrong_rule')
        super().__init__(backend, cfg, **kwargs)

    def _expert_examples(self, selected, cycle):
        examples = [example for _, example in selected]
        rules = [i for i, (item, _) in enumerate(selected)
                 if self.hippo.get(item.episode_id)['meta'].get('family') == 'rule']
        mapping = wrong_rule_mapping(parse_answer(examples[i][1]) for i in rules)
        for i in rules:
            question, target = examples[i]
            examples[i] = (question, 'Final answer: ' + mapping[parse_answer(target)])
        return examples
