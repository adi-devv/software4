"""Offline signature validity gate for the v3 wrong-rule control."""
from baselines.arms import parse_answer
from baselines.v3 import shape_signature
from harness.grading import _extract


def control_validity(answers, *, arm='wrong_rule', expected=None):
    """Require one final-cycle row for every registered rule item.

    ``expected`` must enumerate the full universe, keyed by task ID or
    (2, task ID). Missing/duplicate/unknown rows raise; missing answer text
    is malformed. Row-provided targets cannot shrink or override the universe.
    """
    if not expected:
        raise ValueError('rule validity needs the full expected answer text mapping')
    targets = {}
    for key, target in expected.items():
        if isinstance(key, tuple):
            if len(key) != 2 or key[0] != 2:
                raise ValueError('expected rule items must use final cycle 2')
            key = key[1]
        if not isinstance(key, str) or not key or key in targets:
            raise ValueError('invalid/duplicate expected rule task ID')
        if not isinstance(target, str) or not parse_answer(target):
            raise ValueError('rule validity needs expected answer text for ' + key)
        targets[key] = parse_answer(target)
    if len(targets) != 360:
        raise ValueError('expected mapping must contain all 360 registered rule items')
    seen, well_formed = set(), 0
    for row in answers:
        if row.get('arm') != arm or row.get('day') != 2:
            continue
        key = row.get('task_id')
        is_rule = row.get('family') == 'rule' or row.get('skill', '').startswith('rule:')
        if key not in targets:
            if is_rule:
                raise ValueError('unknown rule row: ' + str(key))
            continue
        if key in seen:
            raise ValueError('duplicate rule row: ' + key)
        seen.add(key)
        answer = row.get('answer')
        text, _ = _extract(answer) if isinstance(answer, str) else (None, '')
        well_formed += int(bool(text) and shape_signature(text.strip()) == shape_signature(targets[key]))
    missing = targets.keys() - seen
    if missing:
        raise ValueError(f'missing rule rows: {len(missing)} of {len(targets)}; {sorted(missing)}')
    total = len(targets)
    rate = well_formed / total
    return dict(arm=arm, rule_items=total, well_formed=well_formed,
                well_formed_rate=rate, valid=rate >= 0.90)
