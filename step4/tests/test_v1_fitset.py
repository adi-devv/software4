import itertools
import random
import string
from collections import Counter

import pytest

from step4 import v1_fitset
from harness.step4_world import Step4Source, smoke_codes
from harness.step5_world import NAMES, RULES, SUFFIX, Step5Source
from step4.v1_fitset import audit_fit_items, fit_items, fit_items_for_cycle


SEEDS = (44, 45, 46, 47)


def rule_items(source):
    return [i for i in source.raw()[1] if i['skill'].startswith('rule:')]


def answer(rule, code):
    return {'A': code.upper() + '-7', 'B': 'B-' + code.upper(),
            'C': 'B-' + code.upper() + '-7'}[rule]


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('per_rule', [0, 3, 80])
def test_counts_shape_and_determinism(seed, per_rule):
    items = fit_items(seed, per_rule)
    assert items == fit_items(seed, per_rule)
    assert len(items) == 3 * per_rule
    assert Counter(i['skill'] for i in items) == Counter({'rule:' + r: per_rule for r in 'ABC'})
    for item in items:
        rule, code = item['skill'][-1], item['input_key']
        assert len(code) == 4 and all(c in string.ascii_lowercase for c in code)
        assert item == dict(type='item', skill='rule:' + rule, kind='probe', key=rule + '|' + code,
                            input_key=code, split='fit',
                            prompt=f'Apply the {NAMES[rule]} rule to code {code}.' + SUFFIX,
                            expected=answer(rule, code), required_tokens=[code],
                            source='trusted', lesson=RULES[rule])


def test_world_seed_plus_1000():
    first = ''.join(random.Random(1044).choices(string.ascii_lowercase, k=4))
    assert fit_items(44)[0]['input_key'] == first


def test_every_forbidden_split_and_seed():
    forbidden = smoke_codes()
    for seed in (41, 42, 43):
        forbidden.update(i['input_key'] for i in rule_items(Step4Source(seed)))
    hidden_identities = set()
    for seed in SEEDS:
        items = rule_items(Step5Source(seed))
        assert {i['split'] for i in items} == {'train', 'hidden', 'gate'}
        forbidden.update(i['input_key'] for i in items)
        hidden_identities.update((i['skill'], i['input_key'], i['prompt'])
                                 for i in items if i['split'] == 'hidden')
    for seed in SEEDS:
        items = fit_items(seed)
        codes = {i['input_key'] for i in items}
        assert len(codes) == len(items)
        assert codes.isdisjoint(forbidden)
        identities = {(i['skill'], i['input_key'], i['prompt']) for i in items}
        assert len(identities) == len(items)
        assert identities.isdisjoint(hidden_identities)


def test_mutual_disjointness_and_order_independence():
    first_47 = fit_items(47)
    then_44 = fit_items(44)
    forward = {seed: fit_items(seed) for seed in SEEDS}
    reverse = {seed: fit_items(seed) for seed in reversed(SEEDS)}
    assert forward == reverse
    assert first_47 == forward[47] and then_44 == forward[44]
    for left, right in itertools.combinations(SEEDS, 2):
        assert {i['input_key'] for i in forward[left]}.isdisjoint(i['input_key'] for i in forward[right])
        assert {(i['skill'], i['input_key'], i['prompt']) for i in forward[left]}.isdisjoint(
            (i['skill'], i['input_key'], i['prompt']) for i in forward[right])
    # Different requested sizes still reserve the same deterministic code streams.
    for seed in SEEDS:
        short = fit_items(seed, 3)
        assert all(i in forward[seed] for i in short)
        for other in SEEDS:
            if other != seed:
                assert {i['input_key'] for i in short}.isdisjoint(i['input_key'] for i in forward[other])


@pytest.mark.parametrize('cycle,rules', [(-1, ''), (0, 'A'), (1, 'AB'), (2, 'ABC'), (3, 'ABC')])
def test_cycle_gating(cycle, rules):
    items = fit_items(44)
    before = [dict(i) for i in items]
    selected = fit_items_for_cycle(items, cycle)
    assert selected == [i for i in items if i['skill'][-1] in rules]
    assert len(selected) == 80 * len(rules)
    assert items == before


@pytest.mark.parametrize('seed', SEEDS)
def test_audit_passes(seed):
    result = audit_fit_items(seed)
    assert result['counts'] == {'rule:A': 80, 'rule:B': 80, 'rule:C': 80}
    assert all(result['checks'].values())
    assert result['checks']['no_fit_code_under_two_rules']
    assert 'rule+code+prompt' in result['composition_identity']
    assert 'own codes' in result['composition_identity']


@pytest.mark.parametrize('hidden_seed', SEEDS)
def test_audit_rejects_hidden_injection(monkeypatch, hidden_seed):
    hidden = next(i for i in rule_items(Step5Source(hidden_seed))
                  if i['split'] == 'hidden' and i['skill'] == 'rule:A')
    injected = fit_items(44)
    injected[0] = dict(hidden, split='fit')
    original = fit_items
    monkeypatch.setattr(v1_fitset, 'fit_items', lambda seed: injected if seed == 44 else original(seed))
    with pytest.raises(ValueError, match=f'no_overlap_step5:{hidden_seed}'):
        audit_fit_items(44)


@pytest.mark.parametrize('change,message', [
    ('count', 'counts'), ('split', 'split_label'), ('duplicate', 'no_duplicate_codes'),
    ('answer', 'expected_answers'), ('other_fit', 'no_overlap_fit:45'),
])
def test_audit_rejects_other_violations(monkeypatch, change, message):
    injected = fit_items(44)
    if change == 'count':
        injected.pop()
    elif change == 'split':
        injected[0]['split'] = 'hidden'
    elif change == 'duplicate':
        injected[1] = dict(injected[0])
    elif change == 'answer':
        injected[0]['expected'] = 'wrong'
    else:
        injected[0] = fit_items(45)[0]
    original = fit_items
    monkeypatch.setattr(v1_fitset, 'fit_items', lambda seed: injected if seed == 44 else original(seed))
    with pytest.raises(ValueError, match=message):
        audit_fit_items(44)


@pytest.mark.parametrize('per_rule', [-1, 1.5, True, 26 ** 4])
def test_invalid_count(per_rule):
    with pytest.raises(ValueError, match='per_rule'):
        fit_items(44, per_rule)
