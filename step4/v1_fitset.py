"""Fresh V1 critic fit items, separated from every registered world."""
import random
import string
from collections import Counter

from harness.step4_world import Step4Source, smoke_codes
from harness.step5_world import NAMES, RULES, SUFFIX, Step5Source

REGISTERED_SEEDS = (44, 45, 46, 47)
INTRO_CYCLES = {'rule:A': 0, 'rule:B': 1, 'rule:C': 2}


def _rule_items(source):
    return [i for i in source.raw()[1] if i['skill'].startswith('rule:')]


def _forbidden_sets(world_seed):
    groups = {'smoke': smoke_codes()}
    for seed in (41, 42, 43):
        groups[f'step4:{seed}'] = {i['input_key'] for i in _rule_items(Step4Source(seed))}
    for seed in dict.fromkeys((*REGISTERED_SEEDS, world_seed)):
        groups[f'step5:{seed}'] = {i['input_key'] for i in _rule_items(Step5Source(seed))}
    return groups


def _answer(rule, code):
    upper = code.upper()
    return {'A': upper + '-7', 'B': 'B-' + upper, 'C': 'B-' + upper + '-7'}[rule]


def fit_items(world_seed, per_rule=80):
    """Allocate all registered seeds in fixed rounds, then return this seed's items.

    Each seed has its own Random(seed + 1000). Rounds draw A, B, C once per
    seed, so increasing per_rule extends each rule's codes without changing
    earlier allocations. No mutable cache or call history affects separation.
    Additional seeds reserve the registered sets, but are not mutually registered.
    """
    if not isinstance(per_rule, int) or isinstance(per_rule, bool) or per_rule < 0:
        raise ValueError('per_rule must be a nonnegative integer')
    seeds = tuple(dict.fromkeys((*REGISTERED_SEEDS, world_seed)))
    used = set().union(*_forbidden_sets(world_seed).values())
    if len(seeds) * 3 * per_rule > 26 ** 4 - len(used):
        raise ValueError('per_rule exceeds the available four-letter code space')
    randomizers = {seed: random.Random(seed + 1000) for seed in seeds}
    codes = {seed: {rule: [] for rule in 'ABC'} for seed in seeds}
    for _ in range(per_rule):
        for seed in seeds:
            for rule in 'ABC':
                while True:
                    code = ''.join(randomizers[seed].choices(string.ascii_lowercase, k=4))
                    if code not in used:
                        used.add(code)
                        codes[seed][rule].append(code)
                        break
    return [dict(type='item', skill='rule:' + rule, kind='probe', key=rule + '|' + code,
                 input_key=code, split='fit',
                 prompt=f'Apply the {NAMES[rule]} rule to code {code}.' + SUFFIX,
                 expected=_answer(rule, code), required_tokens=[code], source='trusted',
                 lesson=RULES[rule])
            for rule in 'ABC' for code in codes[world_seed][rule]]


def fit_items_for_cycle(items, cycle):
    """Select fit rules introduced by this cycle, preserving item order."""
    return [i for i in items if INTRO_CYCLES[i['skill']] <= cycle]


def audit_fit_items(world_seed):
    """Audit default (80 per rule) fit sets; raise on the first failed check."""
    items = fit_items(world_seed)
    checks = {}

    def check(name, passed):
        if not passed:
            raise ValueError(f'V1 fit audit seed {world_seed}: {name}')
        checks[name] = True

    counts = Counter(i['skill'] for i in items)
    check('counts', counts == {skill: 80 for skill in INTRO_CYCLES})
    check('split_label', all(i['split'] == 'fit' for i in items))
    codes = [i['input_key'] for i in items]
    code_set = set(codes)
    check('no_duplicate_codes', len(codes) == len(code_set))
    code_rules = {}
    for item in items:
        code_rules.setdefault(item['input_key'], set()).add(item['skill'])
    check('no_fit_code_under_two_rules', all(len(rules) == 1 for rules in code_rules.values()))
    for name, forbidden in _forbidden_sets(world_seed).items():
        check('no_overlap_' + name, code_set.isdisjoint(forbidden))

    identity = lambda i: (i['skill'], i['input_key'], i['prompt'])
    identities = {identity(i) for i in items}
    check('canonical_identity_unique', len(identities) == len(items))
    for seed in dict.fromkeys((*REGISTERED_SEEDS, world_seed)):
        hidden = [i for i in _rule_items(Step5Source(seed)) if i['split'] == 'hidden']
        check(f'canonical_identity_hidden:{seed}', identities.isdisjoint(map(identity, hidden)))
        if seed != world_seed:
            other = fit_items(seed)
            check(f'no_overlap_fit:{seed}', code_set.isdisjoint(i['input_key'] for i in other))
            check(f'canonical_identity_fit:{seed}', identities.isdisjoint(map(identity, other)))

    check('expected_answers', all(i['expected'] == _answer(i['skill'][-1], i['input_key']) for i in items))
    check('item_shape', all(
        i['type'] == 'item' and i['kind'] == 'probe' and i['source'] == 'trusted'
        and len(i['input_key']) == 4 and all(c in string.ascii_lowercase for c in i['input_key'])
        and i['key'] == i['skill'][-1] + '|' + i['input_key']
        and i['required_tokens'] == [i['input_key']]
        and i['prompt'] == f"Apply the {NAMES[i['skill'][-1]]} rule to code {i['input_key']}." + SUFFIX
        and i['lesson'] == RULES[i['skill'][-1]] for i in items))
    return dict(world_seed=world_seed, counts=dict(counts), checks=checks,
                composition_identity='rule+code+prompt; rule C uses its own codes, not shared with A or B')
