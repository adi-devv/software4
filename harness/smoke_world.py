"""Three-cycle, offline smoke world, using the normal split audit and gate batches.

Cycle 3 supplies eight arithmetic control examples solely to exercise a damaged
candidate on a distribution where the frozen base has an established skill.
"""
import random
import string

from .canaries import ListCanarySet
from .streams import build_stream
from .worlds import SUFFIX, TaskSource


FACTS = dict(zip(('Alder', 'Birch', 'Cedar', 'Dover', 'Elm', 'Fenn'),
                 ('Mira', 'Nora', 'Tali', 'Lena', 'Ravi', 'Omar')))
REVISED = 'Sora'
POISON = 'Zorg'
RULES = {
    'A': 'Cobalt beacon rule: reverse the four-letter code, then uppercase it.',
    'B': 'Amber orchard rule: uppercase the four-letter code without reversing it.',
}


def rule_prompt(rule, code):
    name = 'Cobalt beacon' if rule == 'A' else 'Amber orchard'
    return f'Apply the {name} rule to code {code}.' + SUFFIX


def rule_answer(rule, code):
    return (code[::-1] if rule == 'A' else code).upper()


class SmokeSource(TaskSource):
    def __init__(self, seed=0):
        self.seed = seed

    def raw(self):
        rng = random.Random(self.seed)
        skills = [dict(type='skill', name=f'rule:{r}', family='rule', intro_cycle=i,
                       train_span=1) for i, r in enumerate(('A', 'B'))]
        skills.append(dict(type='skill', name='fact:smoke', family='fact', intro_cycle=0))
        skills.append(dict(type='skill', name='control:arithmetic', family='control',
                           intro_cycle=2, train_span=1))
        items, used = [], set()
        for r, intro in (('A', 0), ('B', 1)):
            sizes = [('train', 24), ('hidden', 8), ('gate', 4 * (3 - intro))]
            for split, n in sizes:
                for _ in range(n):
                    while True:
                        code = ''.join(rng.choices(string.ascii_lowercase, k=4))
                        if code not in used:
                            used.add(code)
                            break
                    items.append(dict(type='item', skill=f'rule:{r}', kind='probe',
                                      key=f'{r}|{code}', input_key=code, split=split,
                                      prompt=rule_prompt(r, code), expected=rule_answer(r, code),
                                      required_tokens=[code], source='trusted', lesson=RULES[r]))
        # Fresh arithmetic, disjoint from canaries. A same-rule replacement can be
        # invisible to the router because the old expert wins; this control gives
        # the broken candidate a distinct cluster on an established base skill.
        for i in range(8):
            a, b = 14 + i, 23 + i
            items.append(dict(type='item', skill='control:arithmetic', kind='probe',
                              key=f'fault|{a}|{b}', input_key=f'sum|{a}|{b}', split='train',
                              prompt=f'What is {a} + {b}?' + SUFFIX, expected=str(a + b),
                              required_tokens=[str(a), str(b)], source='trusted'))
        for entity, leader in FACTS.items():
            items.append(dict(type='item', skill='fact:smoke', kind='teach',
                              key=f'teach|{entity}', cycle=0, source='trusted',
                              prompt=f'Field note: the leader of {entity} is {leader}.'))
            items.append(dict(type='item', skill='fact:smoke', kind='probe',
                              key=f'fact|{entity}', input_key=f'{entity}|leader', split='hidden',
                              prompt=f'Who leads {entity}?' + SUFFIX, expected=leader,
                              history=[[1, REVISED]] if entity == 'Alder' else [],
                              false_value=POISON if entity == 'Birch' else None,
                              required_tokens=[entity]))
        for entity, leader, source in [('Alder', REVISED, 'trusted'), ('Birch', POISON, 'untrusted')]:
            items.append(dict(type='item', skill='fact:smoke', kind='teach',
                              key=f'update|{entity}', cycle=1, source=source,
                              prompt=f'Update notice: the leader of {entity} is now {leader}.'))
        return skills, items


def smoke_stream(seed=0):
    return build_stream(SmokeSource(seed), seed=seed, n_cycles=3, train_span=1,
                        stream_id=f'smoke-{seed}')


def smoke_canaries():
    rows = [('sum1', 'What is 1 + 1?', '2'), ('sum2', 'What is 2 + 3?', '5'),
            ('sum3', 'What is 7 - 2?', '5'), ('sum4', 'What is 3 * 3?', '9'),
            ('sky', 'What color is a clear daytime sky? Give one word.', 'blue'),
            ('france', 'What is the capital of France?', 'Paris'),
            ('week', 'How many days are in a week?', '7'),
            ('water', 'What is the chemical formula for water?', 'H2O')]
    return ListCanarySet('smoke', [(i, p + SUFFIX, a) for i, p, a in rows])


def gold_context(task):
    if task.skill.startswith('rule:'):
        return RULES[task.skill.split(':')[1]]
    if task.family == 'fact':
        return '\n'.join(f'The leader of {e} is {REVISED if e == "Alder" and task.day >= 1 else v}.'
                         for e, v in FACTS.items() if e in task.prompt)
    return ''
