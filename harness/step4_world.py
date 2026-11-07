"""Fresh step-4 world: same two rules and six facts as the smoke world, new codes.

Differences from SmokeSource (pre-registered in step4/README.md):
  * a new seed, and no rule code (train, gate or hidden) ever equals a seed-0 smoke code;
  * at least 100 hidden items per rule (default 120), so a per-family paired bootstrap
    has power; facts are unchanged (six hidden probes with one revision and one poison);
  * no arithmetic control skill (that existed only to exercise fault injection).
Scores are grouped per skill: rule:A, rule:B, fact:smoke.
"""
import random
import string

from .smoke_world import (FACTS, POISON, REVISED, RULES, SmokeSource, rule_answer,
                          rule_prompt)
from .streams import build_stream
from .worlds import SUFFIX, TaskSource

WORLD_SEED = 41
HIDDEN_PER_RULE = 120
TRAIN_PER_RULE = 24
GATE_PER_BATCH = 4


def smoke_codes():
    """Every rule code the seed-0 smoke world (used for development) contains."""
    _, items = SmokeSource(0).raw()
    return {i['input_key'] for i in items if i['skill'].startswith('rule:')}


class Step4Source(TaskSource):
    def __init__(self, seed=WORLD_SEED, n_cycles=3, hidden=HIDDEN_PER_RULE):
        if hidden < 100:
            raise ValueError('step 4 requires at least 100 hidden items per rule')
        self.seed, self.n_cycles, self.hidden = seed, n_cycles, hidden

    def raw(self):
        rng = random.Random(self.seed)
        skills = [dict(type='skill', name=f'rule:{r}', family='rule', intro_cycle=i, train_span=1)
                  for i, r in enumerate(('A', 'B'))]
        skills.append(dict(type='skill', name='fact:smoke', family='fact', intro_cycle=0))
        items, used = [], smoke_codes()
        for r, intro in (('A', 0), ('B', 1)):
            sizes = [('train', TRAIN_PER_RULE), ('hidden', self.hidden),
                     ('gate', GATE_PER_BATCH * (self.n_cycles - intro))]
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


def step4_stream(n_cycles=3, seed=WORLD_SEED):
    return build_stream(Step4Source(seed, n_cycles), seed=seed, n_cycles=n_cycles, train_span=1,
                        stream_id=f'step4-{seed}')
