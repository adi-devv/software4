"""Stage T template-copy rules and the unchanged step-4 facts."""
import random
import string
from .step4_world import Step4Source, smoke_codes, TRAIN_PER_RULE, HIDDEN_PER_RULE, GATE_PER_BATCH
from .streams import build_stream
from .worlds import SUFFIX, TaskSource

WORLD_SEED = 44
NAMES = {"A": "Cobalt beacon", "B": "Amber orchard", "C": "Cobalt-Amber chain"}
RULES = {"A": "Cobalt beacon rule: uppercase the code and add '-7' at the end.",
         "B": "Amber orchard rule: uppercase the code and put 'B-' in front of it.",
         "C": "Cobalt-Amber chain rule: apply the Cobalt beacon rule to the code, then apply the Amber orchard rule to its result."}

class Step5Source(TaskSource):
    def __init__(self, seed=WORLD_SEED, n_cycles=3, hidden=HIDDEN_PER_RULE):
        if hidden < 100:
            raise ValueError("step 5 requires at least 100 hidden items per rule")
        self.seed, self.n_cycles, self.hidden = seed, n_cycles, hidden

    def raw(self):
        used = smoke_codes()
        for seed in (41, 42, 43):
            _, old = Step4Source(seed).raw()
            used.update(i['input_key'] for i in old if i['skill'].startswith('rule:'))
        randomizer = random.Random(self.seed)
        skills, items = [], []
        for intro, rule in enumerate(('A', 'B', 'C')):
            skills.append(dict(type='skill', name='rule:' + rule, family='rule', intro_cycle=intro, train_span=1))
            for split, count in [('train', TRAIN_PER_RULE), ('hidden', self.hidden),
                                 ('gate', GATE_PER_BATCH * (self.n_cycles - intro))]:
                for _ in range(count):
                    while True:
                        code = ''.join(randomizer.choices(string.ascii_lowercase, k=4))
                        if code not in used:
                            used.add(code)
                            break
                    upper = code.upper()
                    answer = {'A': upper + '-7', 'B': 'B-' + upper, 'C': 'B-' + upper + '-7'}[rule]
                    items.append(dict(type='item', skill='rule:' + rule, kind='probe', key=rule + '|' + code,
                                      input_key=code, split=split, prompt=f'Apply the {NAMES[rule]} rule to code {code}.' + SUFFIX,
                                      expected=answer, required_tokens=[code], source='trusted', lesson=RULES[rule]))
        old_skills, old_items = Step4Source(self.seed, self.n_cycles).raw()
        skills += [s for s in old_skills if s['family'] == 'fact']
        items += [i for i in old_items if i['skill'] == 'fact:smoke']
        return skills, items

def step5_stream(n_cycles=3, seed=WORLD_SEED):
    return build_stream(Step5Source(seed, n_cycles), seed=seed, n_cycles=n_cycles, train_span=1,
                        stream_id=f'step5-{seed}')
