"""Deterministic offline table backend, with deliberate candidate failure modes."""
import json
from pathlib import Path
import re

from .llm import FakeLLM


class FakeBackend:
    """Experts memorize normalized task text, ignoring context and polite wrappers.

    This is a wiring test double, not a model of semantic generalization. Set
    next_broken / next_nonfinite before training to fail the next candidate.
    Candidates can generate before commit; old expert tables are never edited.
    """
    def __init__(self):
        self.experts = {}
        self.committed = []
        self.discarded = []
        self.train_calls = []
        self.generate_calls = []
        self.next_broken = self.next_nonfinite = False
        self._next = 0

    @staticmethod
    def _question(prompt):
        return prompt.rsplit('\n\nTask:\n', 1)[-1].strip()

    @classmethod
    def _key(cls, prompt):
        question = cls._question(prompt).split('End your reply', 1)[0]
        return re.sub(r'\W+', ' ', re.sub(r'^please\s+', '', question.lower())).strip()

    def generate(self, prompt, expert=None):
        self.generate_calls.append((prompt, expert))
        if expert is not None:
            record = self.experts[expert]
            if record['broken']:
                return 'Final answer: broken'
            answer = record['table'].get(self._key(prompt))
            if answer is not None:
                return answer
        # A tiny frozen arithmetic skill lets gate tests measure canary damage.
        arithmetic = re.fullmatch(r'What is (\d+) \+ (\d+)\?.*',
                                  self._question(prompt), re.S)
        if arithmetic:
            return 'Final answer: ' + str(sum(map(int, arithmetic.groups())))
        return FakeLLM().generate(prompt)

    def train_expert(self, examples, steps, seed):
        handle = f'expert-{self._next}'
        self._next += 1
        self.train_calls.append({'examples': list(examples), 'steps': steps, 'seed': seed,
                                 'handle': handle})
        self.experts[handle] = {'table': {self._key(q): a for q, a in examples},
                                'broken': self.next_broken, 'finite': not self.next_nonfinite}
        self.next_broken = self.next_nonfinite = False
        return handle

    def commit_expert(self, handle):
        if handle not in self.committed:
            self.committed.append(handle)

    def discard_expert(self, handle):
        self.discarded.append(handle)
        del self.experts[handle]

    def is_finite(self, handle):
        return self.experts[handle]['finite']

    def frozen_fingerprint(self):
        return 'fake-frozen-core-v1'

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {k: getattr(self, k) for k in ('experts', 'committed', 'discarded', '_next',
                                             'next_broken', 'next_nonfinite')}
        (path / 'backend.json').write_text(json.dumps(state), encoding='utf-8')

    def load(self, path):
        state = json.loads((Path(path) / 'backend.json').read_text(encoding='utf-8'))
        for key, value in state.items():
            setattr(self, key, value)
