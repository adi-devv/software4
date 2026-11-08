"""Fixed gold information baseline, built solely from lessons and train examples."""
import random
from .arms import NoMemory
from harness.tasks import grading_scope
from harness.smoke_world import gold_context as fact_context

class GoldContext(NoMemory):
    name = 'gold_context'

    def __init__(self, *args, stream, **kwargs):
        self.gold = {}
        for skill in sorted(s for s in stream.skills if s.startswith('rule:')):
            train = [i for c in range(stream.n_cycles) for i in stream.train_items(c)
                     if i.kind == 'probe' and i.skill == skill]
            if len(train) < 3:
                raise ValueError('gold_context requires three train examples per rule')
            train.sort(key=lambda i: i.key)
            chosen = random.Random(f'{stream.seed}:{skill}').sample(train, 3)
            lessons = [train[0].lesson]
            if skill == 'rule:C':
                from harness.step5_world import RULES
                lessons += [RULES['A'], RULES['B']]
            with grading_scope():
                demos = [f'Question: {i.prompt}\nVerified answer: {i.expected_at(i.cycle)}' for i in chosen]
            self.gold[skill] = '\n'.join(lessons + demos)
        super().__init__(*args, **kwargs)

    def context(self, task):
        text = self.gold.get(task.skill, '') if task.skill.startswith('rule:') else fact_context(task)
        if text and self.llm.count_tokens('Context:\n' + text) > self.context_tokens:
            raise ValueError('gold_context exceeds context_tokens')
        return [(0, text)] if text else []
