"""Text-memory arms for the harness. Only observe() mutates learned state."""
import copy
import re
from hippocampus import Hippocampus, HashingEmbedder
from .llm import FakeLLM


def parse_answer(text, *, correction=False):
    """Extract the last final answer, or the harness correction value."""
    text = (text or '').strip()
    final = re.findall(r'^Final answer:\s*(.+)$', text, re.M)
    if final:
        return final[-1].strip()
    if correction and text.startswith('Correct answer: '):
        return re.split(r'\.(?:\s|$)', text[16:], maxsplit=1)[0].strip()
    return text


def render_entry(text):
    if text.startswith('Fact:'):
        return text
    match = re.fullmatch(r'Question: (.*?)\nResponse: (.*?)\nCorrect: (True|False|None)\nCorrection: (.*)', text, re.S)
    if not match:
        return text
    question, response, correct, correction = match.groups()
    value = parse_answer(response) if correct == 'True' else parse_answer(correction, correction=True)
    if correct == 'True' and not value:
        return text
    return 'Question: ' + question + ('\nVerified answer: ' + value if value else '')


def build_prompt(llm, task, entries, context_tokens, order=None):
    """Select whole entries by priority; optionally present them by chronological keys."""
    chosen = []
    for i, entry in enumerate(entries):
        entry = render_entry(entry)
        trial = chosen + [(i, entry)]
        if order is not None:
            trial.sort(key=lambda x: order[x[0]])
        block = 'Context:\n' + '\n'.join(x[1] for x in trial)
        if llm.count_tokens(block) <= context_tokens:
            chosen = trial
    block = 'Context:\n' + '\n'.join(x[1] for x in chosen)
    return block + '\n\nTask:\n' + task.prompt, [i for i, _ in chosen]


def episode(task, answer, result, correction):
    if task.kind == 'teach':
        return 'Fact: ' + task.prompt
    return '\n'.join(['Question: ' + task.prompt, 'Response: ' + answer,
                      'Correct: ' + str(result), 'Correction: ' + (correction or '')])


class Base:
    name = 'no_memory'
    def __init__(self, llm=None, context_tokens=3000, seed=0, **kwargs):
        if context_tokens < 0:
            raise ValueError('context_tokens must be nonnegative')
        self.llm, self.context_tokens, self.seed = llm or FakeLLM(), context_tokens, seed
        self.reset()

    def reset(self):
        self._pending = {}

    def context(self, task):
        return []

    def answer(self, task):
        entries = self.context(task)
        prompt, selected = build_prompt(self.llm, task, [e[1] for e in entries], self.context_tokens,
                                        order=[e[0] for e in entries])
        retrieved = [entries[i][0] for i in selected]
        if task.mode == 'day':
            self._pending[task.id] = retrieved
        self.answer_trace = {'expert': None, 'retrieved': retrieved, 'prompt': prompt}
        return self.llm.generate(prompt)

    def observe(self, task, answer, result, correction):
        if task.mode == 'test':
            return
        self._pending.pop(task.id, None)

    def sleep(self, cycle):
        return {}

    def save_state(self):
        return copy.deepcopy({'pending': self._pending})

    def load_state(self, state):
        self._pending = copy.deepcopy(state['pending'])


class NoMemory(Base):
    pass


class Rag(Base):
    name = 'rag'
    def __init__(self, *args, k=5, embedder=None, **kwargs):
        self.k, self.embedder = k, embedder or HashingEmbedder()
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.episodes = []

    def context(self, task):
        return self.retrieve(task.prompt)

    def retrieve(self, query, k=None):
        """Read-only ranked entries, without prompt construction or an LLM call."""
        if not self.episodes:
            return []
        vecs = self.embedder.embed([query] + [e['text'] for e in self.episodes])
        scores = vecs[1:] @ vecs[0]
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), -i))[:self.k if k is None else k]
        return [(i, self.episodes[i]['text']) for i in order]

    def observe(self, task, answer, result, correction):
        if task.mode == 'test': return
        super().observe(task, answer, result, correction)
        self.episodes.append({'text': episode(task, answer, result, correction)})

    def save_state(self):
        return copy.deepcopy({**super().save_state(), 'episodes': self.episodes})

    def load_state(self, state):
        super().load_state(state)
        self.episodes = copy.deepcopy(state['episodes'])


class LongContext(Rag):
    name = 'long_context'
    def context(self, task):
        # Select a contiguous newest suffix; present it in chronological order.
        selected = []
        for i in range(len(self.episodes) - 1, -1, -1):
            trial = [(i, self.episodes[i]['text'])] + selected
            if self.llm.count_tokens('Context:\n' + '\n'.join(render_entry(x[1]) for x in trial)) > self.context_tokens:
                break
            selected = trial
        return selected


class TextNotes(Base):
    name = 'text_notes'
    def __init__(self, *args, max_notes=200, prune_after=3, **kwargs):
        self.max_notes, self.prune_after = max_notes, prune_after
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.notes, self.episodes = [], []

    def context(self, task):
        return self.retrieve(task.prompt)

    def retrieve(self, query, k=None):
        """Read-only native note order; query is unused, as in context()."""
        order = sorted(range(len(self.notes)), key=lambda i: (self.notes[i]['uses'], i), reverse=True)
        if k is not None:
            order = order[:k]
        return [(i, self.notes[i]['text']) for i in order]

    def observe(self, task, answer, result, correction):
        if task.mode == 'test': return
        for i in self._pending.pop(task.id, []):
            self.notes[i]['uses'] += 1
            self.notes[i]['successes'] += int(bool(result))
        self.episodes.append(episode(task, answer, result, correction))

    def sleep(self, cycle):
        header = 'Distill lessons:\nWrite short standalone lesson lines from these episodes and corrections.\n'
        prompts, batch = [], []
        for entry in self.episodes:
            if self.llm.count_tokens(header + entry) > self.llm.input_limit:
                raise ValueError('distillation episode exceeds input_limit')
            if batch and self.llm.count_tokens(header + '\n'.join(batch + [entry])) > self.llm.input_limit:
                prompts.append(header + '\n'.join(batch))
                batch = []
            batch.append(entry)
        if batch:
            prompts.append(header + '\n'.join(batch))
        before = len(self.notes)
        self.notes = [n for n in self.notes if n['uses'] < self.prune_after or n['successes'] * 2 >= n['uses']]
        pruned = before - len(self.notes)
        for prompt in prompts:
            lines = self.llm.generate(prompt).splitlines()
            for line in lines:
                line = line.strip().lstrip('-* ').strip()
                if line and line not in [n['text'] for n in self.notes]:
                    self.notes.append({'text': line, 'uses': 0, 'successes': 0})
        if prompts:
            self.notes = self.notes[-self.max_notes:] if self.max_notes else []
        self.episodes = []
        self._pending = {}
        return {'notes': len(self.notes), 'pruned': pruned}

    def save_state(self):
        return copy.deepcopy({**super().save_state(), 'notes': self.notes, 'episodes': self.episodes})

    def load_state(self, state):
        super().load_state(state)
        self.notes, self.episodes = copy.deepcopy(state['notes']), copy.deepcopy(state['episodes'])


class HippoOnly(Base):
    name = 'hippo_only'
    def __init__(self, *args, k=5, **kwargs):
        self.k = k
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.hippo = Hippocampus()
        self.tick = 0

    def context(self, task):
        return [(h.id, h.text) for h in self.hippo.recall(task.prompt, self.k, now=self.tick)]

    def observe(self, task, answer, result, correction):
        if task.mode == 'test': return
        used = self._pending.pop(task.id, [])
        self.tick += 1
        mid = self.hippo.add(episode(task, answer, result, correction), kind='fact' if task.kind == 'teach' else 'episode', source='system' if task.source == 'trusted' else task.source, meta={'task_source': task.source, 'family': task.family, 'skill': task.skill}, t=self.tick)
        if result is not None:
            self.hippo.record_outcome(task.id, used, success=bool(result), verified=True, correction=correction, episode_id=mid, t=self.tick)

    def sleep(self, cycle):
        self.tick += 1
        return self.hippo.nap(now=self.tick)

    def save_state(self):
        return copy.deepcopy({**super().save_state(), 'events': self.hippo.log.events, 'tick': self.tick})

    def load_state(self, state):
        super().load_state(state)
        self.hippo = Hippocampus()
        for event in copy.deepcopy(state['events']):
            self.hippo.log.append(event)
            self.hippo._apply(event)
        self.tick = state['tick']
