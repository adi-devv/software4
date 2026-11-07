"""Thin day/night arm. Model weights live entirely behind Backend."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import re
import tempfile
from typing import Protocol
from uuid import uuid4

import numpy as np

from hippocampus import HashingEmbedder
from .arms import HippoOnly, build_prompt


class Backend(Protocol):
    def generate(self, prompt: str, expert: str | None) -> str: ...
    def train_expert(self, examples: list[tuple[str, str]], steps: int, seed: int) -> str: ...
    def commit_expert(self, handle: str) -> None: ...
    def discard_expert(self, handle: str) -> None: ...
    def is_finite(self, handle: str) -> bool: ...
    def frozen_fingerprint(self) -> str: ...
    def save(self, path) -> None: ...
    def load(self, path) -> None: ...


@dataclass
class SleepConfig:
    max_examples: int = 64  # total, including replay
    replay_size: int = 8
    steps: int = 16
    seed: int = 0
    threshold: float = 0.2
    min_trust: float = 0.5
    k: int = 5
    context_tokens: int = 3000

    def __post_init__(self):
        if self.max_examples < 1 or not 0 <= self.replay_size < self.max_examples:
            raise ValueError('require max_examples > replay_size >= 0')
        if self.steps < 1 or self.k < 0 or self.context_tokens < 0:
            raise ValueError('steps must be positive; k/context_tokens nonnegative')
        if not -1 <= self.threshold <= 1 or not 0 <= self.min_trust <= 1:
            raise ValueError('invalid similarity threshold or minimum trust')


class SleepArm(HippoOnly):
    """HippoOnly day behavior plus isolated, gated experts at night.

    The inherited LLM supplies only a whitespace context-token counter. A real
    backend may supply a tokenizer counter through ``llm=``. Backend save/load
    paths are directories. Snapshots outlive the arm so runner JSON checkpoints
    can reference them; retain those directories alongside the checkpoints.
    """
    name = 'ours'

    def __init__(self, backend: Backend, cfg: SleepConfig | None = None, *, embedder=None,
                 llm=None):
        self.backend, self.cfg = backend, cfg or SleepConfig()
        self.embedder = embedder or HashingEmbedder()
        self.services = None
        self._fingerprint = backend.frozen_fingerprint()
        self._backend_root = Path(tempfile.mkdtemp(prefix='sleep-backend-'))
        self._initial = str(self._backend_root / 'initial')
        backend.save(self._initial)
        super().__init__(llm=llm, k=self.cfg.k, context_tokens=self.cfg.context_tokens,
                         seed=self.cfg.seed)

    def reset(self):
        super().reset()
        self.backend.load(self._initial)
        self._check_core()
        self.active_experts = []
        self.centroids = {}  # handle -> normalized vector
        self.none_centroid = None
        self._accepted_ids = set()
        self._backend_snapshot = self._initial

    def bind(self, services):
        self.services = services

    def _check_core(self):
        if self.backend.frozen_fingerprint() != self._fingerprint:
            raise RuntimeError('backend changed frozen core weights')

    def _prompt(self, task, *, remember=False):
        entries = self.context(task)
        prompt, selected = build_prompt(self.llm, task, [e[1] for e in entries],
                                       self.context_tokens, order=[e[0] for e in entries])
        if remember and task.mode == 'day':
            self._pending[task.id] = [entries[i][0] for i in selected]
        return prompt

    def route(self, prompt: str, *, centroids=None, none_centroid=None):
        if centroids is None:
            centroids, none_centroid = self.centroids, self.none_centroid
        if not centroids:
            return None
        q = self.embedder.embed([prompt])[0]
        handle = max(centroids, key=lambda h: float(np.dot(q, centroids[h])))
        score = float(np.dot(q, centroids[handle]))
        if score < self.cfg.threshold:
            return None
        if none_centroid is not None and float(np.dot(q, none_centroid)) >= score:
            return None  # none wins ties; expert ties keep the oldest expert
        return handle

    def answer(self, task):
        return self.backend.generate(self._prompt(task, remember=True),
                                     expert=self.route(task.prompt))

    def _centroid(self, prompts):
        if not prompts:
            return None
        vec = np.asarray(self.embedder.embed(prompts), dtype=float).mean(axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    @staticmethod
    def _example(item):
        # HippoOnly's serialized episode format; replay pins its outcome version.
        match = re.match(r'Question: (.*?)\nResponse: (.*?)\nCorrect: ', item.text, re.S)
        if not match:
            return None
        question, response = match.groups()
        final = re.findall(r'^Final answer:\s*(.+)$', response, re.M) if item.kind == 'success' else []
        target = final[-1].strip() if final else ''
        if not target:
            correction = (item.correction or '').strip()
            final = re.findall(r'^Final answer:\s*(.+)$', correction, re.M)
            if final:
                target = final[-1].strip()
            elif correction.startswith('Correct answer: '):
                # Harness corrections: "Correct answer: <value>. <optional lesson>".
                target = re.split(r'\.(?:\s|$)', correction[16:], maxsplit=1)[0].strip()
            else:
                target = correction
        if not question.strip() or not target:
            return None
        return question, 'Final answer: ' + target

    def _training_set(self, cycle):
        pool = self.hippo.replay_batch(len(self.hippo.log.events), seed=self.seed + cycle,
                                      min_trust=self.cfg.min_trust)
        fresh, old = [], []
        for item in pool:
            example = self._example(item)
            if example is not None:
                if item.new:
                    fresh.append((item, example))
                elif item.episode_id in self._accepted_ids:
                    old.append((item, example))
        rng = random.Random(self.seed + cycle)
        def pick(items, n):
            return [items[i] for i in sorted(rng.sample(range(len(items)), min(n, len(items))))]
        replay = pick(old, self.cfg.replay_size)
        new = pick(fresh, self.cfg.max_examples - len(replay))
        return new, replay

    def sleep(self, cycle):
        new, replay = self._training_set(cycle)
        if not new:
            return {'accepted': False, 'reasons': ['no new verified examples']}
        if self.services is None:
            raise RuntimeError('bind harness Services before sleep')
        selected = new + replay
        # Empty context deliberately tests learning in weights, independent of recall.
        examples = [('Context:\n\n\nTask:\n' + q, a) for _, (q, a) in selected]
        self._check_core()
        self._backend_snapshot = None
        handle = self.backend.train_expert(examples, steps=self.cfg.steps, seed=self.seed + cycle)
        info = {'candidate': handle, 'examples': len(examples), 'replay': len(replay),
                'accepted': False, 'reasons': []}
        try:
            self._check_core()
            if not self.backend.is_finite(handle):
                info['reasons'] = ['non-finite candidate']
                return info
            session = self.services.gate_session(cycle)
            proposed = dict(self.centroids)
            proposed[handle] = self._centroid([q for _, (q, _) in selected])
            none = self._centroid([t.prompt for t in session.tasks if t.family == 'canary']
                                  + [q for _, (q, _) in replay])
            current, candidate = {}, {}
            for task in session.tasks:
                prompt = self._prompt(task)
                current[task.id] = self.backend.generate(prompt, expert=self.route(task.prompt))
                routed = self.route(task.prompt, centroids=proposed, none_centroid=none)
                candidate[task.id] = self.backend.generate(prompt, expert=routed)
            decision = session.decide(candidate, current)
            info.update(asdict(decision))
            info['accepted'] = decision.accept
            self._check_core()
            if decision.accept:
                self.backend.commit_expert(handle)
                self._check_core()
                self.active_experts.append(handle)
                self.centroids, self.none_centroid = proposed, none
                self._accepted_ids.update(item.episode_id for item, _ in selected)
                self.tick += 1
                self.hippo.mark_sleep(t=self.tick)
                info['nap'] = self.hippo.nap(now=self.tick)
            return info
        finally:
            if handle not in self.active_experts:
                self.backend.discard_expert(handle)

    def save_state(self):
        self._check_core()
        if self._backend_snapshot is None:
            snapshot = str(self._backend_root / uuid4().hex)
            self.backend.save(snapshot)
            self._backend_snapshot = snapshot
        return copy.deepcopy({**super().save_state(), 'config': asdict(self.cfg),
                              'active_experts': self.active_experts,
                              'centroids': {h: v.tolist() for h, v in self.centroids.items()},
                              'none_centroid': (None if self.none_centroid is None
                                                else self.none_centroid.tolist()),
                              'accepted_ids': sorted(self._accepted_ids),
                              'fingerprint': self._fingerprint,
                              'backend_snapshot': self._backend_snapshot})

    def load_state(self, state):
        if state['fingerprint'] != self._fingerprint:
            raise ValueError('checkpoint frozen core fingerprint differs')
        self.backend.load(state['backend_snapshot'])
        self._check_core()
        self.cfg = SleepConfig(**state['config'])
        self.seed, self.k = self.cfg.seed, self.cfg.k
        self.context_tokens = self.cfg.context_tokens
        super().load_state(state)
        self.active_experts = list(state['active_experts'])
        self.centroids = {h: np.asarray(v, dtype=float) for h, v in state['centroids'].items()}
        none = state['none_centroid']
        self.none_centroid = None if none is None else np.asarray(none, dtype=float)
        self._accepted_ids = set(state['accepted_ids'])
        self._backend_snapshot = state['backend_snapshot']
