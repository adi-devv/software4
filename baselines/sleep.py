"""Thin day/night arm. Model weights live entirely behind Backend."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import re
import tempfile
from typing import Protocol
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from hippocampus import HashingEmbedder
from harness.grading import _extract, match
from .arms import HippoOnly, build_prompt, parse_answer


class Backend(Protocol):
    def generate(self, prompt: str, expert: str | None) -> str: ...
    def train_expert(self, examples: list[tuple[str, str]], steps: int, seed: int) -> str: ...
    def commit_expert(self, handle: str) -> None: ...
    def discard_expert(self, handle: str) -> None: ...
    def is_finite(self, handle: str) -> bool: ...
    def adapter_delta_ratio(self, handle: str) -> float: ...
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
    # Pre-gate replay tolerance (pre-registered, fixed, never tuned per run): at most
    # floor(len(replay) * replay_flip_fraction) replay examples may flip from right
    # (current deployment) to wrong (forced candidate). With the default replay_size of 8
    # that is one example. New (non-replay) examples keep zero tolerance, as do the exact
    # canary rule and the delta-norm cap.
    replay_flip_fraction: float = 1 / 8

    def __post_init__(self):
        if self.max_examples < 1 or not 0 <= self.replay_size < self.max_examples:
            raise ValueError('require max_examples > replay_size >= 0')
        if self.steps < 1 or self.k < 0 or self.context_tokens < 0:
            raise ValueError('steps must be positive; k/context_tokens nonnegative')
        if not -1 <= self.threshold <= 1 or not 0 <= self.min_trust <= 1:
            raise ValueError('invalid similarity threshold or minimum trust')
        if not 0 <= self.replay_flip_fraction <= 1 / 8:
            raise ValueError('replay_flip_fraction must be within [0, 1/8]')


class SleepArm(HippoOnly):
    """HippoOnly day behavior plus isolated, gated experts at night.

    The inherited LLM supplies only a whitespace context-token counter. A real
    backend may supply a tokenizer counter through ``llm=``. Backend save/load
    paths are directories. With snapshot_dir, checkpoint references are relative
    to that directory; copy it with checkpoints to retain a relocatable bundle.
    """
    name = 'ours'

    def __init__(self, backend: Backend, cfg: SleepConfig | None = None, *, embedder=None,
                 llm=None, snapshot_dir=None):
        self.backend, self.cfg = backend, cfg or SleepConfig()
        self.embedder = embedder or HashingEmbedder()
        self.services = None
        self._fingerprint = backend.frozen_fingerprint()
        self._portable_snapshots = snapshot_dir is not None
        self._backend_root = (Path(snapshot_dir).resolve() if snapshot_dir is not None
                              else Path(tempfile.mkdtemp(prefix='sleep-backend-')))
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

    def _prompt(self, task, *, remember=False, expert=None):
        entries = self.context(task) if expert is None else []
        prompt, selected = build_prompt(self.llm, task, [e[1] for e in entries],
                                       self.context_tokens, order=[e[0] for e in entries])
        self._prompt_retrieved = [entries[i][0] for i in selected]
        if remember and task.mode == 'day':
            self._pending[task.id] = self._prompt_retrieved
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
        expert = self.route(task.prompt)
        prompt = self._prompt(task, remember=True, expert=expert)
        self.answer_trace = {'expert': expert, 'retrieved': self._prompt_retrieved, 'prompt': prompt}
        return self.backend.generate(prompt, expert=expert)

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
        target = parse_answer(response) if final else ''
        if not target:
            target = parse_answer(item.correction, correction=True)
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

    def _extra_examples(self, selected, cycle):
        return []

    def _expert_examples(self, selected, cycle):
        return [example for _, example in selected] + self._extra_examples(selected, cycle)

    def sleep(self, cycle):
        """Stage an expert without changing deployment until every check passes.

        Finiteness alone permits arbitrarily destructive finite adapters. Before
        spending the gate batch, cap each effective projection delta's Frobenius
        norm at its frozen projection norm. This is only a scale bound, not
        behavioral evidence. Force the candidate on trusted train/replay targets
        and require independent coverage of its routes and displaced skills.
        Also require exact canary output preservation (zero allowed changes),
        a conservative label-free absolute retention check.
        The harness's statistical gate remains an additional, unchanged check.
        """
        new, replay = self._training_set(cycle)
        if not new:
            return {'accepted': False, 'reasons': ['no new verified examples']}
        if self.services is None:
            raise RuntimeError('bind harness Services before sleep')
        selected = new + replay
        # Empty context deliberately tests learning in weights, independent of recall.
        examples = [('Context:\n\n\nTask:\n' + q, a)
                    for q, a in self._expert_examples(selected, cycle)]
        self._check_core()
        self._backend_snapshot = None  # preserve attempt bookkeeping on checkpoint/resume
        handle = self.backend.train_expert(examples, steps=self.cfg.steps, seed=self.seed + cycle)
        info = {'candidate': handle, 'examples': len(examples), 'replay': len(replay),
                'accepted': False, 'reasons': []}
        try:
            self._check_core()
            if not self.backend.is_finite(handle):
                info['reasons'] = ['non-finite candidate']
                return info
            ratio = self.backend.adapter_delta_ratio(handle)
            info['pre_gate'] = {'max_delta_base_ratio': ratio if np.isfinite(ratio) else None,
                                'delta_base_cap': 1.0}
            if not np.isfinite(ratio) or ratio > 1.0:
                info['reasons'] = ['adapter delta exceeds frozen projection norm']
                return info
            session = self.services.gate_session(cycle)
            proposed = dict(self.centroids)
            proposed[handle] = self._centroid([q for _, (q, _) in selected])
            none = self._centroid([t.prompt for t in session.tasks if t.family == 'canary']
                                  + [q for _, (q, _) in replay])
            current, candidate = {}, {}
            coverage = set()
            for task in session.tasks:
                old = self.route(task.prompt)
                current[task.id] = self.backend.generate(self._prompt(task, expert=old), expert=old)
                routed = self.route(task.prompt, centroids=proposed, none_centroid=none)
                coverage.add((task.skill, old, routed))
                candidate[task.id] = self.backend.generate(self._prompt(task, expert=routed), expert=routed)
            changed = [t.id for t in session.tasks if t.family == 'canary'
                       and candidate[t.id] != current[t.id]]
            info['pre_gate']['changed_canaries'] = changed
            if changed:
                info['reasons'] = ['absolute canary output drift']
                return info
            validation = []
            for item, (question, target) in selected:
                prompt = 'Context:\n\n\nTask:\n' + question
                old = self.route(question)
                baseline = self.backend.generate(self._prompt(SimpleNamespace(prompt=question), expert=old), expert=old)
                forced = self.backend.generate(prompt, expert=handle)
                expected, _ = _extract(target)
                def correct(answer):
                    extracted, _ = _extract(answer)
                    return extracted is not None and match(expected, extracted)
                validation.append(dict(task_id=item.task_id, replay=not item.new,
                    current_correct=correct(baseline), candidate_correct=correct(forced)))
            flipped = [v for v in validation
                       if v['current_correct'] and not v['candidate_correct']]
            tolerance = int(len(replay) * self.cfg.replay_flip_fraction + 1e-9)
            tolerated = [v['task_id'] for v in flipped if v['replay']][:tolerance]
            regressions = [v['task_id'] for v in flipped if v['task_id'] not in tolerated]
            info['pre_gate']['validation'] = dict(
                examples=validation, current_correct=sum(v['current_correct'] for v in validation),
                candidate_correct=sum(v['candidate_correct'] for v in validation),
                regressions=regressions, tolerated=tolerated, replay_flip_tolerance=tolerance)
            if regressions:
                info['reasons'] = ['trusted train/replay validation regressed']
                return info
            # Every proposed expert needs independent routed evidence. Also check
            # transitions on observed probes and retained fact queries, so a new
            # none centroid cannot silently displace an old expert/fact family.
            missing = []
            for expert in proposed:
                if not any(new_route == expert for _, _, new_route in coverage):
                    missing.append(dict(expert=expert, reason='no routed gate/canary tasks'))
            for memory in self.hippo._mems.values():
                if not memory.eligible or memory.trust < self.cfg.min_trust:
                    continue
                record = self.hippo.get(memory.id)
                skill = record['meta'].get('skill')
                if skill is None:
                    continue
                example = re.match(r'Question: (.*?)\nResponse: ', record['text'], re.S)
                key = self.hippo._fact_key(record['text']) if record['kind'] == 'fact' else None
                probes = [example.group(1)] if example else []
                if key:
                    relation, subject = key
                    probes.append(f'Who is the {relation} of {subject}?')
                    if relation == 'leader':
                        probes.append(f'Who leads {subject}?')
                for question in probes:
                    old = self.route(question)
                    routed = self.route(question, centroids=proposed, none_centroid=none)
                    if old != routed and (skill, old, routed) not in coverage:
                        entry = dict(skill=skill, previous=old, proposed=routed)
                        if entry not in missing:
                            missing.append(entry)
            info['pre_gate']['coverage'] = dict(
                routes=[dict(skill=s, previous=o, proposed=n)
                        for s, o, n in sorted(coverage, key=str)], uncovered=missing)
            if missing:
                info['reasons'] = ['uncovered candidate routes']
                return info
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
                              'backend_snapshot': (Path(self._backend_snapshot).relative_to(
                                  self._backend_root).as_posix() if self._portable_snapshots
                                  else self._backend_snapshot)})

    def load_state(self, state):
        if state['fingerprint'] != self._fingerprint:
            raise ValueError('checkpoint frozen core fingerprint differs')
        snapshot = Path(state['backend_snapshot'])
        if not snapshot.is_absolute():
            snapshot = self._backend_root / snapshot
        self.backend.load(snapshot)
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
        self._backend_snapshot = str(snapshot)


class ExpertOnly(SleepArm):
    """Ours with an empty context at evaluation: isolates what the weights alone retain.

    Hippocampus storage, sleep, gating and routing are inherited unchanged (routing reads only
    the task prompt). Only the prompt shown to the backend, at answer time and in the gate's
    current-versus-candidate comparison, drops retrieved context -- the same empty-context form
    the experts are trained on.
    """
    name = 'expert_only'

    def _prompt(self, task, *, remember=False, expert=None):
        self._prompt_retrieved = []
        if remember and task.mode == 'day':
            self._pending[task.id] = []
        return build_prompt(self.llm, task, [], self.context_tokens)[0]
