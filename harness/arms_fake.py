"""Fake arms for testing the harness. TEST-ONLY.

The fakes are "pretrained" with an answer key built before the run (`build_key`). That stands in
for model competence; it is not a leak, because the key is built outside any arm call, exactly as
a real model's weights exist before it is run. `Cheater` is the opposite: it tries to read answers
while it is running, and the runner must stop it.
"""
from __future__ import annotations

from .grading import grade
from .tasks import Task, grading_scope, grading_active
from .util import rng, stable_unit


class Key:
    """task_id -> (answer at intro, [(cycle, new answer)])."""

    def __init__(self):
        self.rows = {}

    def answer(self, task_id: str, cycle: int) -> str:
        exp, hist = self.rows[task_id]
        cur = exp
        for c, v in sorted(hist):
            if c <= cycle:
                cur = v
        return cur


def build_key(stream, canaries=()) -> Key:
    """Answers for every probe in the stream (incl. sealed, via the store's internals) and canaries."""
    key = Key()
    items = []
    for c in range(stream.n_cycles):
        items += [i for i in stream.train_items(c) if i.kind == "probe"]
        items += stream.hidden_items(c)
        items += stream.gate_items(c)
    items += list(stream.sealed._items)  # test-only internal read for the oracle's key
    for cs in canaries:
        items += cs.items()
    with grading_scope():
        for it in items:
            exp, hist, _, _ = it._secret
            key.rows[it.id] = (exp, list(hist))
    return key


def say(a: str) -> str:
    return f"Working it out.\nFinal answer: {a}"


WRONG = say("wrong")


class FakeArm:
    name = "fake"

    def __init__(self, key: Key = None, seed: int = 0, name: str | None = None):
        self.key = key
        self.seed = seed
        if name:
            self.name = name
        self.stream_id = ""
        self.reset()

    def reset(self):
        self.seen = set()
        self.last_seen = {}

    def bind(self, services):
        self.stream_id = services.stream_id

    def observe(self, task: Task, answer, result, correction):
        self.seen.add(task.skill)
        self.last_seen[task.skill] = task.day

    def sleep(self, cycle):
        return {"seen_skills": len(self.seen)}

    def save_state(self):
        return {"seen": sorted(self.seen), "last_seen": self.last_seen}

    def load_state(self, s):
        self.seen = set(s["seen"])
        self.last_seen = dict(s["last_seen"])

    @staticmethod
    def is_canary(task):
        return task.family == "canary"

    def truth(self, task):
        return say(self.key.answer(task.id, task.day))

    def answer(self, task):  # pragma: no cover
        raise NotImplementedError


class NoKnowledge(FakeArm):
    name = "no_memory"

    def answer(self, task):
        return say("unknown")


class Oracle(FakeArm):
    name = "oracle"

    def answer(self, task):
        return self.truth(task)


class PerfectAfterSeeing(FakeArm):
    """Right on any skill it has seen at least one task of; perfect on canaries (never forgets)."""
    name = "perfect"

    def answer(self, task):
        if self.is_canary(task) or task.skill in self.seen:
            return self.truth(task)
        return WRONG


class Forgetter(FakeArm):
    """Learns new skills but loses old ones: right on a skill only for `keep` cycles after it last
    saw it, and its canary accuracy decays by `canary_decay` per cycle."""
    name = "forgetter"

    def __init__(self, key, seed=0, name=None, keep=2, canary_decay=0.12):
        super().__init__(key, seed, name)
        self.keep, self.canary_decay = keep, canary_decay

    def answer(self, task):
        if self.is_canary(task):
            level = max(0.0, 1.0 - self.canary_decay * (max(task.day, -1) + 1))
            return self.truth(task) if stable_unit(self.seed, task.id) < level else WRONG
        ls = self.last_seen.get(task.skill)
        if ls is not None and task.day - ls <= self.keep:
            return self.truth(task)
        return WRONG


class Noisy(FakeArm):
    """Right with true probability `p` on stream tasks (independent per task and cycle), plus a
    per-stream effect N(0, stream_sd) shared by all tasks of a stream (so tasks are not independent).
    Canaries: right with probability `canary_p`, FIXED per item (same model, same answers), so a
    stable arm's canary curve is flat."""

    def __init__(self, key, p, seed=0, name=None, stream_sd=0.0, canary_p=0.95):
        super().__init__(key, seed, name or f"noisy{p}")
        self.p, self.stream_sd, self.canary_p = p, stream_sd, canary_p

    def _effect(self):
        k = (self.name, self.seed, self.stream_id)
        if getattr(self, "_eff_key", None) != k:
            self._eff_key = k
            self._eff = float(rng(*k, "effect").normal(0, self.stream_sd))
        return self._eff

    def answer(self, task):
        if self.is_canary(task):
            ok = stable_unit(self.name, self.seed, self.stream_id, task.id) < self.canary_p
        else:
            eff = float(rng(self.name, self.seed, self.stream_id, "effect").normal(0, self.stream_sd)) \
                if self.stream_sd else 0.0
            ok = stable_unit(self.name, self.seed, self.stream_id, task.id, task.day) < self.p + eff
        return self.truth(task) if ok else WRONG


class Cheater(FakeArm):
    """Tries to reach answers while running. mode: item | grade | scope | sealed."""
    name = "cheater"

    def __init__(self, stream, mode="item", **kw):
        super().__init__(None, 0, kw.get("name"))
        self.stream, self.mode = stream, mode

    def answer(self, task):
        if self.mode == "sealed":
            self.stream.sealed.read(milestone="peek", reason="looking for hidden answers now")
        item = next(i for i in self.stream.hidden_items(self.stream.n_cycles - 1))
        if self.mode == "item":
            return say(item.expected_at(0))
        if self.mode == "grade":
            grade(item, "x", 0)
        if self.mode == "scope":
            with grading_scope():
                return say(item.expected_at(0))
        return say("none")
