"""Streams: ordered phases of tasks over many sleep cycles, with disjoint splits.

One cycle = one day of train tasks followed by one night (arm.sleep). A skill is introduced at its
`intro_cycle`; from then on its hidden items are tested after every sleep.

Raw record fields (what a JSONL task file holds; the synthetic generator emits the same):
  skill:  {"type":"skill","name","family","intro_cycle","fracs"?:{train,gate,hidden,sealed}}
  item:   {"type":"item","skill","kind":"probe"|"teach","key" (unique item identity),
           "input_key" (canonical input), "prompt",
           probe: "expected", "history"?:[[cycle,new_value],..], "prior"?:[old values],
                  "false_value"?, "required_tokens"?:[..], "lesson"?, "split"?
           teach: "cycle", "source"?}
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .sealed import SealedStore
from .splits import SPLITS, assign_splits, audit, gate_batches, tokens
from .tasks import Item, grading_scope
from .util import short_id, stable_hash
from .worlds import StreamConfig, SyntheticSource, TaskSource

DEFAULT_FRACS = {"train": 0.30, "gate": 0.25, "hidden": 0.30, "sealed": 0.15}


@dataclass(frozen=True)
class Fingerprint:
    id: str
    split: str
    skill: str
    key_h: int
    input_h: int
    prompt_h: int
    content: frozenset
    missing_required: bool


def _norm_prompt(p: str) -> str:
    return " ".join(tokens(p))


class Stream:
    def __init__(self, *, stream_id, seed, n_cycles, skills, train_by_cycle, hidden, gate, sealed,
                 fingerprints):
        self.stream_id = stream_id
        self.seed = seed
        self.n_cycles = n_cycles
        self.skills = skills                  # name -> {"family","intro_cycle"}
        self._train = train_by_cycle          # cycle -> [Item]
        self._hidden = hidden                 # skill -> [Item]
        self.gate_batches = gate              # skill -> [[Item]] indexed by cycle - intro
        self.sealed = sealed
        self.fingerprints = fingerprints

    def active_skills(self, cycle: int):
        return [s for s, m in self.skills.items() if m["intro_cycle"] <= cycle]

    def train_items(self, cycle: int):
        return list(self._train.get(cycle, []))

    def hidden_items(self, cycle: int):
        out = []
        for s in self.active_skills(cycle):
            out += self._hidden.get(s, [])
        return out

    def gate_items(self, cycle: int):
        out = []
        for s in self.active_skills(cycle):
            b = cycle - self.skills[s]["intro_cycle"]
            batches = self.gate_batches.get(s, [])
            if 0 <= b < len(batches):
                out += batches[b]
        return out

    def correction_for(self, item: Item) -> str:
        """What an arm is told after a train probe: the right answer (and the rule, if any)."""
        with grading_scope():
            ans = item.expected_at(item.intro_cycle if item.cycle is None else item.cycle)
        txt = f"Correct answer: {ans}."
        if item.lesson:
            txt += " " + item.lesson
        return txt

    def audit(self):
        return audit(self)

    def digest(self) -> str:
        """Hash of everything in the stream including answers (used to test reproducibility)."""
        h = hashlib.sha256()
        with grading_scope():
            def feed(it):
                h.update(repr((it.id, it.skill, it.kind, it.split, it.prompt, it.cycle,
                               it.intro_cycle, it.gate_batch, it.source,
                               it._secret)).encode("utf-8"))
            for c in sorted(self._train):
                for it in self._train[c]:
                    feed(it)
            for s in sorted(self._hidden):
                for it in self._hidden[s]:
                    feed(it)
            for s in sorted(self.gate_batches):
                for b in self.gate_batches[s]:
                    for it in b:
                        feed(it)
            for it in self.sealed._items:  # internal read for the digest only, not an access
                feed(it)
        return h.hexdigest()


def build_stream(source: TaskSource, *, seed: int, n_cycles: int, train_span: int = 4,
                 default_fracs=None, stream_id: str | None = None,
                 sealed_log_path=None) -> Stream:
    default_fracs = default_fracs or DEFAULT_FRACS
    stream_id = stream_id or f"s{seed}"
    skills_raw, items_raw = source.raw()
    skills = {s["name"]: {"family": s["family"], "intro_cycle": int(s["intro_cycle"])}
              for s in skills_raw}
    fracs = {s["name"]: (s.get("fracs") or default_fracs) for s in skills_raw}
    span = {s["name"]: int(s.get("train_span") or train_span) for s in skills_raw}
    for sk, m in skills.items():
        if not 0 <= m["intro_cycle"] < n_cycles:
            raise ValueError(f"skill {sk} intro_cycle outside the stream")

    probes, teach = {}, {}
    for d in items_raw:
        if d["skill"] not in skills:
            raise ValueError(f"item for unknown skill {d['skill']}")
        (probes if d["kind"] == "probe" else teach).setdefault(d["skill"], []).append(d)

    train_by_cycle, hidden, gate_out, sealed_items, fps = {}, {}, {}, [], []
    seen_keys = set()

    for sk, plist in probes.items():
        fam, intro = skills[sk]["family"], skills[sk]["intro_cycle"]
        keys = [d["key"] for d in plist]
        if len(set(keys)) != len(keys) or seen_keys & set(keys):
            raise ValueError(f"duplicate item keys in skill {sk}")
        seen_keys |= set(keys)
        forced = {d["key"]: d["split"] for d in plist if d.get("split")}
        auto = assign_splits([k for k in keys if k not in forced], seed, sk, fracs[sk])
        split_of = {**auto, **forced}
        nb = max(1, n_cycles - intro)
        gmap = gate_batches([k for k in keys if split_of[k] == "gate"], seed, sk, nb)
        # df of tokens within the skill; tokens present in every prompt (the answer-format suffix, a rule's name) are dropped
        toks = {d["key"]: set(tokens(d["prompt"])) for d in plist}
        df = {}
        for ts in toks.values():
            for t in ts:
                df[t] = df.get(t, 0) + 1
        gate_lists = [[] for _ in range(nb)]
        train_items = []
        for d in plist:
            k = d["key"]
            sp = split_of[k]
            if sp not in SPLITS:
                raise ValueError(f"bad split {sp}")
            it = Item(id=short_id(stream_id, sk, k), skill=sk, family=fam, kind="probe", key=k,
                      input_key=d["input_key"], prompt=d["prompt"], split=sp, intro_cycle=intro,
                      source=d.get("source", "user"), required=d.get("required_tokens", ()),
                      gate_batch=gmap.get(k), expected=d["expected"],
                      history=[tuple(h) for h in d.get("history", [])], prior=d.get("prior", ()),
                      false_value=d.get("false_value"), lesson=d.get("lesson"))
            content = frozenset(stable_hash(t) for t in toks[k] if df[t] < len(plist))
            low = d["prompt"].lower()
            miss = any(str(t).lower() not in low for t in it.required)
            fps.append(Fingerprint(it.id, sp, sk, stable_hash(k), stable_hash(d["input_key"]),
                                   stable_hash(_norm_prompt(d["prompt"])), content, miss))
            if sp == "train":
                train_items.append(it)
            elif sp == "hidden":
                hidden.setdefault(sk, []).append(it)
            elif sp == "gate":
                gate_lists[gmap[k]].append(it)
            else:
                sealed_items.append(it)
        gate_out[sk] = [sorted(b, key=lambda x: x.id) for b in gate_lists]
        hidden[sk] = sorted(hidden.get(sk, []), key=lambda x: x.id)
        # spread train probes over `span` cycles starting at intro
        order = sorted(train_items, key=lambda x: (stable_hash(seed, "sched", x.key), x.key))
        s_ = max(1, min(span[sk], n_cycles - intro))
        for i, it in enumerate(order):
            c = intro + (i % s_)
            it.cycle = c
            train_by_cycle.setdefault(c, []).append(it)

    for sk, tlist in teach.items():
        fam = skills[sk]["family"]
        for d in tlist:
            c = int(d["cycle"])
            if not 0 <= c < n_cycles:
                continue  # scheduled outside this stream's length
            it = Item(id=short_id(stream_id, sk, d["key"], "teach", c), skill=sk, family=fam,
                      kind="teach", key=d["key"], input_key=d.get("input_key", d["key"]),
                      prompt=d["prompt"], split="train", intro_cycle=skills[sk]["intro_cycle"],
                      cycle=c, source=d.get("source", "trusted"))
            train_by_cycle.setdefault(c, []).append(it)

    # within a day: statements first, then questions, each in a stable shuffled order
    for c, lst in train_by_cycle.items():
        lst.sort(key=lambda x: (x.kind != "teach", stable_hash(seed, "order", c, x.id), x.id))

    return Stream(stream_id=stream_id, seed=seed, n_cycles=n_cycles, skills=skills,
                  train_by_cycle=train_by_cycle, hidden=hidden, gate=gate_out,
                  sealed=SealedStore(sealed_items, sealed_log_path), fingerprints=fps)


def synthetic_stream(seed: int, cfg: StreamConfig | None = None, **kw) -> Stream:
    cfg = cfg or StreamConfig()
    return build_stream(SyntheticSource(seed, cfg), seed=seed, n_cycles=cfg.n_cycles,
                        train_span=cfg.train_span, default_fracs=cfg.fracs, **kw)
