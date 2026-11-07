"""Seeded fictional worlds and the synthetic task source.

Everything is invented (made-up place, person and good names; made-up rules), so a pretrained
model cannot already know any answer. A `TaskSource` hands raw skill/item records to
`streams.build_stream`; `SyntheticSource` generates them from a seed and `JsonlSource` loads the
same records from a file (the "bigger model writes streams once" path).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .util import rng, stable_hash

ONSETS = ["b", "d", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "th", "kr", "gl", "sh",
          "qu", "fr", "br", "dr"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "io", "ou"]
CODAS = ["", "", "n", "l", "r", "s", "th", "x", "m", "k"]

SUFFIX = " End your reply with a line 'Final answer: <answer>'."

DOMAINS = {"village": "village", "guild": "guild", "creature": "clan"}
ATTRS = ["leader", "export", "founded", "population", "ally"]
LEAF_ATTRS = ["leader", "export", "founded", "population"]
ATTR_PHRASE = {"leader": "leader", "export": "main export", "founded": "founding year",
               "population": "population", "ally": "ally"}

Q = {
    "leader": ["Who leads {e}?", "Name the leader of {e}.", "{e} is led by whom?"],
    "export": ["What is the main export of {e}?", "Which good does {e} mainly export?",
               "{e} is known for exporting what?"],
    "founded": ["In what year was {e} founded?", "What is the founding year of {e}?",
                "When was {e} founded? Give the year."],
    "population": ["What is the population of {e}?", "How many people live in {e}?",
                   "State the population of {e}."],
    "ally": ["Which {noun} is the ally of {e}?", "Name the ally of {e}.",
             "Which {noun} is allied with {e}?"],
}
HOP_LEAF = {
    "leader": ["Who leads that ally?", "Name the leader of that ally."],
    "export": ["What is that ally's main export?", "Which good does that ally mainly export?"],
    "founded": ["In what year was that ally founded?", "What is that ally's founding year?"],
    "population": ["What is that ally's population?", "How many people live in that ally?"],
}
HOP_HEAD = ["Take the ally of {e}.", "Look at {e} and find its ally."]
TEACH = ["Field note: the {p} of {e} is {v}.", "Record: {e}, {p}: {v}."]
UPDATE = "Update notice: the {p} of {e} is now {v}."
CLAIM = "Unverified forum post: the {p} of {e} is actually {v}."


class NameFactory:
    def __init__(self, r):
        self.r = r
        self.used = set()

    def _syl(self):
        r = self.r
        return (ONSETS[r.integers(len(ONSETS))] + VOWELS[r.integers(len(VOWELS))]
                + CODAS[r.integers(len(CODAS))])

    def word(self, lo=2, hi=3):
        while True:
            w = "".join(self._syl() for _ in range(int(self.r.integers(lo, hi + 1))))
            if len(w) >= 5 and w not in self.used:
                self.used.add(w)
                return w

    def place(self):
        return self.word().capitalize()

    def person(self):
        while True:
            n = f"{self.word(2, 2).capitalize()} {self.word(2, 3).capitalize()}"
            return n


def _display(attr, v):
    return f"{int(v):,}" if attr == "population" else str(v)


# ---------------------------------------------------------------- rules


@dataclass
class Rule:
    name: str
    kind: str  # affine | digitsum | revadd | shift | compose
    params: dict
    text_input: bool = False

    def apply(self, x: str) -> str:
        p = self.params
        if self.kind == "shift":
            k = p["k"]
            return "".join(chr((ord(c) - 97 + k) % 26 + 97) if c.isalpha() else c for c in x)
        n = int(x)
        if self.kind == "affine":
            return str((p["a"] * n + p["b"]) % p["m"])
        if self.kind == "digitsum":
            return str(n + p["k"] * sum(int(c) for c in str(n)))
        if self.kind == "revadd":
            return str(int(str(n)[::-1]) + p["k"])
        if self.kind == "compose":
            m = int(str(n)[::-1]) + p["k"]
            return str((p["a"] * m + p["b"]) % p["m"])
        raise ValueError(self.kind)

    def describe(self) -> str:
        p = self.params
        if self.kind == "shift":
            return f"move every letter {p['k']} places forward in the alphabet (wrapping z to a)"
        if self.kind == "affine":
            return f"multiply by {p['a']}, add {p['b']}, then take the remainder after dividing by {p['m']}"
        if self.kind == "digitsum":
            return f"add {p['k']} times the sum of the digits to the number"
        if self.kind == "revadd":
            return f"reverse the digits and add {p['k']}"
        return (f"reverse the digits, add {p['k']}, multiply by {p['a']}, add {p['b']}, "
                f"then take the remainder after dividing by {p['m']}")


def make_rule(r, names: NameFactory, kind: str) -> Rule:
    params = {}
    if kind in ("affine", "compose"):
        params.update(a=int(r.integers(2, 10)), b=int(r.integers(1, 50)), m=int(r.integers(97, 998)))
    if kind in ("digitsum", "revadd", "compose"):
        params["k"] = int(r.integers(2, 10)) if kind != "compose" else int(r.integers(3, 30))
    if kind == "shift":
        params["k"] = int(r.integers(1, 25))
    return Rule(names.word(2, 2), kind, params, text_input=(kind == "shift"))


# ---------------------------------------------------------------- config and world


@dataclass
class StreamConfig:
    n_cycles: int = 16
    entities_per_domain: int = 30
    items_per_rule: int = 120
    rule_kinds: tuple = ("affine", "shift", "digitsum", "compose")
    train_span: int = 4          # cycles over which a skill's train probes are spread
    exposure_span: int = 2       # cycles over which a domain's facts are first stated
    revise_fraction: float = 0.25
    poison_fraction: float = 0.25
    fracs: dict = field(default_factory=lambda: {"train": 0.30, "gate": 0.25, "hidden": 0.30,
                                                 "sealed": 0.15})
    poison_fracs: dict = field(default_factory=lambda: {"train": 0.0, "gate": 0.35, "hidden": 0.45,
                                                        "sealed": 0.20})


@dataclass
class World:
    seed: int
    domains: dict   # domain -> {entity: {attr: value}}
    rules: list

    def canonical(self) -> str:
        return json.dumps({"seed": self.seed, "domains": self.domains,
                           "rules": [(r.name, r.kind, r.params) for r in self.rules]},
                          sort_keys=True)


def generate_world(seed: int, cfg: StreamConfig) -> World:
    r = rng("world", seed)
    names = NameFactory(r)
    domains = {}
    for dom in DOMAINS:
        ents = [names.place() for _ in range(cfg.entities_per_domain)]
        table = {}
        for e in ents:
            others = [o for o in ents if o != e]
            table[e] = {
                "leader": names.person(),
                "export": names.word(2, 2),
                "founded": str(int(r.integers(1100, 1900))),
                "population": str(int(r.integers(200, 90000))),
                "ally": others[int(r.integers(len(others)))],
            }
        domains[dom] = table
    rules = [make_rule(r, names, k) for k in cfg.rule_kinds]
    return World(seed, domains, rules)


def _other_value(r, names, world, dom, e, attr, avoid):
    for _ in range(100):
        if attr == "leader":
            v = names.person()
        elif attr == "export":
            v = names.word(2, 2)
        elif attr == "founded":
            v = str(int(r.integers(1100, 1900)))
        elif attr == "population":
            v = str(int(r.integers(200, 90000)))
        else:
            cands = [o for o in world.domains[dom] if o != e]
            v = cands[int(r.integers(len(cands)))]
        if v not in avoid:
            return v
    raise RuntimeError("could not find a distinct value")


# ---------------------------------------------------------------- sources


class TaskSource:
    """Interface: raw() -> (skills, items), plain dicts (the same shape a JSONL file holds)."""

    def raw(self):
        raise NotImplementedError


def _fact_sort(seed, dom, facts):
    return sorted(facts, key=lambda f: stable_hash(seed, "factorder", dom, f[0], f[1]))


class SyntheticSource(TaskSource):
    def __init__(self, seed: int, cfg: StreamConfig | None = None):
        self.seed = seed
        self.cfg = cfg or StreamConfig()
        self.world = generate_world(seed, self.cfg)

    # schedule: skill intro cycles as fractions of the stream
    def _sched(self):
        n = self.cfg.n_cycles
        s = [int(round(f * n)) for f in (0, 0.19, 0.375, 0.5, 0.625, 0.75)]
        return s

    def raw(self):
        cfg, w, seed = self.cfg, self.world, self.seed
        r = rng("tasks", seed)
        names = NameFactory(rng("tasknames", seed))
        names.used = {x for d in w.domains.values() for e in d for x in [e.lower()]}
        s = self._sched()
        skills, items = [], []

        def skill(name, family, intro, fr=None):
            skills.append({"type": "skill", "name": name, "family": family,
                           "intro_cycle": int(intro), "fracs": fr})

        # domain plan: (domain, fact intro)
        fact_intro = {"village": s[0], "guild": s[1], "creature": s[5]}
        revise_dom, rev_cycle = "village", s[3]
        poison_dom, poison_cycle = "guild", s[4]
        hop_plan = [("village", s[2]), ("guild", s[5])]
        rule_intro = [s[0], s[1], s[2], s[5]]

        revised, poisoned = {}, {}
        for dom, table in w.domains.items():
            facts = [(e, a) for e in table for a in ATTRS]
            order = _fact_sort(seed, dom, facts)
            if dom == revise_dom:
                k = int(math.ceil(cfg.revise_fraction * len(facts)))
                revised = {f: None for f in order[:k]}
            if dom == poison_dom:
                k = int(math.ceil(cfg.poison_fraction * len(facts)))
                poisoned = {f: None for f in order[:k]}

        noun = DOMAINS

        def qtext(attr, e, dom, ti):
            return Q[attr][ti % 3].format(e=e, noun=noun[dom])

        def probe_prompt(key, attr, e, dom):
            return qtext(attr, e, dom, stable_hash(seed, key) % 3) + SUFFIX

        for dom, table in w.domains.items():
            intro = fact_intro[dom]
            skill(f"fact:{dom}", "fact", intro)
            order = _fact_sort(seed, dom, [(e, a) for e in table for a in ATTRS])
            # exposures of every fact (original, trusted values)
            for i, (e, a) in enumerate(order):
                cyc = intro + (i % cfg.exposure_span)
                tmpl = TEACH[stable_hash(seed, "t", dom, e, a) % 2]
                items.append({"type": "item", "skill": f"fact:{dom}", "kind": "teach",
                              "key": f"teach|{dom}|{e}|{a}", "input_key": f"teach|{e}|{a}",
                              "prompt": tmpl.format(p=ATTR_PHRASE[a], e=e,
                                                    v=_display(a, table[e][a])),
                              "expected": None, "cycle": cyc, "source": "trusted"})
            for e, a in order:
                if (e, a) in revised or (e, a) in poisoned:
                    continue
                key = f"fact|{dom}|{e}|{a}"
                items.append({"type": "item", "skill": f"fact:{dom}", "kind": "probe", "key": key,
                              "input_key": f"{e}|{a}", "required_tokens": [e],
                              "prompt": probe_prompt(key, a, e, dom),
                              "expected": table[e][a]})

        # ---- revisions
        skill(f"revise:{revise_dom}", "revise", rev_cycle)
        table = w.domains[revise_dom]
        for (e, a) in revised:
            old = table[e][a]
            new1 = _other_value(r, names, w, revise_dom, e, a, {old})
            two = stable_hash(seed, "second", e, a) % 4 == 0
            hist = []
            new2 = None
            items.append({"type": "item", "skill": f"revise:{revise_dom}", "kind": "teach",
                          "key": f"update|{e}|{a}|1", "input_key": f"update|{e}|{a}|1",
                          "prompt": UPDATE.format(p=ATTR_PHRASE[a], e=e, v=_display(a, new1)),
                          "expected": None, "cycle": rev_cycle, "source": "trusted"})
            if two and rev_cycle + 3 < cfg.n_cycles:
                new2 = _other_value(r, names, w, revise_dom, e, a, {old, new1})
                hist = [[rev_cycle + 3, new2]]
                items.append({"type": "item", "skill": f"revise:{revise_dom}", "kind": "teach",
                              "key": f"update|{e}|{a}|2", "input_key": f"update|{e}|{a}|2",
                              "prompt": UPDATE.format(p=ATTR_PHRASE[a], e=e, v=_display(a, new2)),
                              "expected": None, "cycle": rev_cycle + 3, "source": "trusted"})
            key = f"revise|{revise_dom}|{e}|{a}"
            items.append({"type": "item", "skill": f"revise:{revise_dom}", "kind": "probe",
                          "key": key, "input_key": f"rev|{e}|{a}", "required_tokens": [e],
                          "prompt": probe_prompt(key, a, e, revise_dom), "expected": new1,
                          "history": hist, "prior": [old]})

        # ---- poison
        skill(f"poison:{poison_dom}", "poison", poison_cycle, cfg.poison_fracs)
        table = w.domains[poison_dom]
        for (e, a) in poisoned:
            true = table[e][a]
            false = _other_value(r, names, w, poison_dom, e, a, {true})
            for j in range(2):
                items.append({"type": "item", "skill": f"poison:{poison_dom}", "kind": "teach",
                              "key": f"claim|{e}|{a}|{j}", "input_key": f"claim|{e}|{a}|{j}",
                              "prompt": CLAIM.format(p=ATTR_PHRASE[a], e=e, v=_display(a, false)),
                              "expected": None, "cycle": poison_cycle + j, "source": "untrusted"})
            key = f"poison|{poison_dom}|{e}|{a}"
            items.append({"type": "item", "skill": f"poison:{poison_dom}", "kind": "probe",
                          "key": key, "input_key": f"poi|{e}|{a}", "required_tokens": [e],
                          "prompt": probe_prompt(key, a, e, poison_dom), "expected": true,
                          "false_value": false})

        # ---- multi-hop (needs two facts)
        for dom, hintro in hop_plan:
            skill(f"hop:{dom}", "hop", hintro)
            table = w.domains[dom]
            for e in table:
                b = table[e]["ally"]
                for leaf in LEAF_ATTRS:
                    key = f"hop|{dom}|{e}|{leaf}"
                    h = stable_hash(seed, key)
                    prompt = (HOP_HEAD[h % 2].format(e=e) + " " + HOP_LEAF[leaf][(h >> 8) % 2]
                              + SUFFIX)
                    items.append({"type": "item", "skill": f"hop:{dom}", "kind": "probe",
                                  "key": key, "input_key": f"hop|{e}|{leaf}",
                                  "required_tokens": [e], "prompt": prompt,
                                  "expected": w.domains[dom][b][leaf]})

        # ---- rules (inputs unique across all rules)
        n_rules = len(w.rules)
        nums = r.permutation(np_range(10, 10000))[: n_rules * cfg.items_per_rule]
        ni = 0
        for ri, rule in enumerate(w.rules):
            sk = f"rule:{rule.name}"
            skill(sk, "rule", rule_intro[ri % len(rule_intro)])
            if rule.text_input:
                inputs = [names.word(2, 3) for _ in range(cfg.items_per_rule)]
            else:
                inputs = [str(int(x)) for x in nums[ni: ni + cfg.items_per_rule]]
                ni += cfg.items_per_rule
            for x in inputs:
                key = f"rule|{rule.name}|{x}"
                t = stable_hash(seed, key) % 3
                prompt = ["Apply the {n} rule to {x}.", "What does the {n} rule give for {x}?",
                          "Run {x} through the {n} rule and reply with the result."][t].format(
                    n=rule.name, x=x) + SUFFIX
                items.append({"type": "item", "skill": sk, "kind": "probe", "key": key,
                              "input_key": x, "required_tokens": [x], "prompt": prompt,
                              "expected": rule.apply(x),
                              "lesson": f"Rule '{rule.name}': {rule.describe()}."})
        return skills, items


def np_range(a, b):
    import numpy as np
    return np.arange(a, b)


class JsonlSource(TaskSource):
    """Loads {"type":"skill"|"item", ...} lines (see streams.build_stream for the fields)."""

    def __init__(self, path):
        self.path = Path(path)

    def raw(self):
        skills, items = [], []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                (skills if d.get("type") == "skill" else items).append(d)
        return skills, items


def dump_jsonl(source: TaskSource, path) -> None:
    skills, items = source.raw()
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for d in skills + items:
            f.write(json.dumps(d, sort_keys=True) + "\n")
