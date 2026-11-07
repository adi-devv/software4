"""Disjoint splits by stable hash, and an audit that proves they are disjoint.

For each skill the probe items are sorted by sha256(seed, skill, key) and cut into train / gate /
hidden / sealed by quota (so counts are exact and the assignment does not depend on input order).
Gate items are then dealt round-robin into per-cycle batches: a batch is used in one cycle only.
"""
from __future__ import annotations

import re

from .util import stable_hash

SPLITS = ("train", "gate", "hidden", "sealed")


def assign_splits(keys, seed, skill, fracs) -> dict:
    """key -> split. Order of `keys` does not matter."""
    fr = {s: float(fracs.get(s, 0.0)) for s in SPLITS}
    tot = sum(fr.values())
    if tot <= 0:
        raise ValueError("fracs must sum to > 0")
    keys = sorted(set(keys), key=lambda k: (stable_hash(seed, "split", skill, k), k))
    n = len(keys)
    raw = {s: fr[s] / tot * n for s in SPLITS}
    cnt = {s: int(raw[s]) for s in SPLITS}
    rem = n - sum(cnt.values())
    for s in sorted(SPLITS, key=lambda s: -(raw[s] - cnt[s]))[:rem]:
        cnt[s] += 1
    out, i = {}, 0
    for s in SPLITS:
        for k in keys[i: i + cnt[s]]:
            out[k] = s
        i += cnt[s]
    return out


def gate_batches(keys_in_gate, seed, skill, n_batches) -> dict:
    """key -> batch index (round-robin over hash order)."""
    order = sorted(keys_in_gate, key=lambda k: (stable_hash(seed, "gate", skill, k), k))
    return {k: i % max(1, n_batches) for i, k in enumerate(order)}


_TOKEN = re.compile(r"[a-z0-9']+")


def tokens(prompt: str):
    return _TOKEN.findall(prompt.lower())


def audit(stream) -> list:
    """Return a list of violation strings; an empty list means the splits are clean.

    Checks (sealed items are compared by hash only, so the audit does not read sealed content):
      * the same item id or item key in two splits
      * the same canonical input (e.g. a rule's input number, an entity+attribute) in two splits
      * the same normalised prompt text in two splits
      * near-duplicate content: Jaccard >= 0.9 on a skill's prompt tokens (minus tokens every prompt of the skill shares), across splits
      * a declared required token (entity / input) that is missing from the prompt
      * a gate item appearing in more than one per-cycle batch
    """
    bad = []
    by_id, by_key, by_in, by_prompt = {}, {}, {}, {}
    for fp in stream.fingerprints:
        for table, h, what in ((by_id, fp.id, "id"), (by_key, fp.key_h, "item key"),
                               (by_in, fp.input_h, "input"), (by_prompt, fp.prompt_h, "prompt")):
            prev = table.setdefault(h, fp.split)
            if prev != fp.split:
                bad.append(f"{what} shared by splits {prev} and {fp.split} (skill {fp.skill})")
        if fp.missing_required:
            bad.append(f"required token missing from prompt (skill {fp.skill}, split {fp.split})")
    per_skill = {}
    for fp in stream.fingerprints:
        per_skill.setdefault(fp.skill, []).append(fp)
    for skill, fps in per_skill.items():
        for i in range(len(fps)):
            a = fps[i]
            for j in range(i + 1, len(fps)):
                b = fps[j]
                if a.split == b.split or not a.content or not b.content:
                    continue
                inter = len(a.content & b.content)
                if inter and inter / len(a.content | b.content) >= 0.9:
                    bad.append(f"near-duplicate prompts across {a.split}/{b.split} (skill {skill})")
    seen = {}
    for sk, batches in stream.gate_batches.items():
        for bi, batch in enumerate(batches):
            for it in batch:
                if it.id in seen and seen[it.id] != bi:
                    bad.append(f"gate item {it.id} appears in batches {seen[it.id]} and {bi}")
                seen[it.id] = bi
    return bad
