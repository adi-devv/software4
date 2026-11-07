"""Convert raw benchmark rows into deterministic, gradable canary sets."""
import json
import re
from pathlib import Path

from .util import rng, short_id
from .worlds import SUFFIX

_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+")


def _sample(rows, n, seed, name):
    if n < 0:
        raise ValueError("n must be non-negative")
    order = rng("canary_data", name, seed).permutation(len(rows))
    return [rows[int(i)] for i in order[:n]]


def gsm8k_rows(raw_rows, n, seed):
    rows, seen = [], set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        question, answer = raw.get("question"), raw.get("answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or "####" not in answer:
            continue
        value = answer.rsplit("####", 1)[1].strip()
        if not _NUMBER.fullmatch(value):
            continue
        question = question.strip()
        cid = "gsm8k-" + short_id(question, n=24)
        if cid in seen:
            continue
        seen.add(cid)
        rows.append((cid, question + SUFFIX, value.replace(",", "")))
    return _sample(rows, n, seed, "gsm8k")


def arc_rows(raw_rows, n, seed):
    rows, seen = [], set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        cid, question, choices, key = (raw.get(k) for k in ("id", "question", "choices", "answerKey"))
        if not isinstance(cid, str) or not cid.strip() or not isinstance(question, str) or not question.strip() or not isinstance(choices, dict) or not isinstance(key, str):
            continue
        labels, texts = choices.get("label"), choices.get("text")
        if not isinstance(labels, list) or not isinstance(texts, list) or len(labels) != len(texts) or not labels:
            continue
        if not all(isinstance(x, str) and x.strip() for x in labels + texts):
            continue
        mapping = {str(i): chr(64 + i) for i in range(1, 5)}
        normalized = [mapping.get(label, label) for label in labels]
        key = mapping.get(key, key)
        if (len(set(normalized)) != len(normalized) or any(x not in "ABCDE" or len(x) != 1 for x in normalized) or key not in normalized):
            continue
        cid = "arc-" + cid.strip()
        if cid in seen:
            continue
        seen.add(cid)
        options = "\n".join(f"{label}. {text.strip()}" for label, text in zip(normalized, texts))
        rows.append((cid, question.strip() + "\n" + options + SUFFIX, key))
    return _sample(rows, n, seed, "arc")


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for cid, prompt, expected in rows:
            f.write(json.dumps({"id": cid, "prompt": prompt, "expected": expected}, ensure_ascii=False) + "\n")
