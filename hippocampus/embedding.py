"""Pluggable embedder and entity extractor, plus deterministic offline defaults.

Embedder: any object with `embed(texts) -> float array (n, d)` of L2-normalised rows.
Extractor: any object with `extract(text) -> list[str]` of entity phrases.
The dimension is read from the embedder's output, never hardcoded.
"""
from __future__ import annotations

import functools
import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have he her his how i if in into is it its
me my no not of on or our she so than that the their them then there these they this to us was we were what when where
which who whom why will with would you your
""".split())


def normalize_phrase(p: str) -> str:
    return " ".join(p.lower().split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class Extractor(Protocol):
    def extract(self, text: str) -> list[str]: ...


@functools.lru_cache(maxsize=200_000)
def _features(tok: str, dim: int, ngram: int, ngram_w: float):
    def h(s: str):
        d = hashlib.blake2b(s.encode("utf-8"), digest_size=9).digest()
        return int.from_bytes(d[:8], "little") % dim, (1.0 if d[8] & 1 else -1.0)

    feats = [(*h("w:" + tok), 1.0)]
    if ngram_w > 0 and len(tok) >= ngram + 1:
        padded = f"#{tok}#"
        for i in range(len(padded) - ngram + 1):
            feats.append((*h("c:" + padded[i:i + ngram]), ngram_w))
    return tuple(feats)


class HashingEmbedder:
    """Signed feature hashing of content words (plus light character n-grams).

    Deterministic and offline. It has NO real semantic meaning: texts only look
    similar when they share words or word pieces.
    """

    def __init__(self, dim: int = 256, ngram: int = 3, ngram_weight: float = 0.3):
        self.dim, self.ngram, self.ngram_weight = dim, ngram, ngram_weight

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = _tokens(text)
            content = [t for t in toks if t not in STOPWORDS] or toks
            for tok in content:
                for idx, sign, w in _features(tok, self.dim, self.ngram, self.ngram_weight):
                    out[i, idx] += sign * w
            n = float(np.linalg.norm(out[i]))
            if n > 0:
                out[i] /= n
        return out


class RuleExtractor:
    """Capitalised word runs and identifiers/numbers; falls back to content words if too few."""

    _CAP = re.compile(r"\b[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*")
    _IDENT = re.compile(r"\b(?:\w*[\d_]\w*|\d+)\b")

    def __init__(self, max_entities: int = 12, min_entities: int = 2):
        self.max_entities, self.min_entities = max_entities, min_entities

    def extract(self, text: str) -> list[str]:
        found: list[str] = []
        for m in self._CAP.finditer(text):
            words = m.group(0).split()
            while words and words[0].lower() in STOPWORDS:
                words.pop(0)
            if words:
                found.append(" ".join(words))
        found += self._IDENT.findall(text)
        out = list(dict.fromkeys(normalize_phrase(f) for f in found if f.strip()))
        if len(out) < self.min_entities:
            for tok in _tokens(text):
                if tok not in STOPWORDS and len(tok) >= 3 and tok not in out:
                    out.append(tok)
                if len(out) >= self.min_entities + 2:
                    break
        return out[: self.max_entities]
