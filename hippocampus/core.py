"""The hippocampus: append-only log + graph/vector indexes + plastic links.

Every mutation is written to the log as an event and then applied by `_apply`, the
same function that replays the log. So all state (indexes, link weights, fast
index) is rebuildable from the log alone.
"""
from __future__ import annotations

import math
import random
import re
from copy import deepcopy

import numpy as np

from .clock import SystemClock
from .embedding import HashingEmbedder, RuleExtractor, STOPWORDS, _tokens, normalize_phrase
from .eventlog import EventLog
from .graph import personalized_pagerank
from .models import Config, Hit, ReplayItem
from .plasticity import Plasticity


class _Mem:
    __slots__ = ("id", "kind", "source", "created", "trust", "quarantined", "released_at",
                 "tombstoned_at", "versions", "slot", "contrib", "meta")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def eligible(self) -> bool:
        return self.tombstoned_at is None and not self.quarantined


class Hippocampus:
    def __init__(self, log_path=None, *, embedder=None, extractor=None, config=None, clock=None):
        self.cfg = config or Config()
        self.embedder = embedder or HashingEmbedder()
        self.extractor = extractor or RuleExtractor(self.cfg.max_entities)
        self.clock = clock or SystemClock()
        self.log = EventLog(log_path)
        self.stats = {"fast_calls": 0, "deep_calls": 0}
        self.last_trace: dict = {}
        self._reset()
        self._replay_log()

    # ------------------------------------------------------------------ state
    def _reset(self) -> None:
        c = self.cfg
        self._mems: dict[str, _Mem] = {}
        self._order: list[str] = []
        self._slot_ids: list[str] = []
        self._vecs = None
        self._alive = np.zeros(0, dtype=bool)
        self._trust = np.zeros(0)
        self._lt = np.zeros(0)                       # time of latest version, per slot
        self._phr: dict[str, int] = {}
        self._phr_list: list[str] = []
        self._pvecs = None
        self._live: list[int] = []                   # live contributors per phrase
        self._cooc: dict[tuple[int, int], int] = {}
        self._syn: dict[tuple[int, int], float] = {}
        self._plast = Plasticity(c.half_life, c.node_rate, c.link_rate, c.max_strength)
        self._pending: list[dict] = []
        self._outcomes: list[dict] = []
        self._sleeps: list[float] = []
        self._fast: set[int] = set()
        self._recent: list[int] = []
        self._gver = 0
        self._ecache = None
        self._lcache = None
        self._ncache = None
        self._old_vec_cache: dict = {}
        self._add_count = 0
        self._fact_ids: dict[tuple[str, str], str] = {}
        self._fact_members: dict[tuple[str, str], set[str]] = {}
        self.last_nap: float | None = None
        self.last_nap_info: dict = {}

    def _replay_log(self) -> None:
        for ev in self.log.events:
            self._apply(ev)

    def rebuild(self) -> "Hippocampus":
        """Throw away every index and link weight and rebuild them from the log file."""
        self.log.reload()
        self._reset()
        self._replay_log()
        return self

    def _t(self, t) -> float:
        return float(self.clock() if t is None else t)

    def _emit(self, ev: dict) -> dict:
        ev = self.log.append(ev)
        self._apply(ev)
        return ev

    # ------------------------------------------------------------- public writes
    def add(self, text: str, *, kind: str = "fact", source: str = "user", trust: float | None = None,
            entities=None, meta=None, t=None, id: str | None = None, quarantine: bool = False) -> str:
        """Append a memory, or version an identified, trusted fact revision.

        In the controlled ``Field note/Update notice: the REL of SUBJECT is
        VALUE`` grammar, REL+SUBJECT identify a single-valued fact. An explicit
        update notice with no caller-supplied id versions that identity instead
        of adding a contradictory live passage. This is exact grammar matching,
        never embedding similarity. Quarantined or lower-trust assertions cannot
        overwrite trusted facts. Originals remain accessible via get/as_of, and
        log replay reconstructs the identity index just like the vector index.
        Other prose and explicit ids retain the normal append semantics.
        """
        if not text or not text.strip():
            raise ValueError("empty text")
        mid = id or f"m{self._add_count:06d}"
        if mid in self._mems:
            raise ValueError(f"id exists: {mid}")
        trusted = source in self.cfg.trusted_sources
        if trust is None:
            trust = self.cfg.default_trust if trusted else self.cfg.untrusted_trust
        key = self._fact_key(text) if kind == 'fact' else None
        previous = self._mems.get(self._fact_ids.get(key))
        if (id is None and key is not None and 'update notice:' in text.lower()
                and trusted and not quarantine and previous is not None
                and previous.eligible and trust >= previous.trust):
            self.update(previous.id, text, entities=entities, t=t,
                        trust=trust, source=source, meta=meta or {})
            return previous.id
        self._emit({"type": "add", "t": self._t(t), "id": mid, "kind": kind, "source": source,
                    "text": text, "trust": float(trust), "quarantined": bool(quarantine or not trusted),
                    "entities": self._entities(text, entities), "meta": meta or {}})
        return mid

    def update(self, id: str, text: str, *, entities=None, t=None,
               trust=None, source=None, meta=None) -> int:
        m = self._get(id)
        if m.tombstoned_at is not None:
            raise ValueError("memory is tombstoned")
        self._emit({"type": "update", "t": self._t(t), "id": id, "text": text,
                    "entities": self._entities(text, entities),
                    "trust": m.trust if trust is None else float(trust),
                    "source": m.source if source is None else source,
                    "meta": deepcopy(m.meta if meta is None else meta)})
        return len(m.versions)

    def tombstone(self, id: str, reason: str = "", t=None) -> None:
        self._get(id)
        self._emit({"type": "tombstone", "t": self._t(t), "id": id, "reason": reason})

    def release(self, id: str, *, trust: float | None = None, t=None) -> None:
        """Release a quarantined memory into recall and replay."""
        self._get(id)
        self._emit({"type": "release", "t": self._t(t), "id": id, "trust": trust})

    def set_trust(self, id: str, trust: float, t=None) -> None:
        self._get(id)
        self._emit({"type": "trust", "t": self._t(t), "id": id, "trust": float(trust)})

    def record_outcome(self, task_id: str, memory_ids=(), *, success: bool, verified: bool = False,
                       correction: str | None = None, episode_id: str | None = None, t=None) -> None:
        """Log how a task ended. Weights do NOT change here; they change at nap()."""
        ids = sorted(set(memory_ids))
        for i in ids + ([episode_id] if episode_id else []):
            self._get(i)
        self._emit({"type": "outcome", "t": self._t(t), "task_id": str(task_id), "memory_ids": ids,
                    "success": bool(success), "verified": bool(verified),
                    "correction": correction, "episode_id": episode_id})

    def nap(self, now=None) -> dict:
        """Apply pending outcomes to link strengths and refresh the fast index. Trains nothing."""
        self._emit({"type": "nap", "t": self._t(now)})
        return dict(self.last_nap_info)

    def mark_sleep(self, t=None) -> None:
        """Record that a deep sleep ran, so replay can tell new episodes from old."""
        self._emit({"type": "sleep", "t": self._t(t)})

    # ---------------------------------------------------------------- reading
    def get(self, id: str, version: int | None = None, as_of: float | None = None) -> dict:
        m = self._get(id)
        v = self._pick_version(m, version, as_of)
        historical = version is not None or as_of is not None
        return {"id": m.id, "version": v[0], "t": v[1], "text": v[2], "entities": list(v[3]),
                "kind": m.kind, "source": v[4] if historical else m.source,
                "trust": v[5] if historical else m.trust, "quarantined": m.quarantined,
                "tombstoned": m.tombstoned_at is not None,
                "meta": deepcopy(v[6] if historical else m.meta)}

    def strength(self, id: str, now=None) -> float:
        return self._plast.node(id, self._t(now))

    def link_weight(self, a: str, b: str, now=None) -> float:
        return self._plast.link(a, b, self._t(now))

    def link_weights(self, now=None) -> dict:
        now = self._t(now)
        return {k: self._plast.fade(v, t, now) for k, (v, t) in sorted(self._plast.links.items())}

    def raw_state(self) -> dict:
        """Exact plastic state (values and last-strengthened times) for equality checks."""
        s = self._plast.state()
        return {"nodes": sorted(s["nodes"].items()), "links": sorted(s["links"].items()),
                "fast": sorted(self._slot_ids[i] for i in self._fast)}

    def fast_ids(self) -> list[str]:
        return sorted(self._slot_ids[s] for s in self._fast)

    def __len__(self) -> int:
        return len(self._mems)

    # ----------------------------------------------------------------- recall
    def recall(self, query: str, k: int = 5, *, deep: bool | None = None, min_trust: float = 0.0,
               as_of: float | None = None, now=None, entities=None) -> list[Hit]:
        """Check automatic fast recall against the complete eligible vector index.

        A cosine threshold on a bounded cache cannot establish global confidence:
        an omitted passage may score higher. Accept a confident fast winner only
        when it also wins the full-store cosine check. For k > 1, score the full
        store before ranking, rather than returning just the cache's candidates.
        With no learned links this returns exactly vector_search's top-k on the
        confident path; k=1 needs only a matrix-vector product and argmax.

        Confidence also cannot establish that one passage answers a compound
        question. If the winning passage lacks content words from the query,
        use PPR to seek linked evidence, even when cosine is high. This lexical
        coverage check is deliberately conservative and independent of question
        templates, entity names, and embedder scores.

        PPR and learned-link reserves can displace cosine neighbours: no fixed
        k can both retain all k vector hits and insert a new graph-only hit.
        The full-vector candidate guarantee concerns the confident cosine path,
        not a universal relevance guarantee for graph reranking.
        deep=True forces deep only, deep=False forces the bounded fast pass only.
        """
        if k <= 0:
            return []
        now = self._t(now)
        qents = self._entities(query, entities)
        if as_of is not None:
            self.last_trace = {"fast_best": None, "deep_ran": False, "as_of": True}
            return self._recall_as_of(query, qents, k, min_trust, as_of)
        q = self.embedder.embed([query])[0]
        self.last_trace = {"fast_best": None, "deep_ran": False, "vector_checked": False}
        sims = None
        if deep is not True:
            self.stats["fast_calls"] += 1
            fast = self._fast_pass(q, k, min_trust, now)
            self.last_trace["fast_best"] = fast[0][1] if fast else None
            if deep is False:
                return [self._hit(s, sc, "fast") for s, sc in fast]
            if fast and fast[0][1] >= self.cfg.confidence:
                sims = self._vector_scores(q, min_trust)
                self.last_trace["vector_checked"] = True
                best = int(np.argmax(sims))
                consistent = sims[best] > 0 and fast[0][0] == best
                self.last_trace["fast_consistent"] = bool(consistent)
                text = self._mems[self._slot_ids[best]].versions[-1][2]
                uncovered = set(_tokens(query)) - STOPWORDS - set(_tokens(text))
                self.last_trace["uncovered_terms"] = sorted(uncovered)
                if consistent and not uncovered:
                    if k > 1:
                        fast = self._fast_pass(q, k, min_trust, now, full_sims=sims)
                    return [self._hit(s, sc, "fast") for s, sc in fast]
        self.stats["deep_calls"] += 1
        self.last_trace["deep_ran"] = True
        return [self._hit(s, sc, "deep") for s, sc in self._deep_pass(q, qents, k, min_trust, now, sims=sims)]

    def vector_search(self, query: str, k: int = 5, *, min_trust: float = 0.0) -> list[Hit]:
        """Plain cosine search over all eligible memories, also used to check recall."""
        q = self.embedder.embed([query])[0]
        n = len(self._slot_ids)
        if n == 0:
            return []
        sims = self._vector_scores(q, min_trust)
        idx = np.flatnonzero(sims > 0)
        idx = sorted(idx.tolist(), key=lambda s: (-float(sims[s]), s))[:k]
        return [self._hit(s, float(sims[s]), "vector") for s in idx]

    def _vector_scores(self, q, min_trust):
        """One shared eligibility mask for the full-store check and vector baseline."""
        n = len(self._slot_ids)
        sims = self._vecs[:n] @ q
        ok = self._alive[:n] & (self._trust[:n] >= min_trust) & (sims > 0)
        return np.where(ok, sims, 0.0)

    def _hit(self, slot: int, score: float, via: str) -> Hit:
        m = self._mems[self._slot_ids[slot]]
        v = m.versions[-1]
        return Hit(m.id, v[0], v[2], float(score), via, m.kind, m.trust)

    def _fast_pass(self, q, k, min_trust, now, *, full_sims=None):
        cand = (sorted(self._fast | set(self._recent)) if full_sims is None
                else np.flatnonzero(full_sims > 0).tolist())
        cand = [s for s in cand if self._alive[s] and self._trust[s] >= min_trust]
        if not cand:
            return []
        sims = self._vecs[cand] @ q if full_sims is None else full_sims[cand]
        scores = {s: float(x) for s, x in zip(cand, sims) if x > 0}
        self._link_cache()
        seeds = sorted(scores, key=lambda s: (-scores[s], s))[: self.cfg.hop_seeds]
        hops: dict[int, float] = {}
        for s in seeds:                                   # one hop along strong links
            for nbr, key in self._lcache["adj"].get(self._slot_ids[s], ()):
                eff = self._plast.link(key[0], key[1], now)
                if eff < self.cfg.hop_min_link:
                    continue
                ns = self._mems[nbr].slot
                if not (self._alive[ns] and self._trust[ns] >= min_trust):
                    continue
                sc = scores[s] * eff * self.cfg.hop_decay
                if sc > scores.get(ns, 0.0):
                    scores[ns] = sc
                    hops[ns] = sc
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        # linked neighbours are the point of plastic links: keep the strongest few even if cosine
        # candidates outscore them, but never ahead of what they were reached from
        kept = {s for s, _ in ranked}
        extra = [s for s in sorted(hops, key=lambda s: (-hops[s], s)) if s not in kept][: self.cfg.hop_reserve]
        if extra and len(ranked) >= k:
            keep_n = max(len(ranked) - len(extra), 1)
            ranked = ranked[:keep_n] + [(s, scores[s]) for s in extra][: len(ranked) - keep_n]
        return ranked

    def _deep_pass(self, q, qents, k, min_trust, now, *, sims=None):
        S = len(self._slot_ids)
        if S == 0:
            return []
        c = self.cfg
        self._edge_cache()
        ec = self._ecache
        H = len(self._phr_list)
        N = S + H
        node_ok = np.ones(N, dtype=bool)
        node_ok[:S] = self._alive[:S] & (self._trust[:S] >= min_trust)
        # plastic node strength scales passage<->phrase edges
        s_eff = np.zeros(S)
        self._node_cache()
        nc = self._ncache
        if len(nc["slots"]):
            age = np.maximum(now - nc["t"], 0.0)
            s_eff[nc["slots"]] = nc["v"] * 0.5 ** (age / self._plast.half_life)
        w = ec["w"].copy()
        pn = ec["pn"]
        m = pn >= 0
        w[m] *= 1.0 + c.node_boost * s_eff[pn[m]]
        src, dst = ec["src"], ec["dst"]
        # plastic passage<->passage links
        self._link_cache()
        lc = self._lcache
        if len(lc["a"]):
            age = np.maximum(now - lc["t"], 0.0)
            lw = c.link_scale * lc["v"] * 0.5 ** (age / self._plast.half_life)
            src = np.concatenate([src, lc["a"], lc["b"]])
            dst = np.concatenate([dst, lc["b"], lc["a"]])
            w = np.concatenate([w, lw, lw])
        keep = node_ok[src] & node_ok[dst] & (w > 0)
        src, dst, w = src[keep], dst[keep], w[keep]
        # seeds: query-matched phrases + vector-similar passages
        seed = np.zeros(N)
        ph = self._phrase_seeds(qents)
        if ph:
            tot = sum(ph.values())
            for i, wt in ph.items():
                seed[S + i] += c.phrase_seed_weight * wt / tot
        sims = self._vecs[:S] @ q if sims is None else sims
        sims = np.where(node_ok[:S] & (sims > 0), sims, 0.0)
        top = sorted(np.flatnonzero(sims).tolist(), key=lambda s: (-float(sims[s]), s))[: c.passage_seed_k]
        if top:
            tot = float(sum(sims[s] for s in top))
            for s in top:
                seed[s] += c.passage_seed_weight * float(sims[s]) / tot
        if seed.sum() <= 0:
            return []
        seed /= seed.sum()
        x = personalized_pagerank(src, dst, w, N, seed, c.damping, c.ppr_iters, c.ppr_tol)
        sc = np.where(node_ok[:S], x[:S], 0.0)
        idx = np.flatnonzero(sc > 1e-12)
        idx = sorted(idx.tolist(), key=lambda s: (-float(sc[s]), s))[:k]
        top_score = float(sc[idx[0]]) if idx else 1.0
        return [(s, float(sc[s]) / top_score) for s in idx]

    def _phrase_seeds(self, qents) -> dict[int, float]:
        out: dict[int, float] = {}
        H = len(self._phr_list)
        if H == 0:
            return out
        live = np.array(self._live) > 0
        for e in qents:
            i = self._phr.get(e)
            if i is not None and self._live[i] > 0:
                out[i] = max(out.get(i, 0.0), 1.0)
            v = self.embedder.embed([e])[0]
            sims = self._pvecs[:H] @ v
            for j in np.flatnonzero(live & (sims >= self.cfg.phrase_match_threshold)).tolist():
                out[j] = max(out.get(j, 0.0), float(sims[j]))
        return out

    def _recall_as_of(self, query, qents, k, min_trust, as_of):
        """Historical recall: versions as they were at `as_of`; brute force, no graph."""
        q = self.embedder.embed([query])[0]
        qset = set(qents)
        scored = []
        for mid in self._order:
            m = self._mems[mid]
            if m.created > as_of or m.trust < min_trust:
                continue
            if m.tombstoned_at is not None and m.tombstoned_at <= as_of:
                continue
            if m.quarantined or (m.released_at is not None and m.released_at > as_of):
                continue
            ver = self._pick_version(m, None, as_of)
            if ver[0] == len(m.versions):
                vec = self._vecs[m.slot]
            else:
                key = (mid, ver[0])
                if key not in self._old_vec_cache:
                    self._old_vec_cache[key] = self.embedder.embed([ver[2]])[0]
                vec = self._old_vec_cache[key]
            sc = float(vec @ q)
            if qset:
                sc += 0.5 * len(qset & set(ver[3])) / len(qset)
            if sc > 0:
                scored.append((sc, mid, ver))
        scored.sort(key=lambda r: (-r[0], r[1]))
        return [Hit(mid, ver[0], ver[2], sc, "deep", self._mems[mid].kind, self._mems[mid].trust)
                for sc, mid, ver in scored[:k]]

    # ----------------------------------------------------------------- replay
    def replay_batch(self, n: int, *, seed: int = 0, new_fraction: float = 0.5,
                     min_trust: float = 0.5) -> list[ReplayItem]:
        """Episodes for deep sleep: verified successes and corrected failures from trusted,
        non-quarantined, non-tombstoned memories; a mix of new (since last sleep) and old.
        No surprise filter. Deterministic for a given seed."""
        last = self._sleeps[-1] if self._sleeps else None
        new, old = [], []
        for o in self._outcomes:
            if not o["verified"]:
                continue
            if o["success"]:
                kind = "success"
            elif o.get("correction"):
                kind = "corrected_failure"
            else:
                continue
            ids = list(o["memory_ids"]) + ([o["episode_id"]] if o.get("episode_id") else [])
            ms = [self._mems.get(i) for i in ids]
            if any(m is None or not m.eligible or m.trust < min_trust for m in ms):
                continue
            ep = self._mems.get(o["episode_id"]) if o.get("episode_id") else None
            item = ReplayItem(o["task_id"], kind, ep.versions[o["episode_version"] - 1][2] if ep else "", o.get("correction"),
                              list(o["memory_ids"]), o.get("episode_id"), o["t"],
                              last is None or o["t"] > last, min((m.trust for m in ms), default=1.0))
            (new if item.new else old).append(item)
        n_new = min(len(new), int(round(n * new_fraction)))
        n_old = min(len(old), n - n_new)
        n_new = min(len(new), n - n_old)
        rng = random.Random(seed)
        pick_new = sorted(rng.sample(range(len(new)), n_new))
        pick_old = sorted(rng.sample(range(len(old)), n_old))
        return [new[i] for i in pick_new] + [old[i] for i in pick_old]

    # ------------------------------------------------------------ log -> state
    @staticmethod
    def _fact_key(text):
        match = re.fullmatch(
            r'(?:Fact:\s*)?(?:Field note|Update notice):\s*the (.+?) of (.+?) is (?:now )?[^.\n]+\.?',
            text.strip(), re.I)
        return tuple(normalize_phrase(p) for p in match.groups()) if match else None

    def _index_fact(self, m, old_key=None):
        """Reselect affected identities from all members, also during event replay."""
        key = self._fact_key(m.versions[-1][2]) if m.kind == 'fact' else None
        if old_key is not None and old_key != key:
            self._fact_members.get(old_key, set()).discard(m.id)
        if key is not None:
            self._fact_members.setdefault(key, set()).add(m.id)
        for affected in {old_key, key} - {None}:
            eligible = [self._mems[mid] for mid in self._fact_members.get(affected, ())
                        if self._mems[mid].eligible
                        and self._mems[mid].source in self.cfg.trusted_sources]
            if eligible:
                winner = max(eligible, key=lambda x: (x.trust, x.versions[-1][1], x.slot))
                self._fact_ids[affected] = winner.id
            else:
                self._fact_ids.pop(affected, None)

    def _entities(self, text, entities) -> list[str]:
        raw = entities if entities is not None else self.extractor.extract(text)
        out = list(dict.fromkeys(normalize_phrase(e) for e in raw if e and e.strip()))
        return out[: self.cfg.max_entities]

    def _get(self, id: str) -> _Mem:
        try:
            return self._mems[id]
        except KeyError:
            raise KeyError(f"unknown memory id: {id}") from None

    @staticmethod
    def _pick_version(m: _Mem, version, as_of):
        if version is not None:
            if not 1 <= version <= len(m.versions):
                raise KeyError(f"no version {version} of {m.id}")
            return m.versions[version - 1]
        if as_of is None:
            return m.versions[-1]
        ok = [v for v in m.versions if v[1] <= as_of]
        if not ok:
            raise KeyError(f"{m.id} did not exist at {as_of}")
        return ok[-1]

    def _apply(self, ev: dict) -> None:
        h = getattr(self, "_on_" + str(ev.get("type")), None)
        if h is not None:
            h(ev)

    def _set_slot(self, m: _Mem, vec, t: float) -> None:
        s = m.slot
        self._vecs[s] = vec
        self._lt[s] = t
        self._alive[s] = m.eligible
        self._trust[s] = m.trust

    def _on_add(self, ev) -> None:
        mid = ev["id"]
        if mid in self._mems:
            return
        vec = self.embedder.embed([ev["text"]])[0]
        slot = len(self._slot_ids)
        if self._vecs is None:
            self._vecs = np.zeros((64, len(vec)), dtype=np.float32)
            self._alive = np.zeros(64, dtype=bool)
            self._trust = np.zeros(64)
            self._lt = np.zeros(64)
        if slot >= len(self._vecs):
            grow = len(self._vecs)
            self._vecs = np.vstack([self._vecs, np.zeros((grow, self._vecs.shape[1]), np.float32)])
            self._alive = np.concatenate([self._alive, np.zeros(grow, bool)])
            self._trust = np.concatenate([self._trust, np.zeros(grow)])
            self._lt = np.concatenate([self._lt, np.zeros(grow)])
        m = _Mem(id=mid, kind=ev.get("kind", "fact"), source=ev.get("source", ""), created=ev["t"],
                 trust=float(ev.get("trust", 1.0)), quarantined=bool(ev.get("quarantined", False)),
                 released_at=None, tombstoned_at=None,
                 versions=[(1, ev["t"], ev["text"], tuple(ev.get("entities", ())),
                            ev.get("source", ""), float(ev.get("trust", 1.0)),
                            deepcopy(ev.get("meta", {})))],
                 slot=slot, contrib=(), meta=deepcopy(ev.get("meta", {})))
        self._mems[mid] = m
        self._order.append(mid)
        self._slot_ids.append(mid)
        self._add_count += 1
        self._index_fact(m)
        self._set_slot(m, vec, ev["t"])
        if m.eligible:
            self._graph_add(m)
            self._push_recent(slot)
        self._gver += 1

    def _on_update(self, ev) -> None:
        m = self._mems.get(ev["id"])
        if m is None or m.tombstoned_at is not None:
            return
        if m.eligible:
            self._graph_remove(m)
        old_key = self._fact_key(m.versions[-1][2])
        m.trust = float(ev.get('trust', m.trust))
        m.source = ev.get('source', m.source)
        m.meta = deepcopy(ev.get('meta', m.meta))
        m.versions.append((len(m.versions) + 1, ev["t"], ev["text"], tuple(ev.get("entities", ())),
                           m.source, m.trust, deepcopy(m.meta)))
        self._index_fact(m, old_key)
        self._set_slot(m, self.embedder.embed([ev["text"]])[0], ev["t"])
        if m.eligible:
            self._graph_add(m)
            self._push_recent(m.slot)
        self._gver += 1

    def _on_tombstone(self, ev) -> None:
        m = self._mems.get(ev["id"])
        if m is None or m.tombstoned_at is not None:
            return
        if m.eligible:
            self._graph_remove(m)
        m.tombstoned_at = ev["t"]
        self._index_fact(m)
        self._alive[m.slot] = False
        self._fast.discard(m.slot)
        if m.slot in self._recent:
            self._recent.remove(m.slot)
        self._gver += 1

    def _on_release(self, ev) -> None:
        m = self._mems.get(ev["id"])
        if m is None or not m.quarantined:
            return
        m.quarantined = False
        m.released_at = ev["t"]
        if ev.get("trust") is not None:
            m.trust = float(ev["trust"])
        self._trust[m.slot] = m.trust
        self._alive[m.slot] = m.eligible
        self._index_fact(m)
        if m.eligible:
            self._graph_add(m)
            self._push_recent(m.slot)
        self._gver += 1

    def _on_trust(self, ev) -> None:
        m = self._mems.get(ev["id"])
        if m is not None:
            m.trust = float(ev["trust"])
            self._trust[m.slot] = m.trust
            self._index_fact(m)

    def _on_outcome(self, ev) -> None:
        ev = dict(ev)
        ep = self._mems.get(ev.get("episode_id"))
        ev["episode_version"] = len(ep.versions) if ep else None  # bind in log order, including on rebuild
        self._outcomes.append(ev)
        self._pending.append(ev)

    def _on_sleep(self, ev) -> None:
        self._sleeps.append(ev["t"])

    def _on_nap(self, ev) -> None:
        t = ev["t"]
        applied = 0
        for o in self._pending:
            if o["success"] and o["verified"]:        # only a verified success strengthens anything
                ids = [i for i in o["memory_ids"] if i in self._mems and self._mems[i].eligible]
                if ids:
                    self._plast.strengthen(ids, o["t"])
                    applied += 1
        info = {"pending": len(self._pending), "applied": applied}
        self._pending = []
        self._lcache = self._ncache = None
        self._refresh_fast(t)
        self.last_nap = t
        self.last_nap_info = info

    # ------------------------------------------------------------ fast index
    def _push_recent(self, slot: int) -> None:
        if slot in self._recent:
            self._recent.remove(slot)
        self._recent.append(slot)
        if len(self._recent) > self.cfg.fast_recent_cap:
            self._recent.pop(0)

    def _refresh_fast(self, now: float) -> None:
        n = len(self._slot_ids)
        slots = [s for s in range(n) if self._alive[s]]
        c = self.cfg
        prio = []
        for s in slots:
            rec = 0.5 ** (max(now - self._lt[s], 0.0) / c.recent_half_life)
            prio.append(self._plast.node(self._slot_ids[s], now) + c.recency_weight * rec)
        order = np.argsort(-np.array(prio), kind="stable")[: c.fast_size] if slots else []
        self._fast = {slots[i] for i in order}
        self._recent = []

    # ----------------------------------------------------------------- graph
    def _phrase(self, text: str) -> int:
        i = self._phr.get(text)
        if i is not None:
            return i
        v = self.embedder.embed([text])[0]
        i = len(self._phr_list)
        if self._pvecs is None:
            self._pvecs = np.zeros((64, len(v)), dtype=np.float32)
        if i >= len(self._pvecs):
            self._pvecs = np.vstack([self._pvecs, np.zeros_like(self._pvecs)])
        if i > 0:
            sims = self._pvecs[:i] @ v
            cand = np.flatnonzero(sims >= self.cfg.syn_threshold)
            cand = sorted(cand.tolist(), key=lambda j: (-float(sims[j]), j))[: self.cfg.syn_topk]
            for j in cand:
                self._syn[(j, i)] = float(sims[j])
        self._pvecs[i] = v
        self._phr[text] = i
        self._phr_list.append(text)
        self._live.append(0)
        return i

    def _graph_add(self, m: _Mem) -> None:
        ids = tuple(dict.fromkeys(self._phrase(e) for e in m.versions[-1][3]))
        m.contrib = ids
        for p in ids:
            self._live[p] += 1
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                key = (min(ids[a], ids[b]), max(ids[a], ids[b]))
                self._cooc[key] = self._cooc.get(key, 0) + 1

    def _graph_remove(self, m: _Mem) -> None:
        ids = m.contrib
        for p in ids:
            self._live[p] -= 1
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                key = (min(ids[a], ids[b]), max(ids[a], ids[b]))
                n = self._cooc.get(key, 0) - 1
                if n > 0:
                    self._cooc[key] = n
                else:
                    self._cooc.pop(key, None)
        m.contrib = ()

    def _edge_cache(self) -> None:
        if self._ecache is not None and self._ecache["ver"] == self._gver:
            return
        c = self.cfg
        S = len(self._slot_ids)
        src, dst, w, pn = [], [], [], []
        for mid in self._order:
            m = self._mems[mid]
            if not m.eligible:
                continue
            for p in m.contrib:
                node = S + p
                src += (m.slot, node)
                dst += (node, m.slot)
                w += (1.0, 1.0)
                pn += (m.slot, m.slot)
        for (i, j), cnt in self._cooc.items():
            wt = c.cooc_scale * math.log1p(cnt)
            src += (S + i, S + j)
            dst += (S + j, S + i)
            w += (wt, wt)
            pn += (-1, -1)
        for (i, j), s in self._syn.items():
            if self._live[i] > 0 and self._live[j] > 0:
                wt = c.syn_scale * s
                src += (S + i, S + j)
                dst += (S + j, S + i)
                w += (wt, wt)
                pn += (-1, -1)
        self._ecache = {"ver": self._gver, "src": np.array(src, dtype=np.int64),
                        "dst": np.array(dst, dtype=np.int64), "w": np.array(w, dtype=np.float64),
                        "pn": np.array(pn, dtype=np.int64)}

    def _link_cache(self) -> None:
        if self._lcache is not None:
            return
        a, b, v, t = [], [], [], []
        adj: dict[str, list] = {}
        for key, (val, t0) in self._plast.links.items():
            x, y = key
            if x not in self._mems or y not in self._mems:
                continue
            a.append(self._mems[x].slot)
            b.append(self._mems[y].slot)
            v.append(val)
            t.append(t0)
            adj.setdefault(x, []).append((y, key))
            adj.setdefault(y, []).append((x, key))
        self._lcache = {"a": np.array(a, dtype=np.int64), "b": np.array(b, dtype=np.int64),
                        "v": np.array(v), "t": np.array(t), "adj": adj}

    def _node_cache(self) -> None:
        if self._ncache is not None:
            return
        items = [(self._mems[i].slot, v, t) for i, (v, t) in self._plast.nodes.items() if i in self._mems]
        self._ncache = {"slots": np.array([x[0] for x in items], dtype=np.int64),
                        "v": np.array([x[1] for x in items]), "t": np.array([x[2] for x in items])}
