# hippocampus

Long-term memory for Software4: exact recall by day, and the source of replay at night.
Standalone (numpy + standard library only; no torch). Build step 2 of `PLAN.md`.

## How it works

- **The log is the truth.** Every change is one line in an append-only JSONL file: add, update (new
  version, old kept), tombstone, quarantine release, trust change, outcome, nap, sleep. Lines are never
  rewritten. All indexes and link weights are rebuilt by replaying the log (`rebuild()`). A half-written
  last line from a crash is skipped.
- **Memories** have a kind (episode, fact, lesson), time, versions, a source, and a trust score kept
  apart from the text. Memories from sources not in `Config.trusted_sources` are **quarantined**: no recall,
  no replay, no graph influence, no strengthening, until `release()`.
- **Graph + vectors (HippoRAG 2 style).** Passage nodes (memories) and phrase nodes (entities); edges
  passage-phrase, phrase-phrase co-occurrence, and synonym edges between phrases with similar embeddings.
- **Plastic links (ours).** When memories are used together in a task whose outcome is a *verified success*,
  the links between them and each memory's own strength rise (bounded, saturating). Failures and
  unverified outcomes never change anything. Strength fades with a half-life, counted from the last time it
  was strengthened. Nothing is deleted: a faded memory is still found by deep recall, and a successful use
  makes it strong again. Strengths live in a separate index; `record_outcome` only logs, weights change at `nap()`.
- **Recall: fast first.** The fast pass searches a small bounded index (strongest and recent memories)
  plus one hop along strong links. If the best score clears `Config.confidence`, it returns. Otherwise the
  deep pass runs Personalized PageRank (power iteration) over all eligible memories, seeded from query
  phrases and similar passages; strong links carry more activation.
- **Replay supply** for deep sleep: verified successes and corrected failures from trusted, live,
  non-quarantined memories, a mix of new (since the last `mark_sleep`) and old, seeded. No surprise filter.

## Use

```python
from hippocampus import Hippocampus, Config, ManualClock

clock = ManualClock(0.0)
hc = Hippocampus("memory.jsonl", config=Config(), clock=clock)   # path=None keeps the log in memory

a = hc.add("Mireva Tonks founded Quillon Labs.", kind="fact")        # entities=[...] to pass them yourself
b = hc.add("Quillon Labs operates from Pelmont.", kind="fact")
hc.add("Forum post: Pelmont is on the moon.", source="forum")          # untrusted -> quarantined

hits = hc.recall("Where does the company of Mireva Tonks work?", k=3)  # fast, then deep if needed
for h in hits:
    print(h.id, h.version, h.score, h.via, h.text)                      # via is "fast" or "deep"
hc.recall("office", as_of=1000.0)                                       # versions as they were then
hc.recall("office", min_trust=0.7, deep=True)                           # deep=True/False forces a pass

hc.update(a, "Mireva Tonks founded Quillon Labs in 2011.")             # new version; old one stays in the log
hc.tombstone(b, reason="revoked")                                       # gone from recall and replay
hc.release("m000002", trust=0.8)                                        # let a quarantined memory in

hc.record_outcome("task-1", [a, b], success=True, verified=True)      # logged now
hc.nap()                                  # applies pending outcomes, fading, refreshes the fast index
hc.mark_sleep()                           # call after a deep sleep, so replay knows what is new
batch = hc.replay_batch(32, seed=0, new_fraction=0.5, min_trust=0.5)  # episodes for training
hc.rebuild()                              # re-derive everything from the log
```

Pass `now=` to recall/nap/etc. or inject a `clock`. Logic never reads the wall clock by itself;
`SystemClock` is only the default when you inject nothing. Embedder (`embed(texts) -> (n, d)`) and extractor
(`extract(text) -> [phrases]`) are pluggable; the dimension is never hardcoded. Use the **same embedder** when
rebuilding a log.

## Simplified compared with HippoRAG 2

- Default embedder is a hashing embedder and default extractor is rule-based (capitalised runs, numbers,
  fallback content words). The embedder has no real semantic meaning. A real one is meant to replace it.
- No LLM triple extraction or recognition-memory filter; phrase nodes are plain entity strings and
  co-occurrence replaces extracted relations.
- Query phrases are matched by exact string or embedding similarity, not by an LLM.
- Synonym edges are found incrementally when a phrase first appears (cosine >= threshold, top few).
- Passage seeds use cosine similarity only; passage seed weight (0.2) and damping (0.6) were chosen so the
  graph beats vector similarity on the multi-hop test, not tuned on real data.
- `as_of` recall is brute force (cosine + entity overlap) over versions at that time; it does not use the graph
  or plastic links. Quarantine release and tombstones are applied by their log time; trust is the current value.
- Fading is applied lazily on read (exact), not written back at nap. Nap only applies outcomes and refreshes
  the fast index.
- Linked pairs are capped to the first 12 memory ids (sorted) of one outcome.
- Tuning values in `Config` are untested guesses beyond what the tests show.
