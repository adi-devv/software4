# Retrieval-only results

Gate: **FAIL**. Strictly higher hippocampus recall@1 AND recall@5 than
RAG in each of revised, multihop, distractors, and poison, both in the seed mean and every seed.
Ties fail. This gate was specified before measuring; no parameters were tuned.

Command: `python scripts/retrieval_check.py --seed 10 11 12 --queries 40`

3 seeds; 320 queries/seed; 960 total queries.
Cells show mean [min, max] across seeds. Latency is the pooled median per system/category,
in milliseconds at k=5, including query embedding and RAG's native corpus re-embedding,
excluding ingestion. JSON also includes per-seed median latency at k=1 and k=5 and its mean/range.
First queries are included (no warmup); CPU wall times are not deterministic.
Calls run in k=1, k=5, then full-ranking order per system/query; graph caches may
already be warm at k=5. Systems rotate order across queries.

| Category | System | Recall@1 | Recall@5 | MRR | All gold@5 | Median ms@5 |
|---|---|---:|---:|---:|---:|---:|
| plain | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.180 |
| plain | rag | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.905 |
| plain | text_notes | 0.025 [0.025, 0.025] | 0.125 [0.125, 0.125] | 0.107 [0.107, 0.107] | 0.125 [0.125, 0.125] | 0.009 |
| revised | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.224 |
| revised | rag | 0.142 [0.125, 0.175] | 1.000 [1.000, 1.000] | 0.566 [0.556, 0.579] | 1.000 [1.000, 1.000] | 2.082 |
| revised | text_notes | 0.025 [0.025, 0.025] | 0.125 [0.125, 0.125] | 0.107 [0.107, 0.107] | 0.125 [0.125, 0.125] | 0.014 |
| multihop | hippocampus | 0.500 [0.500, 0.500] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.804 |
| multihop | rag | 0.487 [0.475, 0.500] | 0.537 [0.512, 0.562] | 0.988 [0.975, 1.000] | 0.075 [0.025, 0.125] | 1.884 |
| multihop | text_notes | 0.013 [0.013, 0.013] | 0.062 [0.062, 0.062] | 0.100 [0.099, 0.102] | 0.000 [0.000, 0.000] | 0.014 |
| distractors | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.522 |
| distractors | rag | 0.983 [0.950, 1.000] | 1.000 [1.000, 1.000] | 0.992 [0.975, 1.000] | 1.000 [1.000, 1.000] | 11.612 |
| distractors | text_notes | 0.000 [0.000, 0.000] | 0.008 [0.000, 0.025] | 0.010 [0.007, 0.012] | 0.008 [0.000, 0.025] | 0.056 |
| poison | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.245 |
| poison | rag | 0.008 [0.000, 0.025] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 2.527 |
| poison | text_notes | 0.017 [0.000, 0.025] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.016 |
| scale_100 | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.494 |
| scale_100 | rag | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 3.778 |
| scale_100 | text_notes | 0.008 [0.000, 0.025] | 0.033 [0.025, 0.050] | 0.051 [0.048, 0.054] | 0.033 [0.025, 0.050] | 0.024 |
| scale_1000 | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 2.976 |
| scale_1000 | rag | 0.975 [0.950, 1.000] | 1.000 [1.000, 1.000] | 0.988 [0.975, 1.000] | 1.000 [1.000, 1.000] | 31.285 |
| scale_1000 | text_notes | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 0.144 |
| scale_10000 | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 43.460 |
| scale_10000 | rag | 0.783 [0.775, 0.800] | 0.983 [0.975, 1.000] | 0.873 [0.864, 0.883] | 0.983 [0.975, 1.000] | 326.113 |
| scale_10000 | text_notes | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 2.577 |
| overall | hippocampus | 0.938 [0.938, 0.938] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.762 |
| overall | rag | 0.672 [0.667, 0.678] | 0.815 [0.814, 0.817] | 0.801 [0.798, 0.803] | 0.757 [0.753, 0.762] | 3.356 |
| overall | text_notes | 0.011 [0.008, 0.014] | 0.044 [0.042, 0.045] | 0.047 [0.047, 0.047] | 0.036 [0.034, 0.037] | 0.026 |

## Findings

- The strict structural gate failed; overall recall alone does not establish it.
- Multihop both-gold@5: hippocampus 1.000,
  RAG 0.075. Their recall@1 values are
  0.500 and 0.487,
  respectively (the ceiling is 0.5). MRR can be perfect while the second fact is missing.
- Distractor recall@5: hippocampus 1.000,
  RAG 1.000. Graph structure provides no universal advantage here.
- Automatic recall checks a confident fast winner against full-store eligible cosine argmax.
  For k>1, the confident complete-match path re-ranks the full store, retaining the existing
  learned-link hop reserve. k=1 uses an argmax check without a full cosine sort or PPR when accepted.
  Explicit deep=False retains the bounded fast-only override; historical recall is unchanged.
- If the best passage omits non-stopword query tokens, automatic recall runs the existing PPR
  graph pass even when cosine is confident. This coverage rule uses no question templates or
  category names. PPR weights, extraction, trust, quarantine, release, revision and tombstone
  handling are unchanged. Graph dispatch alone recovers the linked multihop evidence here.
- Fixed-k limitation: graph-only evidence cannot be inserted while also retaining every member
  of an arbitrary vector top-k. Exact vector top-k coverage holds on the confident complete-match
  cosine path without learned-link reserves. PPR and learned-link reserves may displace cosine
  neighbours; a universal no-worse recall guarantee for arbitrary relevance labels is not claimed.
  Measured recall@1 and recall@5 are >= RAG in every category and every seed in both runs.
  The strict gate still fails on ties, including multihop recall@1 at its 0.5 ceiling.

See the [original-seed report](../RETRIEVAL_RESULTS.md) for the before/after and latency comparison.

- TextNotes scores reflect native query-independent ordering over verbatim notes, not model quality.

## Protocol and limitations

- CPU only, offline, numpy/stdlib, no model objects or downloads. An inert sentinel raises on any
  language-model access. Neither `answer()` nor `sleep()` is called.
- Default `HashingEmbedder` shared by Hippocampus and RAG. TextNotes uses no embeddings:
  its existing ranking is uses descending then newest first, independent of the query.
  Adding vector ranking to TextNotes would change the arm's behavior, so it is preserved.
- Added small read-only `Rag.retrieve(query, k)` and `TextNotes.retrieve(query, k)` helpers;
  `context()` delegates to them without changing ordering or limits.
- Both baseline corpora are loaded via public `load_state()` with the identical verbatim fact
  versions; TextNotes receives oracle verbatim notes, all uses=0. Model distillation, the default
  200-note cap, pruning, and prompt token budgets are deliberately bypassed. This measures native
  note retrieval from a shared corpus, not end-to-end TextNotes learning or note quality.
- Hippocampus uses public `add/update/tombstone/recall`, default extractor/config and normal fast/deep
  dispatch. No explicit entities, gold-derived links, outcomes, naps, or forced-deep queries.
  Update identities and quarantine/revocation metadata are supplied as structured events.
  RAG/notes retain all text versions because they have no corresponding lifecycle support.
  This tests Hippocampus core lifecycle APIs; `HippoOnly.observe()` currently appends rather than
  calling update/tombstone, so these gains need not appear in the existing full harness arm.
- Fictional names follow `harness.worlds.NameFactory`; fact/update wording and ally chains follow
  the harness pattern. Categories have separate corpora; insertion and query order are seeded.
  Revised originals all precede updates; gold ids include version, so stale hits cannot score.
- Each query has exactly one gold id except multihop, which has exactly two distinct gold ids.
  Recall is the fraction of gold ids in the cutoff (maximum multihop recall@1 = 0.5).
  All-gold@1 is therefore impossible for multihop. JSON includes all-gold@1 and @5 for every row.
- MRR requests k=number of stored fact versions from each system and uses reciprocal rank of
  the FIRST relevant fact, zero if absent from the native returned candidates. This extra call
  is excluded from k=1/5 latency. JSON also includes truncated MRR@5 and full-ranking count,
  first-gold rank, unsafe count, and a ranking hash. All-gold@5 is the stronger multihop measure.
- Any quarantined, untrusted, or tombstoned id retrieved anywhere within a cutoff zeroes that
  query's recall, reciprocal rank, and all-gold score at that cutoff, even if gold is also present.
  For full MRR the cutoff is the entire requested ranking: any unsafe returned candidate also
  zeroes it. JSON includes unsafe@1/@5. Other categories have no forbidden ids.
- Scale corpora are nested at 100/1,000/10,000 live facts, with identical queries targeting facts
  sampled from the first 100. Overall is query-weighted, including all three scale conditions;
  it is not a gate criterion. Data, rankings, scores, and gate are reproducible; latency varies.
- JSON contains the full synthetic corpora, queries, hashes, top-1/5 retrieved ids, per-query metrics,
  per-seed results, and summaries for audit. No retrieval configuration is fitted to these seeds.

## Gate details

- revised recall@1: mean PASS, every seed PASS.
- revised recall@5: mean FAIL, every seed FAIL.
- multihop recall@1: mean PASS, every seed FAIL.
- multihop recall@5: mean PASS, every seed PASS.
- distractors recall@1: mean PASS, every seed FAIL.
- distractors recall@5: mean FAIL, every seed FAIL.
- poison recall@1: mean PASS, every seed PASS.
- poison recall@5: mean PASS, every seed PASS.

## Validation

- `python -m pytest hippocampus/tests baselines/tests scripts/tests -q`: **107 passed**
  (94 existing tests plus 13 regressions for full-store expansion, wrong confident winners,
  single-hit fast dispatch, graph routing, stable ties and lifecycle/trust filters).
- Each full benchmark was run once after the implementation was fixed: seeds 0/1/2 and
  fresh seeds 10/11/12, 40 queries/category/seed. No retrieval settings were fitted to either run.
- Saved-result audit passed for all 5,760 system rows across both runs: dataset hashes,
  valid unique cutoff ids, recall/MRR/all-gold/safety scores, full-MRR metadata, seed summaries
  and the unchanged gate. Original-seed dataset hashes/configuration match the saved before
  artifact; RAG and TextNotes scores, including per-seed scores, are unchanged.
- Hippocampus returned zero unsafe hits at k=1, k=5 and full ranking in both runs.
  Poison remains 1.000/1.000; revised improves from 0.992/1.000 to 1.000/1.000 on 0/1/2,
  with fresh revised and poison also 1.000/1.000. Multihop both-gold@5 improves from 0.400
  to 1.000 on 0/1/2 and is 1.000 on fresh seeds. No category scores decreased.
