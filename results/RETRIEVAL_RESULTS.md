# Retrieval-only results

Gate: **FAIL**. Strictly higher hippocampus recall@1 AND recall@5 than
RAG in each of revised, multihop, distractors, and poison, both in the seed mean and every seed.
Ties fail. This gate was specified before measuring; no parameters were tuned.

Command: `python scripts/retrieval_check.py --seed 0 1 2 --queries 40`

3 seeds; 320 queries/seed; 960 total queries.
Cells show mean [min, max] across seeds. Latency is the pooled median per system/category,
in milliseconds at k=5, including query embedding and RAG's native corpus re-embedding,
excluding ingestion. JSON also includes per-seed median latency at k=1 and k=5 and its mean/range.
First queries are included (no warmup); CPU wall times are not deterministic.
Calls run in k=1, k=5, then full-ranking order per system/query; graph caches may
already be warm at k=5. Systems rotate order across queries.

| Category | System | Recall@1 | Recall@5 | MRR | All gold@5 | Median ms@5 |
|---|---|---:|---:|---:|---:|---:|
| plain | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.182 |
| plain | rag | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.917 |
| plain | text_notes | 0.025 [0.025, 0.025] | 0.125 [0.125, 0.125] | 0.107 [0.107, 0.107] | 0.125 [0.125, 0.125] | 0.009 |
| revised | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.190 |
| revised | rag | 0.158 [0.150, 0.175] | 0.992 [0.975, 1.000] | 0.573 [0.567, 0.577] | 0.992 [0.975, 1.000] | 1.939 |
| revised | text_notes | 0.025 [0.025, 0.025] | 0.125 [0.125, 0.125] | 0.107 [0.107, 0.107] | 0.125 [0.125, 0.125] | 0.014 |
| multihop | hippocampus | 0.500 [0.500, 0.500] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.759 |
| multihop | rag | 0.500 [0.500, 0.500] | 0.537 [0.525, 0.550] | 1.000 [1.000, 1.000] | 0.075 [0.050, 0.100] | 1.687 |
| multihop | text_notes | 0.013 [0.013, 0.013] | 0.062 [0.062, 0.062] | 0.101 [0.099, 0.103] | 0.000 [0.000, 0.000] | 0.014 |
| distractors | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.398 |
| distractors | rag | 0.983 [0.950, 1.000] | 1.000 [1.000, 1.000] | 0.992 [0.975, 1.000] | 1.000 [1.000, 1.000] | 10.998 |
| distractors | text_notes | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.011 [0.010, 0.012] | 0.000 [0.000, 0.000] | 0.054 |
| poison | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.189 |
| poison | rag | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.742 |
| poison | text_notes | 0.008 [0.000, 0.025] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.014 |
| scale_100 | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.299 |
| scale_100 | rag | 0.983 [0.975, 1.000] | 1.000 [1.000, 1.000] | 0.992 [0.988, 1.000] | 1.000 [1.000, 1.000] | 2.421 |
| scale_100 | text_notes | 0.017 [0.000, 0.025] | 0.067 [0.050, 0.075] | 0.061 [0.044, 0.075] | 0.067 [0.050, 0.075] | 0.016 |
| scale_1000 | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 2.716 |
| scale_1000 | rag | 0.950 [0.925, 0.975] | 1.000 [1.000, 1.000] | 0.974 [0.958, 0.988] | 1.000 [1.000, 1.000] | 25.380 |
| scale_1000 | text_notes | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 0.135 |
| scale_10000 | hippocampus | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 35.495 |
| scale_10000 | rag | 0.792 [0.725, 0.825] | 0.983 [0.975, 1.000] | 0.874 [0.840, 0.902] | 0.983 [0.975, 1.000] | 258.206 |
| scale_10000 | text_notes | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 2.185 |
| overall | hippocampus | 0.938 [0.938, 0.938] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.732 |
| overall | rag | 0.671 [0.662, 0.675] | 0.814 [0.811, 0.816] | 0.801 [0.795, 0.805] | 0.756 [0.753, 0.759] | 2.305 |
| overall | text_notes | 0.011 [0.008, 0.014] | 0.047 [0.045, 0.048] | 0.049 [0.047, 0.050] | 0.040 [0.037, 0.041] | 0.016 |

## Findings

- The strict structural gate failed; overall recall alone does not establish it.
- Multihop both-gold@5: hippocampus 1.000,
  RAG 0.075. Their recall@1 values are
  0.500 and 0.500,
  respectively (the ceiling is 0.5). MRR can be perfect while the second fact is missing.
- Distractor recall@5: hippocampus 1.000,
  RAG 1.000. Graph structure provides no universal advantage here.
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
  This tests Hippocampus core lifecycle APIs with explicit identities. Ordinary `add()` also
  versions trusted update notices in the controlled fact grammar; this benchmark uses explicit
  `update()` calls, so that additional ingestion path does not affect its measurements.
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
- revised recall@5: mean PASS, every seed FAIL.
- multihop recall@1: mean FAIL, every seed FAIL.
- multihop recall@5: mean PASS, every seed PASS.
- distractors recall@1: mean PASS, every seed FAIL.
- distractors recall@5: mean FAIL, every seed FAIL.
- poison recall@1: mean PASS, every seed PASS.
- poison recall@5: mean PASS, every seed PASS.
