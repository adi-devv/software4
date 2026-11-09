# V2 pre-registration (DRAFT, not frozen)

Status: draft, NOT frozen. Code is implemented and has been through Codex gpt-6.1-sol review of the code diff; CPU tests only, no GPU spent. Kaggle/GPU needs explicit user approval; the file is frozen by its SHA-256 only at that approval.
Applies to fresh seeds 48, 49, 50 only (47 is the sweep seed). Registered v1 verdicts (seeds 44-46, 30-point bar) remain unchanged; retrospective v2 diagnostics on s44/s45 are labelled exploratory.
Lives in `docs/`, not `step4/RESULTS.md`; docs are not covered by the source digest, so freeze this file by its own hash on approval.

## Why v2 (facts from s44 and s45)

- `ours_generic_only` installed no experts (3/3 sleeps rejected on both seeds) and matched `hippo_only`'s final rule scores. This did not test an installed generic expert. (Not identical to hippo_only in general: s45 final fact accuracy was 0/6 vs 2/6, because a rejected `SleepArm` skips the nap `HippoOnly.sleep()` does.)
- The 30-point bar was reached on neither seed: ours - rag = +20.3 (s44), -1.1 (s45) pooled points.
- `rag` uses `HashingEmbedder`. RAG rule A went 37.5% (s44) to 90.8% (s45). The s44 advantage was concentrated on A (71 of 73 net extra correct) and associated with mixed-rule retrieval (RAG contexts held competing C examples). Seed changes do not isolate retrieval causally.
- On s45, ours trailed rag on B (108/120 case-folded, 106 exact, vs 115) and C (111, 110 exact, vs 117). A pooled number hides that.
- Claim 5 failed on s45 without recorded poison absorption. The `check` in `load_v1_run` uses Birch only (correct and not absorbed); Dover does not decide it. Birch was answered `B-NORA`: right name, wrong format. Cycles 1-2 retrieved the correct fact plus four rule-B worked examples (`B-OJDU`, `B-GLDS`, `B-AGEJ`, `B-FNGZ`), `expert=None`. This is consistent with contamination by the retrieved B examples; a controlled context ablation would be needed to establish causality. `hippo_only` shares the error. Alder revision, A retention and no rollback still held.

## Changes (code outside the v1 hash; v2 gets its own source digest and verdict path)

1. **Forced generic control.** `ours_generic_only` installs all generic experts regardless of the gate, for this arm only; routing evidence is task-derived as in `ours`. Log each bypassed gate check. Training counts, steps, replay schedule and routing evidence are matched across controls.
2. **Control without retrieval.** Add `generic_forced_noret` (forced generic experts, no retrieval), compared with `expert_only`.
   - **Claim 3 (narrowed):** task-specific training beats forced generic fine-tuning without retrieval. It supports task-specific transfer beyond the registered generic control; it does not by itself exclude format-learning explanations, and no format-matched arm is added (the rules are output formats, so such a control would be nearly the treatment).
   - Claim 3 also requires `expert_only` to accept all 3 sleeps and its logged per-sleep training example counts and steps to match `generic_forced_noret` on every valid seed. Any rejection, mismatch or missing evidence makes Claim 3 FAIL for the run; it does not void a seed or authorize replacement. Treatment training stays as in v1.
   - **M = 30 points, LOCKED (user approved).** Pass: pooled final paired difference `expert_only` - `generic_forced_noret` >= 0.30 and the prespecified CI lower bound > 0. M is not lowered after seeing v2 data. Reachability: v1 expert_only pooled is 95.0% / 93.6% (s44/s45), so M needs the forced generic control <= ~65%; its score is unknown.
3. **Real RAG retriever.** Add a sentence-embedding retriever arm (model named and pinned by hash in the registration; same top-k, budget and store policy; no gold skill metadata). Keep `rag_hashing`. Better retrieval is not guaranteed by an embedding model alone.
4. **Non-inferiority (Claim 1), LOCKED (user approved).** Endpoints: A, B, C and pooled. Pass when paired difference (ours minus comparator) >= -5 points AND 95% CI lower bound >= -10 points, inclusive.
   - Comparators fixed up front: semantic RAG, hashing RAG and `hippo_only`; ours must pass against each (all required; intersection-union, no multiplicity correction because it is one conjunction claim).
   - Not Sol's one-sided LCB > -5: at n=120 per rule the paired SE is about 3 points, so a true tie passes about half the time per endpoint and a conjunction of endpoints is worse. The -5/-10 rule is v1's bar.
   - Seed scope (decided): each of A, B, C and pooled is evaluated pooled across valid seeds (v1's method), at least 2 valid seeds required, and per-seed results are always reported alongside. Not a per-seed conjunction: that is about 12 endpoint-seed checks per comparator at n=120 per rule, so a true tie would fail most of the time. For reference, s45 alone had B (-5.83) and C (CI low -10.83) failing vs rag; s44 passed (exploratory).
5. **Inference procedure (fixed before data).** Primary score: case-folded correctness at final cycle 2, 120 hidden items per rule, 360 pooled. Exact-case reported alongside. Pair arms by identical task IDs within a seed; missing or duplicate IDs invalidate that comparison. 10,000 paired item-bootstrap draws, NumPy `default_rng(0)`, sorted IDs, two-sided percentile 95% CI (2.5/97.5, linear interpolation), pooled resampling unstratified, pairs kept within seed. Item intervals are conditional on the trained stacks; three seeds give weak evidence on new-stack variability.
6. **Claim 5 repair.** Report separately: correct, stale, format-corrupted, absorbed, unusable. Semantic absorption must catch variants such as `B-Zorg`. Only absorption counts as a safety failure. Abstention must not pass: keep fact correctness, revision, retention and fail-closed rollback checks. Format-leak rate reported for every retrieval arm.
7. **Logging.** Generic-fit logging measures the actual generic training targets (the existing `forced_train_accuracy` scores trusted world/replay examples, so it shows 0.0 for the generic arm).

## Unchanged

Frozen core, rank-4 experts, 3 cycles, 120 hidden per rule, validity gate (gold_context >= 85% and no_memory <= 10% per rule), the sleep gate for `ours`, exact-case reporting next to the case-folded score.

## Caveats stated in the registration

- Seeds reuse facts and templates; cross-seed code disjointness is not guaranteed.
- C is directly trained, so this is not zero-shot composition.
- Audit overlap with development and sweep (seed 47) items before freezing.
- Invalid-run rule: a seed failing the validity gate is void and replaced by the next fresh seed (51, 52); replacement rule fixed here, no tuning on hidden scores.

## Implementation notes (implemented, CPU tests only)

Implemented in `baselines/semantic.py`, `baselines/generic.py`, `baselines/sleep.py`, `step4/run.py`, `step4/v2_analyze.py` and the packaged `step4/upload/run.py`. v2 is selected by `world == "step5"` with `world_seed` in 48-52; v1 seeds keep the v1 arms, analyzer and digest `308d3bb8...` (the working tree no longer matches that digest by design; archived v1 runs stay valid by their recorded digest).

- **v2 arms (10):** `ours`, `expert_only`, `hippo_only`, `rag_semantic`, `rag_hashing`, `generic_forced_noret`, `ours_generic_forced`, `text_notes`, `no_memory`, `gold_context`.
- **`rag_starved` is removed from v2.** v1's starved-context comparison is not repeated; the v2 analyzer excludes it and its absence does not affect completeness or the verdict.
- **Claim 3 matching rule:** `expert_only` must accept all 3 sleeps, with per-sleep training counts and steps equal to `generic_forced_noret`, on every valid seed. A mismatch is a Claim 3 FAIL (not void) and no replacement seed is run.
- **Semantic retriever:** `sentence-transformers/all-MiniLM-L6-v2`, revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, weights SHA-256 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`, 384-d. Kaggle fetches the snapshot in `setup_runtime` while the hub is reachable (workers run offline) and re-verifies the SHA before any GPU work. Embeddings were checked identical (max abs diff 1e-7) between transformers 4.56.2 and the Kaggle-pinned 5.3.0.
- **Pooled inference:** unstratified, pairs kept within seed.

## Process

Draft here -> Sol reviews code diff -> user approves -> only then any Kaggle/GPU run. Nothing committed.
