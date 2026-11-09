# V3 pre-registration (FROZEN)

Status: FROZEN 2026-11-09 (Aadit approved "Freeze + commit"). Hash and source digest are in `docs/V3_FREEZE.txt`. No v3 run has happened; any Kaggle/GPU run, including the smoke, needs Aadit's separate OK. Written 2026-11-09.
Base: frozen v2 code, branch `v2-frozen` @ `2c2aaf6`. V3 work goes on a new branch off it.

## Why v3

v2 (seeds 48, 49, 50; 3 valid, none excluded) failed overall:
- Claim 1 (non-inferiority) FAIL on rule C only: ours 90.2% pooled vs rag_semantic 92.8%, rag_hashing 86.5%.
- Claim 3 PASS but uninformative: the `generic_forced` control is a broken adapter (loops "Final answer: ..."), scoring 0%.
- Claim 5 FAIL on seed 48, Birch fact only.
- Claims 2 and 4 PASS.

Everything below about *causes* is exploratory (from post-hoc looks at seeds 48-50) and is labelled so. Seeds 48-50 are **development data only** from here on. No v3 verdict uses them.

## What changes (each with its reason)

1. **Retrieval padding rule (fixes Claim 5, hypothesis H1).**
   - *Observed (exploratory):* the correct Birch fact was retrieved, but top-k=5 filled the other slots with rule demos. Contexts made only of B demos made the frozen core copy the "B-" format (12/12 corrupted); mixed contexts were fine.
   - *Rule, registered exactly:* in `SleepArm.context`, if the top-ranked hit has `kind == 'fact'`, return only hits with `kind == 'fact'` (still at most k=5). Otherwise return the v2 top-k unchanged. No numeric cutoff, no tuning, uses no task label.
   - Applies to `ours` (and `expert_only` where it retrieves). Baselines `rag_*` and `hippo_only` keep v2 retrieval, so they are not changed to match. Caveat: this makes `ours` slightly better-engineered than `hippo_only` on facts.

2. **ONE primary rule C fix: format-checked memory fallback (hypothesis H2).**
   - *Observed (exploratory):* day-2 rule C errors, 360 answers per version. v1 vs v2: letter errors 23 vs 38; dropped "-7" 0 vs 17; other garbled format 4 vs 16. In v2, `ours` never retrieved on rule items (0 of 2,160), so on rules Claim 1 is the expert alone vs RAG with no safety net.
   - *Why this and not "more C training examples":* the day-2 C set has only 24 fresh examples (+8 replay = 32). More examples means changing the world/stream, which changes the benchmark. It also does not explain why v1 (same mix, steps, seeds) did better.
   - *Rule, registered exactly:* applies to arm `ours` only (not `expert_only`, so Claim 3 compares the weights alone against the control). When an expert answers, compute the answer's *shape signature* (letters to `a` regardless of case, digits to `0`, punctuation and whitespace kept; scoring is case-folded). Retrieve the top 2 memory hits with the v2 recall, as the trigger only. If the expert's signature matches none of those hits' answer signatures (or the answer is unusable under the strict extractor), answer again on the **frozen core with no expert**, using the context the `hippo_only` arm would build (v2 recall, k=5, no fact filter), and use that answer. Otherwise keep the expert answer. The retry's context tokens count. k=2 for the trigger is fixed now and not tuned.
   - *Why the retry uses the frozen core and not the expert:* the same expert may repeat its error, and expert-with-2-demos has never been measured. Frozen core + k=5 recall is exactly the `hippo_only` answer path, the only setup with evidence.
   - *Known limit (exploratory, v2 logs, dev data):* replaying real top-2 recall on ours' day-2 checkpoints (3 seeds), the case-folded signature caught 64 of 72 rule C errors with 0 correct answers re-asked; 8 escape (anagrams, true substitutions). Rule B: 7 of 22 caught, 0 re-asked; rule A: 8 of 12 caught, 3 re-asked. Proxy only, not a prediction: if retries behaved like `hippo_only`, day-2 C would go from 288 to about 344 of 360; non-inferiority vs `rag_semantic` needs at least 314. Claim 4 estimate is about 11-16% of hashing RAG on day-2 rule items; if the fallback fires often Claim 4 could still fail, and it is reported either way.

3. **Valid generic control (fixes Claim 3 being uninformative).**
   - Add a `wrong_rule` control: same prompts, same answer shape, shuffled (wrong) mapping, not run on seeds 48-50. Claim 3 uses it in place of `generic_forced`.
   - *Validity gate (stricter than v2):* the control must produce well-formed answers (signature matches) on ≥ 90% of rule items. If it does not, Claim 3 is reported **UNINFORMATIVE**, not PASS.
   - Margin stays 30 points, CI low > 0.

4. **Miss-type analyzer (descriptive only).** A separate script that tabulates error types (letter error, dropped suffix, garbled format, other) per arm, from `answers.jsonl`. It carries no verdict. `v2_analyze.py` stays frozen and is reused for claims.

## Unchanged (no bar loosened)

- Claim 1: non-inferiority ≥ −5 with CI low ≥ −10 vs rag_semantic, rag_hashing, hippo_only on A, B, C and pooled.
- Claim 2: ours − no_memory ≥ 50.
- Claim 4: context cost ≤ 25% of hashing RAG.
- Claim 5: facts/safety, `trusted_rollback` False is the required good value.
- Inference: 10,000 paired bootstrap draws, `default_rng(0)`, case-folded final cycle 2.
- Addendum 1 procedure rules (no subset verdicts, void seeds, stop on invalid seed, report duplicate/missing rows, fewer-than-3-seeds wording) carry over.
- Model, router (threshold 0.2), sleep gate, rank-4 LoRA, steps 64 (`S4_STEPS` default), replay 8.

## Seeds

- **Confirmation seeds: 53, 54, 55**, run once, none seen before.
- Replacements if one is void: 56, then 57.
- 48-50 are dev only. 51 and 52 were v2's reserve and were never used; they stay unused here too.

## Hypotheses, not findings

- H1: retrieval padding with rule demos caused the Birch failure.
- H2: expert-2 is fragile (its format errors, not its rule knowledge, drive the C gap). The fallback test and the miss-type table can support or refute this. If the C gap persists after the fallback, H2 is not confirmed.
- Possible outcome: v3 fails again. That is reported as such.

## Caveats

- The v2 Claim 1 miss is on C only; v3 does not claim to fix letter errors.
- The padding rule touches `ours` and `expert_only`; the fallback touches `ours` only. Baselines are not given the same help.
- The padding rule fires on all 12 format-corrupted v2 fact answers (replaying checkpoints through real recall, the fact was ranked first; the replay matched the logged top-5 on 50 of 54 rows). It does NOT address seed 48 day-0 Alder and Cedar, where the fact was not in the top 5.
- Any reviewer-found change after this text is frozen must be a recorded addendum, stricter-only.

## Implementation notes

- Branch/worktree: `software4-v3-git`, off `v2-frozen`. Main checkout never branch-switched.
- CPU tests only before any run: unit tests for the padding rule, the shape signature and fallback, the `wrong_rule` control (flag-gated), the miss-type analyzer. Smoke mode `S4_SMOKE=1` first on Kaggle before any full run (needs Aadit's explicit OK).
- Freeze record (prereg SHA-256, source digest) is written only after Aadit approves this draft.

## Process

Codex writes the code on the v3 branch; Claude verifies; Sol reviews diffs only. Report to Aadit before anything is frozen. No Kaggle/GPU run, commit or push without Aadit's separate OK.
