# Step 4 run ledger

## Step 4c

**Status: code done, needs fresh smoke.** The review fixes change the config hash and the behavior of ours on empty corrections. Results from this branch are not comparable with the seed-41 4b run. The pre-registered stop-rule thresholds are unchanged.

Smoke debugging runs v1-v7 and one full run (2026-11-07/08 IST). The first full run FAILED the stop rule; the 4b full run PASSED it on one seed (see the table).

| Run | Time used | Result | Weekly GPU-hours |
|---|---:|---|---:|
| Package preparation / CPU tests | CPU only | See README; fake integration and optional random-weight HF check | 0 used by this job; account total unknown |

Local validation (2026-11-07; GPUs hidden):

- `python -m pytest step4 hippocampus baselines harness scripts -q`: **385 passed, 3 skipped in 213 s** on Python 3.13 (torch absent); `step4` alone is 72 tests with 2 skipped.
- Python 3.12 with torch, offline `Qwen/Qwen3-0.6B` config: `baselines/tests/test_sleep_hf.py` **6 passed in 25 s** (inference cap 1024, training cap 112).
- `S4_BACKEND=fake S4_SMOKE=0`: all eight arms complete over three cycles (fake backend, no model quality information).
- `S4_BACKEND=tiny S4_SMOKE=1 S4_ARMS=ours` (Python 3.12, random tiny HF model): **ours complete in 15.7 s**.
- Optional HF checks used torch 2.6.0 / Transformers 5.22.2; the future pinned Kaggle runtime remains untested.

After each later Kaggle run, add its UTC timestamp, measured setup/session/worker times, smoke/full
status, output reference, and running weekly account total. Each run also generates its own RESULTS.md.

## Notes before the first Kaggle run (2026-11-07 IST)

- **Canary 7/8 (local smoke, Qwen3-0.6B):** the one missed canary is caused by unrelated retrieved hippocampus context in the `ours` prompt, not by the weights. Canaries route to no expert, and the fault-injected cycle 2 left the model state identical to cycle 1 (rollback correct). No code change.
- **Review fix:** a full run no longer requires the same physical GPU UUIDs as its smoke (Kaggle gives a new machine each session). It now matches mode, device count, GPU model and memory. Each run still needs two distinct T4 UUIDs and each worker re-asserts its UUID. `prepare.py --full` also targets a separate kernel (`-full`) with the smoke kernel as `kernel_sources`. `step4`: 73 passed, 2 skipped.
- Because `run.py` changed, the first smoke (pushed ~21:01 IST on the old code) is information only (timings, 2-T4 behavior). A smoke on the fixed code is needed before the full run.

## Kaggle smoke debugging log (2026-11-07 IST; kernel `adidevv/software4-step4-lean`)

Exact setup/session times for v1-v5 were not recorded; v6 ran 22:17-22:21 IST (about 4 min). Weekly account GPU-hours total: not yet read from Kaggle.

| Ver | Outcome | Cause / fix |
|---|---|---|
| v1-v2 | ERROR | UUID form and receipt hardware match; no per-arm errors in the log, so error/log-tail printing was added |
| v3-v4 | all arms fail canaries | base Qwen3.5-2B is verbose and was cut at 16 tokens; fixed with a "Final answer:" assistant prefill, same for all arms |
| v5 | 6 arms complete; `ours`/`expert_only` fail | expert training: "nonfinite/skipped optimizer step" (~52 s) |
| v6 | same, with diagnostics | `{"loss": 3.01, "skipped_steps": 1, "optimizer_steps": 1, "nonfinite_gradient_steps": 1, "steps": 2}`: loss finite, the fp16 GradScaler (init_scale 256) hit an overflow on step 1 and the strict check counted that normal calibration as a failure |
| v7 | **COMPLETE** (kernel 22:23-22:28 IST, driver wall 132 s; worker s: ours 52.8, expert_only 52.6, others 22-24; 2 distinct Tesla T4s) | fix: on a nonfinite gradient, halve the scale and redo the same batch (up to 8 backoffs); ours: 2/2 optimizer steps, 1 scale backoff, 0 skipped, loss 1.90, canaries pass. Smoke receipt `smoke-0327a3f4c26e2f34`. Weekly GPU-hours: not read from Kaggle (about 0.07 worker-hours this run) |
| full | **COMPLETE, stop rule = STOP (fail)** (kernel `adidevv/software4-step4-full`; QUEUED 22:39 IST, RUNNING 23:03 IST, COMPLETE seen 23:21 IST; driver wall 995 s, setup 41 s, 0.54 worker-hours, 2 distinct T4s) | receipt `full-0327a3f4c26e2f34`, output `step4/upload/output/step4/full-0327a3f4c26e2f34/`. Final hidden acc: ours 31.7%, expert_only 66.7%, hippo_only 53.7%, rag 51.2%, oracle 51.2%, no_memory 10.2%. Ours - rag = -22.5 pts (CI -32.5 to -12.5); ours - hippo_only = -22.5. Rule B accepted, rule A not dropped, forced-train accuracy 1.0/0.97, routing perfect, 0 parse failures. Cause under study: ours rule:B 5% vs expert_only 87.5% (expert trained without retrieved context, tested with it). One seed, so a trend only. Weekly GPU-hours: not read from Kaggle (quota was 28h47m before the run; about 0.54 worker-hours used, but billing is per session wall time) |
| 4b | **Code done** (superseded by the rows below) | Integration fixes after the full-run STOP: clean memory rendering, route-chosen context, per-answer logging (see README "Step 4b"). Code by Codex on branch `step4b` (22f8001); CPU tests pass; Codex review found 3 non-blocking items (empty-correction render, copy-rate ignores `Fact:` lines, `answers.jsonl` write not guarded). Config hash changed, so it needed a fresh smoke (next row) |
| 4b smoke | **COMPLETE** (kernel version 8, finished 00:50 IST 2026-11-08; driver wall 124.7 s, 0.063 worker-hours, 2 distinct Tesla T4s) | receipt `smoke-945f4b75d265a230`; all 8 arms complete; `ours` canaries 8/8, 2/2 optimizer steps, 1 scale backoff, 0 skipped, loss 1.90; `kaggle_full_run_eligible: true`. Plumbing only: smoke does not run scored cycles, so `answers.jsonl` is first checked in the full run. Next: Step 4b full run (seed 41 + 2 seeds), awaiting user approval |
| 4b full | **COMPLETE, stop rule = CONTINUE (pass)** (kernel `adidevv/software4-step4-full` v2, finished 2026-11-07 19:42 UTC; driver wall 999 s, 0.545 worker-hours, 2 distinct T4s) | receipt `full-945f4b75d265a230`, seed 41 only. Final hidden acc: ours 69.1%, expert_only 66.7%, hippo_only 48.4%, rag 45.9%, oracle 51.2%, no_memory 10.2%. On rule items ours beat fair RAG and hippo_only by 21.2 pts (CI 14.6 to 27.9, margin 15 met); rule B accepted (forced fit 1.00/0.97), rule A held at 49.2%. Caveats: ours equals expert_only on rules (59/120 A, 105/120 B), so memory adds only the fact item here; rule A is still only 49%; one seed, a trend not a result. Fix of the 4b rendering took rule B from 5% to 87.5%. Seeds 2 and 3 (about 1.1 worker-hours) need approval |
| 4b smoke, world seed 42 | **COMPLETE** (wall 125.3 s) | receipt `smoke-30d242da79e09c2d`; `kaggle_full_run_eligible: true` |
| 4b full, world seed 42 | **COMPLETE, stop rule = CONTINUE (pass)** (kernel `adidevv/software4-step4-full` v3, finished 2026-11-08 01:48 IST; driver wall 1008 s, 0.554 worker-hours) | receipt `full-30d242da79e09c2d`. Final hidden acc: ours 78.5%, expert_only 76.4%, hippo_only 49.6%, rag 47.2%, oracle 49.6%, no_memory 2.4%. Rule items: ours - rag +30.0 pts (CI 23.8 to 36.3), ours - hippo_only +29.6 pts (CI 23.3 to 35.8); rule B accepted at cycle 1 (forced fit 1.00/1.00), rule A held at 60%. Same caveat as seed 41: ours equals expert_only on rules (72/120 A, 116/120 B) |
| 4b smoke, world seed 43 | **COMPLETE** | receipt `smoke-bef760edea4ef5b6`; `kaggle_full_run_eligible: true` |
| 4b full, world seed 43 | **COMPLETE, stop rule = CONTINUE (pass)** (kernel `adidevv/software4-step4-full` v4, finished 2026-11-07 20:46 UTC; driver wall 1025 s, 0.562 worker-hours) | receipt `full-bef760edea4ef5b6`. Final hidden acc: ours 77.6%, expert_only 75.2%, hippo_only 50.0%, rag 48.0%, oracle 50.4%, no_memory 4.5%. Rule items: ours - rag +27.9 pts (CI 21.3 to 34.6), ours - hippo_only +28.3 pts (CI 21.7 to 34.6); rule B accepted at cycle 1 (forced fit 1.00/0.97), rule A held at 56.7%. Seeds 41, 42, 43 all pass. Same caveat: ours equals expert_only on rules (68/120 A, 117/120 B), so memory adds only the fact item |
| 4b smoke, ours_salient + expert_only, world seed 41 | **COMPLETE** (wall 87.2 s, 0.043 worker-hours; both arms complete; two-arm smoke; kernel status read ERROR with no exception in the log, so superseded by the nine-arm smoke whose receipt is eligible) | receipt `smoke-e840d6be67982d26`; arms baked with `prepare.py --arms`; superseded by the nine-arm smoke and the full run below |
| 4b full + ours_salient, world seed 41 | **COMPLETE, stop rule = CONTINUE (pass)** (kernel `adidevv/software4-step4-full` v5; driver wall 1481 s, 0.701 worker-hours, nine arms) | receipt `full-e840d6be67982d26`; arms baked with `prepare.py --arms`. Final hidden acc: ours_salient 185/246 (75.2%), ours 170/246 (69.1%), expert_only 164/246 (66.7%), hippo_only 119/246, rag 113/246, oracle 126/246, rag_starved 79/246, no_memory 25/246, text_notes 0/246. Rule A: ours_salient 68/120 vs ours 59/120; rule B: 111/120 vs 105/120. The stop rule in the output is computed for `ours` only (+21.2 pts vs rag and hippo_only, CI 14.6 to 27.9, pass); it was not computed for ours_salient (raw rule-item gap vs rag about +27.5 pts, unbootstrapped). Caveats: one seed; ours_salient cycle 0 rule A is 3/120 (the gain appears only at cycle 1); worker time 431 s vs 656 s for ours. Seeds 42 and 43 for ours_salient not run |
| 4b full + ours_salient, world seed 42 | **COMPLETE, ours_salient stop rule = STOP (fail)** (kernel `adidevv/software4-step4-full`; nine arms) | receipt `full-d58ae25c9f79d73f`; ran the pre-fix salient code (single-skill trim bug, so no cycle-0 expert). Final hidden acc: ours 193/246 (78.5%), expert_only 188/246, hippo_only 122/246, rag 116/246, ours_salient 121/246 (49.2%). Rule A: ours_salient 3/120 vs ours 72/120; rule B 113/120 vs 116/120. ours_salient trained at cycle 1 (forced train acc 0.946) but the gate rejected every sleep, so it behaves like hippo_only (+0.0 pts vs rag). Stop-rule table from `analyze.py --arms` (branch `stop-any`): ours +30.0 pts vs rag [23.8, 36.2] continue; ours_salient stop. Caveat: seed 41 and seed 42 disagree for the buggy salient arm (win then no accepted sleep); the fix is on branch `salient-fix` and needs a fresh smoke |

## Three-seed summary for `ours` (seeds 41, 42, 43; added 2026-11-08)

Only `ours` was pre-registered against the stop rule. Rule items, final cycle, paired bootstrap (10,000 draws, seed 0), same code for all three seeds.

| Seed | ours rule acc | rag | hippo_only | ours - rag (95% CI) | ours - hippo_only (95% CI) | Verdict |
|---|---|---|---|---|---|---|
| 41 | 68.3% | 47.1% | 47.1% | +21.2 [14.6, 27.9] | +21.2 [14.6, 27.9] | continue (point estimate; CI low 14.6 is under 15) |
| 42 | 78.3% | 48.3% | 48.8% | +30.0 [23.8, 36.2] | +29.6 [23.3, 35.8] | continue |
| 43 | 77.1% | 49.2% | 48.8% | +27.9 [21.2, 34.6] | +28.3 [21.7, 34.6] | continue |
| **Pooled (720 rule items)** | | | | **+26.4 [22.6, 30.1]** | **+26.4 [22.6, 30.0]** | all three seeds pass individually |

Caveats:

- Seed 41's CI low is 14.6, so "at least 15 pts" holds there only as a point estimate. Seeds 42 and 43 clear 15 on the CI too. The pooled CI low is 22.6.
- The pooled bootstrap treats items as independent across seeds. It is a summary, not a replacement for the per-seed passes.
- `ours` equals `expert_only` on rule items in every seed (the memory adds only the fact item). **Rule A drives the whole gain, rule B almost none** (corrected 2026-11-08; the earlier line had it backwards). Per-rule final hidden accuracy, seeds 41/42/43: rule A ours 59/72/68 of 120 vs rag 3/3/6 and hippo_only 3/4/5; rule B ours 105/116/117 vs rag 110/113/112 and hippo_only 110/113/112 (pooled B: ours 338/360 vs rag 334/360, +1.1 pts). So fair RAG already follows rule B from context (92-94%); the +26 pooled gap is rule A, where ours is 49-60%. Rule A is letter reversal, which the 2B model cannot do from context even with gold notes (`oracle` A 5/120 on seed 41), so a rule-A win cannot be told apart from "the weights learned a skill the context cannot teach". The first v1 design was withdrawn; revision 2 below replaces it.
- `ours_salient` is a null result: on seed 41 it scored 162 vs 164 for `ours` on rule items. Its seed 42 run used buggy code and is void. The fix is merged (e0c9444); no fresh run is planned.
- `oracle` is gold context in a 512-token budget with no training, so it is a baseline, not an upper bound for weight-learning arms (seed 41: A 5/120, B 115/120).
- Seeds 41-43 are the seeds used while debugging; fresh-seed confirmation (44-46) is not run.
- Not run and not merged: `ours_online`, `dream`, `generic`, `oracle_trained`, `merge` arms (no GPU verification yet).

## v1 pre-registration, revision 2 (2026-11-08, before any v1 GPU run)

Supersedes the version in baa8e3b, which tested rule A (letter reversal). That rule is not followable from context by the 2B model, so a "win" on it could not be told apart from weights learning a skill context cannot teach, and the old pass rule (ours >= rag + 15) was met almost entirely by it. Stage T of PLAN.md replaces the test; this section is its pre-registration.

**World (new file, seeds 44, 45, 46; the old world stays for reproducibility).** All rules are template-copy rules the 2B follows from context (local Q4 calibration: 90-100% with 3 demos, 0% with none):

- A, Cobalt beacon (cycle 0): uppercase the code and add `-7` at the end.
- B, Amber orchard (cycle 1): uppercase the code and put 'B-' in front of it (changed 2026-11-08: the old bracket format let a bare echoed code pass the grader).
- C, Cobalt-Amber chain (cycle 2): apply rule A, then rule B to its result. It is a multi-step composition with its own train episodes, so the answer for `abcd` is `B-ABCD-7`.
- Six facts as before. 24 train, 120 hidden and 4 gate items per remaining cycle for each rule. No code appears in the old smoke or step-4 worlds.

**Baseline changes.** `gold_context` (renamed from `oracle`) gets the rule lesson plus 3 verified demos in "Question / Verified answer" form, because a lesson alone makes the 2B write a long reply that the 16-token budget truncates. `text_notes` is repaired: distillation lines that are bare "Final answer:" copies are dropped, a verified question-answer example replaces them when nothing usable is left, and retrieval depends on the query. The old 0/246 came from every prompt carrying the same four copied lines.

**Validity gate (checked before any claim).** In each seed, `gold_context` must score >= 85% on every rule's final hidden set and `no_memory` must score <= 10%. If either fails, the world is broken and that seed is void, not a result.

**What v1 tests, and what it does not.** With rules the model can follow from context, a fair RAG baseline is expected to sit near the ceiling, so v1 does not test "weights beat context". It tests that sleep-trained experts carry the rules at near-zero context cost and that the gain is rule learning rather than answer-format learning. Beating RAG needs a capacity-pressure world (many rules, a binding context budget) and is deferred, not claimed.

- **Seeds:** 44, 45, 46 (fresh).
- **Arms (8):** `ours`, `expert_only`, `rag`, `hippo_only`, `text_notes`, `no_memory`, `gold_context`, and the format-only control `ours_generic_only`.
- **Pass rule (all must hold, on seeds that pass the validity gate; at least 2 of 3 seeds must):**
  1. Non-inferiority: pooled over seeds and rules A-C, final hidden accuracy of `ours` minus the better of `rag` and `hippo_only` is >= -5 pts, paired-bootstrap CI low >= -10 pts.
  2. Rule learning: `ours` beats `no_memory` by >= 50 pts, CI low > 0.
  3. Not format learning: `ours` beats `ours_generic_only` by >= 30 pts, CI low > 0.
  4. Cost: mean context tokens in `ours` prompts <= 25% of those in `rag` prompts (measured, not assumed).
  5. Retention and safety: rule A (final) not down more than 5 pts from its cycle-0 value; the revised fact accepted; the untrusted poison fact rejected; no gate rollback caused by a trusted update.
- **Per-rule results (A, B, C) are always reported separately**, with composition (C) called out. If `ours` fails C while passing A and B, the claim is single-rule only.
- **If claim 3 fails:** the experts learn the format, not the rules; stop and rethink how experts are built. **If claim 1 fails:** weights do not carry the rules as well as context does; stop and rethink consolidation before V1 (critic). No extra seeds either way.
- **Not in v1:** `ours_online`, `dream`, `oracle_trained`, `merge`, `ours_generic_data`, `ours_salient`, the critic (stages V1 to V3).
- **Cost estimate:** one smoke plus three 8-arm full runs, about 1.5 worker-hours; re-estimate from the smoke timing before asking to spend.
- **Process:** Codex gpt-6.1-sol review of this section, the world and the verdict code first; then smoke; then v1 only after explicit approval.

## v2 (seeds 48, 49, 50; replacements 51/52 only if a seed is void) — COMPLETE: overall verdict FAIL (3 valid seeds, none excluded)

Frozen 2026-11-09: prereg `docs/V2_PREREG_DRAFT.md` sha256 `4420f006...8fa90`, source digest `98eb1f1f...5d42` (see `docs/V2_FREEZE.txt`; the digest is recorded in `results.json` and `run.json` of all three runs and matches). Addendum 1 (`docs/V2_ADDENDUM_1.md`) written before any seed finished. Backup: `D:\Aadit\claudezackups\software4-v2-frozen-2026-11-09.zip` (+ `.sha256`). Analyzer: `step4/v2_analyze.py`, run once on all three seeds. Full analyzer output (every per-seed table, intervals, fact classes): `step4/V2_VERDICT_s48-50.md`. Branch `v2-frozen` at `2c2aaf6`.

### Run ledger (UTC from `kaggle kernels list --mine`; IST = UTC+5:30)

| Run | Kernel | Started (UTC / IST) | Finished (UTC / IST, as seen by poll) | Driver wall (s, setup excluded) | Status |
|---|---|---|---|---:|---|
| smoke (seed 48 config) | `adidevv/software4-step4-lean` | | 2026-11-09 08:37:50 | 200.5 | COMPLETE, full-run eligible |
| full s48 | `adidevv/software4-step4-v2-s48` | 09:47 / 15:17 | about 10:48 / 16:18 | 3530.2 | COMPLETE, receipt `full-cef1aadf08f66c86` |
| full s49 | `adidevv/software4-step4-v2-s49` | 09:47 / 15:17 | about 10:49 / 16:19 | 3582.8 | COMPLETE, receipt `full-1eb49de417235644` |
| full s50 | `adidevv/software4-step4-v2-s50` | 12:23 / 17:53 (queued behind the 2-session cap) | about 13:23 / 18:53 | 3489.8 | COMPLETE, receipt `full-c3017792600b907f` |
| replacement | s51 / s52 | not needed (no void seed) | | | |

Session time used: about 1.0 h per seed (about 3.0 h for the three; s48 and s49 ran concurrently on separate accounts' slots). The weekly Kaggle total was not read from Kaggle after the runs.

### Validity gate (gold_context >= 85%, no_memory <= 10% per rule): all 9 checks pass, no void seeds

| Seed | Rule | gold_context | no_memory | Valid? |
|---|---|---:|---:|---|
| 48 | A / B / C | 97.50 / 95.00 / 91.67 | 0 / 0 / 0 | valid |
| 49 | A / B / C | 96.67 / 92.50 / 91.67 | 0 / 0 / 0 | valid |
| 50 | A / B / C | 94.17 / 94.17 / 91.67 | 0 / 0 / 0 | valid |

### Duplicate / missing answer rows outside cycle 2 (Addendum rule 4)

Checked by hand on `answers.jsonl` for every seed and arm (30 files): 770 rows each (cycle 0: 120 rule + 8 canary + 6 fact; cycle 1: 240 + 8 + 6; cycle 2: 360 + 8 + 6; canary day -1: 8). **Duplicate rows: 0. Missing rows (vs the union of tasks across arms): 0**, in all 30 seed-arm files.

### Pooled final-cycle-2 accuracy (case-folded; n = 1080 pooled, 360 per rule)

| Arm | Pooled | A | B | C |
|---|---:|---:|---:|---:|
| ours (identical to expert_only to the digit) | 90.19 | 96.67 | 93.89 | 80.00 |
| rag_semantic | 92.78 | 94.72 | 91.39 | 92.22 |
| rag_hashing (identical to hippo_only) | 86.48 | 71.94 | 95.00 | 92.50 |
| text_notes | 61.67 | 0.28 | 92.22 | 92.50 |
| gold_context | 93.89 | 96.11 | 93.89 | 91.67 |
| no_memory, ours_generic_forced, generic_forced_noret | 0.00 | 0 | 0 | 0 |

Per-seed tables and exact-case numbers are in `step4/V2_VERDICT_s48-50.md`.

### Pooled claims (3 valid seeds, none excluded; 10,000 paired bootstrap draws, default_rng(0))

| Claim | Bar | Result | Verdict |
|---|---|---|---|
| 1 Non-inferiority vs rag_semantic, rag_hashing, hippo_only (A, B, C, pooled) | diff >= -5 and CI low >= -10 | vs rag_semantic: A +1.94, B +2.50, C **-12.22 [-16.67, -7.50]**, pooled **-2.59 [-4.72, -0.46]**. vs rag_hashing and hippo_only (identical): A +24.72, B -1.11, C **-12.50 [-16.94, -8.06]**, pooled +3.70 [1.11, 6.39] | **FAIL** (rule C fails the bar against every comparator) |
| 2 Rule learning (ours - no_memory) | >= 50 pts, CI low > 0 | +90.19 [88.33, 91.94] | PASS |
| 3 Task-specific transfer (expert_only - generic_forced_noret) | >= 30 pts, CI low > 0, 3/3 sleeps accepted, training counts/steps match | +90.19 [88.33, 91.94]; all 3 sleeps accepted and counts/steps match on every seed | PASS (see caveat) |
| 4 Context cost | ours <= 25% of hashing RAG | mean context tokens: ours 17.8, rag_semantic 195.5, rag_hashing 197.4 (9%) | PASS |
| 5 Facts/safety | correct Birch, revised Alder, rule-A retention within 5 pts, no absorption, no rollback | seeds 49, 50 pass per-seed checks; seed 48 fails (Birch not correct: 10 correct, 6 format-corrupted, 2 unusable of 18); `trusted_rollback` is False on all three seeds (False is the required good value: no trusted sleep was rolled back); absorbed = False everywhere | **FAIL** (seed 48 Birch only; see erratum below) |

**Overall verdict: FAIL, on 3 valid seeds, none excluded** (Addendum rule 5 statement).

Fact answer classes (correct / stale / format-corrupted / absorbed / unusable of 54, pooled): ours 40 / 0 / 12 / 0 / 2; rag_semantic 34 / 4 / 16 / 0 / 0; hippo_only 45 / 0 / 6 / 0 / 3; gold_context 54 / 0 / 0 / 0 / 0; ours_generic_forced 20 / 0 / 30 / 0 / 4; rag_hashing, text_notes, no_memory, expert_only, generic_forced_noret 0 correct (all unusable). Format-leak rate (rule wrappers in fact answers): ours 24.1%, rag_semantic 29.6%, rag_hashing 72.2%, hippo_only 14.8%, text_notes 61.1%, ours_generic_forced 61.1%.

### Caveats (facts read from the output, not extra claims)

- **Claim 3's margin equals expert_only's whole score** because the forced generic control scored 0.00% on every rule and seed, including `ours_generic_forced`. The 30-point bar is cleared trivially. Why the control scores 0 (for example, the generic experts destroying the answer format) is a hypothesis, not checked; Claim 3 shows task-specific experts beat this particular generic control, not why.
- `ours` equals `expert_only` exactly on every rule item: in `ours`, no rule answer used retrieval (0 of 2,160), so retrieval adds nothing to its rule answers; memory contributes only the fact items. `rag_hashing` and `hippo_only` have no expert, so their rule scores come entirely from retrieval (per the strategy session they match on 2,159 of 2,160 rule items but differ on facts; not re-checked here). *Erratum 2026-11-09: an earlier version of this line said retrieval adds nothing in all four arms.*
- The Claim 1 failure is entirely rule C (ours 80.0% against about 92% for every baseline); A and B pass. C is directly trained, so this is not zero-shot composition.
- The Claim 5 failure is seed 48 (Birch format-corrupted) only; no poison absorption was recorded. *Erratum 2026-11-09: an earlier version also blamed `trusted_rollback` False on all seeds, but `v2_analyze._safety` requires `rollback is False`, so that value is the passing one. The verdict is unchanged.*
- Intervals are conditional on the trained stacks; 3 seeds give weak evidence on new-stack variability.

### Deviations (anything not as registered)

- None to the registered analysis. No code or analyzer change after freeze; no void seed, replacement, rerun or failed push. Seed 50 was pushed after s48/s49 finished because of the 2-session Kaggle cap (a scheduling delay, not a change).
- Process note: the verdict was run once, after all three seeds finished (Addendum rule 1). Outputs were pulled with CR-normalised copying into `software4-v2`; the v2 tree has no `.git`, and the committed code is on `v2-frozen`.

### Exploratory (labelled; not part of the registered verdict)

*Exploratory notes from the strategy session (analysis of existing logs, no new runs; relayed 2026-11-09 and not re-checked by me; verdict unchanged).*
- **generic_forced 0%:** the control adapter was trained on phrase completion (64 steps, final loss about 7.6e-6) and collapses into a "Final answer: Final answer: ..." loop (83-88 of 120 rule C answers per seed). It is a broken adapter, not a working non-rule adapter, so Claim 3's pass is uninformative about task-specific learning (0% is the same floor as no_memory).
- **ours never retrieves on rules:** 0 of 2,160 rule answers used retrieval, and facts always did (54/54). So on rules Claim 1 is trained expert against RAG, with no memory fallback.
- **Birch / Claim 5 (seed 48):** the correct fact was retrieved. Top-k=5 pads the other slots with rule demos (no relevance cutoff or type filter). When all demos were rule B, the frozen core copied the format (`B-NORA`): 12 of 12 format-corrupted in all-B contexts (s48, s50); mixed A+B contexts answered correctly; s49 had no all-B contexts and went 18/18. This is retrieval padding contaminating the format, not memory loss, staleness or absorption. s48 day-0 wrong answers ("Alder-7", "CEO") look like retrieval misses but are unverified.
- **Rule C (hypothesis, not a finding):** ours day-2 hidden C was 114/111/106 of 120 in v1 vs 98/97/93 in v2 (pooled 91.9% to 80.0%). Ruled out: replay mix (identical 24C+6A+2B; an earlier note saying 8 B was wrong), steps, training seeds, train/hidden code overlap (0), code length, routing (all C items went to expert-2), underfitting (s50 fits 1.0 yet has the worst C). Seed difficulty explains little (gold_context C 115/115/110 in v1 vs 110/110/110 in v2). Day-2 C error types over 360 answers per version: wrong/dropped/extra/reordered letters 23 (v1) vs 38 (v2); dropped -7 0 vs 17; other garbled format 4 vs 16; dropped B- 2 vs 1; total wrong 29 vs 72. Best reading: expert-2 is fragile and the v2 worlds exposed it; three seeds per version cannot prove that.
- **Suggested for a V3 pre-registration (not started):** a relevance cutoff or type filter on retrieval, a memory fallback for the expert on composite rule C or more C training, a format-matched control, and fresh seeds, since 48-50 are now dev data. No Kaggle or GPU run without the user's explicit approval.

## v3 smoke (2026-11-09; kernel `adidevv/software4-step4-lean`, version 23; world seed 53)

*Approved by Aadit ("Run smoke"). The full V3 run (seeds 53/54/55) is NOT yet approved.*
- **Timing (UTC from `kaggle kernels list --mine`; IST = UTC+5:30):** pushed 16:52 UTC (22:22 IST), COMPLETE seen 22:28 IST, finished 16:56:45 UTC (22:26:45 IST). Driver wall 171.8 s; setup 47.1 s (excluded); kernel about 4-5 min.
- **Result:** all 10 arms complete (ours, expert_only, hippo_only, rag_semantic, rag_hashing, wrong_rule, ours_generic_forced, text_notes, no_memory, gold_context). Worker s: ours 54.4, expert_only 54.4, wrong_rule 32.1, ours_generic_forced 32.0, others 22-26.
- **Receipt:** `smoke-d844372d5ac34f14`; `kaggle_full_run_eligible: True`; 2 distinct Tesla T4s (`dual_t4`). Config: protocol v3, all three V3 flags on, source_sha256 `d2bc3554...` (matches `docs/V3_FREEZE.txt`).
- **GPU-hours:** this run 0.088 worker-hours. Weekly total not read from Kaggle (earlier smoke/v2 runs not re-summed here).
- **Packaging note:** V3 flags are baked into the generated script by `prepare.py --env` (new option; `prepare.py` is outside the source digest). First push attempt failed locally (old Kaggle CLI 1.7.4.5 on PATH, no GPU time used); second push used CLI 2.2.5.

## v3 full run (2026-11-09 to 2026-11-10; seeds 53/54/55; kernels `adidevv/software4-step4-v3-s53|s54|s55`)

*Approved by Aadit ("put all runs in queue on kaggle"). Verdict run once over all three seeds, as registered (`v3_report.py`, no subset verdicts). Outputs and verdict files are in `C:\Users\aadit\v3_out` (D: was full).*
- **Timing (finished, from `results.json`; IST = UTC+5:30):** s53 18:00:05 UTC (23:30 IST); s54 17:59:43 UTC (23:29 IST); s55 19:00:49 UTC (00:30 IST, 2026-11-10). Driver wall: s53 3285 s, s54 3174 s, s55 3324 s. All three `kaggle_full_run_eligible: True`.
- **GPU-hours (summed worker-hours):** s53 1.770, s54 1.753, s55 1.797 = 5.32 for the full run (plus 0.088 smoke). Weekly total not read from Kaggle.
- **Verdict: UNINFORMATIVE.** Claim 3 is UNINFORMATIVE because the `wrong_rule` control failed its validity gate on seed 54 (well-formed 67.5% of 360 rule items, gate is at least 90%; s53 99.7%, s55 97.8%). Per the freeze, a failed gate leaves Claim 3 uninformative, not failed. No seed was void or invalid, so the report triggered no replacement (56/57).
- **Claims:** 1 PASS, 2 PASS, 3 UNINFORMATIVE, 4 PASS, 5 PASS (safety passes on all seeds: revised, poison fact correct, no absorption, retention).
- **Accuracy (final cycle, case-folded, pooled over 1080 items):** ours 94.17%; expert_only 90.83%; rag_semantic 92.41%; rag_hashing 90.93%; hippo_only 90.93%. Rule C pooled: ours 93.06% (v2: 80.0%). Gold-context validity 90.8-96.7% on every rule and seed; no_memory 0%.
- **ours minus baselines (pooled, 95% CI):** rag_semantic +1.76% [0.00, 3.61]; rag_hashing +3.24% [1.48, 5.00]; hippo_only +3.24% [1.48, 5.00]; no_memory +94.17% [92.78, 95.56].
- **Fallback:** the format-checked frozen-core fallback fired on `ours` 68 / 91 / 84 times (s53/54/55) and never on other arms.
- **Caveats:** `text_notes` scores 0.00% on rule A on every seed (not investigated here). rag_hashing and hippo_only drop to 75.8% on rule A in seed 55 only. Intervals are conditional on trained stacks; 3 seeds give weak evidence on new-stack variability.
- **Not yet checked:** why `wrong_rule` was malformed on s54 (hypothesis only: the same format collapse seen with the v2 generic control). Whether to re-run Claim 3 with a repaired control is Aadit's decision; it would be a new pre-registration, not a replacement seed.
