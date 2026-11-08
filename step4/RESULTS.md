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
- B, Amber orchard (cycle 1): uppercase the code and wrap it in square brackets.
- C, Cobalt-Amber chain (cycle 2): apply rule A, then rule B to its result. It is a multi-step composition with its own train episodes, so the answer for `abcd` is `[ABCD-7]`.
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
