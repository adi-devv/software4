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
| 4b smoke, ours_salient + expert_only, world seed 41 | **COMPLETE** (wall 87.2 s, 0.043 worker-hours; both arms complete; kernel status read ERROR though outputs and log show no exception, so treat the receipt as unverified until checked) | receipt `smoke-e840d6be67982d26`; arms baked with `prepare.py --arms`; no full `ours_salient` run approved yet |
