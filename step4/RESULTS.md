# Step 4 run ledger

Smoke debugging runs v1-v7 and one full run (2026-11-07/08 IST). The full run FAILED the stop rule (see the table).

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
| 4b | **PLANNED** (not run) | Integration fixes after the full-run STOP: clean memory rendering, route-chosen context, per-answer logging (see README "Step 4b"). Code edited by Codex on branch `step4b`; CPU tests pass. Needs a fresh smoke (config hash changed), then seed 41 + 2 more seeds (~1.6 worker-hours). Awaiting user approval for any Kaggle launch |
| 4b smoke | **COMPLETE** (kernel version 8, finished 00:50 IST 2026-11-08; driver wall 124.7 s, 0.063 worker-hours, 2 distinct Tesla T4s) | receipt `smoke-945f4b75d265a230`; all 8 arms complete; `ours` canaries 8/8, 2/2 optimizer steps, 1 scale backoff, 0 skipped, loss 1.90; `kaggle_full_run_eligible: true`. Plumbing only: smoke does not run scored cycles, so `answers.jsonl` is first checked in the full run. Next: Step 4b full run (seed 41 + 2 seeds), awaiting user approval |
