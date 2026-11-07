# Lean step 4 package — prepared only

Default: real `Qwen/Qwen3.5-2B`, seed 0, **three cycles**, eight arms in this order:
`ours, expert_only, hippo_only, rag, text_notes, no_memory, rag_starved, oracle`.
Uses a fresh step-4 world (`harness/step4_world.py`), `SleepArm`, `HFBackend`, scoring helpers
and the harness runner. Rule A is taught in cycle 0, rule B in cycle 1, and the last cycle adds no
exposures and checks retention (rejected candidates may still retry). No forced broken adapter.
This small run can show a trend; it cannot establish the plan's multi-seed statistical gate.

**Order of checks.** Everything was first hot-checked locally on the small `Qwen/Qwen3-0.6B`
(GTX 1650, plain transformer): unit tests, real sleep, and a local smoke. Only then did it go to
Kaggle: smoke on real Qwen3.5-2B (v1-v7, 2 x T4), then the full run. Local 0.6B results say nothing
about 2B quality; they only prove the code path.

## Design, pre-registered before any real run

- **Fair budget.** Every arm that reads context (`ours`, `hippo_only`, `rag`, `text_notes`) gets the
  same 512-token budget at inference; `expert_only` shows an empty context. `HFBackend` now has
  separate caps: inference 1024 tokens, training and replay still 112. `rag_starved` is RAG at the
  old 64-token budget, kept as a labelled row to show what starvation costs.
- **Two ablations.** `hippo_only` is memory retrieval with no night training. `expert_only` is
  `SleepArm`'s sleep, gating and routing, but the evaluation prompt has an empty context.
- **Fresh world.** Seed 41; no rule code (train, gate or hidden) equals a seed-0 code; at least
  100 hidden items per rule (120 by default; `Step4Source` refuses fewer). Scores are reported per
  family (`rule:A`, `rule:B`, `fact:smoke`) so a strong fact score cannot hide a weak rule.
- **Diagnostics per arm.** Routing confusion per cycle (true skill -> handle chosen, or none),
  forced-expert train accuracy per night (does the candidate fit its own training set), and the
  parse-failure rate (hidden answers with no `Final answer:` line), per cycle and overall.
- **Order.** `ours` is scheduled first and does not wait for any baseline; baselines follow.

## Stop rule (pre-registered; descriptive, never tuned)

Evaluated on the last cycle's hidden **rule** items (`rule:A` and `rule:B`), by `stop_rule` in
`run.py`, and written to `results.json` and `RESULTS.md` only for a valid full run
(eligible hardware, not smoke, `ours`, `rag` and `hippo_only` all complete). Continue only if all hold:

1. `ours` beats fair RAG (512) by at least 15 points, paired-bootstrap 95% CI excluding 0
   (10,000 draws over item ids, seed 0, percentile interval);
2. `ours` beats `hippo_only` by the same margin and test;
3. rule B was accepted on its introduction night (cycle 1) and rule A accuracy in the last cycle
   is not more than 5 points below cycle 0.

If an expert did not fit its own training set (forced-expert train accuracy below 0.9 on any
night), a failed verdict reads `inconclusive: fix expert optimisation, rerun`, not `stop`. That is an
optimisation failure to fix, not a reason to rethink the design. A passing verdict is never
relabelled. Thresholds (0.15, 0.05, 0.9) are constants in `run.py`; do not change them after seeing results.

## Step 4b (pre-registered 2026-11-08, after the Step 4 full run returned STOP)

The Step 4 full run (world seed 41, one seed) is logged as **STOP (fail)** and is not relabelled.
Diagnosis from the run data: `ours` rule:B 5% vs `expert_only` 87.5%. Retrieved memory held the
model's own wrong "Response:" lines (day-phase answers stored as episodes), which `ours` copied, and
experts were trained with an empty context but tested with retrieved context. Step 4b changes only
the integration, not the learning mechanism:

- **Clean memory rendering (all memory arms).** Retrieved episodes are shown as question plus
  verified answer (`render_entry` in `baselines/arms.py`); the wrong `Response:` text is never shown.
  Stored hippocampus text is unchanged.
- **Route-chosen context (`ours`).** A routed expert gets an empty context (matching training);
  route "none" gets the retrieved memory. The gate and validation use the deployed prompt.
- **Per-answer log.** `answers.jsonl` per arm (task, day, family, skill, expert, retrieved ids,
  prompt, answer); `step4/analyze.py` splits failures into routing, context and copied-answer causes.
- **Not changed:** stop-rule thresholds (0.15, 0.05, 0.9), day-phase routing, expert training data.
- **Run plan:** fresh smoke on the new code, then world seed 41 plus two more seeds. Ceiling note:
  fair RAG already gets rule B right about 92.5% of the time, so `ours` must win mainly on rule A.
  A second STOP under Step 4b would count against the design.

## Pre-gate replay tolerance

`baselines/sleep.py` (`SleepConfig.replay_flip_fraction`, default and maximum 1/8) lets a candidate
flip at most 1/8 of the replay examples from correct to wrong before the pre-gate rejects it. It
is documented in a comment there. The exact-canary rule (any canary regression rejects) and the
adapter norm cap are unchanged, and non-replay examples keep zero tolerance for right-to-wrong flips. The tolerance was
set from the plan review, not from any run, and must not be tuned to make a run pass. Tests cover
both sides: within tolerance passes and beyond it rejects.

CPU checks (no GPU or downloads):

```bash
python -m pytest step4 -q
S4_BACKEND=fake S4_SMOKE=0 python -B step4/run.py
# Optional, in a Python environment with torch + transformers:
S4_BACKEND=tiny S4_SMOKE=1 S4_ARMS=ours CUDA_VISIBLE_DEVICES='' python -B step4/run.py
```

Prepare the **smoke** upload, without launching anything:

```bash
python -B step4/prepare.py
```

`step4/upload/` contains a metadata file and one self-contained `run.py`. It freezes read-only
repository imports into an archive; it does not duplicate their implementations. Regenerate
after sleepC finishes, then review that frozen source. Upload **that folder**, not raw `step4/`.
The script defaults to `S4_SMOKE=1`: two optimizer steps, every microbatch using the longest
actual encoded train example across all requested cycles, and short evaluation. Microbatch 1,
accumulation 4, maximum 112 input + 16 target tokens, matching HFBackend. A worker budget of 285
seconds per wave of arms (eight arms on two devices is four waves) stops failed smoke runs; cold package installation/model download is separate setup time.
No run has been launched by this job.

After a successful T4×2 smoke **and review**, prepare the full artifact:

```bash
python -B step4/prepare.py --full
```

Attach the successful smoke output using `step4/upload/kernel-metadata.json`'s `kernel_sources`
before the later full submission. A matching source/model/config smoke receipt is required;
the full artifact refuses to proceed without it. Both preparations use the same frozen source
identity. Reattach full outputs there for reuse too. Do not regenerate against changed upstream
files between smoke and full; a changed source hash correctly requires another smoke.

Switches: `S4_SMOKE`, `S4_MODEL`, `S4_CYCLES` (>=3, default 3), `S4_ARMS` (comma-separated). Optional:
`S4_STEPS` (64), `S4_REVISION` (default model pinned to fetched config revision), `S4_OUT`.
Kaggle cannot conveniently pass environment variables to committed scripts: `prepare.py --full`
bakes the mode; edit defaults in the prepared upload only if needed, and review them.
Real mode refuses to run outside Kaggle. CPU modes require `S4_BACKEND=fake|tiny` explicitly.

Real runs validate the driver inventory before package setup, weight downloads or workers.
The default requires two distinct physical T4 UUIDs. `CUDA_VISIBLE_DEVICES=0` fails;
select two inventory indices or full GPU UUIDs. At least two arms must run, including
`ours` for an eligible smoke. Workers receive one UUID each and verify their actual CUDA
UUID, model and compute capability. `hardware.json`, `run.json` and `results.json` retain
the detected inventory/topology; each arm receipt records its scheduled and observed device.
Both devices must have completed worker work. Full runs require the same physical devices
and assignments valid against that topology in the matching smoke receipt; a different
allocation requires another smoke. `S4_SINGLE_DEVICE_DEV=1` explicitly permits a one-T4
smoke, marks its receipt ineligible, and refuses full mode. It has its own config identity.

Each GPU gets one child process through `CUDA_VISIBLE_DEVICES`. Independent baselines run in
parallel; `ours` is first in the queue and has no dependencies. A failed arm does not block the others.
Completed arms are reused; failed receipts are retained for diagnosis, not silently retried.
Artifacts live under `/kaggle/working/step4/{smoke,full}-<config hash>/`: per-arm scores,
adapters, replay/hippocampus state, portable-together checkpoint snapshots, logs, `results.json`,
`timings.json`, and `RESULTS.md`. Keep the complete tree. Cold weights use `/kaggle/working/step4/hf-cache`.
The setup uninstalls old **torchao**, and pins Transformers 5.3.0 / Accelerate 1.13.0.
Optional local HF checks used the existing torch 2.6.0 / Transformers 5.22.2 environment;
the pinned Kaggle environment still needs its own smoke check.

`attn_implementation="eager"` selects full attention only. Independently, `hf.py` checks
module availability and distribution versions for FLA (`flash-linear-attention`, `fla-core`,
`fla`), `causal-conv1d` and HF `kernels`. Any installed/importable optional library refuses
setup before weights download, and refuses each worker before model loading. Use a clean
environment. Detection, versions and refusal/verification status are saved in
`kernel_environment.json` and the setup/arm receipts. Pinned 5.3.0 DeltaNet instances must
select the Torch convolution update, chunk/recurrent functions and gated norm; selected
functions are recorded. On newer global-dispatch releases, the layer check selects the
original undecorated Torch functions and verifies their source. No compiled optional path
is accepted as a reference-path receipt.

The refusal tests use mocked package discovery for both version strings, including module-only
and metadata-only installations, without importing Torch or Transformers. Separate dispatch
contract tests check all four functions and the norm, independently of the installed release.
Optional offline CPU integration tests exercise tiny random Llama and hybrid Qwen models,
two optimizer steps, reload and generation:

```bash
python -m pytest step4/tests/test_run_cpu.py -q -k 'random_tiny_hf_cpu or random_hybrid_targets_cpu'
```

Run that command in an environment with Torch and the intended Transformers version. The
hybrid fixture uses FP32 CPU weights and synthetic `dt_bias=-4` for both releases.
An initial fp16 CPU check on 5.3.0 hit nonfinite `RsqrtBackward` gradients in its reference
Q/K normalization; its random decay initialization can also destabilize chunk math.
These fixture choices test reference dispatch and adapter integration independently of
release-specific random-model fp16 behavior. Production/pretrained parameters and fp16
are unchanged, and nonfinite/skipped steps still fail the run. Local 5.22.2 success alone
does not validate the deployment pin. The fix log records checks under both versions;
an actual pretrained fp16 T4 smoke remains required.

Estimate only: local 0.6B real sleep was about **172 s**. Size scaling gives
`172 × (2/0.6) = 573 s`; assuming a T4 takes 0.5–1.0× that time gives **287–573 s/sleep**,
per sleeping arm; `ours` and `expert_only` each sleep three times (cycle 2 trains nothing), so
**0.5–1.0 active GPU-hours for sleeps** at most. Every arm also scores about 620 hidden answers
(126, 246, 246) with up to 512 tokens of context. Budget roughly **2–5 active GPU-hours including
evaluation/loading across all eight arms**; this is a guess, not a measurement.
Hybrid reference kernels may change this substantially. Session billing/weekly usage is unknown;
record actual wall and account usage after the run instead of treating this estimate as measured.

Review checklist:

- Inspect [hybrid targets and proposed patch](hfbackend_qwen35.patch.md): 48 wrappers, correct loader, thinking disabled.
- Verify final sleepC APIs and the frozen archive; CPU tests pass after regeneration.
- Confirm two T4s, fp16, successful matching smoke, finite logits/loss and zero skipped steps.
- Inspect longest-batch memory/timing; setup time is excluded from the smoke compute budget.
- Confirm no hidden/gate/sealed training leakage, persisted snapshots, and unchanged frozen core.
- Review before any >30-minute run; only then authorize the later Kaggle submission elsewhere.
- Report ours > no-memory and trend >= RAG descriptively, canaries and accepted sleeps; log real GPU usage.

## Fix log

fixB (Codex job, finished after a usage limit; only `step4/` edited):

- **Two distinct T4 devices.** `validate_topology` requires two distinct UUIDs of `Tesla T4`.
  The topology and each arm's scheduled and observed device are persisted in `hardware.json`,
  `run.json`, `results.json` and the arm receipts. A smoke receipt carries its hardware, and a full
  run must match its devices and assignments. `S4_SINGLE_DEVICE_DEV=1` is explicit, marks the
  receipt ineligible, has its own config hash, and refuses full mode.
- **DeltaNet reference path.** On the pinned 5.3.0 release the layer is asserted to use the Torch
  convolution update, chunk/recurrent functions and gated norm; on newer global-dispatch releases
  the original undecorated Torch functions are selected and their source verified. Optional kernel
  libraries (FLA, `causal-conv1d`, HF `kernels`) are detected by module and distribution version and
  refuse setup if present. Detection is recorded in `kernel_environment.json`. A
  Transformers-version-independent dispatch test runs without torch; see above.

plan-review job (`results/PLAN_REVIEW.md`):

1. **Fair RAG and notes.** 512-token inference budget for every context arm; `HFBackend` inference
   cap 1024, training cap 112 unchanged; a 64-token `rag_starved` row kept. The `text_notes`
   distiller is `BackendLLM`: input limit 480 and 64 generated tokens only for `Distill lessons:` calls.
2. **Ablations.** Added `hippo_only` and `expert_only` (`baselines.ExpertOnly`).
3. **Fresh world and diagnostics.** `harness/step4_world.py`; per-family scores; routing confusion,
   forced-expert train accuracy and parse-failure rate recorded per arm and shown in `RESULTS.md`.
   Parse tallies cover hidden answers only (not canaries) and are persisted in `parse.json` so a
   restored run does not double count.
4. **Defaults.** Three cycles; `ours` first and independent of the baselines.
5. **Stop rule.** Written above, implemented as `stop_rule` and `paired_bootstrap` in `run.py`.
6. **Replay tolerance.** Described above; implemented in `baselines/sleep.py`.

Known limits: the large hidden set is scored every cycle, which costs time; the `text_notes`
distiller is the same small model; the expert trains on empty-context prompts while `ours`
evaluates with context; `ours` and `expert_only` train separately, so their gates can diverge.
Nothing here has run on Kaggle or a GPU.
