# Text-memory baselines

These arms implement build step 4 of the evaluation harness.
Import `NoMemory`, `Rag`, `LongContext`, `TextNotes`, or `HippoOnly` from `baselines`.
Construct an arm with an LLM and optional `context_tokens` and `seed`.
The default context budget is 3000 tokens, measured by the selected LLM.
Pass the arm to `harness.runner.run_arm` with a stream and result path.
`FakeLLM` is a deterministic offline prompt reader for tests.
`HFLLM` lazily loads Qwen/Qwen3.5-2B with transformers and fp16.
HF prompts and token counts use the tokenizer's chat template and assistant prefix,
with thinking disabled when the template supports it.
`input_limit` bounds formatted distillation inputs (HF default: 32768; Fake: 1000000).
Example: `arm = Rag(HFLLM(), context_tokens=3000, k=5)`.
Run `python -m pytest baselines/tests -q` for the offline tests.
Run `python -m pytest harness/tests/test_runner.py -q` for harness regression tests.
Teach statements and day observations are the only learning inputs.
Test answers and observations do not update the saved state.
RAG uses the offline hashing embedder by default, not semantic embeddings.
Long context retains the newest contiguous sequence of complete episodes.
Notes are generated at sleep, deduplicated, capped, and pruned by outcomes.
Distillation batches whole episodes to fit `input_limit`; an oversized single episode
raises `ValueError` before changing state.
All arms use the same Context/Task format, including an empty context when needed.
Selected memories appear chronologically, with the newest evidence last.
Hippo-only uses the in-memory hippocampus log and naps; it trains no weights.
Text-memory arm checkpoints contain memory state, not model weights or LLM configuration.
Note distillation quality has not been evaluated on a real model or GPU.

## Thin sleep arm (offline / part A)

`SleepArm(FakeBackend(), SleepConfig())` is arm `ours`, with no torch dependency.
Day prompts and logging reuse `HippoOnly`; Backend generates with one expert or
`None`. Night trains a separate expert on at most 64 verified trusted examples
(including up to 8 previously accepted replay examples), using empty context.
Normalized prompt centroids route experts; a canary/replay centroid and similarity
threshold select `None`. Finite weights are required before opening the one-shot
harness gate. The gate scores the complete proposed deployment. Acceptance commits,
marks sleep, and naps; rejection discards. Older experts stay frozen. `run_arm`
automatically binds the required harness services.

Backend `save(path)` / `load(path)` use directories. JSON checkpoints reference
immutable snapshots under the system temp directory: retain these directories
alongside the JSON. Reset reloads initial backend state. Restore with the same
router embedder and token counter; the default counts whitespace tokens (`llm=`
can supply a model token counter). All generation and training use Backend.

`FakeBackend` memorizes normalized task text and has tiny frozen arithmetic skills;
it tests wiring, not model generalization. `next_broken` / `next_nonfinite` inject
candidate failures. Harness corrections use the first sentence after `Correct
answer:` (preserving decimals); sentence-containing values can use an explicit
`Final answer:` line. Run `python -m pytest baselines/tests/test_sleep.py -q`.

## Local HF sleep backend (part B)

Import `HFBackend` from `baselines.sleep_hf` explicitly; normal baseline imports
remain torch-free. It loads only cached files (`local_files_only=True`), using
`S4_MODEL` (default `Qwen/Qwen3-0.6B`). The base is frozen fp16. Independent
rank-4, alpha-8 LoRA adapters wrap q_proj/v_proj; their parameters and low-rank
branch stay fp32. Adapters are never merged. Training uses CUDA fp16 autocast,
GradScaler, AdamW at 2e-4, microbatch 1 accumulated four times, prompt labels
masked, 112 prompt tokens plus at most 16 target tokens (128 total). Generation
is greedy with at most 16 new tokens; Qwen thinking is disabled via its template.
Loss/gradient checks skip nonfinite steps; a nonfinite loss or adapter weight
makes the candidate ineligible. Skipped gradient steps are reported separately.

Each candidate/commit is stored as an adapter-only `.pt` state dict. Save/load
directories include JSON metadata and a sampled frozen-core fingerprint; discard
removes the adapter and its working file and releases unused CUDA allocations.
Older adapter weights stay frozen. Backend inference caches exact prompt/adapter
pairs; invariance tests and the smoke's before/after checks clear that cache.
Snapshots still follow SleepArm's immutable-directory convention above.

Run offline in PowerShell using the GPU-enabled Python:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:S4_MODEL='Qwen/Qwen3-0.6B'
$env:S4_SMOKE='1'
& 'C:/Users/aadit/AppData/Local/Programs/Python/Python312/python' scripts/smoke_sleep.py
Remove-Item Env:S4_SMOKE
& 'C:/Users/aadit/AppData/Local/Programs/Python/Python312/python' scripts/smoke_sleep.py
```

The small run uses two training steps; full uses 64. Both use the same three-cycle
world and unchanged harness runner, grading and paired-noise gate (alpha 0.05).
Six fictional facts include a later revision and a quarantined poisoned claim.
Two opaque rules have 24 initial train and eight hidden examples each; gate items
are four fresh examples per active rule per cycle. Cycle 3 adds eight fresh
arithmetic control examples, trains for one step, then injects finite random damage into
only the new adapter's B matrices. This tests whether the actual deployment gate
rejects it; rejection is measured, never forced. No thresholds are adjusted.

Outputs: `results/smoke_sleep/results.jsonl`, `CHECKLIST.json`, `evidence.json`,
adapter/checkpoint directories, and `results/SMOKE_RESULTS.md`. Small-run outputs
live under `results/smoke_sleep/small/`. Use `--output <fresh-directory>` for a
separate run. Eight base canaries must all pass the real model before execution.
Reports include per-cycle held-out scores, no-context and experts-off ablations,
routing, recall, rollback hashes, wall time and allocated/reserved GPU peaks.
This is a smoke test, not the harness's multi-stream success criterion.

Tests: `python -m pytest baselines/tests harness/tests hippocampus/tests -q`.
`test_sleep_hf.py` builds a tiny random Qwen locally and uses no model downloads;
it skips through `pytest.importorskip` when torch is absent (including Python 3.13).
The existing hippocampus, sleep arm, other arms, grader and gate are unchanged.
