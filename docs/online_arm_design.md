# Step 4: `ours_online` design proposal
This is a new experimental arm, with preregistered settings; the running Step 4b job
and existing arms remain untouched. The online effect is provisional until sleep.
## Online mechanism
- Subclass `SleepArm` in a new module. Add one residual sparse memory at the middle
  text block's MLP, following `step0/step0.py:Mem`, `step0b/step0b.py:Mem` and
  `pilot/pilot.py:make_mem_class` / `MemoryArm`. Normalize inputs, dot with fixed
  keys, top-k, softmax, and add the weighted value vectors to the MLP output.
  Start with 256 slots, k=4, zero values and seeded frozen keys. Apply it on all
  routes, including no expert. Freeze the core, keys and all expert weights.
- For each eligible episode do one teacher-forced answer-token cross-entropy step,
  plain SGD, lr=1e-3, only on value rows selected across its tokens. Limit the
  union to 32 rows deterministically by aggregate routing weight; mask other rows.
  SGD has no momentum or weight decay, so unselected rows cannot move. Top-k is
  per token, not necessarily per episode; dense key scoring and backbone backward
  still cost compute. Initial zero values make the residual initially a no-op.
- This reuses the prototypes' mechanism, not their executable scripts or 8192/32
  sizing. A tiny LoRA updates shared projections broadly and overlaps the nightly
  rank-4 experts (`sleep_hf.py:LoRALinear`); DeltaNet state is transient inference
  cache, not persistent trainable weights, and the 0.6B test model lacks DeltaNet.
  Neither prototype establishes that these sizes or a single step learn usefully.
## Data boundary and sleep transaction
- Override `observe`, call `HippoOnly.observe` via `super()`, then update only if
  `task.mode == 'day'`, `result` is a boolean, source is trusted, and a verified
  target can be parsed: correct response with a valid final answer, or a trusted
  correction for a wrong response. Skip teach (`result is None`), missing/ambiguous
  targets and all other sources. Reuse `SleepArm._example`/`parse_answer` semantics;
  do not train on the raw wrong answer. `harness/runner.py:_train_one` grades before
  observe and supplies correction; verify correction provenance in tests before use.
  No hidden, gate, sealed, or unverified examples enter the update or replay buffer.
- Snapshot the last accepted deployment at day start. At sleep train the existing
  replay-based rank-4 expert from trusted episodes (`SleepArm._training_set`), with
  the online residual disabled during training. Stage a candidate containing the
  proposed expert/router and **zeroed online values**: consolidation is through
  shared verified examples, not an algebraic merge of incompatible architectures.
  Compare it against the last accepted deployment, also with zero online values,
  using the existing route, prompt, quantization and generation path. The gate thus
  measures cumulative drift from the safe state, rather than blessing day drift.
- Adapt `SleepArm.sleep`'s finite/norm checks, forced train/replay checks, exact
  canary-output check, route coverage and one `GateSession.decide` to those exact
  two complete states. Generate both maps before the one decision; never train on
  gate data. On acceptance atomically install the tested expert/router/zero-memory
  candidate; on rejection, error, or no candidate restore the last accepted state
  and zero online values. Save that state and hashes with the gate receipt. The
  gate does not certify intermediate day answers; zeroing may lose online gains.
  Explicitly test state identity after commit and save/load, including residual on
  every route. `SleepArm.save_state` currently caches backend snapshots: invalidate
  on every update and include memory, logs and transaction state in checkpoints.
## Bounds, audit and failure handling
- Clip the actual concatenated selected-row parameter delta to L2 <= 0.01 per step;
  reject nonfinite loss/gradients/weights and restore the pre-step rows. Cap at one
  step per eligible episode, 32 steps and cumulative delta norm <= 0.1 per cycle;
  stop updates for that cycle when a cap is reached. These are initial engineering
  limits, not calibrated behavioral guarantees; record clipping and skips.
- Before applying each update durably append an intent with cycle/attempt, episode
  ID, source, target digest, seed, selected row IDs, pre/post hashes, learning rate,
  norms and exact old row values (or a durable snapshot reference); append commit
  after it succeeds. Include model/config identity, monotonic sequence and rollback
  events. Flush before mutation, recover incomplete intents by restoring preimage,
  and distinguish abandoned attempts on runner resume. Protect logs as training
  data; no gate answers or secrets. Reverse committed preimages to undo any suffix.
- Quarantine untrusted/poisoned inputs even when the grader reports a result. A
  verified trusted label can still be wrong: rate limits, rollback and the existing
  poison gate reduce exposure but cannot guarantee safety. On inconsistent labels,
  nonfinite state or drift alarm, suspend updates and restore the day-start snapshot;
  log the cause. Day-time canary evaluation would spend/reuse protected data, so do
  not add it; retain the existing sleep checks and inspect poisoning in held-out tests.
## Evaluation and identity
- Report existing pre-update `phase=day` scores from `_train_one` in order, per
  cycle/skill and exposure index, plus a matched, disjoint same-day probe after
  eligible updates and before sleep (runner-owned grading, no `observe`). Report
  probe timing and counts; do not confuse training-item accuracy with generalization.
  Add post-sleep hidden accuracy and canary change from cycle -1, per cycle/skill,
  plus gate decisions, update counts/norms, wall time and peak memory. Same probe
  schedule for all comparison arms; never use probes to choose updates.
- Apply `step4/run.py:stop_rule` unchanged in substance to a separately named
  `ours_online` verdict: >=15 points over both fair RAG and hippo_only, paired
  95% bootstrap CI low > 0, B accepted at cycle 1, A drop <=5 points; failed
  forced fit <0.9 is inconclusive. Preserve original `ours` verdict. PLAN's <=2
  point canary goal is distinct and current small canary set cannot establish it.
- Pair exactly the same stream, world/model seeds, items and hardware with existing
  arms. Current `settings` and `make_arm` hard-code model seed 0 and world seed 41;
  README calls for two more world seeds, but their values and seed plumbing need
  owner agreement. Do not claim multi-seed support already exists.
- `settings` hashes `source_digest` into the common config, so editing runner code
  normally changes even existing-arm hashes. Preserve the existing digest algorithm
  and historical artifacts; run this as a separately versioned experiment with an
  online-specific identity containing its full source and parameters. Reuse old
  receipts only after exact original source/config/stream identity verification;
  never relabel them under a new hash. New code cannot keep the current shared hash
  by simply excluding itself without weakening provenance.
## First implementation and checks
Add `baselines/online.py` for the arm and sparse layer; extend the HF backend with
opt-in residual/training/snapshot support and fake backend with deterministic state.
Add narrowly scoped arm registration, probe/reporting and separate identity in
`step4/run.py` and `harness/runner.py`; add focused tests under `step4/tests/`.
Keep existing defaults, arm behavior and stop rule intact. Confirm hook target and
model output shape for Qwen3.5 before declaring support; `HFBackend` currently has
no sparse-memory API. The change requires a fresh source-matched smoke receipt.
CPU: fake end-to-end tests for eligibility, one-step ordering, caps, poison/teach
skips, exact gate state, accept/reject/no-candidate rollback, logs and interrupted
resume; assert other arms' results/identity stay unchanged. Offline random-weight
tiny Qwen3 configuration and, when locally available, Qwen3-0.6B architecture on
CPU: forward/backward, touched rows only, finite clipping, routed generation and
save/reload equality. Random weights test plumbing, not accuracy; no download is
assumed. Then run the pinned real-model GPU smoke before any full comparison.
Cost per seed: existing PLAN estimates 20–30 min per deep sleep; at three sleeps
that is roughly 1–1.5 GPU-hours for the arm, plus day updates and evaluation.
Budget an additional 0.1–1 GPU-hour for <=96 online steps and probes, giving
roughly 1.1–2.5 GPU-hours/seed on one T4, excluding setup and other arms. This is
an unmeasured order-of-magnitude estimate; dense top-k scoring, sequence length,
backprop and gate inference may dominate. Measure a smoke before scheduling.
Owner decisions: approve source/correction trust policy and inconsistent-label
handling; choose two additional world seeds and probe construction/count; approve
initial caps and whether resetting on a failed or absent sleep is acceptable; agree
on separate identity and historic receipt reuse. Confirm target hardware/model and
measure cost before treating the proposed limits or efficacy as established.
