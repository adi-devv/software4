Repo: D:/Aadit/claudez/software4 (branch step4b; a Kaggle run is using this code, so DO NOT edit any existing file). Task: design note only. Write ONE new file docs/online_arm_design.md (max 120 lines). Do not write code outside that note.

Goal: add an arm `ours_online` to the Step 4 harness: after each VERIFIED day-phase episode, take one or a few gradient steps on a small sparse set of weights (fast, local, logged, undoable), then the normal sleep (replay, rank-4 expert, router, gate) consolidates or resets them.
Read: PLAN.md, baselines/sleep.py, baselines/sleep_hf.py, baselines/arms.py, step4/run.py, harness/arms.py (arm interface), and the sparse memory layer code in step0/step0.py, step0b/step0b.py, pilot/pilot.py.
The note must decide, with reasons and file/function pointers:
1. What is updated online (sparse memory layer slots top-k vs a tiny LoRA vs DeltaNet-state-only) and why; reuse existing step0 memory-layer code where possible.
2. Where it hooks in: SleepArm.observe, with only verified episodes (never hidden/gate/sealed data); what happens to the online weights at sleep (merge into the expert, reset, or keep) and how the gate tests the exact deployed state.
3. Safety: per-step update-norm and step-count limits, a log of every update so it can be rolled back, poisoning/drift handling.
4. Metrics: same-day accuracy (new), post-sleep accuracy, forgetting on canaries; same stop-rule thresholds and the same seeds, no change to existing arms or config hash of existing arms.
5. CPU test plan (fake backend and Qwen3-0.6B random-weight/tiny) and a rough GPU cost per seed.
6. Smallest first implementation (files to add/change) and open questions for the owner.
Be concrete and skeptical; flag anything you could not verify.
