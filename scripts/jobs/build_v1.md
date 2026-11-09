Your working tree is D:/Aadit/claudez/software4-arms (git worktree, branch `arms`). NEVER read-write or run anything in D:/Aadit/claudez/software4 or the other software4-* folders. Do not commit; leave changes in the working tree. Ignore the `.pytest-tmp`/`.pytest-work` permission warnings; do not try to delete them (use `-p no:cacheprovider --basetemp` under a fresh new folder name if pytest needs a tmp dir).

Read first: baselines/sleep.py (SleepArm, _training_set, how sleep trains the expert), baselines/generic.py, baselines/dream.py, baselines/tests/test_generic.py, step4/run.py (settings, make_arm, AVAILABLE_ARMS, validate_smoke_receipt, the smoke/full flow), step4/prepare.py.

Two jobs.

JOB 1 - format-only control arm `ours_generic_only`.
Purpose: show whether the `ours` gain on rule items comes from learning the rules, or merely from a LoRA expert learning the answer format ("Final answer: <short answer>"). Existing `ours_generic_data` adds generic examples ON TOP of the episode training set, so it does not isolate this.
Implement a new opt-in arm `ours_generic_only` (add to AVAILABLE_ARMS and make_arm; NOT in default ARMS) that runs the identical sleep pipeline as `ours` (same admission, gate, rollback, router, steps, lr, max_examples, same number of training examples per cycle as `ours` would use for the same admitted episodes) but every training example comes from data/generic/pool.jsonl (format 'Final answer: ' + answer, cycling/shuffling deterministically with seed+cycle) instead of the episode pairs. Memory/hippocampus still stores episodes and retrieval at inference time is identical to `ours`, so the ONLY difference to `ours` is what the expert is trained on. The count per cycle = the number of examples `ours` would train on (len(new)+len(replay) from the standard _training_set). If the pool is too small for the count, cycle through it (it may repeat). Document the rule in the docstring. Add tests (fake backend): the arm trains on pool examples only, none of which contain episode content; count equals ours' count for the same episodes; default ARMS unchanged. Keep ours_generic_data and all existing arms working.

JOB 2 - smoke receipt independent of world seed.
Today a full run needs a smoke receipt whose config equals the full run's config including world_seed, so each seed needs its own smoke. Change this so ONE smoke on any world seed licenses full runs on other world seeds, without weakening anything else:
- In step4/run.py settings(): the SMOKE directory key must be computed from cfg WITHOUT world_seed; the FULL run directory key keeps world_seed (distinct output dir per seed). Return both keys (keep settings() callers/tests working; adapt them).
- validate_smoke_receipt: compare previous['config'] and cfg with world_seed removed from both. Everything else (source digest, model, steps, hardware profile, arms executed, topology) stays as strict as now.
- Add tests: smoke at seed 41 receipt accepted for full at seed 44; rejected if source digest/model/steps differ; full output dirs differ per seed.
- Do not change prepare.py behaviour except if needed; `prepare.py --full --world-seed N --arms a,b,...` must still work (arms list is baked into the script; the smoke arms must include every arm of the full run - this check exists in validate_execution, keep it).

Finally run `python -m pytest baselines step4 harness -q -x` and fix failures. Final message: files changed, what you chose, test counts, anything unverified.
