# v2 fixes (single owner of all files below). Work only in D:\Aadit\claudez\software4-v2. Do not touch D:\Aadit\claudez\software4. No commits, no Kaggle/GPU, no downloads.
Use Python 3.12 (C:\Users\aadit\AppData\Local\Programs\Python\Python312\python.exe), `-I` for tests. Delete any temp dirs you create.

Opus-approved fixes for three Sol findings:

## P1: training divergence (baselines/sleep.py ~500)
Do NOT change how `ours` or `expert_only` train (treatment must stay as in v1). Instead:
- Log per-sleep training example counts and steps in every sleep's info (all SleepArm-based arms: ours, expert_only, ours_generic_forced, generic_forced_noret).
- step4/v2_analyze.py: Claim 3 requires `expert_only` to have accepted ALL 3 sleeps AND per-sleep training example counts (and steps) matching `generic_forced_noret` on every valid seed. Any mismatch => Claim 3 FAIL for the run (not void, does not authorize a replacement seed). Make the verdict/markdown say why.
- Add a test reproducing Sol's case (identical observations, `expert_only` rejects cycles 0-1 while forced control accepts: 14 vs 9 training examples) and assert Claim 3 FAILs via the mismatch rule. Test the passing-match case too.
- Record the rule in docs/V2_PREREG_DRAFT.md (Claim 3 section, short).

## P2: rag_starved
`rag_starved` is not registered in docs/V2_PREREG_DRAFT.md, so remove it from `V2_ARMS` in step4/run.py (also from v2 dispatch/smoke/wiring tests). Keep it in v1 ARMS and v1 behaviour unchanged. Update step4/tests/test_v2_wiring.py accordingly.

## P3: punctuation normalization
`_semantic_value` and `_format_leaked` in step4/v2_analyze.py must use the same normalization as ordinary grading (reuse the grading normalizer; do not reimplement differently). Tests: `Final answer: Nora-7.` and `B-NORA.` => format_corrupted AND format_leaked=True; `B-Zorg.` => absorbed. Make sure plain `Nora-7` still classes as before and abstention still does not pass.

## Doc check
Make docs/V2_PREREG_DRAFT.md item 5 say the bootstrap is "unstratified, pairs kept within seed" (matches paired_item_bootstrap's (seed, task_id) keys).

Finally run the whole CPU suite. Report: files changed, test counts, the pre-existing failure test_random_hybrid_targets_cpu should be the only failure (Transformers lacks Qwen3_5TextConfig). Also confirm the v1 digest of the software4 tree is untouched (you don't edit it).
