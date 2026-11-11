# Job: Direction D, teachability screen: design + code (CPU-tested, no GPU, no Kaggle push)
Work ONLY in the git worktree D:/Aadit/claudez/software4-screen (branch screen-d). Commit there with message ending "Co-Authored-By: Codex <noreply@openai.com>". Do not touch D:/Aadit/claudez/software4 (main checkout) or push.

Read first: D:/Aadit/claudez/software4/docs/DIRECTION_REVIEW_2026-11-11.md (direction D), PLAN.md Direction section, CLAUDE.md "Kaggle" rules, step4b (rule A letter reversal: context could not teach, training could), step5 world/task generators, and the existing frozen-core inference code (Qwen model, fp16 on T4; find model id and loader used by step4/v3).

Goal of the screen (no training): for ~25 procedurally generated skill families, measure the frozen core's accuracy under context arms: (a) no context, (b) rule text only, (c) rule text + 5 retrieved/worked examples, (d) 16-shot, (e) 32-shot examples. A family is a "D candidate" if best context arm < 50%. Training is not in this job.

Deliver in the worktree under screen/:
1. screen/SKILLS.md: the ~25 families, one line each: what it is, why it might be hard to teach by context (e.g. character-level ops, multi-step algorithmic, arbitrary mappings, tokenizer-hostile), deterministic generator + exact-match scorer. Spread from easy (context should solve) to hard. Include letter reversal (rule A) as anchor.
2. screen/skills.py: generators + scorers, seeded, held-out item split.
3. screen/run_screen.py: loads the frozen core, runs all arms on N items per family (default 40), batched, greedy, writes screen_results.json + a summary table. Reuse the repo's existing prompt format ("Final answer: <x>") and scorer. Supports S4_SMOKE=1 (2 families, 4 items, longest 32-shot prompt) and a tiny random-weight model mode for CPU testing. Two T4s: shard families across 2 processes via CUDA_VISIBLE_DEVICES.
4. screen/kaggle/: kernel folder compatible with scripts/kaggle_run.sh (look at an existing step's upload folder for the pattern), uninstall torchao as other steps do.
5. tests: run the CPU tiny-model test and smoke mode on CPU; paste the output. Estimate T4 time per arm from prompt lengths.
Output (markdown): file list, test output, estimated GPU-h, open design questions.
