# Software4

**A brain-inspired LLM that keeps learning after deployment, in its own weights.** It has a frozen hybrid core (the cortex), a learnable layer that adds a new small expert each sleep, and a hippocampus memory whose links strengthen with use. At night it replays, dreams, checks its work against tests, and trains. Text notes and RAG are only baselines it has to beat.

Status: planning is done (v4). Step 0 (feasibility) passed; see `step0/RESULTS.md`. Next: the hippocampus and the evaluation harness.

- **Plan + papers page:** https://claude.ai/artifact/7CPfNetLBiXViGD5zQt5DM
- **Sleep paper explainer:** https://claude.ai/artifact/65vqoJu3D9HznJVg9FmqLT

## The idea: Software 4.0

| Era | What you write | What the machine does |
|---|---|---|
| 1.0 | Code | Runs it |
| 2.0 | Data and a loss | Learns weights |
| 3.0 | Prompts | A frozen model follows them |
| 4.0 | Graders and environments | Finds the *how* itself, and keeps what it learns |

**The test:** when it fails, does a human edit the prompt (3.0), or does the grader catch it and the system fix itself for good (4.0)?

**Why it's hard:** training on new things erases old skills; a model that changes in use is hard to check and trust; and updating weights constantly is expensive.

## What's here

| Path | What |
|---|---|
| `PLAN.md` | Part 1: the current plan (v4: architecture, sleep, core merge, experiment, build order). Part 2: experiments to try later |
| `CLAUDE.md` | Instructions for Claude: answer style, status %, running the 3 Codex accounts |
| `research/` | Paper notes (`papers_deep.md`, `papers_xsa.md`), reviews (`critique.md`, `opus_max_review.md`, `sol_review.md`), and the HTML pages for the papers and the sleep paper |
| `scripts/run_rotate.sh` | Runs Codex jobs from a queue and moves to the next account when one hits its limit |
| `scripts/codex_usage.sh` | Shows 5h and weekly usage % for each Codex account |

## Key ideas
- **The memory log is the truth; weights are a cache.** Any new base model can be rebuilt from the log.
- **Learning goes into the weights:** new experts each sleep, merged into the core when a metric says so. Text notes are a baseline, not the product.
- **Brain-like memory:** links between memories strengthen with verified use and fade with disuse, so strongly connected memories are recalled first (plastic links on top of HippoRAG 2).
- **Grounded checks only:** tests, execution and user corrections, never self-grading.
- **A safety gate every cycle:** fresh held-out items, canaries and poison checks, with auto-rollback.

## Hardware
Free Kaggle T4s for experiments and training, run from the Kaggle CLI. A GTX 1650 (4 GB) locally for development, memory and live use.
