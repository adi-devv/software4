# Software4

**A brain-inspired LLM that keeps learning after deployment.** A frozen small model works by day. A memory graph and a skill library hold what it learns. At night it sleeps: it reflects, dreams and checks its work, and trains small adapters only when they beat plain text memory.

Status: planning is done (v3), and step 0 (failure labelling + hardware benchmark) is next.

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
| `PLAN.md` | The current plan (v3): architecture, night loop, experiment, build order |
| `CLAUDE.md` | Instructions for Claude: answer style, status %, running the 3 Codex accounts |
| `research/` | Paper notes (`papers_deep.md`, `papers_xsa.md`), reviews (`critique.md`, `opus_max_review.md`, `sol_review.md`), and the HTML pages for the papers and the sleep paper |
| `scripts/run_rotate.sh` | Runs Codex jobs from a queue and moves to the next account when one hits its limit |
| `scripts/codex_usage.sh` | Shows 5h and weekly usage % for each Codex account |

## Key ideas
- **The memory log is the truth; weights are a cache.** Any new base model can be rebuilt from the log.
- **Text first:** the lessons and skills layer must win before any weight training.
- **Grounded checks only:** tests, execution and user corrections, never self-grading.
- **A safety gate every cycle:** fresh held-out items, canaries and poison checks, with auto-rollback.

## Hardware
GTX 1650 (4 GB) locally, plus free Kaggle T4s for the night training.
