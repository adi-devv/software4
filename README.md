# software4

Small-scale experiments toward **Software 4.0**: models that keep learning from their own runs.

## The idea

| Era | What you write | What the machine does |
|---|---|---|
| 1.0 | Code | Runs it |
| 2.0 | Data and a loss | Learns weights |
| 3.0 | Prompts | A frozen model follows them |
| 4.0 | Graders and environments | Finds the *how* itself, and keeps what it learns |

A 3.0 agent with memory files and a test loop *looks* like 4.0, but the model never gets better: the
"learning" is notes stuffed into the prompt. 4.0 starts when the agent's own experience updates the model.

**The test:** when it fails, does a human edit the prompt (3.0), or does the grader catch it and the system
fix itself for good (4.0)?

## Why it's hard

- **Forgetting:** training on new things erases old skills
- **Safety:** a model that changes in use is hard to check and trust
- **Cost:** updating weights constantly, or per user, is expensive

## Plan

1. Measure forgetting: train a small model on task A, then B, and track A's accuracy
2. Benchmarks: Split-MNIST, Permuted-MNIST
3. Compare: naive fine-tuning, EWC, replay buffer, one LoRA adapter per task
4. Then: a grader-driven loop where a small model improves on a task from its own scored attempts

Laptop-scale on purpose: the goal is understanding the mechanism, not a leaderboard.

## Status

Just started.
