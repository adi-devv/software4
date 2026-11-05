# Software4 plan (v4, 2026-11-05)

**One model that learns in its own weights.** It has a frozen hybrid core (the cortex), a learnable layer that grows each sleep, and a hippocampus memory. A sleep phase replays, dreams, checks and trains. Text notes, RAG and long context are **baselines to beat**, not the product.

- **Papers page (plan diagram + paper library):** https://claude.ai/artifact/7CPfNetLBiXViGD5zQt5DM. Local copy: `research/software4-papers.html`.
- **Sleep paper explainer:** https://claude.ai/artifact/65vqoJu3D9HznJVg9FmqLT. Local copy: `research/sleep-paper.html`.
- **Reviews folded in:**
  - `research/critique.md`: v1 red team.
  - `research/opus_max_review.md` and `research/sol_review.md`: v2 reviews.
  - Their safety fixes are kept.
  - Their push towards "text first, weights optional" (v3) was reversed in v4. It drifted from the goal: by the README's test, a model that only gets better notes is Software 3.0.

---

# Part 1 — Current plan (what we build)

## Architecture

| Part | Brain analogy | Choice | Source |
|---|---|---|---|
| **Core** | Cortex | **Frozen Qwen3.5-2B**: a hybrid with 18 Gated DeltaNet layers and 6 full-attention layers, 262K context, Apache-2.0, about 12 KiB of KV cache per token. Fallback: dense **Qwen3-1.7B** | HF cards; hybrid analyses 2507.06457, 2510.04800 |
| **Learnable layer** | Fast-learning cortex | **One new low-rank expert per deep sleep.** Older experts are frozen, so new learning never overwrites them. Capped at N experts, which triggers a core merge. Rival arm: a **sparse memory layer** retrofitted into the core, where only the slots relevant to new data update | Sleep paper 2606.03979; sparse memory FT 2510.15103, 2604.05248 |
| **Hippocampus** | Hippocampus | Memory graph plus vectors (HippoRAG 2 style), typed and timestamped, with provenance and trust. Handles exact recall by day and supplies the sleep loop's replay | HippoRAG 2 2502.14802 |
| Connection strength | Synapses | **Hebbian link weights:** links used together in a *verified success* get stronger, and unused ones slowly decay. Kept in a separate index that can be rebuilt | Our idea |
| Hot/cold tiers | Recent vs old memories | Frequently used memories are fast to reach. Old ones are demoted, never deleted, and pulled back up when used | Our idea |
| Memory rules | — | **Append-only:** an update saves a new version, and revoked content gets a tombstone | Reviews |
| Source of truth | — | The memory log. Experts can be rebuilt from it, and so can the whole learned state on a new base | Reviews |

## Day
- The core plus its experts does the tasks, pulling from the hippocampus.
- Every episode is logged with its result (pass/fail, correction).
- **Nap** every ~50 tasks, or when idle: update the link weights and tiers. No weight training; runs on the GTX 1650.

## Night (deep sleep)

**Trigger:** enough new verified episodes, or repeated failures on a skill. At most once a day.

| Step | What it does | Source |
|---|---|---|
| Select | Verified usefulness, corrected failures and trusted sources. Trust is scored separately, new inputs are quarantined, and there's **no surprise filter** | 2405.05904 |
| Replay | Mix new episodes with old ones from the hippocampus so old skills stay in practice. Replay answers are written by the pre-update model | RL's Razor 2509.04259 |
| Dream | **The Sleep paper's dreaming stage.** The model writes self-edits and practice data (SEAL loop), with gradient-based dream selection (top-k plus some random picks) and novelty from temperature or perturbed retrieval. A dream is rewarded only for a grounded gain. Their ablation: SQuAD drops from 48.1 to 35.7 without dreaming | 2606.03979, SEAL 2506.10943 |
| Check | Grounded only: hidden tests written independently, execution, user corrections. No self-judge | 2507.08794 |
| Train | Train the **new expert** on the dreams that passed plus the replay. Distil from the pre-update model, with a KL penalty to the base | 2606.03979, 2509.04259 |
| Gate | Test the **exact quantized model + experts** that will be deployed, with NaN checks (fp16 on T4/GTX 16-series). Add fresh gate items each cycle, plus canaries and poison checks. Roll back automatically on any regression | Thresholdout 1506.02629 |

## Core update (merge)
- **Triggered by a metric, not a fixed schedule:** the expert cap is reached, forgetting on canaries rises, or the replay backlog grows too big.
- Merge the experts into the core's weights to make a new core version, which then starts a fresh expert stack.
- Run the full gate plus the sealed test set, keep the old core for rollback, and merge at most about once a week.
- The reviews warned that merges can cause forgetting, which is why every merge is gated and can be rolled back.
- **New base models:** rebuild from the memory log; no weight merging across bases.

## Core experiment

**Setup:**
- Synthetic task streams with hidden tests.
- 15+ sleep cycles and 3+ seeds.
- Report 95% confidence intervals.
- Keep a sealed test set and use it only at milestones.

**Arms:**
1. No memory
2. RAG only
3. Long context (same token limit as the others)
4. Text-notes stack (ACE / ReasoningBank-style lessons): the strongest 3.0 baseline
5. **Ours:** hippocampus + experts + sleep
6. **Ours with a sparse memory layer** in place of the experts
7. Ours without dreaming (ablation)
8. Ours trained on an equal amount of generic data (control)
9. Oracle (correct info given in context)

**Success:**
- Ours beats the best baseline by **≥ 3 points**, with the 95% CI lower bound **> 0**.
- Old skills on the canaries drop **≤ 2 points** over all cycles.
- Detecting 3 points needs about 1.7k paired tasks, which is why the streams are synthetic.
- If the text notes win, we report that honestly as a result.

## Helpers (not part of the model)
- **Task and test writer:** a bigger model writes the synthetic streams and hidden tests once, before the experiments. Sarvam credits can cover this later.
- **Grader:** tests and execution, not a model.

## Hardware
- **Kaggle (T4×2, ~30 GPU-h/week):** experiments, expert training, gates.
  - Driven from the CLI, which works, logged in as `adidevv`.
  - One account only, per Kaggle's terms.
- **GTX 1650 (4 GB, no bf16):** development, debugging, the hippocampus, naps, and live use later.

## Build order
1. **Step 0, a feasibility check on Kaggle:**
   - Qwen3.5-2B in fp16 on a T4, checking for NaNs.
   - LoRA/expert adapters on the Gated DeltaNet layers.
   - Whether a sparse memory layer can be retrofitted.
   - llama.cpp speed on the 1650.
   - A Kaggle job runner script.
2. Hippocampus: graph, link weights, tiers, append-only log.
3. Evaluation harness: synthetic streams, hidden tests, sealed set, canaries, CIs.
4. Baselines (arms 1–4).
5. Sleep loop with experts: select, replay, check, train, gate.
6. Dreaming.
7. Sparse memory arm (6) and ablations.
8. Core merge.
9. Later experiments (Part 2).

---

# Part 2 — Experiments to try later

Run each one on the same harness, after build step 7.

| # | Experiment | Why | Source |
|---|---|---|---|
| E1 | **Tutor-assisted dreams:** a bigger model helps write the dreams | Small models fix their own mistakes poorly without a strong verifier, so a 2B model may dream weakly | 2404.17140 |
| E2 | **Retrain from base vs stacking experts:** v3's method, where one LoRA is retrained from the base on the full log each sleep | The Opus review warned that distilling from the old model can lock in its errors | Opus review |
| E3 | **Dense vs hybrid core:** Qwen3-1.7B vs Qwen3.5-2B, with LFM2-1.2B as a third option | Hybrids are cheaper at long context but weaker at copying exactly from the prompt | 2402.01032 |
| E4 | **Sleep cadence:** nap/deep sleep vs immediate updates vs fixed counts | The sol review called the thresholds arbitrary | sol review #4 |
| E5 | **Link weights / tiers off** (ablation) | Do our two memory ideas really help? | Our bet |
| E6 | **Bigger hybrid cores** (Qwen3-Next-80B, Nemotron-H, Kimi Linear 48B) | Out of budget now. Switching means rebuilding from the log | — |
| E7 | **JEPA world model** (V-JEPA 2) | For predicting how the world changes, not just text. Add it once the loop works | V-JEPA 2 |
| E8 | **Our own pretrained core** (with XSA, gated attention) | Only if we ever pretrain; XSA can't be added to a frozen model | 2603.09078, 2505.06708 |

## Dropped (and why)
- **Surprise-based selection:** it attracts poison, and training on unknown facts raises hallucination (2405.05904).
- **Self-judge as a gate:** one token can fool an LLM judge (2507.08794).
- **Text notes as the product (v3):** kept only as baseline arm 4.
- **A fixed 8B reflector in the loop (v3):** the model learns by itself now. A bigger model only writes tasks and tests (and is tried for dreams in E1).
- **Weight merging across base models:** we rebuild from the log instead.
