# Software4 plan (v4, 2026-11-05)

**One model that learns in its own weights.** It has a frozen hybrid core (the cortex), a learnable layer that grows each sleep, and a hippocampus memory. A sleep phase replays, dreams, checks and trains. Text notes, RAG and long context are **baselines to beat**, not the product.

- **Architecture and milestones (diagrams, design choices, progress):** https://claude.ai/artifact/DjoyAzMJP5NK9QZMXxbJNz. Local copy: `docs/architecture.html`.
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
| **Learnable layer** | Fast-learning cortex | **One new low-rank expert per deep sleep**, with a **router** so only the relevant expert(s) switch on. Older experts are frozen, so new learning never overwrites them. Capped at N experts, which triggers a core merge. Rival arm: a **sparse memory layer** retrofitted into the core, where only the slots relevant to new data update. **Step 0: the memory layer forgot far less (GSM8K 70 → 64 vs 70 → 10 for a plain expert), so it is now co-lead, not just a rival** | Sleep paper 2606.03979; sparse memory FT 2510.15103, 2604.05248; `step0/RESULTS.md` |
| **Hippocampus** | Hippocampus | Memory graph plus vectors (HippoRAG 2 style), typed and timestamped, with provenance and trust. Handles exact recall by day and supplies the sleep loop's replay | HippoRAG 2 2502.14802 |
| **Connection strength** (our addition to HippoRAG 2) | Synapses | **Plastic links:** HippoRAG 2's links are fixed once built. Ours strengthen when memories are used together in a *verified success*, and slowly fade with disuse (a forgetting curve). Recall uses HippoRAG 2's spreading activation (Personalized PageRank), so strong links are reached first. Search is **fast first, deep only if needed**: a shallow pass along strong links, then a wider search if nothing fits. No strict tiers: "cold" just means outside the small fast path (context budget, top-k, the RAM index). Nothing is deleted, and a cold memory that gets used becomes strong again. Link weights sit in a separate index that can be rebuilt | HippoRAG 2 + our idea |
| Memory rules | — | **Append-only:** an update saves a new version, and revoked content gets a tombstone | Reviews |
| Source of truth | — | The memory log. Experts can be rebuilt from it, and so can the whole learned state on a new base | Reviews |

## Day
- The core plus its experts does the tasks, pulling from the hippocampus.
- Every episode is logged with its result (pass/fail, correction).
- **Nap** every ~50 tasks, or when idle: update link strengths (strengthen and fade) and refresh the fast index. No weight training; runs on the GTX 1650.

## Night (deep sleep)

**Trigger:** enough new verified episodes, or repeated failures on a skill. At most once a day.

| Step | What it does | Source |
|---|---|---|
| Select | Verified usefulness, corrected failures and trusted sources. Trust is scored separately, new inputs are quarantined, and there's **no surprise filter** | 2405.05904 |
| Replay | Mix new episodes with old ones from the hippocampus so old skills stay in practice. Replay answers are written by the pre-update model | RL's Razor 2509.04259 |
| Dream | **The Sleep paper's dreaming stage.** The model writes self-edits and practice data (SEAL loop), with gradient-based dream selection (top-k plus some random picks) and novelty from temperature or perturbed retrieval. A dream is rewarded only for a grounded gain. Their ablation: SQuAD drops from 48.9 to 35.7 without dreaming | 2606.03979, SEAL 2506.10943 |
| Check | Grounded only: hidden tests written independently, execution, user corrections. No self-judge | 2507.08794 |
| Train | Train the **new expert** on the dreams that passed plus the replay. Distil from the pre-update model, with a KL penalty to the base | 2606.03979, 2509.04259 |
| Gate | Test the **exact quantized model + experts** that will be deployed, with NaN checks (fp16 on T4/GTX 16-series). Add fresh gate items each cycle, plus canaries and poison checks. Roll back automatically on a regression larger than a margin set from measured noise, so chance swings don't throw out good updates | Thresholdout 1506.02629 |

## Core update (merge)
- **Triggered by a metric, not a fixed schedule:** the expert cap is reached, forgetting on canaries rises, or the replay backlog grows too big.
- Merge the experts into the core's weights to make a new core version, which then starts a fresh expert stack.
- Run the full gate plus the sealed test set, keep the old core for rollback, and merge at most about once a week.
- The reviews warned that merges can cause forgetting, which is why every merge is gated and can be rolled back.
- **New base models:** rebuild from the memory log; no weight merging across bases.

## Core experiment

**Setup:**
- Synthetic task streams with hidden tests.
- 15+ sleep cycles and 3+ seeds, with more seeds if the intervals come out too wide.
- Report 95% confidence intervals, calculated per stream because tasks within a stream aren't independent.
- Keep a sealed test set and use it only at milestones.
- **Rough compute (estimate, not measured):** 5 arms train weights, so 15 cycles × 3 seeds is 225 deep sleeps. At 20–30 minutes each, that's about 75–110 GPU-hours, before task runs and extra seeds.

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
10. **Hippocampus only, no weight learning:** the key baseline, since it shows whether the weights add anything beyond better memory
11. **Ours with core merges,** compared with arm 5, to measure whether merging causes forgetting

**Success:**
- Ours beats **every baseline (arms 1–4 and 10)** by **≥ 3 points**, with each paired stream-level CI lower bound **> 0**, using Bonferroni correction for 95% family confidence across the five comparisons (99% per CI).
- Old skills on the canaries drop **≤ 2 points** over all cycles, using enough canaries to measure 2 points. **Harness finding:** 120 canaries cannot verify a 2-point drop; the normal approximation needs **≥ 308** (335 for 80% exact McNemar power, at 2% discordance, one-sided α=.05), so we use 400 each of GSM8K and ARC.
- Detecting 3 points needs about 1.7k tasks, **but only in the paired design** (same tasks for both arms; ≈ 1,742 at 20% discordance, two-sided; 1,716 at 25%, one-sided; α=.05 and power=.8, before multiple-comparison correction). Unpaired it is about 4.4k per arm. Non-independent tasks will need more.
- If the text notes win, we report that honestly as a result.

## Helpers (not part of the model)
- **Task and test writer:** a bigger model writes the synthetic streams and hidden tests once, before the experiments. Sarvam credits can cover this later.
- **Grader:** tests and execution, not a model.

## Hardware
- **Kaggle (T4×2, ~30 GPU-h/week):** experiments, expert training, gates.
  - Driven from the CLI, which works, logged in as `adidevv`.
- **GTX 1650 (4 GB, no bf16):** development, debugging, the hippocampus, naps, and live use later.

## Build order
1. **Step 0, a feasibility check on Kaggle — DONE, all 7 pass** (`step0/RESULTS.md`):
   - fp16 on T4 is clean (99.7% agreement with fp32, no NaNs); baseline GSM8K 70%.
   - Experts train on the Gated DeltaNet layers, but a plain expert forgets badly (GSM8K → 10%), and experts that are all on at once interfere (skill A 90 → 53%). Hence the router, and replay must be proven in step 5.
   - The sparse memory layer retrofits and forgets little (GSM8K 64%).
   - llama.cpp on the 1650: 80 tok/s. The Kaggle runner works from the CLI.
   - **Step 0b, forgetting test — DONE** (`step0b/RESULTS.md`): replay keeps GSM8K at 71–74% (expert A 94%, B 76%). The router keeps both skills, while both experts on at once drop B to 0%. Memory layer + replay: skill A 82%, GSM8K 77%. Toy test: replay overlaps the math test, and the router's task was easy.
1b. **Pilot, before building the rest:** a few sleep cycles on small hand-made tasks, with experts and the memory layer, replay and a simple gate. It must show new skills being kept without old ones dropping. If not, rethink before step 2.
2. Hippocampus: graph, plastic link strengths, fast-then-deep recall, append-only log.
3. Evaluation harness: synthetic streams, hidden tests, sealed set, canaries, CIs.
4. Baselines (arms 1–4 and 10).
5. Sleep loop with experts and the sparse memory layer (arms 5 and 6): select, replay, check, train, gate.
6. Dreaming.
7. Ablations (arms 7 and 8).
8. Core merge (arm 11).
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
| E5 | **Connection strength on vs plain HippoRAG 2** (ablation) | Do plastic links really beat fixed ones? Risk: recall strengthens links, which makes the same recall more likely (a feedback loop), so links strengthen only on verified success | Our bet; Opus review #7 |
| E6 | **Bigger hybrid cores** (Qwen3-Next-80B, Nemotron-H, Kimi Linear 48B) | Out of budget now. Switching means rebuilding from the log | — |
| E7 | **JEPA world model** (V-JEPA 2) | For predicting how the world changes, not just text. Add it once the loop works | V-JEPA 2 |
| E8 | **Our own pretrained core** (with XSA, gated attention) | Only if we ever pretrain; XSA can't be added to a frozen model | 2603.09078, 2505.06708 |
| E9 | **Same approach on a world model (very late):** frozen pretrained world model + learnable layer + episode memory + deep sleep, tried first in a small simulated world | Keep-learning may not depend on predicting text, and reality grades predictions automatically. Untested; do only after the text version works and after E7 | Our idea |
| E10 | **BDH-style short-term layer:** a small Hebbian layer on top of the frozen core, inspired by Dragon Hatchling | BDH state is short-term and its weights stay fixed, so test whether it helps within a session. Not as the core: no released weights, and it must be trained from scratch | 2509.26507 (partly read) |

## Dropped (and why)
- **Surprise-based selection:** it attracts poison, and training on unknown facts raises hallucination (2405.05904).
- **Self-judge as a gate:** one token can fool an LLM judge (2507.08794).
- **Text notes as the product (v3):** kept only as baseline arm 4.
- **A fixed 8B reflector in the loop (v3):** the model learns by itself now. A bigger model only writes tasks and tests (and is tried for dreams in E1).
- **Weight merging across base models:** we rebuild from the log instead.
