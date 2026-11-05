# Software4 plan (v3, 2026-11-05)

A brain-inspired LLM system that keeps learning after deployment. A frozen core is paired with a memory graph and a skill library. A "night" loop then consolidates what was learned: text lessons first, and weights only if they prove they help.

- **Papers page (plan diagram + paper library):** https://claude.ai/artifact/7CPfNetLBiXViGD5zQt5DM. Local copy: `research/software4-papers.html`.
- **Sleep paper explainer:** https://claude.ai/artifact/65vqoJu3D9HznJVg9FmqLT. Local copy: `research/sleep-paper.html`.
- **Reviews folded in:**
  - `research/critique.md`: v1 red team.
  - `research/opus_max_review.md` and `research/sol_review.md`: v2 reviews.

---

# Part 1 — Current plan (what we build)

## Architecture

| Layer | Choice | Source |
|---|---|---|
| Core | **One frozen base shared by every arm.** First choice: **Qwen3.5-2B**, a hybrid with 18 Gated DeltaNet layers and 6 full-attention layers (2 KV heads × 256), 262K context, Apache-2.0, about 12 KiB of KV cache per token. Fallback: dense **Qwen3-1.7B** (about 112 KiB per token), with the same token limit for every arm | HF cards |
| Day runtime | llama.cpp with a quantized model on the GTX 1650 | Opus review |
| Memory | Graph + vectors, HippoRAG 2 style, typed and timestamped, with provenance and trust | HippoRAG 2 |
| Memory rules | Episodes are **append-only**: reconsolidation saves a new version, and revoked content gets a tombstone. Hot/cold tiers: demote rather than delete. Link weights sit in a separate, decaying index that can be rebuilt; they're **off by default** (see the experiments section) | Both reviews |
| Skills | Text lesson and skill library | ACE 2510.04618, ReasoningBank 2509.25140, GEPA 2507.19457 |
| Source of truth | The memory log. Weights are a cache that gets rebuilt | — |

## Night loop

| Step | What it does |
|---|---|
| Nap | Every ~50 tasks or when idle: memory linking and text lessons. Runs on the GTX 1650 |
| Deep sleep trigger | Repeated failures with enough diverse verified examples, at most once a day |
| Select | Verified usefulness, corrected failures and trusted sources. Trust is scored separately from novelty, and new inputs are quarantined |
| Reflect | A **fixed, stronger reflector (Qwen3-8B on Kaggle)**, the same for every arm. ACE was tested on 70B+, GEPA on 8B+ and ReasoningBank on 12B+, and small models self-correct poorly (2404.17140) |
| Dream | **The dreaming stage of the Sleep paper** (2606.03979). It is SEAL's loop: write a self-edit, LoRA-SFT a copy, reward = downstream gain, ReST outer loop. The paper adds **gradient-based dream selection** (top-k plus some random picks) and **novelty injection**. With no MoE router, our novelty comes from temperature or perturbed retrieval. The reward uses only grounded checks, not the paper's reward model. Runs in a sandbox. Their ablation: SQuAD accuracy drops from 48.1 to 35.7 without dreaming |
| Check | Grounded only: hidden tests written independently, execution, user corrections. No self-judge (2507.08794) |
| Train (optional) | **Retrain the LoRA from the base on the full verified log** each deep sleep. Target skills where the lesson is retrieved but the first attempt still often fails. Replay answers are written by the frozen base (2509.04259), with a KL penalty to the base |
| Gate | Test the **exact quantized model + adapter** that will be deployed, with NaN checks (fp16 on GTX 16-series). Add fresh gate items each cycle, plus canaries and poison checks. Roll back automatically on any regression |
| Upgrades | Rebuild from the log onto each new base. No weight merging |

## Core experiment

**Setup:**
- Synthetic task streams with hidden tests.
- 15+ cycles and 3+ seeds.
- Report 95% confidence intervals.
- Keep a sealed test set and use it only at milestones (1506.02629).

**Arms:**
1. No memory
2. RAG only
3. Long context (same token limit as the others)
4. Text stack
5. Text + LoRA
6. Text + LoRA trained on an equal amount of generic data (control)
7. Text + LoRA + self-judge (checks whether a judge hurts)
8. Oracle (correct info given in context)

**Kill rule:** drop weight learning unless the LoRA gain over the text stack is ≥ 3 points and its 95% CI lower bound is > 0. Detecting 3 points needs about 1.7k paired tasks, which is why the streams are synthetic.

## Hardware
- **GTX 1650 (4 GB, no bf16, Windows):** day loop, memory, naps, llama.cpp inference.
- **Free Kaggle (T4×2, ~30 GPU-h/week):** 8B reflector, LoRA training, gate runs.

## Build order
1. **Step 0:**
   - Label real failures as missing info, didn't use it, or lacked a skill.
   - Benchmark hardware: VRAM use, fp16 NaNs, llama.cpp speed, and fp16 LoRA for Qwen3.5-2B on a T4.
2. Memory graph + skill library.
3. Evaluation harness: synthetic streams, sealed test set, CIs.
4. Night loop, text only.
5. LoRA arm (needs steps 1–4 working).
6. Later experiments (Part 2).
7. Pick the deploy core.

---

# Part 2 — Experiments to try later

Run each one as an extra arm on the core experiment's harness, after build step 5.

| # | Experiment | Why | Source |
|---|---|---|---|
| E1 | **Sleep-paper consolidation vs our LoRA.** Add a new low-rank expert each cycle with everything else frozen, distil from the pre-update model, use the Learning-to-Imitate RL loss, and reset old experts. Compare against "retrain LoRA from base" | Their math tests used plain Qwen3-1.7B/8B and beat GRPO and OPSD (Qwen3-1.7B AIME-24: 53.2 vs base 49.8). The Opus review advised against distilling from the old model, so let data decide | 2606.03979 |
| E2 | **Sparse memory finetuning** in place of LoRA | Forgetting: 11% vs 71% for LoRA in their setup. Memory layers can be added to an existing model. It's built for facts, which our design keeps in memory | 2510.15103, 2604.05248, 2605.03229 |
| E3 | **Dense vs hybrid core:** Qwen3-1.7B vs Qwen3.5-2B, with LFM2-1.2B as a third option | Hybrids are cheaper at long context but weaker at copying exactly from the prompt | 2402.01032 |
| E4 | **Hebbian link weights on** vs plain HippoRAG 2 | Our own bet. Risk: retrieving something strengthens it, which strengthens retrieval again, a loop that can entrench poison. Strengthen only on verified success | Our bet; Opus review #7 |
| E5 | **Hot/cold tiers + reconsolidation** vs a flat store | Our own bet: retrieval speed and accuracy as the store grows | Our bet |
| E6 | **Sleep cadence:** nap/deep-sleep vs immediate text updates vs fixed counts | Our own bet. The sol review called the thresholds arbitrary | Our bet; sol review #4 |
| E7 | **Bigger hybrid deploy cores** (Qwen3-Next-80B, Nemotron-H, Kimi Linear 48B) | Out of budget now. Revisit if compute appears. Switching just means rebuilding from the log | — |

## Dropped (and why)
- **Surprise-based selection:** it attracts poison, and training on unknown facts raises hallucination (2405.05904).
- **Self-judge as a gate:** one token can fool an LLM judge (2507.08794). It stays only as arm 7.
- **Weight merging on base upgrades:** we rebuild from the log instead.
- **JEPA:** not needed for now.
- **A 0.6B/1.7B model as the night reflector:** too small for ACE, GEPA or ReasoningBank.
