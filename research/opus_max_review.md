# Opus (max effort) review of plan v2, 2026-11-05

1. **The reflector is too small (VERIFIED).** ACE was tested on 70B+ models, GEPA on 8B+ and ReasoningBank on 12B+. Small models only self-correct well with a strong verifier (2404.17140).
   - Fix: run night reflection on a fixed, stronger model (e.g. Qwen3-8B on Kaggle) for every arm.
2. **The kill rule can't be tested at this scale (JUDGEMENT).** Detecting 3 points takes about 1.7k paired tasks. With 200 tasks the error bar is about 3 points.
   - Fix: synthetic streams with hidden tests (15+ cycles), 3+ seeds and a confidence-interval rule.
   - Add a no-memory arm, an oracle arm, and LoRA on an equal amount of generic data.
3. **The long-context arm is limited by VRAM, and the listed hybrids can't run (VERIFIED).**
   - Qwen3-1.7B needs ~112 KiB of KV cache per token, about 3.5 GiB at 32K. Qwen3.5-2B needs ~12 KiB per token.
   - Qwen3-Next-80B and Kimi Linear 48B are out of budget.
   - Fix: one base for all arms. Test fp16 LoRA on Qwen3.5-2B on a T4. Otherwise give every arm the same token limit.
4. **The model the gate tests isn't the model you deploy (VERIFIED).**
   - A LoRA trained on 0.6B can't be used with a 1.7B core, and fp16 on Kaggle isn't the same as 4-bit locally.
   - GTX 16-series cards have reports of fp16 NaNs (pytorch#58123).
   - vLLM doesn't run natively on Windows and needs compute capability 7.0+.
   - Fix: one base per experiment, llama.cpp locally, gate the exact quantized model plus adapter, and check for NaNs.
5. **Updating the adapter step by step contradicts "weights are a cache" (JUDGEMENT).**
   - Fix: at each deep sleep, retrain the adapter from the base on the full verified log.
   - Drop "distil from pre-update model". Have the frozen base write the replay answers (2509.04259).
   - Train on skills where the lesson is retrieved but the first attempt still often fails.
6. **Reusing one held-out gate every cycle overfits it (VERIFIED, 1506.02629).**
   - Fix: add fresh gate items each cycle, and keep a sealed test set for milestones only.
7. **Hebbian strengthening, reconsolidation and utility selection feed each other (JUDGEMENT).**
   - Fix: never overwrite episodes; save each reconsolidation as a new version.
   - Keep link weights in a separate, decaying index that can be rebuilt, is off by default, and is tested against plain HippoRAG 2.
   - Reward verified success, not how often something is retrieved.

**Minor:** memory layers can be added to an existing model (2604.05248 does it on Qwen2.5-0.5B). The real reason to wait on SMF is that it's built for facts.

**Verdict:** start step 0 and the memory/text stack now. Fix issues 1–4 before building the LoRA arm.
