# Step 0 results (2026-11-05)

Script: `step0/step0.py`, run on Kaggle (2× T4) with `scripts/kaggle_run.sh step0`. Raw output: `step0/results.json`. Run 1 crashed at test 3 on Kaggle's old torchao; run 2 uninstalls it.

## Verdict: feasible. Every part runs on our hardware. Two lessons change the plan.

| # | Test | Result | Numbers |
|---|---|---|---|
| 1 | Qwen3.5-2B in fp16 on T4, NaN check | PASS | 1.882B params; logits finite at 1K tokens; fp16 vs fp32 top-1 agreement 99.7%, max logit diff 0.10; 21 tok/s (HF, batch 1); 5.7 GB peak |
| 2 | Baseline scores | PASS | GSM8K (100) **70%**; toy skill A 0%, toy skill B 0% (unseen, as expected) |
| 3 | LoRA expert A (r=16, all layers incl. Gated DeltaNet, 16.8M params) | PASS mechanically, **forgets badly** | Skill A 0 → **90%**; no NaN steps; 8.9 GB peak. GSM8K 70 → **10%** |
| 4 | Expert B trained with A frozen, then both on | PASS mechanically, **experts interfere** | B alone: skill B **88%**. Both on: skill A 90 → **53%**, skill B 82%, GSM8K **8%** |
| 5 | Sparse memory layer retrofit (layer 12, 8192 slots, top-32, adapters off) | PASS, **forgets little** | Skill A 0 → **83%**; GSM8K 70 → **64%** |
| 6 | llama.cpp on GTX 1650 (Q4_K_M) | PASS | 383 tok/s prompt, 80 tok/s generation; answers "Paris" |
| 7 | Kaggle job runner from CLI | PASS | Push, poll, download all scripted; a full run takes ~25 min |

## Lessons

1. **A plain expert trained only on new data wrecks old skills** (GSM8K 70 → 10). The model starts answering everything in the new format. This is the forgetting the plan's replay + KL-to-base step is meant to stop, so step 5 must show replay fixes it, not assume it.
2. **Stacked experts that are all switched on at once interfere** (A drops 90 → 53 when B is added). Summing every expert does not scale. Each expert needs a **router**: only the expert(s) relevant to the input switch on.
3. **The sparse memory layer learned almost as well (83% vs 90%) and kept most of the old skill** (64% vs 10%), with no replay at all. This matches the sparse memory finetuning papers (2510.15103). It is now the stronger candidate for the learnable layer.

Caveats: toy skills, 600 examples, 2–3 epochs, one seed, 100 GSM8K items. Good enough for "does it run, and which way does it lean", not for claims.
