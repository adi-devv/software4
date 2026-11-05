# XSA + other architecture papers (checked 2026-11-05)

## Task A: XSA = Exclusive Self Attention (verified; only XSA found in search)
- Title: "Exclusive Self Attention". Author: Shuangfei Zhai (Apple), sole author listed. arXiv:2603.09078, v1 10 Mar 2026, cs.LG/cs.CL. https://arxiv.org/abs/2603.09078 (Apple page: machinelearning.apple.com/research/exclusive-self-attention)
- Mechanism: run normal causal softmax attention to get y_i, then subtract the part of y_i pointing along the token's own value vector: z_i = y_i - (y_i . v_i) v_i / ||v_i||^2. Output is orthogonal to own v_i. Rationale (empirical only, no theory): standard attention outputs are often highly similar to v_i ("attention similarity bias"), duplicating what the FFN/residual already carries; XSA forces attention to carry only info from OTHER tokens. About 2 lines of code, no new parameters.
- Results (trained from scratch on FineWeb, ~100B tokens, 2048 ctx, nanoGPT + RoPE; avg of 8 tasks ARC-E, BoolQ, HellaSwag, LAMBADA, OBQA, PIQA, SIQA, WinoGrande):
  - 0.7B: 53.22 -> 53.48 (+0.26)
  - 1.3B (called 1.4B in Table 1): 56.16 -> 57.19 (+1.03)
  - 2.7B: 58.06 -> 59.42 (+1.36)
  - Lower train/val loss at all sizes; gain grows with model size and sequence length (tested 512-16k, 1.3B). Some tasks drop (BoolQ 1.3B 65.47 -> 62.29; OBQA 0.7B 35.0 -> 32.2). Acts as implicit attention sink; stable across learning rates.
- Cost: "minimal" speed/memory overhead (B200, bf16; numbers only in a figure).
- Code: no official repo found; pseudocode only (Algorithm 1). A diffusers issue (#13447) asks for it. No independent replication found.
- Pretrained model? Paper only trains from scratch; NO fine-tuning/retrofit experiments. Retrofitting a pretrained model changes its attention outputs, so it would need adaptation training (untested). Untested: Muon, >2.7B, non-text, hybrids/SSM.

## Task B table
| Paper | Org | Date | URL | Key idea | Use in our design? |
|---|---|---|---|---|---|
| XSA | Apple | Mar 2026 | arxiv.org/abs/2603.09078 | remove self-value component from attn output | Later/no for frozen core (needs training; unproven at scale, unproven on hybrids). Only if we ever pretrain our own core |
| Gated Attention for LLMs (NeurIPS 2025 Best Paper) | Qwen/Alibaba | May 2025 (2505.06708) | arxiv.org/abs/2505.06708 ; github.com/qiuzh20/gated_attention | head-wise sigmoid gate after SDPA; adds nonlinearity+sparsity, kills attention sink, fewer loss spikes; tested 1.7B dense, 15B MoE, 3.5T tokens; used in Qwen3-Next | Use by choosing a base that already has it (Qwen3-Next family). Cannot bolt onto frozen core without training; small gate-only fine-tune possible but untested |
| Kimi Linear (KDA) | Moonshot | Oct 30 2025 (2510.26692) | arxiv.org/abs/2510.26692 | Kimi Delta Attention (finer-gated Gated DeltaNet) in 3:1 ratio with MLA; 48B total/3B active; claims -75% KV cache, up to 6x decode at 1M ctx; kernels+checkpoints open | Yes as candidate frozen core (open weights, hybrid, KV-light). 48B MoE is heavy; check hardware |
| Systematic Analysis of Hybrid Linear Attention | Dustin Wang, Zhu, Eshraghian et al. | Jul 2025 (2507.06457) | arxiv.org/abs/2507.06457 | 72 models (340M, 1.3B); LM quality flat across ratios, recall needs more full attention below ~3:1; best 3:1 to 6:1 (1/4 to 1/7 attention layers); standalone strength != hybrid strength | Yes: design guide for choosing/placing attention layers; informs which base. Small scale, placement details not checked |
| Hybrid Architectures for LMs: Systematic Analysis | FAIR Meta + KAIST | Oct 2025 (2510.04800), revised 2026 | arxiv.org/abs/2510.04800 | Attention+Mamba hybrids: inter-layer vs intra-layer (parallel) fusion; intra-layer on Pareto frontier; block ratio/order study; MoE compatible | Yes: guide for Transformer-Mamba core choice |
| DeepSeek-V3.2 (DSA) | DeepSeek | Dec 2025 (2512.02556) | arxiv.org/abs/2512.02556 | DeepSeek Sparse Attention: lightning indexer + top-k token selection, retrofitted onto V3.1-Terminus by continued pretraining (indexer warm-up 1000 steps x 16 x 128k = 2.1B tokens, then sparse adaptation) | Later: proves attention can be retrofitted cheaply, but needs a lot of compute; a retrieval-like sparse selector is conceptually close to our episodic memory |
| Native Sparse Attention | DeepSeek/PKU/UW | Feb 2025 (2502.11089) | arxiv.org/abs/2502.11089 | trainable hardware-aligned 3-branch sparse attention | No for frozen core (train from scratch); reference only |
| Tiny Recursive Model (TRM) | Samsung SAIL Montreal (Jolicoeur-Martineau) | Oct 2025 (2510.04871) | arxiv.org/abs/2510.04871 | 7M params, single 2-layer net recursing on latent+answer; 45% ARC-AGI-1, 8% ARC-AGI-2; simplifies HRM | Later/no: puzzle-specific, not a language core; maybe as a tiny planner/verifier module |
| Hierarchical Reasoning Model (HRM) | Sapient (G. Wang et al.) -- org unverified | Jun 2025 (2506.21734) -- ID-inferred, not fetched | arxiv.org/abs/2506.21734 (UNVERIFIED) | two recurrent modules at different timescales, deep supervision | Same as TRM; low priority |
| Mamba in the Llama | Princeton/Together/Cartesia etc (Wang, Paliotta, May, Rush, Dao) | Aug 2024 (2408.15237), NeurIPS 2024 -- just outside window | arxiv.org/abs/2408.15237 ; github.com/jxiw/MambaInLlama | distill a pretrained Transformer into hybrid keeping ~1/4 attention layers, reusing projection weights | Yes, high relevance: recipe to MAKE a hybrid core from an existing frozen-quality model without pretraining |
| Dream 7B / diffusion LMs (LLaDA 2502.09992 unfetched) | HKU et al. | Aug 2025 (2508.15487) | arxiv.org/abs/2508.15487 | AR checkpoint (Qwen2.5-7B) converted to masked-diffusion LM by fine-tuning | No for now (breaks Mamba-hybrid + LoRA/memory pipeline); watch item. Shows AR->other-paradigm conversion works |
| Jet-Nemotron / PostNAS | NVIDIA | ~Aug 2025 | UNVERIFIED (search returned nothing) | search best attention layer placement on a frozen pretrained model, replace others with linear attention | Would be very relevant (placement on frozen model); verify first |

Skipped: Differential Transformer (Oct 2024, outside window); MLA (DeepSeek-V2, 2024). Not verified: Dream's LLaDA numbers, HRM org.

## Takeaways for our design
1. For a frozen hybrid core, pick an existing open hybrid (Kimi Linear, Qwen3-Next with gated attention, or Nemotron/Falcon-H1 already known) rather than inventing; XSA and NSA need pretraining.
2. Hybrid ratio guidance converges on ~1/4 to 1/7 attention layers; recall suffers below ~3:1 linear:full, which matters because our episodic memory should cover long-term recall externally, but in-context recall still needs enough attention layers.
3. If a custom core is ever needed, distill via Mamba-in-the-Llama recipe, and optionally add gating (cheap, proven) and XSA (cheap, unproven).
