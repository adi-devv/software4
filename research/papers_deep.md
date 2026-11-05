# Deep read: six sleep / continual-learning papers

All IDs resolved with abstract page and HTML full text. Corrections to your list are flagged.

## Identity check
- 2606.03979 "Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories". Behrouz, Hashemi (Cornell), Javanmard, Mirrokni (Google). Submitted 2026-06-02, v2 10 Jul 2026, cs.LG. Says an earlier version has been on OpenReview since Sept 2025.
- 2605.26099 is NOT the Google paper. Current (v3) title: "Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference". Lee (CMU), McLeish, Goldstein (UMD), Fanti (CMU). Submitted 2026-05-25, v3 5 Jun 2026. Originally titled "Language Models Need Sleep"; renamed to avoid overlap with the Google paper.
- 2506.10943 SEAL, "Self-Adapting Language Models". Zweiger, Pari, Guo, Akyurek, Kim, Agrawal (MIT). v2 18 Sep 2025.
- 2510.15103 "Continual Learning via Sparse Memory Finetuning". Lin, Zettlemoyer, Ghosh, Yih, Markosyan, Berges, Oguz (FAIR at Meta, UC Berkeley). 2025-10-16.
- 2512.23675 "End-to-End Test-Time Training for Long Context". Tandon, Dalal, ... Yu Sun (Astera, Stanford, Berkeley, UCSD, possibly NVIDIA; affiliation list truncated, unverified). 2025-12-31.
- Nested Learning / HOPE = arXiv 2512.24695 "Nested Learning: The Illusion of Deep Learning Architectures". Behrouz, Razaviyayn, Zhong, Mirrokni (Google). NeurIPS 2025 version; arXiv 2025-12-31.

---
## 1. Google "Language Models Need Sleep" (2606.03979), top priority

**Problem.** Deployed LLMs are static. In-context learning vanishes when the context ends (anterograde amnesia analogy). Continual finetuning/LoRA causes catastrophic forgetting. Re-pretraining is too expensive.

**Method.** Two phases.
- Wake: online consolidation through the Continuum Memory System (CMS), a chain of MLP blocks updated at different frequencies. Fast blocks absorb new info and pass it to slower, more stable blocks.
- Sleep: model takes little or no external input and processes its own knowledge. Triggered at chunk-length boundaries per CMS level.

Stage 1, memory consolidation (NREM analogy):
1. Parameter expansion. MLP blocks are sparse MoE. A new low-rank expert (A, B) is added to the next, more stable block, so new knowledge does not overwrite old. Implementation pre-allocates experts and masks them until first activation.
2. Knowledge Seeding, a compute-consolidate-update protocol.
   - Compute: sender block's base-weight update from accumulated gradients is computed but not applied.
   - Consolidate: teacher = pre-update model including accumulated experts. Student = prospective updated weights, sender's experts reset, new expert added. ONLY the new low-rank params train; everything else is frozen (the anti-forgetting mechanism). Data = "dreams" sampled from the teacher.
   - Update: base update applied, sender's old low-rank experts reset ("synaptic pruning"), new expert activated.
3. Losses: (a) on-policy distillation, GKD-style divergence between teacher and student, mixing fraction lambda of student samples, no backprop through sampling; (b) Learning to Imitate (LTI), an RL loss: sample a prefix from a teacher dream, student continues, reward = semantic reward (frozen reward model, 0/1) + absolute reward (Levenshtein similarity vs threshold). A coefficient balances distillation vs LTI.

Stage 2, dreaming (REM analogy), built on SEAL with three fixes:
- Dreams generated from task context; each MoE router also picks a random expert to inject novelty.
- Dreams selected by gradient-based importance (top-k plus random for diversity).
- Each selected dream: isolated model copy gets LoRA-SFT; reward = downstream improvement; outer loop ReST (as SEAL).

**Architecture.** Hope (CMS + self-modifying module) on Llama-3B/8B for class-incremental; Transformer and "four-level" variants for knowledge incorporation; Qwen3-1.7B/8B for math; Llama-3.2-1B for ARC.

**Experiments (exact where in tables).**
Table 2, math, average@16 (AIME-24 / AIME-25 / HMMT-25):
- Qwen3-1.7B: Base 49.8/34.5/25.7; SFT 47.3/36.1/22.9; GRPO 51.0/38.6/26.1; OPSD 51.6/40.0/28.1; Sleep 53.2/40.2/29.3.
- Qwen3-8B: Base 73.8/68.1/42.4; SFT 75.5/66.4/43.7; GRPO 76.4/68.1/44.9; OPSD 76.6/67.4/45.1; Sleep 79.2/69.0/46.1.

Table 1 ablation (Qwen3-8B): no imitation learning 76.8/67.9/45.0; no semantic reward 78.9/69.2/44.5; no expansion 78.2/67.9/44.9; OPSD+Expansion 77.9/68.2/45.9.

Table 3, SQuAD knowledge incorporation, no-context acc (single passage n=1 / CPT n=200): base 31.9/31.9; finetune w/o dreaming 33.4/32.0; SEAL 46.7/43.2; Sleep Transformer 48.1/44.3; Sleep four-level 48.9/46.2; w/o gradient selection 47.1/45.2; w/o random expert 48.0/44.7; w/o dreaming 35.7/36.2.

Table 4, few-shot ARC success %: ICL 0, TTT 10, SEAL 72.5, Sleep 80 (Llama-3.2-1B, 11 train / 8 held-out tasks).

Table 5 config: LoRA rank 64, alpha 128, batch 32, 500 steps GRPO, 100 steps SFT and Sleep.
B.5 efficiency: SFT 4x cheaper per step but needs 4.3x / 3.6x / 4.8x wall-clock to match Sleep on AIME-24/25/HMMT-25.

Figure-only results (no exact numbers in text, unverified values): class-incremental CLINC/Banking/DBpedia (Hope beats ICL, EWC, InCA; Fig 3); long context MK-NIAH/LongHealth/QASPER (beats ICL, DuoAttention, Cartridges; more consolidation stages helps; Fig 4); CTNL on Kalamang/Manchu translation (Hope-1/2/3 improve monotonically, SFT and Cartridges forget; Figs 5-6); BABILong "almost perfect" to 10M tokens vs GPT-4, GPT-4o-mini, Llama-8B+RAG, RMT, ARMT, Titans.

**Stated limitations.** No dedicated limitations section in the HTML (verified absent). Motivating caveats: SEAL-style loops forget and inner-loop SFT is costly; related work says naive iterated self-distillation can leak and collapse. My inference (unverified as authors' claims): only modest scale, headline architecture is Hope/MoE not a plain frozen LLM, and key class-incremental results are figure-only.

**Figures worth recreating.** Fig 1 wake/sleep timeline with fast-to-slow memory flow; Fig 2 parameter expansion then knowledge seeding (teacher/student/frozen vs trained); a compact table of Table 2 and Table 3; Fig 5/6 CTNL forgetting curves.

**Mapping to your design.**
- Wake = log episodes to vector DB; sleep = your offline loop. Direct match.
- "Only the new low-rank expert trains, all else frozen" = your frozen core + adapter. Their expert is per-level and reset after merge; yours maps to one adapter generation per sleep cycle.
- Dreams (teacher samples) = replay/synthetic data; gradient-based dream selection is a cheap surprise-like filter alternative to embedding prediction error. Random-expert novelty injection has no direct analogue; consider temperature or perturbed retrieval.
- Teacher (pre-update) vs student distillation = old-data replay anchor; use teacher outputs as targets to preserve behaviour.
- Reward from downstream gain (ReST) = your judge-verify plus metric gate.
- Gap: they merge by in-model update schedules, not a metric-gated promotion. The gate is your addition. Ablation shows each piece (imitation, expansion, selection) is worth about 0.5-3 points, so expect modest gains.

---
## 2. CMU/UMD "Do LMs Need Sleep?" (2605.26099)
**Problem.** SSM-attention hybrids store evicted context in fast weights but cannot reason deeply over it. Accuracy falls with reasoning depth even at fixed capacity: bottleneck is compute, not capacity.

**Method.** When the attention window of length L fills, model "sleeps": N recurrent passes (looped blocks) over the accumulated context update the SSM fast weights via the learned outer-product rule (Mamba2/GDN style, Eq 3). KV cache then cleared (hard eviction) or partly kept (sliding). Prediction afterwards is one forward pass. Trained end-to-end by backprop through the whole process with masked cross-entropy (Algorithm 1). N=1 is a vanilla hybrid.

**Models.** 4-layer GDN-attention hybrid (Rule 110 automaton); 10-layer model from scratch for Depo; pretrained Jet-Nemotron 2B (loop middle 14 of 28 blocks) and Ouro 1.4B (+6 Jet layers, under 10% extra params) for GSM-Infinite. Muon optimizer; 1 A6000 day (automaton), 1-2 H100 days per run (others).

**Results.**
- Rule 110 (t=32): no loop about 10% exact after about 5B tokens; 2 loops about 20%; 3-4 loops above 30%.
- Depo (75-node cycles, 300 tokens, T=360, L=75): more loops help at 4+ hops; only 4-loop starts to improve at 16 hops.
- GSM-Infinite (2k-3.3k tokens, no CoT, 1600 held-out): Jet 6 loops 6-op 0.742 to 0.812, 8-op 0.351 to 0.388; Ouro 4 loops 6-op 0.419 to 0.615, 8-op 0.210 to 0.272. Sliding-window eviction with Ouro 2-op 0.596 to 0.905.
- Training throughput roughly inversely proportional to N.

**Limitations (their Sec 7).** Slower, can be unstable; sequential across windows and depth (they suggest implicit gradients, truncated BPTT); synthetic tasks, modest scale.

**Meaning for you.** Not lifelong learning: converts context to fast weights, no persistent adapter. Takeaway only: spending extra offline compute to re-process memory improves later reasoning, supporting a "reason" step in your sleep loop. Do not conflate with the Google paper.

---
## 3. SEAL (2506.10943)
- Model writes a "self-edit" (synthetic data, optionally hyperparameters/tool calls); LoRA SFT applied; reward = post-update downstream performance; outer loop ReST^EM (rejection sampling + SFT).
- Knowledge incorporation, Qwen2.5-7B, SQuAD, Table 2 (n=1 / n=200 / n=2067): Base 32.7/32.7/29.0; train on passage 33.5/36.0/31.2; +synthetic 39.7/50.6/43.4; +GPT-4.1 synthetic 46.3/59.4/49.2; SEAL 47.0/58.2/46.4. Two ReST rounds (50 contexts x 5 self-edits) beat GPT-4.1 data in single-passage.
- Few-shot ARC, Llama-3.2-1B-Instruct: ICL 0%, TTT with self-edit no RL 20%, SEAL 72.5%, oracle TTT 100% (11 train / 8 eval tasks).
- Stated limits: catastrophic forgetting under sequential edits (Fig 6); about 30-45 s per self-edit evaluation; every context needs a paired task.
- Design meaning: the template for your reason + judge-verify + train-adapter step, with reward from measured gain. Its forgetting is exactly why you need old-data replay and the metric gate.

## 4. Sparse Memory Finetuning (2510.15103)
- Memory-layer model (Berges et al.): 1M slots, k=32 accesses per head, 4 heads, value dim 1024, at layer 12 of 22 in a 1.3B model.
- Per batch, count slot accesses and rank against a background corpus (1000 DCLM batches) by TF-IDF; only top-t slots get gradient (t=500 enough for fact learning). SGD best for this; baselines used AdamW wd 0.1.
- Headline: NaturalQuestions F1 drops 89% (full finetune), 71% (LoRA), only 11% (sparse), at equal new-knowledge gain.
- Tasks: 1000 TriviaQA facts in sequence (held-out NQ, HellaSwag, GSM8K); document QA on 100 SimpleQA questions with Active Reading data.
- Limits: facts only; RAG is a natural alternative; needs larger models and more tasks; naive memory finetuning still forgets.
- Meaning: strongest evidence for a sparse-memory adapter over LoRA for forgetting. Needs a memory-layer base model, so it is not a drop-in on a stock frozen LLM. TF-IDF slot gating is a cheap "what to update" rule that fits surprise-gated training.

## 5. TTT-E2E (2512.23675)
- Standard Transformer with sliding-window attention (k=8K) keeps learning at test time by next-token prediction on context, writing into MLPs of the last 1/4 of blocks; mini-batch TTT b=1K. Each updated block has an extra static MLP preserving pretrained knowledge. Trained by meta-learning (gradients of gradients). TTT-naive (no E2E) is far worse.
- Headline: 3B models on 164B tokens, scales with context like full attention while Mamba 2 and Gated DeltaNet do not; prefill 2.7x faster than full attention at 128K, constant latency. TTT-E2E is the only method with lower loss than full attention across the whole context (Fig 6).
- Data: DCLM at 8K, Books extension to 128K; 125M to 3B.
- Limits: training 3.4x slower than full attention at 8K (1.2x faster at 128K); Needle-in-a-haystack (Table 2) is dramatically worse than full attention because compression drops detail.
- Meaning: weights-as-memory works, and the "static safe MLP next to the updating one" mirrors frozen core + adapter. Poor exact recall is the argument for keeping the vector DB for episodic facts.

## 6. Nested Learning / HOPE (2512.24695)
- Verified from the full text: recasts models and optimizers as nested optimization problems with different update frequencies; Adam/momentum are associative memories compressing gradients; Continuum Memory System = chain of MLP blocks, block l updated every C(l) steps; Hope = self-modifying sequence module + CMS.
- Verified numbers (NIAH, Table 1, 4K/8K/16K): S-NIAH-1 Hope 100/100/100 vs Transformer 88.6/76.4/79.8; S-NIAH-2 Hope 99.2/88.4/78.2 vs Transformer 100/98.8/94.2; S-NIAH-3 Hope 73.2/46.2/24.8 vs Transformer 78.0/69.2/40.8. Hope beats Titans and RWKV-7 but not full attention on harder recall. Other tables (language modelling, reasoning) not read; unverified.
- Meaning: supplies the multi-frequency principle for your design: fast adapter, slow adapter, frozen core, with consolidation moving knowledge slow-ward. Hope itself is a new architecture, not retrofittable to a frozen LLM.

---
## Overall for your design
1. Best template: Google Sleep (consolidation + dreaming) plus SEAL's reward-from-gain loop.
2. Anti-forgetting: freeze everything but a new low-rank piece, distil from the pre-update teacher, and replay old data; consider sparse-slot updates (Sparse Memory Finetuning) if you can use a memory-layer model.
3. Keep the vector DB for exact recall (TTT-E2E, Hope NIAH both show weights lose detail).
4. Metric-gated merge is not provided by any paper; treat as your own contribution and validate it.
5. Unverified: Google paper limitations, exact class-incremental numbers, Nested Learning non-NIAH tables, TTT-E2E full affiliations.
