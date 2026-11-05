# Red-team critique: "software4" brain-inspired continual learning plan

Reviewed 2026-11-05. Verification tags used throughout:
- **[V]** checked this session (arXiv abstract/HTML or search hit on the primary record)
- **[P]** from the provided deep-read files (papers_deep.md / papers_xsa.md / local paper text)
- **[UNVERIFIED]** ID or claim from my own memory, or only from a secondary source; not re-checked
- **[E]** my own estimate / arithmetic, not from a paper

---

## 0. Verdict in one paragraph

The plan is a stack of borrowed mechanisms, each validated in isolation on small, curated benchmarks that **never compare against simply putting the information in context**. The single most important number for this project is not in any paper you cite: how much of the gain does "core + good memory + text-level reflection" already give, at ~0 training cost? 2026 evidence (CL-Bench) says even the best current systems capture only ~25% of learnable headroom, and that **naive full-context ICL beats dedicated memory systems**. That means the bottleneck is *using* experience, not *storing* it — and weight consolidation is the most expensive, least safe, least portable way to attack it. Several components are internally inconsistent (a "frozen" core that gets merged into; an expert per night with no budget; pruning the only exact copy of a memory after a lossy consolidation; a surprise filter that preferentially promotes poison and hallucination-inducing facts). The hybrid core and JEPA are distractions from the research question. Keep the sleep loop idea, but make it write **text artifacts first** and treat weights as a rebuildable cache of the memory log, not the place knowledge lives.

---

## 1. Overall framing ("complementary learning systems", "keeps learning without going dumb/rogue")

**Strongest objection.** The brain analogy is decoration, not mechanism, and in one place it is inverted. In CLS theory (McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis & McClelland 2016, *TICS*) [UNVERIFIED, classic non-arXiv refs] the transferable computational content is exactly one thing: *a slow learner must be trained with interleaved replay so new items don't overwrite old ones*. In CLS the neocortex **does learn** — slowly. Freezing the "cortex" and making consolidation-into-cortex a rare, scary event (the merge gate) inverts the theory. The rest of the mapping (cortex = frozen LLM, hippocampus = FAISS, sleep = nightly cron, REM = SEAL) is labeling. Also, the modern consolidation literature (multiple-trace / trace-transformation theory, Nadel & Moscovitch 1997 [UNVERIFIED]) says the hippocampus stays necessary for detailed episodic recall — which argues *against* your "prune memories" step.

Second objection: the goal "doesn't go dumb or rogue" has no operational definition. There is no stated metric, threshold, or benchmark for either "learned" or "dumb/rogue". Without that the project cannot fail, which means it cannot succeed either.

**Evidence.**
- CL-Bench (2606.05661, Asawa, Zaharia, Gonzalez et al., Jun 2026) [V]: normalized gain (share of learnable headroom captured): ICL + Claude Sonnet 4.6 = **25.4%** (best); GPT-5.4 ICL 20.1% vs Mem0 20.2% (tie); ACE 8.6% at 3.4x the cost ($62.8 vs $18.4/run); "ICL Notepad" *worse* than plain ICL. Conclusion: "naive ICL outperforms systems dedicated to memory management"; memory modules introduce "spurious generalizations and stale beliefs". Parametric methods were **not evaluated** — an open gap you could fill, but it also means nobody has shown weights beat ICL there.
- None of your three core papers (2606.03979 Sleep, 2506.10943 SEAL, 2510.15103 SMF) report a passage-in-context or RAG baseline on their knowledge tasks [P, grepped local text]. SEAL explicitly evaluates "no-context" SQuAD and claims the n=200 CPT regime is where "placing data directly in context is no longer feasible" — but 200 SQuAD passages are roughly 30–40k tokens [E], which fits in any 128k context, and 2,067 passages is a trivial RAG index.

**Better alternative.** Frame the project as: *"Can offline consolidation into weights beat the best non-parametric learner (long context + structured memory + reflected text lessons) at equal compute, on held-out tasks, without capability/safety regression?"* That is falsifiable. Use the brain only as a hypothesis generator.

**Severity: major.** (Not fatal, because the fix is a reframing, but every downstream choice inherits the missing objective.)

---

## 2. Cortex: frozen open-weight hybrid Transformer–Mamba core

### 2a. Why freeze?
Freezing is actually the most defensible choice in the plan — it is the only real anti-forgetting guarantee, and it gives you a fixed teacher/anchor for distillation. The objection is not "freezing caps capability" (it does — LoRA learns less, full-FT perturbations are 10–100x higher rank, 2405.09673 [V] — but that's an acceptable trade). The real objection is **strategic**:

**Base-model churn strands everything learned in weights.** Open-weight frontier bases turn over every ~3–6 months [E]. Every LoRA expert, router, and merged delta is tied to one base checkpoint and does not transfer to Qwen-next-version. Text memories, skill files, and lesson libraries port to any model for free. On a one-person/small-team budget, the next base release will usually beat a year of your consolidation on general capability, and you'll have to throw the experts away.

**Better alternative.** Make the **memory log the source of truth and weights a cache**: everything consolidated into weights must be re-derivable by replaying the log onto a new base. Then "freeze" is fine, and "merge" is replaced by "rebuild on the next base".

### 2b. Why a hybrid at all?
**Strongest objection.** The hybrid choice optimizes long-context throughput, which is not the bottleneck for this goal — and it actively hurts the read path your design depends on.
- If memory is retrieved (RAG), prompts are short; the hybrid's KV savings (Kimi Linear "-75% KV, up to 6x decode at 1M" [P]) buy little.
- SSM/linear layers are provably and empirically worse at copying/retrieving from context (2402.01032 "Repeat After Me" [V]: Pythia Transformers beat Mamba on copying/context retrieval despite Mamba's lower perplexity). Your system's main loop is "retrieve text → read it faithfully". The hybrid analysis you cite says recall degrades below ~3:1 linear:full (2507.06457 [P]).
- **All of the continual-learning evidence you rely on used dense Transformers or bespoke architectures**: Sleep paper = Hope/CMS on Llama, Qwen3-1.7B/8B dense, Llama-3.2-1B; SEAL = Qwen2.5-7B, Llama-3.2-1B; SMF = a 1.3B memory-layer model [P]. Zero evidence that LoRA-MoE expansion / SEAL loops behave the same when adapters sit on Mamba-2 / Gated-DeltaNet projections.
- Tooling: I found nothing confirming vLLM multi-LoRA serving on Nemotron-H's Mamba-2 layers; the vLLM GDN module special-cases LoRA for Qwen3.5 [V, docs snippet only]. PEFT support for SSM blocks is thin [UNVERIFIED]. Expect weeks of plumbing that has nothing to do with the research question.
- Most named candidates are large MoEs (Qwen3-Next 80B-A3B, Kimi Linear 48B-A3B [P]). LoRA training needs all weights resident (~160 GB bf16 for 80B) [E] — 3B-active doesn't save memory during the inner-loop training SEAL requires per candidate.

**Better alternative.** Research on a dense Qwen3-4B/8B (what the evidence used; cheap; mature PEFT/vLLM multi-LoRA). Choose the deployment core last, after the loop works. If you later need a hybrid, Granite 4 / Falcon-H1 small sizes exist, or convert via Mamba-in-the-Llama (2408.15237 [P]).

**Severity: major** (distraction + cost multiplier + confound: a failure won't tell you whether the loop or the hybrid broke it).

---

## 3. Hippocampus: vector DB, instant writes, RAG

**Strongest objection.** Flat chunk-embedding RAG is the weakest memory design on every 2024–2026 memory benchmark, and "instant write" is an open poisoning door. The plan treats the hippocampus as solved infrastructure; it is actually where most of the measurable gain (and most of the risk) lives.

**Evidence.**
- LongMemEval (2410.10813, ICLR 2025) [V]: ~30% accuracy drop for assistants on sustained-interaction memory; failure modes are multi-session reasoning, temporal reasoning, knowledge updates — exactly what naive vector top-k misses. Round-level granularity beats session-level.
- MemoryAgentBench (2507.05257) [V]: four competencies (accurate retrieval, test-time learning, long-range understanding, conflict resolution); no method masters all; RAG agents fail conflict resolution/updates.
- HippoRAG 2 (2502.14802, ICML 2025) [V]: a *hippocampus-inspired* non-parametric continual learner (KG + Personalized PageRank) gains ~7 points on associative tasks over standard RAG with no factual-memory loss. Note: this is a more faithful CLS implementation than yours, and it has no weight updates.
- Mem0 (2504.19413) [V for abstract]: 91% lower p95 latency, >90% token savings vs full context; but a secondary table shows full-context **more accurate** on LOCOMO (72.9 vs 66.9 judge score) [UNVERIFIED numbers]. Memory systems trade accuracy for cost; they are not free accuracy.
- Poisoning: MINJA (2503.03704, NeurIPS 2025) [V]: query-only attacker injects malicious memory records with 98.2% success and 76.8% end attack success. PoisonedRAG (2402.07867, USENIX Sec 2025) [V]: ~90% ASR with 5 injected texts into a DB of millions; paraphrase and perplexity defenses insufficient. AgentPoison (2407.12784) [UNVERIFIED ID].
- "Prune memories" after consolidation: weights lose exact detail (TTT-E2E NIAH much worse than full attention [P]; Hope loses to full attention on hard NIAH [P]; parametric RAG shows "substantial performance gaps" vs token RAG, best is combining both — 2510.12668 [V]). Pruning destroys the only exact copy, your provenance trail, and your ability to rebuild onto a new base.

**Better alternative.** Structured, typed, time-stamped memory (episodes, entities, facts with supersession/validity intervals, skills/lessons as separate stores), retrieval with temporal and entity filters, a graph layer (HippoRAG-2 style) for association, and **trust/provenance on every write** (source, user vs tool vs web, confidence). Never hard-delete: demote to cold storage. Compare explicitly against MemGPT/Letta (2310.08560 [V via secondary]), Mem0, A-MEM (2502.12110 [V]), and plain long context.

**Severity: major.**

---

## 4. Learnable layer: a new low-rank expert per sleep cycle; A/B vs Sparse Memory Finetuning

**Strongest objection.** You copied the "add an expert" half of the Sleep paper and dropped the half that keeps it bounded. In 2606.03979 the new expert goes into the next, slower CMS block; then the sender's base update is **applied** and the sender's old experts are **reset** ("synaptic pruning") [P]. Their design does base updates on purpose, in an architecture built for it. Yours keeps the core frozen and never resets, so experts grow monotonically.

**Growth arithmetic [E].** A rank-64 LoRA on all linear layers of a Llama-3-8B-shaped model is ~168M params (≈5.2M/layer × 32). One per night → 365/year ≈ **61B adapter params, 7.7x the base**. Even with top-k routing (compute bounded), memory, checkpointing, eval, and routing difficulty all grow without bound.

**Routing drift.** The router is a shared, trained component — i.e. a non-frozen part of the "frozen" system. Continual MoE-LoRA work reports that as experts accumulate, routers fail to index the pool, causing "routing collapse and drift of old experts" and accelerated forgetting; router–expert co-drift is a named failure (2601.13020 PASs-MoE [V via search snippet]; 2506.21035 MoRA/MoRAM, ICML 2026 [V via search]). Old experts can be *perfectly preserved and still forgotten* because the router no longer sends traffic to them.

**Per-cycle trigger is wrong.** Most nights contain little worth consolidating; some contain a lot. "One expert per night" ties capacity to the calendar instead of to content.

**The SMF A/B is not apples-to-apples.** SMF needs a memory-layer base model (only research checkpoints at ~1.3B in the paper [P]); your LoRA-MoE arm sits on a hybrid MoE. Different base, different data scale — a "win" either way is uninterpretable. SMF is also facts-only [P], and facts are precisely what RAG already does well.

**Better alternative.** Fixed expert budget (e.g. 8–16 slots); new learning goes into a scratch adapter; consolidation is *expert-to-expert distillation* within the budget (core stays truly frozen); experts are keyed to skill clusters, not nights. Or rank-1 / slot-level sparse experts (MoRAM, SMF-style TF-IDF gating) if you're willing to change base. Run the SMF comparison only on SMF's own base, as a separate study.

**Severity: major.**

---

## 5. Sleep loop (replay → surprise filter → SEAL dreams → keep-if-helps → judge/tests → distill + replay → prune)

### 5a. Offline vs online
Offline batch consolidation is defensible (batching compute, gating, rollback). But note the consequence: anything learned today is only in weights tomorrow, so memory/context must carry the system in the meantime anyway — which again makes weights matter only for what memory cannot do. Online alternatives (Titans 2501.00663 [V], TTT-E2E 2512.23675 [P], Hope [P]) require custom architectures and can't be bolted onto a frozen open base. **Severity: minor.** Do make the trigger data-driven (value/volume), not nightly.

### 5b. Surprise filter (embedding prediction error, "JEPA-like")
**Strongest objection.** Surprise ≠ value, and it is adversarially aligned with your worst inputs.
- *Noisy-TV problem*: prediction-error novelty is maximized by unpredictable noise (Burda et al., 1808.04355 [UNVERIFIED ID]).
- *Poison amplifier*: injected/malicious/off-distribution memories are surprising by construction, so the filter preferentially promotes them from the DB (reversible) into weights (much less reversible). Combined with MINJA/PoisonedRAG rates above, this is the single most dangerous interaction in the plan.
- *Hallucination amplifier*: Gekhman et al. (2405.05904, EMNLP 2024) [V]: fine-tuning on facts the model doesn't know (= high surprise) is learned slowly and, once learned, **linearly increases hallucination**; filtering out unknown examples reduces risk "without hurting performance". Your filter selects exactly the "unknown" class.
- *Wrong tool*: "JEPA-like embedding prediction" requires a separately trained predictor that itself must be continually trained and will drift. The frozen LM's own token loss already *is* a calibrated surprise signal, for free.
- Even the paper you're copying uses **gradient-based importance + random sampling** for dream selection, not embedding surprise [P]; SMF uses **TF-IDF vs background** [P]; the Sleep paper notes consolidation "strengthens the parts of a memory that are repeatedly recalled" [P, local text].

**Better alternative.** Select by *utility and verification*: (1) items the agent needed and retrieved repeatedly, (2) items tied to an observed failure that was later corrected with ground truth, (3) procedural traces that passed a verifier, (4) trusted provenance only. Use LM loss only as a tie-breaker, and *exclude* unverified high-surprise facts from weight training (leave them in memory). **Severity: major.**

### 5c. SEAL-style self-edits + "keep only if it helps" + Claude judge
**Strongest objection: the reward is self-referential in deployment.** In SEAL the "downstream score" is dataset-provided held-out QA for each passage [P]. After deployment there is no paired task. If the model writes the passage, writes the self-edit, writes the test questions, and a judge grades them, you are measuring self-consistency, not correctness — a textbook self-confirmation loop. SEAL itself lists "every context needs a paired task" as a limitation [P].
- Judges: LLM evaluators recognize and favor their own generations, linearly with self-recognition ability (2404.13076) [V]. Single content-free "master key" strings ("Thought process:") fool generative judges/reward models at high false-positive rates (2507.08794) [V; 80–90% FPR figure UNVERIFIED]. A Claude judge (different family) reduces self-preference but any fixed judge inside an optimization loop will be Goodharted, and it's an external, versioned, costly dependency.
- Self-distillation loops fail: SDPO "exhibits greater forgetting and can collapse… on-policy data alone is insufficient for continual learning", failures driven by "high-frequency artifacts amplified by a self-reinforcing teacher-student loop" (2607.01763, Jul/Oct 2026) [V]. On-Policy Replay (2605.29495) [V] works — but its gain comes from the **high-reward filter** (low-reward on-policy samples are worse than vanilla replay), i.e. it needs a real task reward.
- Subliminal learning (2507.14805) [V]: a teacher transmits traits (including misalignment) to a student via semantically unrelated data when they share initialization; filtering and LLM classifiers fail to detect it. "Distil the new expert from the pre-update model on its own dreams" is precisely the shared-init setting. Drift can be invisible to the judge. (Mitigation work exists, 2609.22215 [V via search, not read].)
- Sequential SEAL edits forget (SEAL Fig. 6) [P]; model editing at scale shows gradual-then-catastrophic collapse (2401.07453, ACL Findings 2024) [V; note: the commonly cited ID 2401.04700 is wrong].
- Even benign fine-tuning erodes safety alignment; 10 adversarial examples suffice to jailbreak (2310.03693, ICLR 2024) [V]. Narrow fine-tuning can produce broad misalignment (2502.17424, ICML 2025) [V].

**What the Sleep paper's own numbers say.** Gains over the strongest baseline (OPSD) on Qwen3-8B math: +2.6 / +1.6 / +1.0 (AIME-24/25/HMMT) [P]; ablations are each worth ~0.5–3 points [P]. Those are post-training gains on a fixed task, not deployment continual learning, at n≈30-problem benchmarks where 1 point ≈ noise [E]. ARC result is 8 held-out tasks [P] — 80% vs 72.5% is a difference of one task (8 × 0.075 = 0.6) [E].

**Better alternative.** Only consolidate with **grounded rewards**: unit tests, execution, environment outcomes, explicit user corrections, cross-checked sources. Keep a frozen, never-trained-on held-out set and canaries; use judge *ensembles* from different families only for tie-breaking; log judge disagreements. Replace "SEAL inner-loop per candidate" with batched candidate generation + cheap proxies (loss reduction on held-out paraphrases) — the per-candidate LoRA train-and-eval is the cost killer (below).

**Severity: fatal** in domains without verifiable outcomes (open-ended chat, preferences, world knowledge from the web); **major** where tests exist (code, math, tool use).

### 5d. Compute per sleep cycle [E unless noted]
- Measured anchor: SEAL appendix — one ReST round = 50 passages × 5 completions × 3 runs = 750 inner loops ≈ **6 h on H100s** for Qwen2.5-7B; each TTT per ARC problem "half a minute to a few minutes" [P, local text]. ≈ 29 s per inner loop at 7B.
- 1,000 candidate memories/night × 5 self-edits × 3 seeds × ~29 s ≈ **120 H100-hours/night at 7B dense**.
- On Qwen3-Next-80B-A3B: ≥2–4 H100s per inner-loop worker just to hold weights, slower MoE backward → plausibly **1,000–2,500 H100-hours/night**, i.e. roughly **$2k–7k per night** at $2–3/H100-hr. Plus nightly regression suite (capability + safety + memory benchmarks) ~5–20 GPU-hours.
- RAG/memory write cost for the same day: embeddings + index insert, cents.
- So the loop must deliver gains that RAG + reflection cannot, at 3–5 orders of magnitude more compute. That's the bar.

### 5e. "Distill from pre-update model plus old-data replay"
Good instinct (consistent with RL's Razor 2509.04259 [V]: forgetting tracks forward KL from the base; on-policy methods stay closer; SDFT 2601.19897, ICML 2026 [V]: demo-conditioned self-teacher reduces forgetting at ~2.5x FLOPs / 4x wall-clock [secondary, UNVERIFIED]; Thinking Machines on-policy distillation recovered IFEval 79→83 after mid-training on internal docs [secondary, UNVERIFIED]). But "old-data replay" of *what*? You don't have the base model's post-training data. Replaying your own past outputs is self-distillation again (5c risks). Use a fixed, curated public replay mix (instruction + safety + reasoning) plus KL-to-base, and measure.

---

## 6. Merge gate: "rarely fold experts into the core"

**Strongest objection: it contradicts the design.** The core is frozen *because* it's the anti-forgetting anchor and the distillation teacher. Folding experts in means:
1. The anchor moves; KL-to-base now measures distance from an already-drifted model; drift compounds across merges (gradual→catastrophic pattern, 2401.07453 [V]).
2. Every unmerged expert and the router were trained against the old core and are now off-distribution (LoRA deltas don't transfer cleanly across base changes; Trans-LoRA 2405.17258 exists precisely because of this [UNVERIFIED ID]).
3. Summing many low-rank deltas interferes (TIES-merging 2306.01708 [UNVERIFIED ID]); sequential merges erode safety alignment (2310.03693 [V]).
4. "Full test suite + rollback" doesn't detect subliminal/latent trait transfer (2507.14805 [V]) or backdoors that only fire on triggers (2502.17424 backdoor variant [V]).
5. The triggers ("expert saturation", "memory backlog") are capacity triggers — they say *when you ran out of room*, not *that the merge is safe*.

The Sleep paper's analogue (apply base update, reset sender experts) happens inside CMS blocks designed to be updated at fixed frequencies [P]; it is not evidence that merging into a frozen pretrained hybrid is safe.

**Better alternative.** Never merge. Hold a fixed expert budget and consolidate experts into experts. Treat **base-model swaps as the merge**: periodically rebuild — replay the memory log + skill library onto the newest open base, re-derive the few experts that still earn their keep, run the full suite, A/B against the old stack. This is also how you capture base-model progress.

**Severity: major** (fatal if merges become frequent).

---

## 7. Later: JEPA world model (V-JEPA 2)

**Strongest objection.** Scope creep with no integration path. V-JEPA 2 (2506.09985) [V] is a ~1B video encoder + action-conditioned predictor trained on 1M+ hours of video and 62 h of robot data, demonstrated on Franka pick-and-place. It does not plug into a text LLM's continual-learning loop, and there's no task in the plan that needs video world modeling. Separately, using "JEPA-like" for the text surprise filter conflates a representation-learning objective with an anomaly score (see 5b).

**Better alternative.** Delete it from the plan until there's an embodied/video task. If you want "prediction of how the world changes" for an agent, the cheap version is outcome prediction in text (predict tool/environment result, log prediction error, use that as a grounded learning signal).

**Severity: minor** (it's labeled "later"; the cost is attention, not compute).

---

## 8. Cross-cutting: is "learning in weights" needed at all?

Ladder of alternatives, cheapest first, all of which accumulate learning after deployment without touching weights:
1. **Long context** (just keep it) — CL-Bench winner [V]; Cartridges (2506.06266) [V] show offline-trained KV "self-study" matches ICL at 38.6x less memory / 26.4x throughput [secondary numbers, UNVERIFIED] — a "sleep" that compresses context without touching core weights, composable at inference.
2. **Structured memory** — MemGPT/Letta, Mem0, A-MEM, HippoRAG 2 (§3).
3. **Reflected lessons / skill libraries** — Voyager skill library (2305.16291 [UNVERIFIED ID]); ACE evolving playbooks (2510.04618, ICLR 2026) [V]: +10.6% agent / +8.6% domain benchmarks, beats GPT-4.1-based top AppWorld agent with open DeepSeek-V3.1, warns of "context collapse" from monolithic rewrites; ReasoningBank (2509.25140, ICLR 2026) [V]: memory of distilled strategies from successes *and* failures beats raw-trajectory memory; GEPA (2507.19457, ICLR 2026 oral) [V]: reflective prompt evolution beats GRPO by ~6% avg with up to 35x fewer rollouts.
4. **Sleep-time compute** (2504.13171, Letta/Berkeley) [V]: idle-time preprocessing of context cuts test-time compute ~5x on stateful GSM-Symbolic/AIME. This *is* "sleep", in text.
5. **Weights** — justified only for (a) procedural skills that need many demonstrations and don't fit or transfer via text (ARC-style TTT is the cleanest evidence [P]), (b) amortizing a huge static corpus (Cartridges/EntiGraph; EntiGraph 2409.07431 [V] needed **455M synthetic tokens from 1.3M real** to recover most of the in-context gain — knowledge-in-weights is extremely data-hungry; Physics of LMs 3.1, 2309.14316 [V]: without augmentation, knowledge can be memorized yet 0% extractable), (c) latency/cost.

**Position:** weights are needed for a narrow slice (procedural skill, amortization). For declarative facts, RAG beats fine-tuning (2312.05934 [V]) and parametric injection loses detail (2510.12668 [V]); RING (2608.01630, Aug 2026) [V] claims an internal-memory model "matches or surpasses" RAG, but via continued pretraining + SFT + RL of a dedicated knowledge expert — not a nightly LoRA. **Facts should never be the target of consolidation.**

---

## 9. Evaluation: how to actually measure gain and forgetting

The plan has no evaluation spec. Minimum viable spec:

**Protocols**
- Streamed "days" with held-out test items per day, never used for selection, judging, or early stopping.
- Report every learning metric in **two conditions: with retrieval and without**. Only the *with-retrieval* delta over the RAG/ICL baseline counts as value added by weights.
- Standard CL metrics: average accuracy, backward transfer (BWT), forward transfer (GEM, 1706.08840 [UNVERIFIED ID]); CL-Bench-style normalized gain [V]; retention curve over cycles (day-1 items re-tested on day 30).
- Compute-normalized: gain per GPU-hour vs baseline.
- Distinguish true vs "spurious" forgetting (2501.13453, ICLR 2025) [V]: drops can be task-alignment loss, not knowledge loss — probe both.

**Benchmarks**
- Memory/continual: LongMemEval (2410.10813) [V], MemoryAgentBench (2507.05257) [V], CL-Bench (2606.05661) [V], LoCoMo (2402.17753 [UNVERIFIED ID]).
- Fresh knowledge (no pretraining leakage): RING's News-2025 (2608.01630) [V], HybridRAG-Bench (2602.10210) [V via search, not read], TemporalWiki (2204.14211 [UNVERIFIED ID]).
- Continual post-training: TRACE (2310.06762 [UNVERIFIED ID]) as used by OPR (2605.29495) [V].
- Retained capability: MMLU-Pro, GSM8K/MATH, IFEval, HumanEval/LiveCodeBench (subsets fine).
- Hallucination/calibration: SimpleQA-style with abstention (critical given Gekhman 2405.05904 [V]).
- Safety: HarmBench (2402.04249 [UNVERIFIED ID]) / StrongREJECT refusal, XSTest over-refusal, plus a fixed misalignment probe set (2502.17424 eval questions [V]).
- Red team: MINJA-style memory injection (2503.03704) [V] and PoisonedRAG (2402.07867) [V] attacks run against the full loop — measure whether poison reaches weights.

---

## 10. Safety (drift, poisoning)

Summary of the threat chain unique to this plan: **untrusted instant write → surprise filter promotes it → self-generated "implications" amplify it → self-judge approves it → distilled into expert (possibly subliminally) → merge folds it into the core → memory pruned so provenance is gone.** Every arrow is supported by a cited result above (MINJA/PoisonedRAG; noisy-TV/Gekhman; 2404.13076/2507.08794; 2507.14805; 2401.07453/2310.03693). Fixes: provenance + trust tiers on writes; only trusted, verified items are eligible for weights; no pruning; per-cycle safety + canary gate with automatic rollback; adapters isolated per source so a poisoned source can be excised; periodic rebuild from the log excluding quarantined sources.

---

## 11. Severity table

| # | Part | Strongest objection | Severity |
|---|---|---|---|
| 0 | Framing | No falsifiable objective; analogy inverts CLS (cortex should learn); evidence never compares to in-context | major |
| 1 | Frozen hybrid core | Hybrid optimizes the wrong bottleneck, hurts context copying, no CL evidence on hybrids, huge MoE training memory; weights stranded on base swaps | major |
| 2 | Vector DB | Weakest memory design on LongMemEval/MemoryAgentBench; instant writes = 90–98% poisoning success; pruning destroys exact copies | major |
| 3 | Expert per cycle | Unbounded growth (~61B params/yr at 8B scale); router drift; dropped the paper's reset mechanism; SMF A/B confounded | major |
| 4 | Sleep loop | Self-referential reward/judge, self-distillation collapse, subliminal transfer; surprise filter promotes poison & hallucination-inducing facts; ~120–2,500 H100-h/night | **fatal** without grounded rewards; major otherwise |
| 5 | Merge gate | Contradicts "frozen"; moves the anchor; invalidates other experts; tests can't see latent traits | major |
| 6 | JEPA later | No integration path; scope creep | minor |
| — | Offline vs online sleep | Fine; make trigger data-driven | minor |

---

## 12. Top 5 changes (ranked)

1. **Invert the burden of proof: build the non-parametric learner first and make it strong.** Long context + structured, provenance-tagged memory + reflected lesson/skill library (ACE/ReasoningBank/Voyager style) + sleep-time reflection that writes *text*. Weight consolidation is an add-on that must beat this on *with-retrieval* held-out metrics at reported compute. (The current "RAG-only baseline" is a strawman; the real baseline is this stack.)
2. **Weights are a cache of the memory log, not the store.** Never prune; every expert must be rebuildable from the log onto a new base. Delete the merge gate; replace it with "rebuild on the next open base". Consolidate only procedural skills and frequently-needed, verified knowledge — never raw facts.
3. **Replace the surprise filter + self-judge with grounded selection and grounded reward.** Utility (retrieval frequency, failure→correction), verifier outcomes (tests, execution, environment), trusted provenance. Hold out a never-touched eval set and canaries. Judges from other families only as tie-breakers.
4. **Drop the hybrid core for the research phase.** Dense Qwen3-4B/8B, mature PEFT/vLLM multi-LoRA. Revisit core choice only after the loop shows value.
5. **Bound the learnable layer.** Fixed expert budget keyed to skill clusters, not nights; expert-to-expert consolidation; KL-to-base + fixed public replay mix; per-cycle capability/safety/poisoning gate with automatic rollback. Run SMF only as a separate study on its own base.

(Also: delete JEPA from the roadmap until there's a video/embodied task.)

---

## 13. Minimal falsification experiment (~2–3 weeks, ~$300–600 compute) [E]

**Question:** Does nightly weight consolidation add anything on top of a strong non-parametric learner?

- **Model:** Qwen3-4B dense (fallback Qwen3-1.7B). 1× H100 80GB (rented).
- **Streams (20 "days"):**
  - *Procedural* (where weights should win, if anywhere): a synthetic new API/DSL — 50 functions introduced over 20 days, each with hidden unit tests; tasks require composing functions from earlier days. Ground-truth reward = tests.
  - *Declarative* (where RAG should win): ~2,000 post-cutoff facts/doc chunks (News-2025 from 2608.01630 if available, else scraped post-cutoff news), incl. ~10% facts that get *updated* later (tests conflict resolution), with held-out paraphrase and 2-hop questions.
  - *Poison*: 2% of writes are MINJA/PoisonedRAG-style injections from an "untrusted" source.
- **Arms:**
  - A. RAG-only (BM25 + dense hybrid)
  - B. Long context (stuff everything that fits; truncate oldest)
  - C. A + structured memory + ACE-style lesson library + sleep-time text reflection
  - D. C + nightly LoRA consolidation (SEAL-lite self-edits, test-verified for procedural, KL-to-base + fixed public replay, one bounded adapter)
  - E. D with self-judge instead of tests (measures self-confirmation/reward hacking)
- **Metrics:** held-out with-retrieval accuracy (declarative, incl. updated facts and 2-hop); procedural pass@1 on held-out composition tasks; BWT on day-1–5 items at day 20; capability delta (MMLU-Pro subset, GSM8K, IFEval); safety delta (HarmBench subset + misalignment probe); hallucination on unknowns; poison reach (fraction of poisoned claims asserted without retrieval); GPU-hours per point of gain.
- **Budget:** ~100 consolidation candidates/night × 4 edits × ~20 s inner loop at 4B ≈ 2–3 GPU-h + ~1 GPU-h eval ⇒ ~60–80 H100-h for 20 cycles for D, similar for E ⇒ ~150 GPU-h ≈ $300–450, plus small API judge costs.
- **Kill / continue rule (pre-registered):** If D beats C by **< 3 points absolute** on held-out procedural pass@1 **and** < 3 points on with-retrieval declarative accuracy, **or** costs > 1 point on capability/safety, or poison reach in D > C: drop weight learning and invest in the non-parametric stack. If D wins only on procedural: keep weights for skills only (as recommended). If E ≈ D: your judge is fine; if E > D on judge scores but < D on tests: you've measured reward hacking — never ship self-judged consolidation.
- **Even cheaper pre-step (free):** log 2 weeks of your own real usage and label each failure as *didn't have the info* vs *had it but didn't use it* vs *lacked a skill*. If most failures are "had it but didn't use it", no amount of consolidation helps — fix retrieval/reflection.

---

## Sources checked this session
- CL-Bench https://arxiv.org/abs/2606.05661 · SDPO limits https://arxiv.org/abs/2607.01763 · RING https://arxiv.org/abs/2608.01630 · Parametric RAG https://arxiv.org/abs/2510.12668 · SDFT https://arxiv.org/abs/2601.19897 · On-Policy Replay https://arxiv.org/abs/2605.29495
- Fine-tuning vs retrieval https://arxiv.org/abs/2312.05934 · Gekhman https://arxiv.org/abs/2405.05904 · LoRA learns less https://arxiv.org/abs/2405.09673 · RL's Razor https://arxiv.org/abs/2509.04259 · Spurious forgetting https://arxiv.org/abs/2501.13453 · Model editing at scale https://arxiv.org/abs/2401.07453
- Sleep-time compute https://arxiv.org/abs/2504.13171 · ACE https://arxiv.org/abs/2510.04618 · ReasoningBank https://arxiv.org/abs/2509.25140 · GEPA https://arxiv.org/abs/2507.19457 · Cartridges https://arxiv.org/abs/2506.06266 · HippoRAG 2 https://arxiv.org/abs/2502.14802 · Mem0 https://arxiv.org/abs/2504.19413 · A-MEM https://arxiv.org/abs/2502.12110
- LongMemEval https://arxiv.org/abs/2410.10813 · MemoryAgentBench https://arxiv.org/abs/2507.05257
- MINJA https://arxiv.org/abs/2503.03704 · PoisonedRAG https://arxiv.org/abs/2402.07867 · Self-preference https://arxiv.org/abs/2404.13076 · One-token judge attack https://arxiv.org/abs/2507.08794 · Subliminal learning https://arxiv.org/abs/2507.14805 · Emergent misalignment https://arxiv.org/abs/2502.17424 · Fine-tuning compromises safety https://arxiv.org/abs/2310.03693
- Repeat After Me https://arxiv.org/abs/2402.01032 · Titans https://arxiv.org/abs/2501.00663 · V-JEPA 2 https://arxiv.org/abs/2506.09985 · EntiGraph https://arxiv.org/abs/2409.07431 · Physics of LMs 3.1 https://arxiv.org/abs/2309.14316
- MoE-LoRA router drift (search snippets only): https://arxiv.org/abs/2601.13020 · https://arxiv.org/abs/2506.21035
