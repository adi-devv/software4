# Independent direction review: software4

Reviewer: Claude (Fable 5.2), read-only pass, 2026-11-11. Branch `step4b`; nothing in the repo was changed.

## Bottom line

- **Pivot the test, park the critic, narrow the headline claim.**
- The project has not shown that learning in weights beats context on anything the frozen model can do from context. Every large gap comes from a skill the 2B model cannot do from context, or from a weak retriever.
- That "artefact" is the only real signal in the data. Stop treating it as a nuisance and make it the experiment: measure where context cannot teach, and test weights there against strong context baselines.
- The critic line (V1 → V2 → V3) should stop now. It is close to inconclusive by construction and breaks the plan's own Stage T rule.

## What I read, and what I did not

- **Read:** `README.md`, `PLAN.md`, `CLAUDE.md`, `step4/RESULTS.md`, the v1 and v3 verdicts, the v2 and v3 pre-registrations, `docs/V1_CRITIC_PLAN_DRAFT.md`, `docs/online_arm_design.md`, `step0`, `step0b` and `pilot` results, `harness/step5_world.py`, the three reviews in `research/` (`critique.md` only lines 1-140), and in the capacity worktree the pre-registration (first 90 lines), the review and the package README.
- **Not read:** `docs/architecture.html`, `docs/CAPACITY_WORLD_DESIGN.md`, `step4/V2_VERDICT_s48-50.md` (I used the v2 section of `step4/RESULTS.md`), `research/papers_deep.md`, `research/papers_xsa.md`, the training code in `baselines/`, and the raw answer files.
- **Not verified:**
  - Any arXiv citation in the docs. I checked none of them.
  - The per-answer effect of the v3 fallback. I rely on the verdict tables, not the raw answers.
  - My claim that the V1 eligibility gate will not be met. It is inferred from final-cycle accuracies, not from the fit set.
  - All GPU-hour estimates below. They are scaled from the logged v2 and v3 run times.

## 1. What has been demonstrated, and what has not

### Solid

| Finding | Evidence |
|---|---|
| The engineering works: fp16 on T4, LoRA on the hybrid core, router, replay, sleep gate and rollback, Kaggle pipeline | `step0/RESULTS.md`, `step0b/RESULTS.md`; 634 CPU tests (`docs/V3_FREEZE.txt`) |
| A rank-4 expert per sleep learns three template rules to the gold-context ceiling | v3: `expert_only` 90.83%, `ours` 94.17%, `gold_context` 93.89% (`C:\Users\aadit\v3_out\v3-verdict.md`) |
| It does so with about a tenth of the context | v2: 17.8 context tokens against about 196 (`step4/RESULTS.md` 105-180); v3: 24.5 against about 195 |
| Pre-registration discipline is real: frozen hashes, paired bootstrap, validity gates, and failures reported as failures | v1 FAIL, v2 FAIL, v3 UNINFORMATIVE all recorded |
| Training lifts a skill that context cannot teach from about 3% to about 55% | step 4b rule A: `ours` 59/72/68 of 120 against `rag` 3/3/6; gold context 5/120 on seed 41 (`step4/RESULTS.md` 68) |

### Artefacts of the test world

- **The +26.4 points in step 4b is rule A only.** Rule B gives +1.1 points (338 against 334 of 360). Rule A is letter reversal, which the frozen model cannot do even with the rule in context (`step4/RESULTS.md` 68; `PLAN.md` 21).
- **The pilot's 91/70/48% against 7% has the same cause.** The oracle peer scored 0% on all three skills (`pilot/RESULTS.md`).
- **The v1 +6.4 points is a retriever weakness.** Hashing RAG scored 71.1% on rule A and beat `ours` on B and C (`docs/V1_VERDICT_s44r-s45-s46.md`).
- **With a fair retriever the expert alone is below RAG.**
  - v2: `ours` 90.19% against `rag_semantic` 92.78%; rule C −12.2 points [−16.7, −7.5].
  - v3: `expert_only` 90.83% against `rag_semantic` 92.41%.
- **The v3 result of +1.76 [0.00, 3.61] is not a weights result.**
  - It comes from a format-checked fallback that hands 68, 91 and 84 answers per seed to the frozen core plus retrieval.
  - No baseline was given the same help (`docs/V3_PREREG_DRAFT.md` states this caveat).
  - The interval's lower bound is exactly 0.00.
- **Everything sits at the ceiling.** Gold context is about 94%, so differences of 2 points are within noise for 360 items per seed.
- **In the step5 world the rules are output formats** (`CODE-7`, `B-CODE`, `B-CODE-7`; `harness/step5_world.py`). "Rule learning, not format learning" cannot be separated there, and `docs/V2_PREREG_DRAFT.md` line 19 says so.

### Not shown

| Claim | Status |
|---|---|
| Weights beat a fair context baseline on a skill the model can follow from context | Not shown. Three confirmatory rounds: FAIL, FAIL, UNINFORMATIVE |
| Rule learning rather than format learning (Claim 3) | Never validly tested: control installed nothing (v1), was a broken adapter (v2), was malformed on seed 54 at 67.5% (v3) |
| Continual learning over many sleeps | Not tested. Three cycles and three rules; the plan's core experiment needs 15+ cycles and about 1.7k paired tasks (`PLAN.md` 102-128) |
| Forgetting bounded | Not tested at scale. The plan asks for 308-400 canaries |
| Long-context baseline | Arm does not exist |
| The memory adds to the expert on rules | No. `ours` never retrieved on rules in v2 (0 of 2,160) and equals `expert_only` in v1 and v2 |
| Online learning, dreaming, merging, memory-layer experts | Unrun, or failed its gate in the pilot |
| Anything about a critic or value function | Nothing built |

## 2. Is the current path the best way to the goal?

**No.** Three reasons.

- **The step5 world cannot discriminate weights from context.**
  - The project's own registration says this: "v1 does not test 'weights beat context'" (`step4/RESULTS.md` 89).
  - Three rules, three cycles and a 195-token RAG context put no pressure on capacity, interference, horizon or cost.
  - A fourth round on this world would cost about 5 GPU-hours and could only move a 2-point difference at the ceiling.
- **The critic is the wrong place for original work right now.**
  - The plan's Stage T rule is "if weights can't beat working baselines, stop and rethink consolidation; don't build V on top" (`PLAN.md` 29-34). Stage T has not passed.
  - The world is single-step and each rule is near-deterministic per arm. The critic draft says per-rule success rate "already predicts almost everything".
  - The V1 eligibility gate needs at least 6 groups at 20-80% accuracy. Observed group accuracies are 88-97% or 0%, so V1 is likely INCONCLUSIVE before any fitting (inferred, not checked against the fit set).
  - `ours` is at 94.17% and the gold ceiling is 93.89%, so a critic has nothing left to gain by day.
  - The v3 fallback is already a hand-built critic, and it took the gain that was available.
  - The plan itself concedes prior art for hidden-state and grounded critics (`PLAN.md` 22-23). Citations unverified.
- **The capacity world asks the right question but is over-built and scoped away from the goal.**
  - 449-line pre-registration, 17 review findings with 2 blockers, 950,400 hidden evaluations, zero model results, and the files are uncommitted.
  - It redefines the system as one pooled adapter retrained from base, and states that no continual-learning claim follows.
  - The review found little real pressure: required support uses 0.58-0.70 of the budget in regime R, and a compact table of the whole 36-rule library is only 420-435 tokens.
  - Power is weak: a true 5-point gain passes about half the time.

## 3. Candidate directions

| | Direction | What it would show | Cost | Main risk | Kill criterion |
|---|---|---|---|---|---|
| A | Current path: finish T, then critic V1 → V2 → V3 | A probe on hidden states predicts correctness and improves routing or sleep selection | 15-25 GPU-h, 4-6 weeks | V1 inconclusive by construction; at most 0-6 points of headroom; prior art | V1 eligibility gate unmet on the existing fit set (a CPU check, zero GPU) |
| B | Capacity world as drafted | Weights hold up when the context budget binds | 40+ GPU-h, 5-8 weeks including the fixes | A tuned retriever or a compact table removes the pressure; no continual-learning claim even on success | A one-seed pilot at the largest rule count shows tuned RAG within 3 points of gold context |
| C | Narrowed claim: consolidation as amortisation | Parity with RAG at about 10% of context tokens, held over 15+ sleeps with bounded forgetting | 10-15 GPU-h, 2-3 weeks | True but unexciting; prompt caching weakens the cost argument; this is ordinary fine-tuning | Non-inferiority fails at 15 cycles, or early rules lose more than 5 points |
| D | Teachability-graded world (recommended) | A map: how much weights gain as a function of how well context can teach the skill, against strong context baselines | Screen 1-2 GPU-h; confirm 12-18 GPU-h; 3-4 weeks | Many-shot context closes the gap; the finding is specific to a 2B model | Fewer than 5 of about 25 screened skills have context below 50% and training above 80%; or many-shot context gets within 10 points of the expert |

Notes on D:
- It keeps what the project already built (sleep loop, gate, router, analysis scripts) and drops only the step5 rules.
- It uses the one robust effect in the data (rule A, the pilot skills) instead of designing it out.
- It folds in C for free: on the high-teachability skills the result is the parity-at-lower-cost claim.
- A low-teachability gap is still fine-tuning. The honest framing is "automated, gated, post-deployment skill acquisition where instructions and examples fail", not "a new learning mechanism".

## 4. Recommendation

**Take direction D. Park A. Do not start B as drafted. Keep C as the fallback claim.**

### Next steps

1. **Stop three things today.**
   - No fourth round on the step5 world.
   - Park V1-V3. Before shelving, run the eligibility count on the existing fit data (CPU only). If it fails, record V1 as INCONCLUSIVE and close it.
   - Freeze the capacity world where it is; commit the files so they are not lost, and spend no GPU on it.
2. **Run a teachability screen with the frozen core only (1-2 GPU-h, no training).**
   - About 25 candidate skill families: character-level edits, multi-step string procedures, small arithmetic conventions, lookup conventions, the three step5 templates as anchors.
   - Score each under three context conditions: rule text, 5 retrieved examples, and many-shot (16 and 32 examples).
   - The output is one table of context accuracy per skill. This table alone decides whether D is alive.
3. **Run a one-seed, unregistered training pilot on 8 skills spanning the table (3-4 GPU-h).**
   - Arms: `expert_only`, `ours`, `rag_semantic`, many-shot context, `gold_context`, `no_memory`.
   - Give the format-checked fallback to every arm or to none.
   - Use 8-10 sleeps so the router and forgetting are exercised.
4. **Only then pre-register a short confirmation (3 seeds, 12-18 GPU-h).**
   - One primary claim: on skills where the best context arm is below 50%, `expert_only` beats it by at least 20 points.
   - One secondary claim: on skills where context is above 85%, `ours` is non-inferior at no more than 15% of the context tokens.
   - Keep the registration to one page. The capacity pre-registration shows what happens otherwise.
5. **Rewrite the headline claim in `README.md` and `PLAN.md` to match what the data can support**, and move the critic to "future work, needs a multi-step world".

### What would change my mind

- The V1 eligibility count comes back with 6 or more partly-learned groups from 2 seeds. Then V1 is a cheap CPU job and worth finishing.
- The screen finds almost no skills in the low-teachability band beyond letter reversal. Then D is dead and C is the honest paper.
- Many-shot context closes the rule A gap. Then the step 4b result is a baseline weakness too, and the weights claim has no support left.
- A cheap one-seed capacity pilot shows tuned RAG falling well below gold context as rules grow. Then B, cut down to that pilot's size, becomes the better test.
