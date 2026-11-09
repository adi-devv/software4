# Stage V1 plan: fit the critic on same-stack data (DRAFT, not started)

Status: draft, revision 4 (revision 3 folded in the Codex gpt-6.1-sol review; the 2026-11-09 feature changes still need a sol re-review before any capture). Starts only after stage T passes (smoke, then v1 on seeds 44/45/46). V1 is a **separate pass after v1**: no v1 code changes, no change to answers or behaviour. This file is committed and frozen (commit hash recorded in the report) before any capture runs on seeds 44/45/46.

## Question
Does a small critic V, trained only on grounded outcomes, predict verified success of an attempt better than cheap baselines on fresh items, **beyond knowing which rule it is**, under the same expert stack that produced the features? If not, it is only a rule detector: stop. If the world has too little variance to tell, the answer is "inconclusive, need a harder world", not "fail".

## Main risk, stated up front
In the step5 world every attempt is a single step and each rule is near-deterministic per arm (gold_context ~100%, no_memory ~0%). Per-rule success rate then already predicts almost everything, and pooled AUROC can reward ranking groups rather than attempts. V can only show value where a rule is partly learned under the current stack, and only by discriminating **within** a group. Facts (6 items) are too few to fit on, so they are diagnostic only.

## Setup: separate pass, same stack for features and labels
For each registered v1 seed (44/45/46) and each cycle c (0, 1, 2), restore the expert stack S_c as it stood after sleep c, then on that stack only:
1. **Fit set F_c**: 80 fresh items per rule introduced so far (rules A, B, C enter at cycles 0, 1, 2), run through `ours` under S_c and graded. Features and labels come from the same forward pass, same stack. Labels from earlier stacks are never reused.
2. **Eval set E_c**: the v1 hidden items of cycle c (120 per rule introduced so far), re-run under S_c with feature capture.
3. A critic is fit on F_c only and scored on E_c only. One critic per cycle. No pooling of F across cycles, because hidden states shift with each expert.

**Item separation.** Fit codes come from a V1-only generator (seeded by world seed + 1000), disjoint from every smoke, step4 (41/42/43), v1 (44/45/46) and sweep (47) code, and from each seed's train, gate and hidden codes, across all rules. The fit generator is the same class, prompt template and suffix as the v1 world, differing only in seed and count, so fit and eval differ by sampling only. A test audits canonical task identity (rule, code, full prompt) and that no code appears under two rules (rule C uses its own codes, so composition identity is checked by rule, code and prompt), not just code strings. Rule answers: A = CODE-7 uppercased, B = `B-CODE`, C = `B-CODE-7`; the world was changed from bracket answers on 2026-11-08 because the grader strips edge brackets and a bare echoed code passed rule B (seed 44 first attempt, no_memory rule B 14.2%, kept as evidence). Fit-set code lives in `step4/v1_fitset.py`. Hidden items are the same set re-scored each cycle, so a task appears in several E_c; this is documented and handled by clustering in the bootstrap.

**Restoring stacks and validating capture.** v1 writes adapters under `<out>/adapters` (`step4/hf.py:138`). Precondition, checked on the first v1 output before V1 starts: for every seed and cycle, the versioned adapter set, router state, gate state and generation settings are all present. If any cell is incomplete, V1 must re-run the sleeps (extra Kaggle spend), which needs the user's approval and a new matched evaluation, since a retrained stack is a different stack.
**Capture validation:** the capture wrapper must reproduce the saved v1 answers token for token on E_c (not only the 0/1 outcome, because "almost always wrong" cells would hide a missing stack). Any unexplained mismatch on a deterministic item invalidates that seed-cycle; the only tolerance allowed is a numeric one fixed in advance for float nondeterminism, and it applies to logprobs, never to emitted tokens. **All 9 registered cells (3 seeds x 3 cycles) must be valid for a PASS.** An invalid cell cannot be dropped to rescue a PASS; it makes the run INVALID and is reported.

V1 capture code lives in a new module that wraps the backend; the v1 run path is untouched.

## What gets logged per attempt
- core hidden states for a few layers, float16, side `.npz` per seed-cycle, in two pools kept separate: prompt tokens (last token, mean of all, mean of the last ~10) and the model's own answer tokens (mean). `step4/v1_capture.py` pools prompt tokens only today, so it must add the answer pool before capture
- mean and min answer-token log-prob, first-token entropy (all V inputs)
- retrieval features: top score, hop count, deep-search flag, context tokens
- rule id, cycle, set (fit or eval), expert-stack version, verified outcome (0/1)
The backend must return hidden states and token logprobs; today it does not. The wrapper adds this without changing generation.

## Layer and hyperparameter sweep: seed 47 only
Seed 47 is a V1-only world seed that is never scored. Before any scored-seed capture, this file freezes: the feature set, the full search space (layers from mid-depth to final, the three prompt poolings listed above, L2 strengths, PCA dims), the selection objective (mean 5-fold CV AUROC **within group**, groups defined by rule), fold construction (stratified by outcome, grouped by code so no code spans folds), the preprocessing procedure, and the fallback for undefined scores (a fold or group with a single class scores 0.5 and is counted as such; if fewer than 3 usable mixed groups exist in seed 47, the sweep is unusable and the default layer, mean pooling and L2=1.0 are used, stated up front). The sweep must run the same capture and restore pipeline as the scored seeds. Selection is made on seed 47's F_c only. Scored seeds supply no tuning signal; if any choice is ever guided by scored-seed results, those seeds are burned and fresh scoring seeds are required. Note v1's own per-rule results will already be known when V1 starts; they are not used for any V1 choice.

## Critic
Logistic regression (L2, standardised, optional PCA to at most 64 dims, all frozen from seed 47). V's features are the prompt pool, the answer pool, mean and min answer log-prob, first-token entropy and the retrieval features. An MLP is out of scope for V1. Monte Carlo target only (single step).

## Prior-work findings behind these choices
- **Answer pool is required.** With prompt pools alone, V's hidden features are nearly the same as baseline B4, so V could hardly beat it.
- **Linear first, separate pools, entropy stats** (POISE, 2605.07579). A linear probe on the model's own hidden states, with prompt and response pools plus entropy stats, beat a separate critic on DAPO-Math held-out Pearson r (1.5B: .834 vs .723). The separate critic won on AceCoder and IF-RLVR, so this is not universal. Its best MLP barely moved MAE (.117 vs .124) and had worse r.
- **Within-group AUROC is required** (2511.14773). AUC .84 at t=4 fell to .72-.77 within a difficulty level, so part of the signal is difficulty detection.
- **Sweep must include mid-depth layers** (2509.10625). The best layer ranged from .44 to .95 of depth. GSM8K AUROC was only about .5-.6, so a weak or absent signal is a real possible outcome.

## Baselines to beat (all fit on the same F_c)
- B1: per-rule success rate in F_c, shrunk toward the pooled F_c rate with a Beta(1,1) prior (add-one smoothing), so tiny classes do not give 0 or 1.
- B2: model's mean answer log-prob, isotonic-calibrated on F_c. Isotonic ties can weaken AUROC, so the raw (uncalibrated) mean log-prob AUROC is also reported and B2's AUROC uses the better of raw and isotonic.
- B4: prompt-only critic: same logistic, same F_c, features from the prompt alone (pooled hidden state of the prompt, no answer tokens, no retrieval or expert output). Tests whether V reads anything beyond the question.
- B5: metadata/length baseline: logistic on rule id, cycle, prompt length, context tokens, answer length, retrieval features only (no hidden states, no log-probs).
- B3: logistic on [B1 rate, mean log-prob] fit on F_c. **B3 is gating**, because it is the strongest cheap predictor that knows the rule and the model's confidence. V must beat it, otherwise V is only a rule detector or a confidence readout.

## Strata
- **Partly-learned**: a (rule, cycle, seed) group whose accuracy in F_c is between 20% and 80% inclusive. Defined from the fit set only, before any critic is scored. Eval results never move a group in or out.
- Rule C (composition): a priori, reported.
- Facts: diagnostic only.

## Eligibility gate (from F_c alone) and evaluation validity (separate)
**Eligibility, computed from F_c only**, over the three registered seeds. A group is one (seed, rule, cycle); rule A appears in 3 cycles, B in 2, C in 1, so there are 6 groups per seed and 18 in total:
- at least 6 partly-learned groups, and
- they come from at least 2 different seeds.
Seeds that contribute at least one partly-learned group are the **sign-agreement seeds**; every one of them must have eligible data, and seeds with none are excluded from the sign check and reported.
**Feasibility check**: before scoring, report the same F_c counts for seed 47, as an informal indication whether the world has any variance. It is not a gate.
**Evaluation validity (separate from eligibility)**: the pooled partly-learned eval sets contain at least 100 successes and 100 failures, and, inside eligible groups, both outcomes occur, so within-group discrimination is defined.
If eligibility or evaluation validity fails, the verdict is **INCONCLUSIVE**: the world cannot show whether V adds anything. That is not a fail and not licence to skip V2. Every verdict, INCONCLUSIVE included, is locked and reported in full. A harder-world retry is a new registration and must keep and cite the prior results; it cannot be used to quietly replace an inconclusive or failing outcome. The gate result is written to the report before any critic metric.

## Metrics and pass rule
Metrics: Brier score and AUROC on the eval sets, plus **within-group AUROC** (mean over partly-learned groups, groups with one class excluded and listed). Differences are V minus baseline for AUROC and baseline minus V for Brier, so positive always means V is better.
Bootstrap: paired, 95% two-sided percentile intervals, 10,000 draws, fixed seed, items equally weighted, stratified by (seed, group). Resampling clusters by **(seed, code) across cycles and rules**, resampled with replacement within each seed, because the same hidden task appears in several E_c; this preserves group sizes in expectation rather than stratifying exactly by (seed, group). Inference is conditional on the fitted critics and the registered stacks; it says nothing about new stacks.
- **PASS** requires all of the following (an intersection of prespecified comparisons, so no multiplicity correction is applied; there are no optional subsets, no alternative winning metrics and no repeated attempts):
  1. All 9 cells valid, eligibility and evaluation validity met.
  2. V beats B1, B2 and B3 on Brier, AUROC and within-group AUROC (3 baselines x 3 metrics), each with CI low > 0, on the pooled eval set and on the partly-learned stratum.
  3. Sign agreement on the partly-learned stratum for each of those comparisons: positive in at least 2 of 3 seeds, or in all of them if only 2 seeds are sign-agreement seeds. Sign agreement is descriptive support, not a generalization claim across seeds.
- **FAIL** (gate and validity met): anything else. Write up as a rule or confidence detector, skip V2.
- **INCONCLUSIVE**: eligibility or evaluation validity not met. Needs a harder world.
- **INVALID**: any registered cell fails restoration or capture validation. Not a result; fix and re-capture, with a note in the report.
- Prior-art additions (2026-11-08): V must also beat B4 and B5 on the same 3 metrics with CI low > 0; within-group AUROC is a pass criterion because published outcome-grounded critics gain mostly by spotting hard tasks (2610.18614). Leakage audit before any fit: no feature may contain the gold answer, the grade, engineered answer-correctness checks, future-cycle data or outcome-correlated ids. The model's own generated-answer activations, log-probs and entropy are allowed V inputs, because this is an answer-time checkpoint. The audit is a test that lists each feature column.
- **Scope statement:** the step5 world is single-step, so there is no verified mid-task state. V1 therefore tests a rule/confidence detector on answer-time features, **not** the grounded-state critic claim. That claim needs a separate variant where an environment checks step 1 and returns the real observation (the capacity world has only model-generated intermediate states, which are not grounded) and its own registration; do not report V1 as a grounded-critic result.
- ECE <= 0.05 reported, not gating. Seed-clustered CIs are not used: with 3 seeds they are too crude; per-seed signs are reported instead.

## Cost
CPU fit. Capture needs about 1,200 forward passes per seed (480 fit, 720 eval) plus the same for seed 47; no Kaggle spend if the adapters survive.

## Resolved decisions
1. Separate pass after v1, reuse saved adapters, no v1 code change.
2. Layer sweep on seed 47, frozen for 44/45/46, with a pre-fixed search space and fallback.
3. Weak within-rule variance is handled by the eligibility gate and an INCONCLUSIVE verdict, not by waiting; verdicts are locked.
4. B3 (rule rate + log-prob) is gating, and within-group AUROC is required, so V cannot pass as a rule detector.

5. Fallback re-run (user-approved): if the adapter precondition fails, V1 may re-run the sleeps on Kaggle for the `ours` arm only, capped at about 1.2 worker-hours total. A retrained stack is a different stack, so the re-run needs its own matched evaluation (see "Restoring stacks"). If the cap would be exceeded, stop and ask. Earlier runs already kept adapters, so this is not expected to be needed.

## Still open (needs the user)
- Nothing on V1 planning. Seed 44 is being re-run on the fixed world; seeds 45 and 46 follow if the validity gate (gold_context >= 85%, no_memory <= 10% on each rule's final hidden set) passes.
