You are a methodology reviewer. Everything you need is in this message; you do NOT need to run commands or read files (your shell is broken, do not try it). Do not modify anything.

Review the pre-registered plan below for stage V1 of a research project (a frozen 2B LLM with sleep-trained rank-4 LoRA experts; V is a small critic that predicts whether an attempt will be verified correct). Look for: (1) leakage between fit and eval sets (same code, same stack, same items, tuning on scored seeds); (2) unfair or weak baselines (B1 per-rule rate, B2 logprob+isotonic) so that V could "win" for the wrong reason, or lose unfairly; (3) the 20-80% partly-learned definition and the minimum variance gate numbers (>=6 groups, >=2 seeds, >=100 successes and >=100 failures): are they achievable, too lax, or gameable; is INCONCLUSIVE handled so it cannot be used to dodge a fail; (4) the seed-47 sweep: does it really isolate tuning from scoring, and is the sweep world representative; (5) the bootstrap and pass rule (stratification by seed and group, paired, CI low > 0, sign agreement 2 of 3, multiple comparisons across 2 baselines x 2 metrics x 2 strata); (6) the 99% re-run agreement check and the adapter-persistence precondition; (7) anything that makes the plan unfalsifiable or makes a broken run able to PASS.

Context facts: world is step5 (rules A, B, C, template-copy; gold_context ~100%, no_memory ~0% per rule), 3 cycles, v1 seeds 44/45/46, 120 hidden items per rule per cycle, each rule near-deterministic per arm. Sleep c produces expert stack S_c; hidden states change with each expert. Greedy decoding. Fit set F_c = 80 fresh items per rule introduced so far, graded under S_c; eval set E_c = v1 hidden items of cycle c re-run under S_c. One critic per cycle, fit on F_c only.

Output: a short ranked list of findings (blocking first) with the section, failing scenario, and a concrete suggested fix. Say "no blocking issues" if none. Do not pad.

=== PLAN: docs/V1_CRITIC_PLAN_DRAFT.md (revision 2) ===
# Stage V1 plan: fit the critic on same-stack data (DRAFT, not started)

Status: draft, revision 2. Starts only after stage T passes (smoke, then v1 on seeds 44/45/46) and review of this pre-registration. V1 is a **separate pass after v1**: no v1 code changes, no change to answers or behaviour.

## Question
Does a small critic V, trained only on grounded outcomes, predict verified success of an attempt better than two cheap baselines on fresh items, under the same expert stack that produced the features? If not, it is only a rule detector: stop. If the world has too little variance to tell, the answer is "inconclusive, need a harder world", not "fail".

## Main risk, stated up front
In the step5 world every attempt is a single step and each rule is near-deterministic per arm (gold_context ~100%, no_memory ~0%). Per-rule success rate then already predicts almost everything. V can only win where a rule is partly learned under the current stack. That is checked mechanically before any critic is scored (see "Minimum variance gate"). Facts (6 items) are too few to fit on, so they are diagnostic only.

## Setup: separate pass, same stack for features and labels
For each registered v1 seed (44/45/46) and each cycle c (0, 1, 2), restore the expert stack S_c as it stood after sleep c, then on that stack only:
1. **Fit set F_c**: 80 fresh items per rule introduced so far (rules A, B, C enter at cycles 0, 1, 2), run through `ours` under S_c and graded. Features and labels come from the same forward pass, same stack. Labels from earlier stacks are never reused.
2. **Eval set E_c**: the v1 hidden items of cycle c (120 per rule introduced so far), re-run under S_c with feature capture. The re-run's outcomes must agree with the saved v1 outcomes on at least 99% of items (greedy decoding is deterministic); otherwise that seed-cycle is invalid and reported.
3. A critic is fit on F_c only and scored on E_c only. One critic per cycle. No pooling of F across cycles, because hidden states shift with each expert.

Fit codes come from a V1-only generator (seeded by world seed + 1000), disjoint from every smoke, step4 (41/42/43), v1 (44/45/46) and sweep (47) code, and from each seed's train, gate and hidden codes. A test checks this. Split is by code: no code appears in both a fit and an eval set.

Restoring stacks: v1 writes adapters under `<out>/adapters`. Precondition, checked on the first v1 output before V1 starts: every per-cycle adapter version plus router and gate state is present for all three seeds. If yes, V1 reuses them and needs no training. If not, V1 must re-run the sleeps (extra Kaggle spend); ask the user first. V1 capture code lives in a new module that wraps the backend; the v1 run path is untouched.

## What gets logged per attempt
- pooled core hidden state at the last prompt token for a few layers, float16, side .npz per seed-cycle
- mean and min answer-token log-prob, first-token entropy
- retrieval features: top score, hop count, deep-search flag, context tokens
- rule id, cycle, set (fit or eval), expert-stack version, verified outcome (0/1)
The backend must return hidden states and token logprobs; today it does not. The wrapper adds this without changing generation.

## Layer and hyperparameter sweep: seed 47 only
Seed 47 is a V1-only world seed that is never scored. Run the same pipeline on it. Pick the layer, pooling (mean vs last token), and L2 strength by 5-fold cross-validated AUROC within the F_c sets of seed 47. Freeze them before touching seeds 44/45/46. Scored seeds supply no tuning signal.

## Critic
Logistic regression on pooled features (L2, standardised, optional PCA to at most 64 dims, all frozen from seed 47). An MLP is out of scope for V1. Monte Carlo target only (single step).

## Baselines to beat (all fit on the same F_c)
- B1: per-rule success rate in F_c (same stack). Not earlier cycles.
- B2: model's mean answer log-prob, calibrated by isotonic regression on F_c.
- B3 (diagnostic only): B1 + B2 combined logistic, fit on F_c.

## Strata
- **Partly-learned**: a (rule, cycle) group whose accuracy in F_c is between 20% and 80% inclusive. Defined from the fit set only, before any critic is scored. Eval results never move a group in or out.
- Rule C (composition): a priori, reported.
- Facts: diagnostic only.

## Minimum variance gate (pre-registered, checked before scoring)
Computed from F_c alone, over the three registered seeds (up to 18 rule-and-cycle groups):
- at least 6 groups are partly-learned, from at least 2 different seeds, and
- the pooled partly-learned part of the eval sets contains at least 100 successes and 100 failures.
If the gate is not met, the V1 verdict is **INCONCLUSIVE**: the world cannot show whether V adds anything. That is not a fail and not licence to skip V2. The next step is a harder world, then repeat V1. The gate result is written to the report before any critic metric.

## Metrics and pass rule
Brier score and AUROC on the eval sets. Paired bootstrap over eval items, stratified by (seed, group), fixed bootstrap seed, 10,000 draws. Also report the sign per seed.
- **PASS** (gate met): V beats B1 and B2 on both Brier and AUROC, CI low > 0 on the pooled eval set **and** on the partly-learned stratum, and the sign agrees in at least 2 of 3 seeds on the partly-learned stratum.
- **FAIL** (gate met): anything else. Write up as a rule detector, skip V2.
- **INCONCLUSIVE**: gate not met. Needs a harder world.
- ECE <= 0.05 reported, not gating. Seed-clustered CIs are not used: with 3 seeds they are too crude; per-seed sign agreement is reported instead.

## Cost
CPU fit. Capture needs about 1,200 forward passes per seed (480 fit, 720 eval) plus the same for seed 47; no Kaggle spend if the adapters survive.

## Open
If v1 does not keep per-cycle adapters, may V1 spend Kaggle quota to re-run the sleeps?

=== PLAN.md line 43 (as edited) ===
Features and labels must come from the same stack: after each sleep, run and grade a fresh fit set under the new stack and fit V on that only; older labels are never reused, because each expert shifts the hidden states. V is versioned and rolled back with its stack.
