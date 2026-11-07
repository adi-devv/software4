# Pilot design (build step 1b)

**Question.** Over 3 sleep cycles, are new skills kept without old ones dropping, and is that better than text-based help? Does it survive step 0b's two weaknesses (replay overlapped the math test; the router's task was easy)?

Script: `pilot/pilot.py`. Run with `scripts/kaggle_run.sh pilot` on 2x T4. Smoke first: `S4_SMOKE=1`. One seed: `S4_SEED` (default 0).

## Tasks
- **3 skills, one template.** `Run the procedure <name> on this list: w1, w2, ... Reply with the result only.` Same word lists (step 0b's), 3-5 words. Only the invented name differs, so the router must tell them apart by one word.
  - `zorvel`: sort alphabetically, capitalize, join with ` ~ `, append ` [n]` (0b's skill A).
  - `quillon`: first letters in order, uppercase, then `/` and the last word.
  - `mardeth`: words reversed, joined with `-`, then ` #n` = number of words starting a-m.
- **Grader:** exact match after trimming outer whitespace. No model judging.
- **Sets:** per skill 100 train, 100 hidden test, 30 gate. Made with separate seeds; no word list (as a set) is reused anywhere, so train/test/gate and all skills are disjoint (checked by `S4_ROLE=selftest`).
- **Canaries (100 each):** GSM8K test rows 0-99 (replay contains GSM8K train, as in 0b); ARC-Challenge test, shuffled with seed 0, first 100 (not in replay). Gate sets: GSM8K test rows 100-129, ARC rows 100-129 of the same shuffle.

## Arms
| Arm | What changes at a sleep | At test time |
|---|---|---|
| Experts + router | add LoRA expert `e<k>` (r=16, all Linear in `.layers.N`, no visual/mtp), trained on skill k + generic replay. Older experts frozen and inactive | nearest centroid on the frozen core's mean last hidden state (adapters off) picks one expert or none |
| Memory layer | one shared layer (hook on `layers[L//2].mlp`, 8192 slots, top-32, `out + mem(input)`), trained on skill k + all earlier skills' training tasks + generic replay | no router |
| Plain base | nothing | - |
| RAG | nothing; top 3 training examples (frozen-core embedding) put in the prompt | corpus = skills seen so far |
| Notes | base model writes one short lesson per skill from 10 examples; top-1 lesson by embedding put in the prompt | notes for skills seen so far |
| Oracle | true rule written in the prompt (skills only; upper bound, not a competitor) | - |

**Same steps in both weight arms:** 80 steps x batch 16 (micro-batches of 2, length cap 384) per sleep. Learning rate is per method (experts 2e-4, memory 1e-3, as in 0b). Data differs by design: memory also replays earlier skills (experts keep them in frozen experts). Every sleep's settings, row counts, loss, NaN steps and peak GPU memory are recorded in `results.json`.

**Generic replay:** 200 GSM8K train + 200 Alpaca prompts, answers written once by the base model (0b did the same), saved. Each sleep samples 150 of them. This is a simplification: for the memory arm the "pre-update model" at sleeps 2-3 is the earlier memory state, but its answers are not regenerated (cost). Worth noting if the memory arm drifts.

**Router.** Centroids from the training prompts of each active expert plus all replay prompts (class "none"). Cosine to normalised centroids (0b used un-normalised mean dot; changed so a diffuse "none" class is not biased). Router accuracy is reported overall, macro, per class, and as a confusion table on the test prompts (skills seen + GSM8K + ARC as "none"). A skill whose expert was rolled back has no class and is excluded from router accuracy.

## Each sleep
Train, gate, keep or roll back, then score on the hidden test sets.
- **Gate:** the deployed system (experts: with router rebuilt including the candidate) on the 30-item gate sets of all earlier accepted skills, the new skill, GSM8K and ARC. Roll back if any old skill or canary falls **more than 10 points** from its value before the sleep (value after the last decision; for sleep 1, the base model). Also roll back if trained parameters are not finite. Every decision is logged (`gate_log` and per sleep).
- **Rollback:** experts: the new expert is never routed to (older experts are checked to be bit-identical). Memory: state restored from a snapshot taken before the sleep (`pilot/memory/sleep<k>_pre.pt`).
- **After every sleep, even if rolled back:** all seen skills + GSM8K + ARC on the 100-item hidden test sets (retention table), in the decided state.

## Go / no-go (fixed before running; `go_nogo()` computes it)
**Go** if at least one weight arm, after sleep 3, meets all of:
1. **Learned:** each skill >= 30 points above plain base.
2. **Kept:** each skill <= 10 points below its score right after its own sleep.
3. **Old skills:** GSM8K and ARC each <= 10 points below base.

**Flagged separately:** does that arm beat, tie (within 9 points) or lose to the best text peer on the 3 skills. Compared on the mean over the 3 skills at cycle 3; "best text peer" = the better of RAG and notes (oracle is an upper bound, shown separately). A loss is a valid result and the decision goes to the user.

## Failure handling and reuse
- Each step is wrapped. A failed step marks everything depending on it `skipped` (no numbers); other arms keep going. Results are written after every step; each process has its own `results_<role>.json`; the driver merges them to `/kaggle/working/results.json`.
- Phase 1 (parallel): GPU0 replay set, GPU1 baselines (plain base on all test and gate sets). Phase 2: GPU0 experts, GPU1 memory. Text peers start on whichever GPU frees first.
- Saved to `/kaggle/working/pilot/`: `baseline.json`, `replay.json`, per sleep `experts/e<k>_accepted|rejected.pt`, `experts/sleep<k>.json`, `memory/after<k>.pt`, `memory/sleep<k>_pre.pt`, `memory/sleep<k>.json`. Attach an earlier run with `kernel_sources` and they are found under `/kaggle/input/**/pilot/` and reused, but only if seed, settings and replay set match (otherwise recomputed).

## Run-time estimate (from step 0b's 47 min; real timings only come from the Kaggle run)
Step 0b: ~5 GSM8K-100 evals, one 400-prompt replay generation, 2 x 2000 training rows, ~5 min setup = 47 min. Splitting that into G (min per GSM8K-100 eval) and T (s per training row), G=2.5-4 min gives T=0.16-0.32 s.
- Phase 1: ~10 min (replay ~6-10, baselines ~6, parallel).
- Per weight arm: 3 x 80 x 16 = 3840 training rows = 10-20 min, plus ~5.5 G-equivalents of eval (3 test + 3 small gate GSM, plus skills/ARC) = 14-22 min, plus ~3 min load: **~35-45 min**, both arms in parallel.
- Text peers: ~15 min, overlapped with the slower arm's tail.
- Setup (pip, fla check) ~5 min.
- **Total ~65-80 min wall, ~2.2-2.7 GPU-hours if both GPUs are counted.** Under 3 hours. If it runs long, cut in this order: `steps` 80->60, canary eval size at sleeps 1-2 (GSM 100->50), skip notes.
