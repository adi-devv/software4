# Step-4 plan review (read-only, 2026-11-07)

Scope: PLAN.md, CLAUDE.md, READMEs, results/{RETRIEVAL,SMOKE}_RESULTS.md, results/smoke_sleep/run2/DIAGNOSIS.md,
results/REVIEW1.md, step4/README.md + run.py, baselines/sleep.py + arms.py, harness/ (smoke_world, gate, streams).
No code edited. One CPU check run: Qwen3 tokenizer count of a RAG rule episode (below).

## 1. Can step 4 tell "ours" apart from RAG? No: as built, the gate passes almost by construction.

1. **RAG never sees the rule lessons (measured).** `context_tokens=64` (step4/run.py `settings`). The shortest possible
   rule episode (Question + 4-token Response + Correction with lesson), chat-formatted, is **75 tokens**; a fact entry is 28.
   `build_prompt` skips any entry over budget, so RAG and the hippocampus get facts only, never rules. On rules RAG = no_memory.
   The 112-token input cap in `HFBackend._ids` (left-truncation) would also cut longer contexts. "Ours >= RAG" on rules
   then just means "the expert learned anything". This is the main confound.
2. **"Ours" bundles the hippocampus.** SleepArm subclasses HippoOnly. On facts, the hippocampus versions Alder->Sora and
   quarantines Birch->Zorg; plain RAG has no versioning/trust and keeps both. So ours > RAG by up to 2/6 facts with zero
   weight learning, and ours > no_memory by 6/6 facts from the store alone. The pass condition cannot fail on facts.
3. **The eval set is tiny and was looked at during development.** Hidden = 8 A + 8 B + 6 facts = 22 items, single seed,
   point-estimate `>` / `>=` on the final cycle only (`report()`). A tie passes ">= RAG". step4 builds `SmokeSource(0)`, the
   **same codes** used for the LR/diagnosis work (REVIEW1 already flagged that the 8 A hold-outs are not a fresh test).
4. **The second skill is likely to be rejected.** In the latest smoke the B expert was rejected because 1 of 32 replay
   examples flipped under the forced new expert (`trusted train/replay validation regressed`, zero tolerance), so ours
   ended B 0/8 (below no_memory's 2/8). Step 4 can easily become "one skill learned, second skill blocked by the pre-gate",
   which says nothing about continual learning and could be misread as a method failure.
5. **The retrieval "FAIL" is a ceiling artifact, not hippocampus losing.** The hippocampus is >= RAG in every cell; the strict
   ties-fail rule fails on 1.000 vs 1.000 ties. The real lesson: that benchmark is too easy to rank stores, and the only
   real win (revision/poison) is bookkeeping a RAG with a trust filter + latest-version key would match.
6. **Format confound.** Experts train on `Final answer: X` targets with a 16-token generation cap. Some of ours' gain on
   a 2B model can be "learned to answer tersely" rather than "learned the rule". Parse failures are not logged per arm.

## 2. Missing arms, metrics and stop rules (cheap ones first)
- **hippo_only** (PLAN arm 10, "the key baseline"; already in baselines, inference only). Without it, step 4 cannot say
  whether weights add anything beyond the store. Also **expert-only** (ours with empty context) at eval, as the smoke did.
- **RAG with a fair budget** (e.g. 512 context tokens, inference-only input cap raised; training cap can stay 112) and
  **text_notes** at the same budget. Keep 64-token RAG too if wanted, but label it "starved".
- **Per-family scores** (rule A / rule B / facts) and a **routing confusion matrix** per cycle; **forced-expert train
  accuracy** (did it fit?) and **parse-failure rate** per arm. All are already computed or nearly free.
- **Stop/continue, pre-registered per family:** continue only if on *rules*, ours (expert) beats fair-RAG and hippo_only
  by >= 15 points on >= 100 fresh hidden items per rule with a paired bootstrap CI excluding 0, *and* B is accepted
  without A dropping. If the expert did not fit its training set, it is an optimisation failure (tune LR/steps),
  not a "rethink". If fair-RAG ties on B and on A, weights add nothing in this world: stop and rethink.

## 3. Cheapest changes for more information per GPU-hour
1. **Fresh world seed(s)** and **100-200 hidden codes per rule** (codes are free; evals are 16-token greedy calls).
   Score the big set only at cycles 1, 2 and final to cap eval cost.
2. **Cut cycles 4-5** (no new exposures; retention of 2 rules over 2 idle nights is near-free information you already
   get from cycles 1-2) and spend that on **2-3 world seeds x 3 cycles**, or on 3-4 rule families instead of 2.
3. **Do not make ours wait for no_memory/rag** (`dependencies` in `schedule`). Ours is the long pole; start it first on
   GPU0 and run the inference-only baselines in series on GPU1.
4. **Relax the replay-validation pre-gate** from zero tolerance to the same paired-noise rule as the gate (or >1 flip),
   or at minimum report "would-have-accepted" outcomes so a gate artifact is visible.
5. **One CPU sanity pass** before Kaggle: fake/tiny run asserting that RAG at the fair budget actually includes the
   lesson text in its prompt.

## 4. Architecture risks this run would not reveal
- **Router:** hashing-embedding centroid, threshold 0.2. Already 16/82 routing errors at 2 experts (check c FAIL). It
  works here only because rule prompts differ by a name token. Top-1 routing also means experts never compose. N-expert
  scaling is untested; misrouting, not weight drift, is where forgetting will come from.
- **Gate power:** 4 gate items per rule per cycle + 8 canaries; the paired t-test cannot detect anything at that n.
  Safety rests on exact canary-string equality, which only protects prompts routed to the new expert and rejects good
  changes. The planned 400 GSM8K/ARC canaries are not in step 4.
- **One expert per sleep:** linear growth, merge (arm 11) unbuilt; step0's "plain expert forgets badly" is masked here
  by routing, not solved.
- **Task realism:** 4-letter reversal is a tokenizer-hard toy; success or failure on it does not transfer to real
  skills. Facts are trivially solved by any store. Neither tests the README claim ("the grader catches it and the
  system fixes itself") on a task where in-context instructions are not enough.

## 5. Over-built / cut
- The Kaggle receipt machinery (GPU UUID matching, two-T4 execution proofs, source digests) is heavy for a 1-seed trend
  run; keep it but stop adding to it. Hardware checks should not block a useful run.
- PLAN still carries 11 arms, sparse-memory co-lead, dreaming, merges, plastic links and E1-E10 before any evidence that
  weights beat a *fair* RAG. Freeze: plastic-link work (retrieval is at ceiling, it cannot be measured), sparse-memory
  arm, dreaming and merge, until a fair-baseline rule result exists.
- Retrieval superiority gate on synthetic, ceiling-level data: retire it or replace it with a harder set.

## Verdict: CHANGE FIRST
1. Fair RAG/text_notes context budget (and inference input cap) so lessons fit; keep 64 only as a labelled extra.
2. Add hippo_only and expert-only (no-context) arms/evals.
3. Fresh world seed(s); >= 100 hidden items per rule; per-family scoring; drop idle cycles 4-5.
4. Pre-register the per-family stop rule in section 2 (with CI), separating "did not fit" from "did not beat RAG".
5. Relax or report the zero-tolerance replay pre-gate so a second skill can be accepted.
6. Start ours first; do not serialize it behind the baselines.
All six are small code/config edits; with them the same ~5-10 GPU-hours answers the real question.
