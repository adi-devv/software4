# Step 0b results: can we stop the forgetting? (2026-11-06)

Script: `step0b/step0b.py`, run on Kaggle with `scripts/kaggle_run.sh step0b`, on one of the 2 T4s. Raw output: `step0b/results.json` (run 3) and `step0b/results_run2.json` (memory layer).

## Verdict: yes, on this toy test. Replay stops the forgetting and the router stops experts interfering.

GSM8K (school math, 100 questions) stands in for old skills. The base model scores 70%.

| Setup | New skill | GSM8K | Without replay or router (step 0) |
|---|---|---|---|
| Expert A + replay | Skill A **94%** | **74%** | A 90%, GSM8K 10% |
| Expert B + replay | Skill B **76%** | **71%** | B 88% |
| Both experts on at once | A 95%, B **0%** | 72% | A 53%, B 82%, GSM8K 8% |
| Both experts + router | A **94%**, B **76%** | **70%** | Not tested |
| Memory layer + replay (run 2) | Skill A **82%** | **77%** | A 83%, GSM8K 64% |

Router accuracy was 100% on all three question types.

## Lessons
1. **Replay works.** GSM8K stays at 71–74% instead of falling to 10%.
2. **The router is needed.** With both experts on, skill B drops to 0%. With the router, both skills keep their solo scores.
3. **Experts learned skill A better than the memory layer** (94% vs 82%). Both kept GSM8K about equally well (74% vs 77%, within noise).
4. **Missed target:** we wanted both skills near 90%, but skill B reached 76%. It was 88% without replay, so replay may have cost it some learning. This is one run, so the cause is unknown.

## Caveats: read before relying on these
- **Replay may look better than it would in real use.** Half the replay set is GSM8K training questions, and the test is GSM8K too. Old skills not in the replay set could fare worse.
- **The router test was easy.** The two toy skills and math look nothing like each other. Similar tasks would be harder to separate.
- **Routed scores are not new measurements of forgetting.** The router sent every math question to no expert, so 70% is just the base model. Likewise, routed A and B equal the solo scores.
- **Not a fair contest yet.** The memory layer learned one skill with no router, in a different run, with different settings (3 epochs at lr 1e-3 vs 2 epochs at lr 2e-4).
- One seed and 100 GSM8K questions, so differences under about 9 points could be noise. Toy skills only.

## Run log
| Run | Time used | Result |
|---|---|---|
| 1 | 20 min | Out of memory on all training |
| 2 | 37 min | Memory layer valid. Experts ran out of memory, so their router numbers are invalid |
| 3 | 47 min | Experts and router valid, no NaN steps, 9.25 GB peak. Micro-batches of 2, length capped at 384 |

Step 0b used 1.7 GPU-hours. The week's total is about 2.2, plus step 0's crashed first run, whose time wasn't recorded.
