# Pilot results log (build step 1b)

All times UTC. Outputs: `pilot/output-smoke/` (smoke), `pilot/output/` (latest run).

## Run 1: smoke (S4_SMOKE=1), 2026-11-07, started 07:29 UTC, T4 x2
- Finished in about 9 minutes (script wall time 542 s), status COMPLETE.
- 2 training steps per sleep, 2 sleeps, 8 test and 4 gate items per skill. Not a quality test.
- Checks that passed: no NaN steps, older experts unchanged, parameters finite, router picks the right expert (8/8 on skill, 16/16 on none), gate logic runs, replay builds, all three text peers run, driver correctly refuses go/no-go with only 2 sleeps.
- Peak GPU memory: experts 9.22 GB, memory layer 7.81 GB, peers 4.61 GB (T4 has 15 GB).
- Scores are near zero as expected after 2 steps (skill 0.0 everywhere, GSM8K 0.125, ARC 0.625).
- Oddity: the oracle peer (rule text in the prompt) scored 0.0 on 8 items for both skills, even for the easy "quillon" rule. The full run now saves 3 oracle samples per skill so we can tell whether it is a prompt or grader problem or just a weak 2B model.
- Not fixed (known): replay answers cut at 256 tokens; 30-item gate is noisy; ARC fallback regex counts a stray "A"; peer GSM/ARC canaries are meaningless; ~1 GB model download.

## Run 2: full (S4_SMOKE=0), started 2026-11-07 07:44 UTC, finished about 08:27 UTC (13:14 to 13:57 IST), T4 x2
- Total 2550 s (about 42.5 min). Baseline 218 s, replay 343 s. Peak GPU memory: experts 9.32 GB, memory layer 7.76 GB, peers 5.98 GB, baseline 4.37 GB.
- **Driver verdict: GO for the experts arm** (`go_arms: [experts]`). Canaries unchanged.
- Experts arm after 3 sleeps: skill 1 = 91%, skill 2 = 70%, skill 3 = 48%. Best text peer (RAG) = 7.33% on the same skills, so experts win by 62 points. RAG per cycle: s1 16%, s2 5%, s3 1%. Text notes scored 0 (the notes it wrote were wrong rules).
- Memory-layer arm: failed. The gate rejected sleeps 2 and 3, so it only kept skill 1 (and dropped 20 then 40 points on it). Not a GO.
- Oracle peer scored 0% on all three skills. Its samples show the 2B model answering the right items in the wrong order, so this looks like a weak model on hard rules, not a grader bug. Still unverified.
- Caveats: tiny tasks, one seed, 30-item gate is noisy, router `per_class` was not saved. Treat as a green light to continue, not as evidence for the claim.
