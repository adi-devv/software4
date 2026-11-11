# Job: V1 critic eligibility count (CPU only, no GPU, no model loading)
Repo: D:/Aadit/claudez/software4 (read-only for you except new files under D:/Aadit/claudez/software4/scripts/analysis_d/). Do not edit tracked files, no git commands that change state.

Read docs/V1_CRITIC_PLAN_DRAFT.md sections "Strata" and "Eligibility gate". A group = (seed, rule, cycle); partly-learned = accuracy between 20% and 80% inclusive. Gate: >= 6 partly-learned groups over 3 seeds, and pooled >= 100 successes and >= 100 failures in those groups.

Data:
- step-4 v1 runs: D:/Aadit/claudez/software4/output_v1_s44r, output_v1_s45, output_v1_s46 (each has output/step4/full-*/results.json, v1-verdict.json; maybe more). Use the arm "ours" (also report expert_only if present).
- step-4 v3 runs: C:/Users/aadit/v3_out/s53, s54, s55 (step4/full-*/<arm>/answers.jsonl, records.jsonl, result.json). Arms: ours, expert_only.
Find how correctness is scored per item (look at step4 code in the repo: grep for the scorer used by results.json, case-folded exact match of 'Final answer:' line). Reuse the repo's own scorer function if importable; else replicate and say so.

Write scripts/analysis_d/elig_count.py, run it, and produce a table per (dataset, seed, arm, rule, cycle): n, accuracy, partly-learned yes/no. Then the gate result per dataset/arm. If per-item data is missing for v1, use whatever group-level accuracy exists and say exactly what was used.

Output (markdown, short): the table, the gate verdict per dataset/arm, and one line: is the V1 critic runnable on existing data? Note any assumption.
