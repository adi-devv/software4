# Sol review: v2 CODE DIFF (not the plan)

Repo copy: D:\Aadit\claudez\software4-v2. Baseline = D:\Aadit\claudez\software4 (v1-frozen; do not edit it).
Review ONLY the code diff between the two trees (`diff -ru` them; skip .codex-jobs, output_*, scripts/jobs, results, data):
baselines/generic.py, baselines/sleep.py, baselines/semantic.py (new), step4/v2_analyze.py (new), step4/run.py, step4/prepare.py, and the new tests.
Do not review or comment on docs/V2_PREREG_DRAFT.md. Spec for intent: docs/V2_PREREG_DRAFT.md (read for requirements only).

Look for real defects: forced-generic arm really installs experts regardless of gate and logs actual generic fit; generic_forced_noret has no retrieval;
matched training counts/replay; rag_hashing/rag_semantic same top-k/budget/store policy, no gold skill metadata leak; v1 paths and defaults unchanged (v1 seeds 44-46 still validated);
v2 seeds 48-52 dispatch; v2_verdict math (paired bootstrap 10,000 draws, default_rng(0), sorted IDs, percentile 95% CI, M=0.30 pooled Claim 3, Claim 1 non-inferiority >= -5 and CI low >= -10 vs semantic RAG, hashing RAG, hippo_only for A,B,C,pooled, >=2 valid seeds);
answer classes (correct, stale, format_corrupted, absorbed, unusable; semantic absorption catches B-Zorg; abstention must not pass); packaging/hash allowlists include new files; semantic embedder pin/sha check.
Known open items: analyzer must include `rag_starved` in report/format-leak summaries; strict count/replay matching after differing gate decisions; real MiniLM weights never executed; one pre-existing failing test test_random_hybrid_targets_cpu (Transformers lacks Qwen3_5TextConfig).

Run the CPU tests with Python 3.12 (-I) if useful. Do NOT edit any file. Output a findings list ranked by severity: file:line, defect, concrete failing scenario. Say "no findings" where none.
