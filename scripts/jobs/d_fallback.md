# Job: does v3's ours-vs-rag_semantic edge come from the ours-only fallback? (CPU only)
Repo: D:/Aadit/claudez/software4 (main) and D:/Aadit/claudez/software4-v3-git (v3 code). Write only new files under D:/Aadit/claudez/software4/scripts/analysis_d/. Don't edit tracked files.

Context: v3 reported ours 94.17% vs rag_semantic 92.41%, diff +1.76% [0.00, 3.61] pooled over 1080 rule items (step4/RESULTS.md ~line 191-215). In software4-v3-git/baselines/sleep.py (~lines 140-165) arm 'ours' falls back to a retrieval prompt without the expert when a trigger fires (FALLBACK_TRIGGER_K in baselines/v3.py). Only 'ours' has this.

Data: C:/Users/aadit/v3_out/s53, s54, s55 -> step4/full-*/<arm>/answers.jsonl (+ records.jsonl, result.json). Answers traces may have fallback=True or prompts=[prompt, fallback].

Do:
1. Identify per final-cycle eval item whether ours used the fallback. Use the same final-cycle item set and the same scorer as the verdict (find the verdict/analysis script in the v3 repo and reuse it).
2. Report per seed and pooled: fallback rate; ours accuracy on fallback vs non-fallback items; expert_only and rag_semantic accuracy on the same items.
3. Counterfactual: ours with fallback items scored using expert_only's answer on that item (= no fallback). Recompute ours - rag_semantic with the same paired bootstrap / CI method as the verdict.
4. Also: on fallback items, does ours's answer equal rag_semantic's answer? (fallback ~= RAG?)
Write scripts/analysis_d/fallback_check.py and run it.

Output (markdown, short): numbers table + a 2-line verdict: is the +1.76 mainly from the fallback (i.e., the expert alone is below RAG)? State any assumption.
