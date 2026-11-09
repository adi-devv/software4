Review-only task. Do not edit any file, no git write, no Kaggle, no GPU.
docs/V2_PREREG_DRAFT.md was just updated (re-read it). Two numbers are now LOCKED by the user: (a) M = 30 points pooled A/B/C for expert_only minus generic_forced_noret; (b) per-rule and pooled non-inferiority vs best RAG: paired difference >= -5 points AND 95% CI lower bound >= -10 points.
Using v1 numbers in output_v1_s44r/output/step4/full-146cd91286261a7e and output_v1_s45/output/step4/full-ad4be0d1780575fb (results.json, answers.jsonl), answer in max 20 lines:
1. Is the updated text internally consistent and unambiguous (which CI method, paired how, which RAG arm is "best", how ties and multiple comparisons are handled)? Give exact wording fixes.
2. Is M = 30 reachable? Flag it only if clearly not; do not change it.
3. Would the -5/-10 rule have passed or failed on s44 and s45 per rule (ours vs best of rag/rag_hashing arms present)? One line per seed and rule.
