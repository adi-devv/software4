Review-only task. Do not edit any file, run no git write, no Kaggle, no GPU.
Read docs/V2_PREREG_DRAFT.md in this repo. Check its claims against: step4/analyze.py (v1_verdict, claims 1-5, load_v1_run safety checks), step4/run.py (arms, sleep gate, ours_generic_only), baselines/, and the v1 results in output_v1_s44r/output/step4/full-146cd91286261a7e and output_v1_s45/output/step4/full-ad4be0d1780575fb (results.json, v1-verdict.md, answers.jsonl).
Answer briefly:
1. Is each factual statement in "Why v2" true? List any that are wrong or unsupported.
2. Is the claim-5 explanation right (Birch/Dover answered B-NORA/B-LENA because four rule-B worked examples were retrieved with the fact)? Check s45 ours/answers.jsonl.
3. Is the generic-forced-no-retrieval vs expert_only comparison with a margin M a valid test of format learning? Propose a reachable M from the v1 numbers and say how to fix it before data.
4. Per-rule non-inferiority: is the proposed design sound (margin, CI, multiple comparisons)?
5. What code changes are needed and which hashed files would change? Any leakage or confound the draft misses?
Give concrete edits to the draft. Max 60 lines.
