Repo: D:/Aadit/claudez/software4, branch step4b. READ-ONLY review: do NOT edit, create or commit any file. Review commit 22f8001 (`git show 22f8001 -- baselines step4 ':!step4/README.md' ':!step4/RESULTS.md'`) against this intent:
1. render_entry (baselines/arms.py): memory arms show "Question + Verified answer", never the model's wrong Response; stored episode text and SleepArm._example parsing unchanged.
2. SleepArm (baselines/sleep.py): a routed expert gets an EMPTY context (matches training); route "none" gets retrieved memory; the gate and validation loops use the prompt actually deployed; _pending records retrieved ids only when memory was used.
3. step4/run.py: answers.jsonl logging must not change scoring or records.jsonl; duplicate (task_id, day) lines after a resume.
4. step4/analyze.py: correctness of accuracy, copy-rate, and paired bootstrap.
Also check: any way these changes leak hidden-test answers into prompts or training, or advantage `ours` unfairly vs rag/hippo_only (the stop rule compares them), and any path where answer_trace is stale from a previous call.
Report only real bugs or fairness risks, ranked, each with file:line and a concrete failing scenario. If none, say so. Max 15 lines.
