You are a reviewer. Do NOT modify, create or delete any tracked file; write nothing except your final message. Working tree: D:/Aadit/claudez/software4-staget (git worktree, branch stage-t, commit 8609469). Read only; change nothing; run no GPU, Kaggle or network jobs. If you run pytest use `-p no:cacheprovider --basetemp <fresh new folder>`; ignore the .pytest-* permission warnings.

Review for CORRECTNESS and for tests that prove nothing. Spec: the last section of step4/RESULTS.md ("v1 pre-registration, revision 2") and PLAN.md stage T. Code: harness/step5_world.py, baselines/gold_context.py, the TextNotes changes in baselines/arms.py, v1_verdict/load_v1_run in step4/analyze.py, the world/context_tokens wiring in step4/run.py and step4/prepare.py, and the tests (step4/tests/test_stage_t.py, baselines/tests/test_notes_repair.py, step4/tests/test_analyze.py, step4/tests/test_run_cpu.py).

Check specifically:
1. World validity: is every rule answer exactly what the lesson says; no code collisions with SmokeSource(0) or Step4Source seeds 41-43; train/hidden/gate counts; does rule C (introduced at the last cycle) get train, hidden and gate items and valid stream checks; fact items and revision/poison unchanged.
2. gold_context: leakage (never uses hidden or gate items; train-only demos), determinism, fails loudly over budget, fact tasks.
3. TextNotes: is the fallback note built only from question and verified answer (never the lesson field); is retrieval fair, not an oracle.
4. v1_verdict vs the pre-registration: validity gate, each of claims 1-5, "at least 2 of 3 valid seeds", pooling, paired bootstrap use, fail-closed on missing data. Report any claim that is implemented differently from the text, and any way a broken run could PASS.
5. context_tokens logging: does it measure the context block as claim 4 requires; do old receipts and arms still work.
6. Which tests would still pass if the code were wrong.

Output: a short ranked list of findings, each with file:line, the failing scenario, and a suggested fix. Say "no blocking issues" if there are none. Do not pad.
