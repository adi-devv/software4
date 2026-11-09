Your working tree is D:/Aadit/claudez/software4-arms (git worktree, branch `arms`, uncommitted + one commit 977c459). NEVER read-write or run anything in D:/Aadit/claudez/software4 or the other software4-* folders. Do not commit; leave changes in the working tree.

Read first: baselines/dream.py, baselines/generic.py, baselines/tests/test_dream.py, test_generic.py, step4/run.py (make_arm).

Problem: `ours_generic_data` is the control for `ours_dream` (does the gain come from consolidation, or just from extra training examples?). Right now the number of generic examples added per cycle is NOT guaranteed to equal the number of dream variants that survive verification in the same cycle, because the two arms run as independent runs and dream's kept-count depends on the model. Fix this so the control is honest:
- The generic arm must add, per sleep cycle, exactly the count the dream arm would add given the same admitted episodes. Make the count a deterministic function of the admitted episodes alone (e.g. a fixed number of variants per admitted episode, capped by cfg.max_examples), NOT of model-dependent verification outcomes; in dream, unverified variants must then be replaced or the count rule must be defined on candidates generated, so both arms are provably equal. Pick the simplest rule that makes equality hold by construction, and document it in both docstrings.
- Add a test that runs both arms on the fake backend over the same episodes and asserts equal extra-example counts every cycle, including a case where verification drops some dream variants.
- Keep existing arms, default ARMS and stop rule unchanged.
Run `python -m pytest baselines step4 harness -q -x`. Final message: what rule you chose, files changed, test counts, anything unverified.
