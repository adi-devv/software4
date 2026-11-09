Code review from the text below only. Do NOT run any commands and do not edit files; everything you need is inline.
Spec: PLAN.md row E11 and the task described below.
Task given to the builder: add opt-in arm `ours_salient`, a SleepArm subclass overriding only `_training_set`: corrected failures weight 2, successes 1, cap 3 copies per episode and 50% of examples per rule/skill, total <= cfg.max_examples, trusted and verified episodes only, no surprise scoring. Register it in step4/run.py only when selected via S4_ARMS; the default ARMS list, the other arms, the stop rule and the config/source digest behaviour for default runs must stay unchanged.
Check for: correctness bugs; any change to default behaviour or to existing arms' identity/hash; untrusted or unverified or hidden/gate data leaking into training; caps violated (per-episode, per-skill, total); determinism (stable ordering, no randomness without seed); the "only one skill in a batch gives an empty training set" edge case (is returning nothing right, or should it fall back?); weak or missing tests.
Report at most 8 findings, most severe first, each with file:line, the failure scenario, and a one-line fix. Say clearly if nothing blocks merging.

=== DIFF of step4/run.py and step4/tests/test_run_cpu.py, then full text of baselines/salient.py and baselines/tests/test_salient.py ===
diff --git a/step4/run.py b/step4/run.py
index e6cfbc1..a4af34f 100644
--- a/step4/run.py
+++ b/step4/run.py
@@ -28,6 +28,7 @@ DEFAULT_WORLD_SEED = ""   # "" keeps the harness default (41); prepare.py --worl
 # `ours` first, then the other long-running weight arm; the inference-only baselines follow.
 # Nothing waits for anything else: `ours` is the long pole and must start immediately.
 ARMS = ("ours", "expert_only", "hippo_only", "rag", "text_notes", "no_memory", "rag_starved", "oracle")
+AVAILABLE_ARMS = ARMS + ("ours_salient",)
 FAIR_CONTEXT_TOKENS = 512    # every context-using arm; `rag_starved` alone keeps the old 64
 STARVED_CONTEXT_TOKENS = 64
 FIT_THRESHOLD = 0.9          # forced-expert train accuracy below this = optimisation failure
@@ -77,8 +78,8 @@ def settings(root):
     steps = int(os.environ.get("S4_STEPS", "64"))
     if backend not in ("fake", "tiny", "hf") or not arms or len(set(arms)) != len(arms):
         raise ValueError("invalid backend or empty/duplicate S4_ARMS")
-    if set(arms) - set(ARMS) or cycles < 3 or steps < 1:
-        raise ValueError("S4_ARMS must use " + ",".join(ARMS) + "; cycles >= 3; steps >= 1")
+    if set(arms) - set(AVAILABLE_ARMS) or cycles < 3 or steps < 1:
+        raise ValueError("S4_ARMS must use " + ",".join(AVAILABLE_ARMS) + "; cycles >= 3; steps >= 1")
     model = os.environ.get("S4_MODEL", "Qwen/Qwen3.5-2B")
     revision = os.environ.get("S4_REVISION", QWEN35_REVISION if model == "Qwen/Qwen3.5-2B" else "main")
     from harness.step4_world import WORLD_SEED
@@ -268,6 +269,9 @@ def make_arm(name, backend, cfg, Oracle):
     sleep_cfg = SleepConfig(steps=cfg["steps"], seed=0, context_tokens=budget)
     if name == "ours":
         return SleepArm(backend, sleep_cfg, llm=backend)
+    if name == "ours_salient":
+        from baselines.salient import SalientSleepArm
+        return SalientSleepArm(backend, sleep_cfg, llm=backend)
     if name == "expert_only":
         return ExpertOnly(backend, sleep_cfg, llm=backend)
     if name == "rag_starved":
@@ -309,7 +313,7 @@ def smoke_arm(name, arm, backend, cfg, stream, canary, measure, prompt_for, out)
                            + json.dumps(wrong, ensure_ascii=True))
     longest = None
     expert = None
-    if name in ("ours", "expert_only"):
+    if name in ("ours", "ours_salient", "expert_only"):
         n, prompt, target, tid = longest_example(stream, backend, prompt_for)
         # HFBackend accumulates FOUR microbatches per optimizer step. Repeating
         # the single longest actual example prevents random shorter selections.
@@ -730,7 +734,7 @@ def report(run_dir, cfg, arms, results, smoke, wall, hardware=None):
 
 def main():
     parser = argparse.ArgumentParser(description=__doc__)
-    parser.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
+    parser.add_argument("--worker", choices=AVAILABLE_ARMS, help=argparse.SUPPRESS)
     args = parser.parse_args()
     root = source_root()
     sys.path.insert(0, str(root))
diff --git a/step4/tests/test_run_cpu.py b/step4/tests/test_run_cpu.py
index b444316..c148307 100644
--- a/step4/tests/test_run_cpu.py
+++ b/step4/tests/test_run_cpu.py
@@ -86,6 +86,17 @@ def test_fake_full_and_reuse(cpu_dir):
     assert (out / "timings.json").is_file()
 
 
+def test_salient_is_opt_in_and_runs(cpu_dir, monkeypatch):
+    assert run.ARMS == ("ours", "expert_only", "hippo_only", "rag", "text_notes",
+                        "no_memory", "rag_starved", "oracle")
+    monkeypatch.delenv("S4_ARMS", raising=False)
+    assert run.settings(ROOT)[1] == list(run.ARMS)
+    proc, payload, out = invoke(cpu_dir, arms="ours_salient")
+    assert proc.returncode == 0, proc.stdout + proc.stderr
+    assert payload["results"]["ours_salient"]["status"] == "complete"
+    assert "ours_salient" in (out / "RESULTS.md").read_text(encoding="utf-8")
+
+
 def test_smoke_trains_only_longest_twice(cpu_dir):
     proc, payload, out = invoke(cpu_dir, smoke="1")
     assert proc.returncode == 0, proc.stdout + proc.stderr
"""Opt-in error-weighted replay over the standard trusted selection."""
from collections import Counter

from .sleep import SleepArm


class SalientSleepArm(SleepArm):
    name = 'ours_salient'

    def _training_set(self, cycle):
        new, replay = super()._training_set(cycle)
        # Keep the parent's admission and selection, including its replay quota.
        # There is no cheap correctness hook: generating again here would add
        # inference cost, so only the recorded corrected-failure signal is used.
        candidates = [(pair, False) for pair in new] + [(pair, True) for pair in replay]
        expanded = list(candidates)
        expanded += [entry for entry in candidates if entry[0][0].kind == 'corrected_failure']

        def skill(entry):
            item = entry[0][0]
            return self.hippo.get(item.episode_id)['meta'].get('skill')

        # Trimming the tail retains the original selection before extra copies.
        expanded = expanded[:self.cfg.max_examples]
        while expanded:
            counts = Counter(skill(entry) for entry in expanded)
            dominant, count = counts.most_common(1)[0]
            if 2 * count <= len(expanded):
                break
            index = next(i for i in range(len(expanded) - 1, -1, -1)
                         if skill(expanded[i]) == dominant)
            expanded.pop(index)
        return ([pair for pair, old in expanded if not old],
                [pair for pair, old in expanded if old])
from collections import Counter

from baselines import FakeBackend, SleepConfig
from baselines.salient import SalientSleepArm
from harness.tasks import Task


def add(arm, skill, number, *, success=False, source='trusted'):
    task = Task(f'{skill}-{number}', f'Question {skill} {number}?', 'fact', skill,
                mode='day', kind='probe', source=source)
    arm.observe(task, 'Final answer: yes' if success else 'Final answer: no',
                success, None if success else 'Correct answer: yes.')


def selected(arm):
    new, replay = arm._training_set(0)
    return new + replay


def test_weighting_caps_and_determinism():
    arm = SalientSleepArm(FakeBackend(), SleepConfig(max_examples=20, replay_size=0))
    for skill in ('A', 'B'):
        add(arm, skill, 0)
        add(arm, skill, 1, success=True)
    first = selected(arm)
    counts = Counter(item.task_id for item, _ in first)
    assert counts == {'A-0': 2, 'A-1': 1, 'B-0': 2, 'B-1': 1}
    assert max(counts.values()) <= 3
    assert first == selected(arm)
    assert len(first) <= arm.cfg.max_examples


def test_skill_cap_and_admission():
    arm = SalientSleepArm(FakeBackend(), SleepConfig(max_examples=8, replay_size=0))
    for i in range(5):
        add(arm, 'A', i)
    add(arm, 'B', 0, success=True)
    add(arm, 'B', 1, source='untrusted')
    # A recorded but unverified outcome cannot enter the parent selection.
    episode = arm.hippo.add('Question: Unverified?\nResponse: Final answer: yes\nCorrect: True',
                            kind='episode', source='system', meta={'skill': 'B'})
    arm.hippo.record_outcome('unverified', success=True, verified=False, episode_id=episode)
    result = selected(arm)
    counts = Counter(arm.hippo.get(item.episode_id)['meta']['skill'] for item, _ in result)
    assert result and max(counts.values()) * 2 <= len(result)
    assert all(item.task_id not in ('B-1', 'unverified') for item, _ in result)
    assert len(result) <= arm.cfg.max_examples
    assert max(Counter(item.episode_id for item, _ in result).values()) <= 3
