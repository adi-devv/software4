Code review from the text below only. Do NOT run any commands and do not edit files; everything you need is inline.
Task given to the builder: three small fixes plus additive cost logging. (1) baselines/arms.py: a memory item with an empty correction must not put the raw wrong Response back in the prompt. (2) step4/analyze.py: the copy-rate check must also look at `Fact:` lines. (3) step4/run.py: a failure writing answers.jsonl must not abort scoring (count it in the worker result instead). (4) step4/analyze.py: report per-arm worker seconds, GPU-seconds, accuracy points per worker-hour, total worker-hours. Also a "4c" note in step4/RESULTS.md saying results are not comparable with the seed-41 run. Stop-rule thresholds must not change.
Check for: correctness bugs; the fixes not actually fixing the stated problem; behaviour changes beyond the task; the analyzer crashing on old outputs that lack the new fields; division by zero in cost metrics; swallowing errors too broadly in run.py (e.g. hiding a real scoring error); weak or missing tests.
Report at most 8 findings, most severe first, each with file:line, failure scenario, one-line fix. Say clearly if nothing blocks merging.

=== git diff of the fixes worktree ===
diff --git a/baselines/arms.py b/baselines/arms.py
index c9c1e1b..e0ee864 100644
--- a/baselines/arms.py
+++ b/baselines/arms.py
@@ -24,9 +24,7 @@ def render_entry(text):
         return text
     question, response, correct, correction = match.groups()
     value = parse_answer(response) if correct == 'True' else parse_answer(correction, correction=True)
-    if not question.strip() or not value:
-        return text
-    return 'Question: ' + question + '\nVerified answer: ' + value
+    return 'Question: ' + question + ('\nVerified answer: ' + value if value else '')
 
 
 def build_prompt(llm, task, entries, context_tokens, order=None):
diff --git a/baselines/tests/test_step4b.py b/baselines/tests/test_step4b.py
index 3c0e508..3eb86de 100644
--- a/baselines/tests/test_step4b.py
+++ b/baselines/tests/test_step4b.py
@@ -10,6 +10,8 @@ def task(mode='day'):
 def test_render_entries():
     wrong = 'Question: Q\nResponse: WRONG\nCorrect: False\nCorrection: Correct answer: yes. lesson'
     assert render_entry(wrong) == 'Question: Q\nVerified answer: yes'
+    assert render_entry(wrong.replace('Correct answer: yes. lesson', '')) == 'Question: Q'
+    assert render_entry(wrong.replace('False', 'None').replace('Correct answer: yes. lesson', '')) == 'Question: Q'
     assert render_entry(wrong.replace('False', 'None').replace('Correct answer: yes. lesson', 'Final answer: yes')) == 'Question: Q\nVerified answer: yes'
     assert render_entry('Question: Q\nResponse: reasoning\nFinal answer: yes\nCorrect: True\nCorrection: ') == 'Question: Q\nVerified answer: yes'
     assert render_entry('Fact: hello') == 'Fact: hello'
diff --git a/step4/RESULTS.md b/step4/RESULTS.md
index 055aa83..05405e5 100644
--- a/step4/RESULTS.md
+++ b/step4/RESULTS.md
@@ -1,5 +1,9 @@
 # Step 4 run ledger
 
+## Step 4c
+
+**Status: code done, needs fresh smoke.** The review fixes change the config hash and the behavior of ours on empty corrections. Results from this branch are not comparable with the seed-41 4b run. The pre-registered stop-rule thresholds are unchanged.
+
 Smoke debugging runs v1-v7 and one full run (2026-11-07/08 IST). The first full run FAILED the stop rule; the 4b full run PASSED it on one seed (see the table).
 
 | Run | Time used | Result | Weekly GPU-hours |
diff --git a/step4/analyze.py b/step4/analyze.py
index cc50944..bc759ee 100644
--- a/step4/analyze.py
+++ b/step4/analyze.py
@@ -18,6 +18,7 @@ from harness.results import load_records
 RESPONSE = re.compile(r'^\s*Response:\s*(.*?)(?=^\s*(?:Correct|Question|Verified answer|Response|Correction):|\Z)', re.I | re.M | re.S)
 VERIFIED = re.compile(r'^\s*Verified answer:\s*(.*?)\s*$', re.I | re.M)
 CORRECT = re.compile(r'^\s*Correct:\s*(False|True|None)\s*$', re.I | re.M)
+FACT = re.compile(r'^\s*Fact:[ \t]*(.*)$', re.I | re.M)
 
 
 def context_matches(prompt, extracted):
@@ -35,9 +36,50 @@ def context_matches(prompt, extracted):
         value, _ = _extract(m.group(1))
         if value is not None and value.strip() == extracted.strip():
             copied = verified = True
+    for m in FACT.finditer(prompt):
+        if re.search(r'(?<!\w)' + re.escape(extracted.strip()) + r'(?!\w)', m.group(1), re.I):
+            copied = True
     return copied, wrong, verified
 
 
+def read_json(path):
+    try:
+        return json.loads(path.read_text(encoding='utf-8'))
+    except (OSError, ValueError):
+        return {}
+
+
+def cost_stats(run_dir, arms):
+    timings = read_json(run_dir / 'timings.json')
+    results = read_json(run_dir / 'results.json')
+    seconds = timings.get('arms_seconds', {})
+    saved = results.get('results', {})
+    costs = {}
+    for arm, data in arms.items():
+        worker = read_json(run_dir / arm / 'result.json') or saved.get(arm, {})
+        elapsed = seconds.get(arm, worker.get('elapsed_seconds'))
+        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
+            elapsed = None
+        # Each HF worker is assigned exactly one device; CPU workers use none.
+        backend = results.get('config', {}).get('backend')
+        devices = 1 if worker.get('worker_device') else 0 if backend in ('fake', 'tiny') else None
+        hours = elapsed / 3600 if elapsed is not None else None
+        costs[arm] = {'worker_seconds': elapsed, 'devices': devices,
+                      'gpu_seconds': elapsed * devices if elapsed is not None and devices is not None else None,
+                      'accuracy_points_per_worker_hour': data['accuracy'] * 100 / hours
+                      if hours and data['accuracy'] is not None else None}
+    extra = set(seconds) | set(saved)
+    for arm in extra - set(costs):
+        worker = read_json(run_dir / arm / 'result.json') or saved.get(arm, {})
+        elapsed = seconds.get(arm, worker.get('elapsed_seconds'))
+        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0:
+            costs[arm] = {'worker_seconds': elapsed, 'devices': None, 'gpu_seconds': None,
+                          'accuracy_points_per_worker_hour': None}
+    known = [v['worker_seconds'] for v in costs.values() if v['worker_seconds'] is not None]
+    return {'arms': costs, 'total_worker_hours': sum(known) / 3600 if known else None,
+            'total_is_partial': len(known) < len(costs)}
+
+
 def answer_stats(path, rows, cycle):
     last = {}
     with path.open(encoding='utf-8') as f:
@@ -107,6 +149,7 @@ def analyze(run_dir, cycle=None):
             pair = paired_bootstrap(rules[first], rules[second])
             if pair:
                 result['pairwise_rule'].append({'first': first, 'second': second, **pair})
+    result['cost'] = cost_stats(run_dir, result['arms'])
     return result
 
 
@@ -118,7 +161,16 @@ def markdown(result):
         lines.append(f"| {arm} | overall | {data['n']} | {pct(data['accuracy'])} |")
         for skill, stats in data['skills'].items():
             lines.append(f"| {arm} | {skill} | {stats['n']} | {pct(stats['accuracy'])} |")
-    lines += ['', '## Rule comparisons (first minus second)', '', '| First | Second | Paired N | Difference | 95% CI |', '|---|---|---:|---:|---:|']
+    def number(x):
+        return 'N/A' if x is None else f'{x:.2f}'
+    cost = result['cost']
+    lines += ['', '## Worker time and cost proxy', '',
+              'Accuracy points per worker-hour uses overall hidden accuracy for this cycle; GPU-seconds count one T4 per assigned worker. This is not a billing estimate.', '',
+              '| Arm | Worker seconds | Devices | GPU-seconds | Accuracy points / worker-hour |', '|---|---:|---:|---:|---:|']
+    for arm, c in cost['arms'].items():
+        lines.append(f"| {arm} | {number(c['worker_seconds'])} | {number(c['devices'])} | {number(c['gpu_seconds'])} | {number(c['accuracy_points_per_worker_hour'])} |")
+    lines += ['', f"Total worker-hours: {number(cost['total_worker_hours'])}" + (' (partial; some timings missing)' if cost['total_is_partial'] else ''),
+              '', '## Rule comparisons (first minus second)', '', '| First | Second | Paired N | Difference | 95% CI |', '|---|---|---:|---:|---:|']
     for p in result['pairwise_rule']:
         lines.append(f"| {p['first']} | {p['second']} | {p['n']} | {pct(p['difference'])} | [{pct(p['ci_low'])}, {pct(p['ci_high'])}] |")
     for arm, data in result['arms'].items():
diff --git a/step4/run.py b/step4/run.py
index e6cfbc1..2438275 100644
--- a/step4/run.py
+++ b/step4/run.py
@@ -372,7 +372,7 @@ def worker(name, cfg, run_dir, smoke):
     out.mkdir(parents=True, exist_ok=True)
     started = time.perf_counter()
     result = dict(arm=name, status="failed", started_utc=datetime.now(timezone.utc).isoformat(),
-                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
+                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), answer_log_failures=0)
     try:
         if cfg["backend"] == "hf":
             import torch
@@ -420,9 +420,13 @@ def worker(name, cfg, run_dir, smoke):
                                skill=task.skill, expert=trace.get("expert"),
                                retrieved=trace.get("retrieved", []), prompt=trace.get("prompt"),
                                answer=response)
-                    with (out / "answers.jsonl").open("a", encoding="utf-8") as log:
-                        log.write(json.dumps(row) + "\n")
-                        log.flush()
+                    try:
+                        serialized = json.dumps(row) + "\n"
+                        with (out / "answers.jsonl").open("a", encoding="utf-8") as log:
+                            log.write(serialized)
+                            log.flush()
+                    except (OSError, TypeError, ValueError, OverflowError):
+                        result["answer_log_failures"] += 1
                 if task.mode == "test" and task.day >= 0 and task.id not in canary_ids:
                     tally = parse.setdefault(task.day, dict(answers=0, parse_failures=0))
                     tally["answers"] += 1
diff --git a/step4/tests/test_analyze.py b/step4/tests/test_analyze.py
index ce6ee5b..5a1451e 100644
--- a/step4/tests/test_analyze.py
+++ b/step4/tests/test_analyze.py
@@ -2,6 +2,7 @@
 import subprocess
 import sys
 from pathlib import Path
+from step4.analyze import analyze, context_matches, markdown
 
 
 def write_jsonl(path, rows):
@@ -48,3 +49,30 @@ def test_analysis_cli(tmp_path):
     pair = data['pairwise_rule'][0]
     assert pair['n'] == 4 and pair['difference'] == -0.25 and pair['draws'] == 10000
     assert 'rule:B' in run.stdout and (tmp_path / 'analysis.md').is_file()
+
+
+def test_fact_copy_matches_whole_values():
+    prompt = 'Context:\nFact: Field note: the leader of Alder is Blue.\n\nTask:\nBlue'
+    assert context_matches(prompt, 'Blue') == (True, False, False)
+    assert context_matches(prompt, 'Blu') == (False, False, False)
+    assert context_matches('Task:\nFactless Blue', 'Blue') == (False, False, False)
+
+
+def test_costs_from_artifacts_and_missing_values(tmp_path):
+    for arm in ('ours', 'rag'):
+        folder = tmp_path / arm
+        folder.mkdir()
+        write_jsonl(folder / 'records.jsonl', [dict(type='score', arm=arm, stream='s', cycle=0,
+                    attempt=1, phase='hidden', task_id='x', skill='fact:smoke', correct=True),
+                    dict(type='cycle_done', arm=arm, stream='s', cycle=0, attempt=1)])
+    (tmp_path / 'timings.json').write_text(json.dumps({'arms_seconds': {'ours': 1800}}))
+    (tmp_path / 'ours' / 'result.json').write_text(json.dumps({'worker_device': {'scheduled': {}}}))
+    (tmp_path / 'rag' / 'result.json').write_text(json.dumps({'elapsed_seconds': 900}))
+    (tmp_path / 'results.json').write_text(json.dumps({'config': {'backend': 'fake'}}))
+    result = analyze(tmp_path)
+    assert result['cost']['arms']['ours'] == {'worker_seconds': 1800, 'devices': 1, 'gpu_seconds': 1800, 'accuracy_points_per_worker_hour': 200}
+    assert result['cost']['arms']['rag']['gpu_seconds'] == 0
+    assert result['cost']['total_worker_hours'] == 0.75
+    assert 'Accuracy points / worker-hour' in markdown(result)
+    (tmp_path / 'rag' / 'result.json').unlink()
+    assert analyze(tmp_path)['cost']['total_is_partial']
diff --git a/step4/tests/test_run_cpu.py b/step4/tests/test_run_cpu.py
index b444316..ad939e9 100644
--- a/step4/tests/test_run_cpu.py
+++ b/step4/tests/test_run_cpu.py
@@ -86,6 +86,30 @@ def test_fake_full_and_reuse(cpu_dir):
     assert (out / "timings.json").is_file()
 
 
+@pytest.mark.parametrize("error", [OSError("disk full"), TypeError("bad JSON")])
+def test_answer_log_failure_does_not_abort_scoring(cpu_dir, monkeypatch, error):
+    monkeypatch.setenv("S4_BACKEND", "fake")
+    monkeypatch.setenv("S4_CYCLES", "3")
+    monkeypatch.setenv("S4_STEPS", "2")
+    cfg, _, _ = run.settings(ROOT)
+    original_open = Path.open
+    original_dumps = json.dumps
+    def failing_open(path, *args, **kwargs):
+        if path.name == 'answers.jsonl' and isinstance(error, OSError):
+            raise error
+        return original_open(path, *args, **kwargs)
+    def failing_dumps(value, *args, **kwargs):
+        if isinstance(error, TypeError) and isinstance(value, dict) and 'task_id' in value and 'answer' in value:
+            raise error
+        return original_dumps(value, *args, **kwargs)
+    monkeypatch.setattr(Path, 'open', failing_open)
+    monkeypatch.setattr(json, 'dumps', failing_dumps)
+    assert run.worker('no_memory', cfg, cpu_dir, False) == 0
+    result = json.loads((cpu_dir / 'no_memory' / 'result.json').read_text(encoding='utf-8'))
+    assert result['status'] == 'complete' and result['answer_log_failures'] > 0
+    assert result['scores'] and result['parse_by_cycle']
+
+
 def test_smoke_trains_only_longest_twice(cpu_dir):
     proc, payload, out = invoke(cpu_dir, smoke="1")
     assert proc.returncode == 0, proc.stdout + proc.stderr
