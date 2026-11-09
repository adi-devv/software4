You are a code reviewer. Everything you need is in this message; you do NOT need to run commands or read files (your shell is broken, do not try it). Do not modify anything.
Review for CORRECTNESS and for tests that prove nothing. Check: (1) world validity: answers match lessons, no code collisions with SmokeSource(0)/Step4Source seeds 41-43, counts, rule C introduced at last cycle gets train/hidden/gate items; (2) gold_context leakage (train-only demos, deterministic, fails loudly over budget); (3) TextNotes fallback uses only question+verified answer, retrieval fair; (4) v1_verdict vs the pre-registration: validity gate, claims 1-5, >=2 valid seeds, fail-closed, any way a broken run could PASS or a claim implemented differently from the text; (5) context_tokens logging measures the context block (claim 4); old receipts/arms still work; (6) tests that would pass even if code were wrong.
Output: a short ranked list of findings with file:line, failing scenario, suggested fix. Say "no blocking issues" if none. Do not pad.

=== SPEC: v1 pre-registration, revision 2 (step4/RESULTS.md) ===
## v1 pre-registration, revision 2 (2026-11-08, before any v1 GPU run)

Supersedes the version in baa8e3b, which tested rule A (letter reversal). That rule is not followable from context by the 2B model, so a "win" on it could not be told apart from weights learning a skill context cannot teach, and the old pass rule (ours >= rag + 15) was met almost entirely by it. Stage T of PLAN.md replaces the test; this section is its pre-registration.

**World (new file, seeds 44, 45, 46; the old world stays for reproducibility).** All rules are template-copy rules the 2B follows from context (local Q4 calibration: 90-100% with 3 demos, 0% with none):

- A, Cobalt beacon (cycle 0): uppercase the code and add `-7` at the end.
- B, Amber orchard (cycle 1): uppercase the code and put 'B-' in front of it (changed 2026-11-08: the old bracket format let a bare echoed code pass the grader).
- C, Cobalt-Amber chain (cycle 2): apply rule A, then rule B to its result. It is a multi-step composition with its own train episodes, so the answer for `abcd` is `B-ABCD-7`.
- Six facts as before. 24 train, 120 hidden and 4 gate items per remaining cycle for each rule. No code appears in the old smoke or step-4 worlds.

**Baseline changes.** `gold_context` (renamed from `oracle`) gets the rule lesson plus 3 verified demos in "Question / Verified answer" form, because a lesson alone makes the 2B write a long reply that the 16-token budget truncates. `text_notes` is repaired: distillation lines that are bare "Final answer:" copies are dropped, a verified question-answer example replaces them when nothing usable is left, and retrieval depends on the query. The old 0/246 came from every prompt carrying the same four copied lines.

**Validity gate (checked before any claim).** In each seed, `gold_context` must score >= 85% on every rule's final hidden set and `no_memory` must score <= 10%. If either fails, the world is broken and that seed is void, not a result.

**What v1 tests, and what it does not.** With rules the model can follow from context, a fair RAG baseline is expected to sit near the ceiling, so v1 does not test "weights beat context". It tests that sleep-trained experts carry the rules at near-zero context cost and that the gain is rule learning rather than answer-format learning. Beating RAG needs a capacity-pressure world (many rules, a binding context budget) and is deferred, not claimed.

- **Seeds:** 44, 45, 46 (fresh).
- **Arms (8):** `ours`, `expert_only`, `rag`, `hippo_only`, `text_notes`, `no_memory`, `gold_context`, and the format-only control `ours_generic_only`.
- **Pass rule (all must hold, on seeds that pass the validity gate; at least 2 of 3 seeds must):**
  1. Non-inferiority: pooled over seeds and rules A-C, final hidden accuracy of `ours` minus the better of `rag` and `hippo_only` is >= -5 pts, paired-bootstrap CI low >= -10 pts.
  2. Rule learning: `ours` beats `no_memory` by >= 50 pts, CI low > 0.
  3. Not format learning: `ours` beats `ours_generic_only` by >= 30 pts, CI low > 0.
  4. Cost: mean context tokens in `ours` prompts <= 25% of those in `rag` prompts (measured, not assumed).
  5. Retention and safety: rule A (final) not down more than 5 pts from its cycle-0 value; the revised fact accepted; the untrusted poison fact rejected; no gate rollback caused by a trusted update.
- **Per-rule results (A, B, C) are always reported separately**, with composition (C) called out. If `ours` fails C while passing A and B, the claim is single-rule only.
- **If claim 3 fails:** the experts learn the format, not the rules; stop and rethink how experts are built. **If claim 1 fails:** weights do not carry the rules as well as context does; stop and rethink consolidation before V1 (critic). No extra seeds either way.
- **Not in v1:** `ours_online`, `dream`, `oracle_trained`, `merge`, `ours_generic_data`, `ours_salient`, the critic (stages V1 to V3).
- **Cost estimate:** one smoke plus three 8-arm full runs, about 1.5 worker-hours; re-estimate from the smoke timing before asking to spend.
- **Process:** Codex gpt-6.1-sol review of this section, the world and the verdict code first; then smoke; then v1 only after explicit approval.

=== DIFF (commit 8609469, includes new files) ===
diff --git a/baselines/arms.py b/baselines/arms.py
index 3af7ed5..b241d07 100644
--- a/baselines/arms.py
+++ b/baselines/arms.py
@@ -146,6 +146,7 @@ class TextNotes(Base):
     name = 'text_notes'
     def __init__(self, *args, max_notes=200, prune_after=3, **kwargs):
         self.max_notes, self.prune_after = max_notes, prune_after
+        self.embedder = HashingEmbedder()
         super().__init__(*args, **kwargs)
 
     def reset(self):
@@ -156,8 +157,12 @@ class TextNotes(Base):
         return self.retrieve(task.prompt)
 
     def retrieve(self, query, k=None):
-        """Read-only native note order; query is unused, as in context()."""
-        order = sorted(range(len(self.notes)), key=lambda i: (self.notes[i]['uses'], i), reverse=True)
+        """Rank notes by query similarity, then use count and recency."""
+        if not self.notes:
+            return []
+        vecs = self.embedder.embed([query] + [n["text"] for n in self.notes])
+        scores = vecs[1:] @ vecs[0]
+        order = sorted(range(len(self.notes)), key=lambda i: (float(scores[i]), self.notes[i]["uses"], i), reverse=True)
         if k is not None:
             order = order[:k]
         return [(i, self.notes[i]['text']) for i in order]
@@ -170,25 +175,32 @@ class TextNotes(Base):
         self.episodes.append(episode(task, answer, result, correction))
 
     def sleep(self, cycle):
-        header = 'Distill lessons:\nWrite short standalone lesson lines from these episodes and corrections.\n'
-        prompts, batch = [], []
+        header = ("Distill lessons:\nExample: Question: What color is snow? Verified answer: white -> Snow is white. Write a short standalone note for the episode.\n")
+        prompts = []
         for entry in self.episodes:
             if self.llm.count_tokens(header + entry) > self.llm.input_limit:
                 raise ValueError('distillation episode exceeds input_limit')
-            if batch and self.llm.count_tokens(header + '\n'.join(batch + [entry])) > self.llm.input_limit:
-                prompts.append(header + '\n'.join(batch))
-                batch = []
-            batch.append(entry)
-        if batch:
-            prompts.append(header + '\n'.join(batch))
+            prompts.append((header + entry, entry))
         before = len(self.notes)
         self.notes = [n for n in self.notes if n['uses'] < self.prune_after or n['successes'] * 2 >= n['uses']]
         pruned = before - len(self.notes)
-        for prompt in prompts:
-            lines = self.llm.generate(prompt).splitlines()
-            for line in lines:
+        for prompt, entry in prompts:
+            rendered = render_entry(entry)
+            match = re.search(r'^Verified answer: (.*)$', rendered, re.M)
+            answer = match.group(1) if match else ''
+            usable = []
+            for line in self.llm.generate(prompt).splitlines():
                 line = line.strip().lstrip('-* ').strip()
-                if line and line not in [n['text'] for n in self.notes]:
+                beyond = re.sub(re.escape(answer), '', line, flags=re.I) if answer else line
+                beyond = re.sub(r'\b(?:final|verified|correct|answer|response)\b', '', beyond, flags=re.I)
+                if (line and not line.lower().startswith('final answer')
+                        and re.search(r'[A-Za-z]', beyond)):
+                    usable.append(line)
+            if not usable and match and rendered.startswith('Question: '):
+                question = rendered.split('\n', 1)[0][len('Question: '):]
+                usable = [f'Example: {question} -> {answer}']
+            for line in usable:
+                if line not in [n['text'] for n in self.notes]:
                     self.notes.append({'text': line, 'uses': 0, 'successes': 0})
         if prompts:
             self.notes = self.notes[-self.max_notes:] if self.max_notes else []
diff --git a/baselines/gold_context.py b/baselines/gold_context.py
new file mode 100644
index 0000000..8a35793
--- /dev/null
+++ b/baselines/gold_context.py
@@ -0,0 +1,32 @@
+"""Fixed gold information baseline, built solely from lessons and train examples."""
+import random
+from .arms import NoMemory
+from harness.tasks import grading_scope
+from harness.smoke_world import gold_context as fact_context
+
+class GoldContext(NoMemory):
+    name = 'gold_context'
+
+    def __init__(self, *args, stream, **kwargs):
+        self.gold = {}
+        for skill in sorted(s for s in stream.skills if s.startswith('rule:')):
+            train = [i for c in range(stream.n_cycles) for i in stream.train_items(c)
+                     if i.kind == 'probe' and i.skill == skill]
+            if len(train) < 3:
+                raise ValueError('gold_context requires three train examples per rule')
+            train.sort(key=lambda i: i.key)
+            chosen = random.Random(f'{stream.seed}:{skill}').sample(train, 3)
+            lessons = [train[0].lesson]
+            if skill == 'rule:C':
+                from harness.step5_world import RULES
+                lessons += [RULES['A'], RULES['B']]
+            with grading_scope():
+                demos = [f'Question: {i.prompt}\nVerified answer: {i.expected_at(i.cycle)}' for i in chosen]
+            self.gold[skill] = '\n'.join(lessons + demos)
+        super().__init__(*args, **kwargs)
+
+    def context(self, task):
+        text = self.gold.get(task.skill, '') if task.skill.startswith('rule:') else fact_context(task)
+        if text and self.llm.count_tokens('Context:\n' + text) > self.context_tokens:
+            raise ValueError('gold_context exceeds context_tokens')
+        return [(0, text)] if text else []
diff --git a/baselines/tests/test_notes_repair.py b/baselines/tests/test_notes_repair.py
new file mode 100644
index 0000000..5e91c70
--- /dev/null
+++ b/baselines/tests/test_notes_repair.py
@@ -0,0 +1,28 @@
+from baselines import TextNotes, FakeLLM
+from harness.tasks import Task
+
+class Junk(FakeLLM):
+    def generate(self, prompt, max_new_tokens=256):
+        return 'Final answer: WHITE'
+
+def test_junk_fallback_and_similarity():
+    arm = TextNotes(Junk())
+    for i, (q, a) in enumerate([('What is the color of snow?', 'WHITE'), ('Who leads Alder?', 'SORA')]):
+        task = Task(str(i), q, 'fact', 'fact')
+        arm.observe(task, 'wrong', False, 'Correct answer: ' + a + '.')
+    arm.sleep(0)
+    assert len(arm.notes) == 2
+    assert all(n['text'].startswith('Example: ') for n in arm.notes)
+    assert all(not n['text'].startswith('Final answer') for n in arm.notes)
+    assert 'snow' in arm.retrieve('What is the color of snow?')[0][1]
+    assert 'Alder' in arm.retrieve('Who leads Alder?')[0][1]
+
+def test_no_lesson_attribute_access():
+    class TaskWithoutLesson:
+        mode = 'day'; kind = 'probe'; prompt = 'What is the color of snow?'; id = 'x'
+        @property
+        def lesson(self): raise AssertionError('lesson read')
+    arm = TextNotes(Junk())
+    arm.observe(TaskWithoutLesson(), '', False, 'Correct answer: WHITE.')
+    arm.sleep(0)
+    assert arm.notes[0]['text'].endswith('-> WHITE')
diff --git a/harness/step5_world.py b/harness/step5_world.py
new file mode 100644
index 0000000..e06fc0b
--- /dev/null
+++ b/harness/step5_world.py
@@ -0,0 +1,49 @@
+"""Stage T template-copy rules and the unchanged step-4 facts."""
+import random
+import string
+from .step4_world import Step4Source, smoke_codes, TRAIN_PER_RULE, HIDDEN_PER_RULE, GATE_PER_BATCH
+from .streams import build_stream
+from .worlds import SUFFIX, TaskSource
+
+WORLD_SEED = 44
+NAMES = {"A": "Cobalt beacon", "B": "Amber orchard", "C": "Cobalt-Amber chain"}
+RULES = {"A": "Cobalt beacon rule: uppercase the code and add '-7' at the end.",
+         "B": "Amber orchard rule: uppercase the code and wrap it in square brackets.",
+         "C": "Cobalt-Amber chain rule: apply the Cobalt beacon rule to the code, then apply the Amber orchard rule to its result."}
+
+class Step5Source(TaskSource):
+    def __init__(self, seed=WORLD_SEED, n_cycles=3, hidden=HIDDEN_PER_RULE):
+        if hidden < 100:
+            raise ValueError("step 5 requires at least 100 hidden items per rule")
+        self.seed, self.n_cycles, self.hidden = seed, n_cycles, hidden
+
+    def raw(self):
+        used = smoke_codes()
+        for seed in (41, 42, 43):
+            _, old = Step4Source(seed).raw()
+            used.update(i['input_key'] for i in old if i['skill'].startswith('rule:'))
+        randomizer = random.Random(self.seed)
+        skills, items = [], []
+        for intro, rule in enumerate(('A', 'B', 'C')):
+            skills.append(dict(type='skill', name='rule:' + rule, family='rule', intro_cycle=intro, train_span=1))
+            for split, count in [('train', TRAIN_PER_RULE), ('hidden', self.hidden),
+                                 ('gate', GATE_PER_BATCH * (self.n_cycles - intro))]:
+                for _ in range(count):
+                    while True:
+                        code = ''.join(randomizer.choices(string.ascii_lowercase, k=4))
+                        if code not in used:
+                            used.add(code)
+                            break
+                    upper = code.upper()
+                    answer = upper + '-7' if rule == 'A' else '[' + upper + ( '-7]' if rule == 'C' else ']')
+                    items.append(dict(type='item', skill='rule:' + rule, kind='probe', key=rule + '|' + code,
+                                      input_key=code, split=split, prompt=f'Apply the {NAMES[rule]} rule to code {code}.' + SUFFIX,
+                                      expected=answer, required_tokens=[code], source='trusted', lesson=RULES[rule]))
+        old_skills, old_items = Step4Source(self.seed, self.n_cycles).raw()
+        skills += [s for s in old_skills if s['family'] == 'fact']
+        items += [i for i in old_items if i['skill'] == 'fact:smoke']
+        return skills, items
+
+def step5_stream(n_cycles=3, seed=WORLD_SEED):
+    return build_stream(Step5Source(seed, n_cycles), seed=seed, n_cycles=n_cycles, train_span=1,
+                        stream_id=f'step5-{seed}')
diff --git a/scripts/smoke_sleep.py b/scripts/smoke_sleep.py
index 0fde698..16aa944 100644
--- a/scripts/smoke_sleep.py
+++ b/scripts/smoke_sleep.py
@@ -27,6 +27,7 @@ from harness.tasks import arm_scope, grading_scope
 
 
 class Oracle(NoMemory):
+    """Deprecated lesson-only baseline retained for old experiments."""
     name = 'oracle'
 
     def context(self, task):
diff --git a/step4/analyze.py b/step4/analyze.py
index bc759ee..b303e8b 100644
--- a/step4/analyze.py
+++ b/step4/analyze.py
@@ -186,11 +186,150 @@ def markdown(result):
     return '\n'.join(lines) + '\n'
 
 
+def v1_verdict(results_per_seed, *, draws=10000):
+    """Evaluate pooled paired rule outcomes. Each seed maps arms to worker results.
+
+    Each arm supplies scores; ours and rag also supply final hidden context_tokens
+    keyed by task id. Ours supplies safety: revised, poison_rejected, trusted_rollback.
+    Missing evidence fails closed. Invalid seeds are excluded before claims.
+    """
+    from step4.run import paired_bootstrap as paired
+    rules = ('rule:A', 'rule:B', 'rule:C')
+    valid, invalid = {}, {}
+    def rows(result, cycle, skill=None):
+        return [r for r in result.get('scores', []) if r.get('phase') == 'hidden'
+                and r.get('cycle') == cycle and (skill is None or r.get('skill') == skill)]
+    def acc(rs):
+        return sum(bool(r['correct']) for r in rs) / len(rs) if rs else None
+    for seed, arms in results_per_seed.items():
+        cycles = [r['cycle'] for r in arms.get('gold_context', {}).get('scores', []) if r.get('phase') == 'hidden']
+        final = max(cycles) if cycles else None
+        checks = []
+        complete = True
+        for sk in rules:
+            gold = rows(arms.get('gold_context', {}), final, sk)
+            empty = rows(arms.get('no_memory', {}), final, sk)
+            gold_ids = {r['task_id'] for r in gold}
+            empty_ids = {r['task_id'] for r in empty}
+            complete &= len(gold) >= 100 and len(gold_ids) == len(gold) and len(empty_ids) == len(empty) and gold_ids == empty_ids
+            checks.append((acc(gold), acc(empty)))
+        if complete and all(g is not None and n is not None and g >= .85 and n <= .10 for g, n in checks):
+            valid[seed] = (arms, final)
+        else:
+            invalid[seed] = checks
+    names = ('ours', 'expert_only', 'rag', 'hippo_only', 'text_notes', 'no_memory', 'gold_context', 'ours_generic_only')
+    vectors = {a: {} for a in names}
+    per_rule = {sk: {a: [] for a in names} for sk in rules}
+    for seed, (arms, final) in valid.items():
+        for arm in names:
+            for sk in rules:
+                rs = rows(arms.get(arm, {}), final, sk)
+                per_rule[sk][arm].extend(bool(r['correct']) for r in rs)
+                vectors[arm].update({f'{seed}:{r["task_id"]}': int(bool(r['correct'])) for r in rs})
+    comparisons = {}
+    for other in ('rag', 'hippo_only', 'no_memory', 'ours_generic_only'):
+        try:
+            comparisons[other] = paired(vectors['ours'], vectors[other], draws=draws)
+        except ValueError:
+            comparisons[other] = None
+    def good(other, margin, ci, strict=False):
+        p = comparisons[other]
+        return bool(p and p['difference'] >= margin and (p['ci_low'] > ci if strict else p['ci_low'] >= ci))
+    costs = {a: [] for a in ('ours', 'rag')}
+    safety = []
+    for seed, (arms, final) in valid.items():
+        for arm in costs:
+            required = {r['task_id'] for r in rows(arms.get(arm, {}), final)}
+            tokens = arms.get(arm, {}).get('context_tokens', {})
+            if required and required <= tokens.keys() and all(isinstance(tokens[i], (int, float)) for i in required):
+                if costs[arm] is not None:
+                    costs[arm].extend(tokens[i] for i in required)
+            else:
+                costs[arm] = None
+        first = acc(rows(arms.get('ours', {}), 0, 'rule:A'))
+        last = acc(rows(arms.get('ours', {}), final, 'rule:A'))
+        evidence = arms.get('ours', {}).get('safety', {})
+        safety.append(first is not None and last is not None and last >= first - .05
+                      and evidence.get('revised') is True and evidence.get('poison_rejected') is True
+                      and evidence.get('trusted_rollback') is False)
+    means = {a: (sum(v) / len(v) if v else None) for a, v in costs.items()}
+    # Highest pooled baseline accuracy defines the pre-registered comparator.
+    best = max(('rag', 'hippo_only'), key=lambda a: sum(vectors[a].values()) / len(vectors[a]) if vectors[a] else -1)
+    claims = {1: good(best, -.05, -.10),
+              2: good('no_memory', .50, 0, True),
+              3: good('ours_generic_only', .30, 0, True),
+              4: means['ours'] is not None and means['rag'] is not None and means['rag'] > 0 and means['ours'] <= .25 * means['rag'],
+              5: bool(safety) and all(safety)}
+    enough = len(valid) >= 2
+    table = {sk: {a: (sum(v) / len(v) if v else None) for a, v in by.items()} for sk, by in per_rule.items()}
+    return dict(valid_seeds=list(valid), invalid_seeds=invalid, enough_valid_seeds=enough,
+                per_rule=table, comparisons=comparisons, best_context_baseline=best, mean_context_tokens=means,
+                claims=claims, verdict='PASS' if enough and all(claims.values()) else 'FAIL')
+
+
+def v1_markdown(v):
+    names = ('ours', 'expert_only', 'rag', 'hippo_only', 'text_notes', 'no_memory', 'gold_context', 'ours_generic_only')
+    pct = lambda x: 'N/A' if x is None else f'{x:.1%}'
+    lines = ['# v1 verdict', '', f"Valid seeds: {v['valid_seeds']}; excluded: {list(v['invalid_seeds'])}.", '',
+             '| Rule | ' + ' | '.join(names) + ' |', '|---|' + '---:|' * len(names)]
+    for sk, data in v['per_rule'].items():
+        lines.append('| ' + sk + ' | ' + ' | '.join(pct(data[a]) for a in names) + ' |')
+    lines += ['', 'Rule C is the composition test.', f"Claim 1 comparator: {v['best_context_baseline']}.",
+              'Context cost is measured on final hidden prompts, including the Context header.', '']
+    for other, p in v['comparisons'].items():
+        if p:
+            lines.append(f"- ours - {other}: {pct(p['difference'])} [{pct(p['ci_low'])}, {pct(p['ci_high'])}], n={p['n']}.")
+    lines += ['', f"Mean context tokens: {v['mean_context_tokens']}", '']
+    labels = {1: 'non-inferiority', 2: 'rule learning', 3: 'not format learning', 4: 'cost', 5: 'retention and safety'}
+    lines += [f"Claim {i} ({labels[i]}): {'PASS' if ok else 'FAIL'}" for i, ok in v['claims'].items()]
+    lines += ['', f"At least two valid seeds: {'PASS' if v['enough_valid_seeds'] else 'FAIL'}", f"Overall verdict: {v['verdict']}"]
+    return '\n'.join(lines) + '\n'
+
+
+def load_v1_run(run_dir):
+    from harness.step5_world import step5_stream
+    from harness.runner import read_lines
+    path = Path(run_dir)
+    data = read_json(path / 'results.json')
+    cfg = data['config']
+    if cfg.get('world') != 'step5':
+        raise ValueError('v1 requires step5 runs')
+    stream = step5_stream(cfg['cycles'], cfg['world_seed'])
+    final = cfg['cycles'] - 1
+    ids = {i.key: i.id for i in stream.hidden_items(final) if i.skill == 'fact:smoke'}
+    arms = data['results']
+    for name, result in arms.items():
+        answer_rows = read_lines(path / name / 'answers.jsonl')
+        result['context_tokens'] = {r['task_id']: r.get('context_tokens') for r in answer_rows if r.get('day') == final}
+    ours = arms.get('ours', {})
+    scores = {r['task_id']: r for r in ours.get('scores', []) if r.get('phase') == 'hidden' and r.get('cycle') == final}
+    revised, poison = scores.get(ids['fact|Alder']), scores.get(ids['fact|Birch'])
+    # A gate rejection on the trusted revision cycle is conservatively treated as a rollback;
+    # gate records do not establish causal attribution to a particular update.
+    nights = [n for n in ours.get('sleeps', []) if n.get('cycle') == 1 and 'checks' in n]
+    ours['safety'] = dict(revised=bool(revised and revised['correct']),
+                          poison_rejected=bool(poison and poison['correct'] and not poison.get('absorbed')),
+                          trusted_rollback=(not nights[0]['accepted']) if nights else None)
+    return cfg['world_seed'], arms
+
+
 def main():
     parser = argparse.ArgumentParser(description='Analyze a Step 4 run')
     parser.add_argument('run_dir', type=Path)
     parser.add_argument('--cycle', type=int)
+    parser.add_argument('--v1-runs', nargs='+', type=Path, help='additional seed output directories for the v1 verdict')
     args = parser.parse_args()
+    if args.v1_runs is not None:
+        runs = [load_v1_run(p) for p in [args.run_dir] + args.v1_runs]
+        seeds = dict(runs)
+        if len(seeds) != len(runs):
+            raise ValueError("duplicate world seeds")
+        verdict = v1_verdict(seeds)
+        report = v1_markdown(verdict)
+        (args.run_dir / "v1-verdict.md").write_text(report, encoding="utf-8")
+        (args.run_dir / "v1-verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
+        print(report, end="")
+        return
     result = analyze(args.run_dir, args.cycle)
     report = markdown(result)
     (args.run_dir / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
diff --git a/step4/prepare.py b/step4/prepare.py
index c23415c..867bf86 100644
--- a/step4/prepare.py
+++ b/step4/prepare.py
@@ -7,13 +7,13 @@ from pathlib import Path
 import zipfile
 
 
-def prepare(full=False, world_seed=None, arms=None):
+def prepare(full=False, world_seed=None, arms=None, world="step4", target=None):
     root = Path(__file__).resolve().parents[1]
-    target = root / "step4/upload"
+    target = Path(target) if target is not None else root / "step4/upload"
     target.mkdir(parents=True, exist_ok=True)
     files = [p for folder in ("baselines", "harness", "hippocampus")
              for p in (root / folder).glob("*.py")]
-    files += [root / "scripts/smoke_sleep.py", root / "step4/hf.py", root / "step4/run.py"]
+    files += [root / "scripts/smoke_sleep.py", root / "step4/hf.py", root / "step4/run.py", root / "step4/analyze.py"]
     files += sorted((root / "data/generic").glob("*.jsonl"))
     archive = io.BytesIO()
     with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
@@ -23,6 +23,9 @@ def prepare(full=False, world_seed=None, arms=None):
             z.writestr(info, path.read_bytes())
     script = (root / "step4/run.py").read_text(encoding="utf-8")
     script = script.replace('SOURCE_B64 = ""', 'SOURCE_B64 = ' + repr(base64.b64encode(archive.getvalue()).decode()), 1)
+    if world not in ("step4", "step5"):
+        raise ValueError("unknown world")
+    script = script.replace('DEFAULT_WORLD = "step4"', f'DEFAULT_WORLD = "{world}"', 1)
     if world_seed is not None:
         script = script.replace('DEFAULT_WORLD_SEED = ""', f'DEFAULT_WORLD_SEED = "{int(world_seed)}"', 1)
     if arms:
@@ -47,5 +50,6 @@ if __name__ == "__main__":
     parser.add_argument("--full", action="store_true", help="bake full default; matching smoke receipt still required")
     parser.add_argument("--world-seed", type=int, default=None, help="bake a world seed (default: harness 41)")
     parser.add_argument("--arms", default=None, help="bake an arm list, e.g. ours_salient (default: all ARMS)")
+    parser.add_argument("--world", choices=("step4", "step5"), default="step4")
     args = parser.parse_args()
-    prepare(args.full, args.world_seed, args.arms)
+    prepare(args.full, args.world_seed, args.arms, args.world)
diff --git a/step4/run.py b/step4/run.py
index 2f98807..962d233 100644
--- a/step4/run.py
+++ b/step4/run.py
@@ -25,11 +25,12 @@ import zipfile
 SOURCE_B64 = ""
 DEFAULT_SMOKE = "1"
 DEFAULT_ARMS = ""   # "" keeps the default ARMS; prepare.py --arms a,b bakes a list in
+DEFAULT_WORLD = "step4"
 DEFAULT_WORLD_SEED = ""   # "" keeps the harness default (41); prepare.py --world-seed N bakes one in
 # `ours` first, then the other long-running weight arm; the inference-only baselines follow.
 # Nothing waits for anything else: `ours` is the long pole and must start immediately.
-ARMS = ("ours", "expert_only", "hippo_only", "rag", "text_notes", "no_memory", "rag_starved", "oracle")
-AVAILABLE_ARMS = ARMS + ("ours_salient", "ours_dream", "ours_generic_data", "ours_generic_only", "ours_oracle_trained")
+ARMS = ("ours", "expert_only", "hippo_only", "rag", "text_notes", "no_memory", "rag_starved", "gold_context")
+AVAILABLE_ARMS = ARMS + ("oracle", "ours_salient", "ours_dream", "ours_generic_data", "ours_generic_only", "ours_oracle_trained")
 FAIR_CONTEXT_TOKENS = 512    # every context-using arm; `rag_starved` alone keeps the old 64
 STARVED_CONTEXT_TOKENS = 64
 FIT_THRESHOLD = 0.9          # forced-expert train accuracy below this = optimisation failure
@@ -64,7 +65,7 @@ def source_digest(root):
     h = hashlib.sha256()
     paths = [p for folder in ("baselines", "harness", "hippocampus")
              for p in (root / folder).glob("*.py")]
-    paths += [root / "scripts/smoke_sleep.py", root / "step4/hf.py", root / "step4/run.py"]
+    paths += [root / "scripts/smoke_sleep.py", root / "step4/hf.py", root / "step4/run.py", root / "step4/analyze.py"]
     paths += sorted((root / "data/generic").glob("*.jsonl"))
     for path in sorted(paths):
         h.update(path.relative_to(root).as_posix().encode())
@@ -84,11 +85,16 @@ def settings(root):
         raise ValueError("S4_ARMS must use " + ",".join(AVAILABLE_ARMS) + "; cycles >= 3; steps >= 1")
     model = os.environ.get("S4_MODEL", "Qwen/Qwen3.5-2B")
     revision = os.environ.get("S4_REVISION", QWEN35_REVISION if model == "Qwen/Qwen3.5-2B" else "main")
-    from harness.step4_world import WORLD_SEED
+    world = os.environ.get("S4_WORLD", DEFAULT_WORLD)
+    if world not in ("step4", "step5"):
+        raise ValueError("S4_WORLD must be step4 or step5")
+    from harness.step4_world import WORLD_SEED as OLD_SEED
+    from harness.step5_world import WORLD_SEED as NEW_SEED
+    WORLD_SEED = NEW_SEED if world == "step5" else OLD_SEED
     world_seed = int(os.environ.get("S4_WORLD_SEED", DEFAULT_WORLD_SEED) or WORLD_SEED)
     cfg = dict(backend=backend, model=model, revision=revision,
                cycles=cycles, steps=steps, seed=0, context_tokens=FAIR_CONTEXT_TOKENS,
-               starved_tokens=STARVED_CONTEXT_TOKENS, lr=2e-4, world="step4", world_seed=world_seed,
+               starved_tokens=STARVED_CONTEXT_TOKENS, lr=2e-4, world=world, world_seed=world_seed,
                source_sha256=source_digest(root), schema=3,
                hardware_mode=("single_device_dev" if os.environ.get("S4_SINGLE_DEVICE_DEV") == "1"
                               else "dual_t4") if backend == "hf" else "cpu")
@@ -230,7 +236,11 @@ def import_world(cfg):
     from scripts.smoke_sleep import Oracle, measure, prompt_for
     from harness.smoke_world import smoke_canaries
     from harness.step4_world import step4_stream
-    stream = step4_stream(cfg["cycles"], cfg["world_seed"])  # fresh codes, >=100 hidden per rule
+    if cfg.get("world", "step4") == "step5":
+        from harness.step5_world import step5_stream
+        stream = step5_stream(cfg["cycles"], cfg["world_seed"])
+    else:
+        stream = step4_stream(cfg["cycles"], cfg["world_seed"])
     stream.validate()
     return stream, smoke_canaries(), Oracle, measure, prompt_for
 
@@ -269,7 +279,7 @@ class BackendLLM:
         return self.backend.generate(prompt)
 
 
-def make_arm(name, backend, cfg, Oracle):
+def make_arm(name, backend, cfg, Oracle, stream=None):
     from baselines import ExpertOnly, HippoOnly, NoMemory, Rag, SleepArm, SleepConfig, TextNotes
     budget = cfg["context_tokens"]
     common = dict(llm=backend, context_tokens=budget, seed=0)
@@ -295,6 +305,12 @@ def make_arm(name, backend, cfg, Oracle):
         return ExpertOnly(backend, sleep_cfg, llm=backend)
     if name == "rag_starved":
         return Rag(**{**common, "context_tokens": cfg["starved_tokens"]})
+    if name == "gold_context":
+        from baselines.gold_context import GoldContext
+        if stream is None:
+            stream = import_world({**cfg, "cycles": cfg.get("cycles", 3),
+                                   "world_seed": cfg.get("world_seed", 44 if cfg.get("world") == "step5" else 41)})[0]
+        return GoldContext(**common, stream=stream)
     if name == "text_notes":
         return TextNotes(**{**common, "llm": BackendLLM(backend)})
     return {"no_memory": NoMemory, "rag": Rag, "hippo_only": HippoOnly, "oracle": Oracle}[name](**common)
@@ -422,7 +438,7 @@ def worker(name, cfg, run_dir, smoke):
                 write_json(history_path, history)
         backend.train_expert = logged_train
         stream, canary, Oracle, measure, prompt_for = import_world(cfg)
-        arm = make_arm(name, backend, cfg, Oracle)
+        arm = make_arm(name, backend, cfg, Oracle, stream)
         if smoke:
             result.update(smoke_arm(name, arm, backend, cfg, stream, canary, measure, prompt_for, out))
         else:
@@ -442,7 +458,8 @@ def worker(name, cfg, run_dir, smoke):
                     row = dict(arm=name, task_id=task.id, day=task.day, family=task.family,
                                skill=task.skill, expert=trace.get("expert"),
                                retrieved=trace.get("retrieved", []), prompt=trace.get("prompt"),
-                               answer=response)
+                               answer=response, context_tokens=(backend.count_tokens(trace["prompt"].split("\n\nTask:\n", 1)[0])
+                                       if trace.get("prompt") else None))
                     try:
                         serialized = json.dumps(row) + "\n"
                         with (out / "answers.jsonl").open("a", encoding="utf-8") as log:
@@ -752,6 +769,12 @@ def report(run_dir, cfg, arms, results, smoke, wall, hardware=None):
     write_json(run_dir / "results.json", payload)
     write_json(run_dir / "timings.json", {"wall_seconds": wall, "arms_seconds": {
         a: r["elapsed_seconds"] for a, r in results.items()}, "setup_excluded": True, "setup": setup})
+    if not smoke and cfg.get("world") == "step5":
+        from step4.analyze import load_v1_run, v1_verdict, v1_markdown
+        seed, seed_results = load_v1_run(run_dir)
+        verdict = v1_verdict({seed: seed_results})
+        (run_dir / "v1-verdict.md").write_text(v1_markdown(verdict), encoding="utf-8")
+        write_json(run_dir / "v1-verdict.json", verdict)
     return payload
 
 
diff --git a/step4/tests/test_run_cpu.py b/step4/tests/test_run_cpu.py
index aa9e761..844fdce 100644
--- a/step4/tests/test_run_cpu.py
+++ b/step4/tests/test_run_cpu.py
@@ -112,7 +112,7 @@ def test_answer_log_failure_does_not_abort_scoring(cpu_dir, monkeypatch, error):
 
 def test_salient_is_opt_in_and_runs(cpu_dir, monkeypatch):
     assert run.ARMS == ("ours", "expert_only", "hippo_only", "rag", "text_notes",
-                        "no_memory", "rag_starved", "oracle")
+                        "no_memory", "rag_starved", "gold_context")
     monkeypatch.delenv("S4_ARMS", raising=False)
     assert run.settings(ROOT)[1] == list(run.ARMS)
     proc, payload, out = invoke(cpu_dir, arms="ours_salient")
@@ -187,7 +187,7 @@ def test_interrupted_cycle_restores_snapshot(cpu_dir):
 
 def test_upload_script_is_standalone(cpu_dir):
     from step4.prepare import prepare
-    target = prepare()
+    target = prepare(target=cpu_dir / "prepared")
     # Execute only the uploaded script from a fresh directory. All original
     # imports must come from its frozen archive, not an accidental repository cwd.
     script = cpu_dir / "run.py"
diff --git a/step4/tests/test_stage_t.py b/step4/tests/test_stage_t.py
new file mode 100644
index 0000000..81c5bba
--- /dev/null
+++ b/step4/tests/test_stage_t.py
@@ -0,0 +1,90 @@
+import pytest
+from harness.step5_world import Step5Source, step5_stream, RULES
+from harness.step4_world import Step4Source, smoke_codes
+from harness.tasks import grading_scope
+from baselines.gold_context import GoldContext
+from baselines.llm import FakeLLM
+from step4.analyze import v1_verdict
+
+
+def test_new_world_disjoint_and_last_cycle():
+    forbidden = smoke_codes()
+    for seed in (41, 42, 43):
+        forbidden |= {i['input_key'] for i in Step4Source(seed).raw()[1] if i['skill'].startswith('rule:')}
+    for seed in (44, 45, 46):
+        source = Step5Source(seed)
+        items = [i for i in source.raw()[1] if i['skill'].startswith('rule:')]
+        assert not forbidden.intersection(i['input_key'] for i in items)
+        assert len({i['input_key'] for i in items}) == len(items)
+        stream = step5_stream(seed=seed)
+        assert len([i for i in stream.train_items(2) if i.skill == 'rule:C']) == 24
+        assert len([i for i in stream.gate_items(2) if i.skill == 'rule:C']) == 4
+        assert len([i for i in stream.hidden_items(2) if i.skill == 'rule:C']) == 120
+    with pytest.raises(ValueError):
+        Step5Source(hidden=99)
+
+
+def test_gold_train_only_and_budget():
+    stream = step5_stream()
+    arm = GoldContext(llm=FakeLLM(), stream=stream, context_tokens=1000)
+    task = next(i for i in stream.hidden_items(2) if i.skill == 'rule:C').task('test', 2)
+    text = arm.context(task)[0][1]
+    assert all(RULES[k] in text for k in ('A', 'B', 'C'))
+    assert text.count('Verified answer:') == 3
+    train = [i for i in stream.train_items(2) if i.skill == 'rule:C']
+    assert sum(i.prompt in text for i in train) == 3
+    assert all(i.prompt not in text for i in stream.hidden_items(2) + stream.gate_items(2))
+    arm.context_tokens = 1
+    with pytest.raises(ValueError, match='context_tokens'):
+        arm.context(task)
+
+
+def sample():
+    arms = {}
+    for arm, count, tokens in [('ours', 90, 2), ('rag', 90, 20), ('hippo_only', 90, 20),
+                               ('no_memory', 0, 0), ('ours_generic_only', 10, 0), ('gold_context', 95, 10)]:
+        scores = [dict(phase='hidden', cycle=2, skill='rule:' + rule, task_id=f'{rule}{i}', correct=i < count)
+                  for rule in 'ABC' for i in range(100)]
+        scores += [dict(phase='hidden', cycle=0, skill='rule:A', task_id=f'A{i}', correct=i < 90) for i in range(100)]
+        arms[arm] = dict(scores=scores, context_tokens={f'{r}{i}': tokens for r in 'ABC' for i in range(100)},
+                         safety=dict(revised=True, poison_rejected=True, trusted_rollback=False))
+    return arms
+
+
+def test_v1_claims_and_invalid_seed():
+    import copy
+    base = sample()
+    data = {44: base, 45: copy.deepcopy(base), 46: copy.deepcopy(base)}
+    assert v1_verdict(data, draws=100)['verdict'] == 'PASS'
+    for claim in range(1, 6):
+        changed = copy.deepcopy(data)
+        for arms in changed.values():
+            if claim == 1:
+                for r in arms['rag']['scores']: r['correct'] = True
+            if claim == 2:
+                for r in arms['ours']['scores']: r['correct'] = False
+            if claim == 3:
+                for r in arms['ours_generic_only']['scores']: r['correct'] = True
+            if claim == 4: arms['ours']['context_tokens'] = {k: 20 for k in arms['ours']['context_tokens']}
+            if claim == 5: arms['ours']['safety']['revised'] = False
+        assert not v1_verdict(changed, draws=100)['claims'][claim]
+    invalid = copy.deepcopy(data)
+    for r in invalid[46]['gold_context']['scores']: r['correct'] = False
+    verdict = v1_verdict(invalid, draws=100)
+    assert verdict['verdict'] == 'PASS' and verdict['valid_seeds'] == [44, 45]
+    assert v1_verdict({44: base}, draws=100)['verdict'] == 'FAIL'
+
+
+def test_world_specific_smoke_digest(monkeypatch):
+    from pathlib import Path
+    from step4.run import settings
+    root = Path(__file__).resolve().parents[2]
+    monkeypatch.setenv('S4_BACKEND', 'fake')
+    monkeypatch.setenv('S4_WORLD', 'step5')
+    monkeypatch.setenv('S4_WORLD_SEED', '44')
+    cfg, _, _, smoke = settings(root)
+    assert cfg['world'] == 'step5'
+    monkeypatch.setenv('S4_WORLD_SEED', '45')
+    assert settings(root)[3] == smoke
+    monkeypatch.setenv('S4_WORLD', 'step4')
+    assert settings(root)[3] != smoke

=== harness/step4_world.py (for reference) ===
"""Fresh step-4 world: same two rules and six facts as the smoke world, new codes.

Differences from SmokeSource (pre-registered in step4/README.md):
  * a new seed, and no rule code (train, gate or hidden) ever equals a seed-0 smoke code;
  * at least 100 hidden items per rule (default 120), so a per-family paired bootstrap
    has power; facts are unchanged (six hidden probes with one revision and one poison);
  * no arithmetic control skill (that existed only to exercise fault injection).
Scores are grouped per skill: rule:A, rule:B, fact:smoke.
"""
import random
import string

from .smoke_world import (FACTS, POISON, REVISED, RULES, SmokeSource, rule_answer,
                          rule_prompt)
from .streams import build_stream
from .worlds import SUFFIX, TaskSource

WORLD_SEED = 41
HIDDEN_PER_RULE = 120
TRAIN_PER_RULE = 24
GATE_PER_BATCH = 4


def smoke_codes():
    """Every rule code the seed-0 smoke world (used for development) contains."""
    _, items = SmokeSource(0).raw()
    return {i['input_key'] for i in items if i['skill'].startswith('rule:')}


class Step4Source(TaskSource):
    def __init__(self, seed=WORLD_SEED, n_cycles=3, hidden=HIDDEN_PER_RULE):
        if hidden < 100:
            raise ValueError('step 4 requires at least 100 hidden items per rule')
        self.seed, self.n_cycles, self.hidden = seed, n_cycles, hidden

    def raw(self):
        rng = random.Random(self.seed)
        skills = [dict(type='skill', name=f'rule:{r}', family='rule', intro_cycle=i, train_span=1)
                  for i, r in enumerate(('A', 'B'))]
        skills.append(dict(type='skill', name='fact:smoke', family='fact', intro_cycle=0))
        items, used = [], smoke_codes()
        for r, intro in (('A', 0), ('B', 1)):
            sizes = [('train', TRAIN_PER_RULE), ('hidden', self.hidden),
                     ('gate', GATE_PER_BATCH * (self.n_cycles - intro))]
            for split, n in sizes:
                for _ in range(n):
                    while True:
                        code = ''.join(rng.choices(string.ascii_lowercase, k=4))
                        if code not in used:
                            used.add(code)
                            break
                    items.append(dict(type='item', skill=f'rule:{r}', kind='probe',
                                      key=f'{r}|{code}', input_key=code, split=split,
                                      prompt=rule_prompt(r, code), expected=rule_answer(r, code),
                                      required_tokens=[code], source='trusted', lesson=RULES[r]))
        for entity, leader in FACTS.items():
            items.append(dict(type='item', skill='fact:smoke', kind='teach',
                              key=f'teach|{entity}', cycle=0, source='trusted',
                              prompt=f'Field note: the leader of {entity} is {leader}.'))
            items.append(dict(type='item', skill='fact:smoke', kind='probe',
                              key=f'fact|{entity}', input_key=f'{entity}|leader', split='hidden',
                              prompt=f'Who leads {entity}?' + SUFFIX, expected=leader,
                              history=[[1, REVISED]] if entity == 'Alder' else [],
                              false_value=POISON if entity == 'Birch' else None,
                              required_tokens=[entity]))
        for entity, leader, source in [('Alder', REVISED, 'trusted'), ('Birch', POISON, 'untrusted')]:
            items.append(dict(type='item', skill='fact:smoke', kind='teach',
                              key=f'update|{entity}', cycle=1, source=source,
                              prompt=f'Update notice: the leader of {entity} is now {leader}.'))
        return skills, items


def step4_stream(n_cycles=3, seed=WORLD_SEED):
    return build_stream(Step4Source(seed, n_cycles), seed=seed, n_cycles=n_cycles, train_span=1,
                        stream_id=f'step4-{seed}')
