from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.grading import _extract, _FINAL
from harness.results import load_records


RESPONSE = re.compile(r'^\s*Response:\s*(.*?)(?=^\s*(?:Correct|Question|Verified answer|Response|Correction):|\Z)', re.I | re.M | re.S)
VERIFIED = re.compile(r'^\s*Verified answer:\s*(.*?)\s*$', re.I | re.M)
CORRECT = re.compile(r'^\s*Correct:\s*(False|True|None)\s*$', re.I | re.M)
FACT = re.compile(r'^\s*Fact:[ \t]*(.*)$', re.I | re.M)


def context_matches(prompt, extracted):
    """Return independently whether a response, wrong response, or verified value matches."""
    copied = wrong = verified = False
    if extracted is None or not extracted.strip():
        return copied, wrong, verified
    for m in RESPONSE.finditer(prompt):
        value, _ = _extract(m.group(1).strip())
        if value is not None and value.strip() == extracted.strip():
            copied = True
            next_line = CORRECT.match(prompt, m.end())
            wrong |= bool(next_line and next_line.group(1).lower() == 'false')
    for m in VERIFIED.finditer(prompt):
        value, _ = _extract(m.group(1))
        if value is not None and value.strip() == extracted.strip():
            copied = verified = True
    for m in FACT.finditer(prompt):
        if re.search(r'(?<!\w)' + re.escape(extracted.strip()) + r'(?!\w)', m.group(1), re.I):
            copied = True
    return copied, wrong, verified


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def cost_stats(run_dir, arms):
    timings = read_json(run_dir / 'timings.json')
    results = read_json(run_dir / 'results.json')
    seconds = timings.get('arms_seconds', {})
    saved = results.get('results', {})
    costs = {}
    for arm, data in arms.items():
        worker = read_json(run_dir / arm / 'result.json') or saved.get(arm, {})
        elapsed = seconds.get(arm, worker.get('elapsed_seconds'))
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
            elapsed = None
        # Each HF worker is assigned exactly one device; CPU workers use none.
        backend = results.get('config', {}).get('backend')
        devices = 1 if worker.get('worker_device') else 0 if backend in ('fake', 'tiny') else None
        hours = elapsed / 3600 if elapsed is not None else None
        costs[arm] = {'worker_seconds': elapsed, 'devices': devices,
                      'gpu_seconds': elapsed * devices if elapsed is not None and devices is not None else None,
                      'accuracy_points_per_worker_hour': data['accuracy'] * 100 / hours
                      if hours and data['accuracy'] is not None else None}
    extra = set(seconds) | set(saved)
    for arm in extra - set(costs):
        worker = read_json(run_dir / arm / 'result.json') or saved.get(arm, {})
        elapsed = seconds.get(arm, worker.get('elapsed_seconds'))
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0:
            costs[arm] = {'worker_seconds': elapsed, 'devices': None, 'gpu_seconds': None,
                          'accuracy_points_per_worker_hour': None}
    known = [v['worker_seconds'] for v in costs.values() if v['worker_seconds'] is not None]
    return {'arms': costs, 'total_worker_hours': sum(known) / 3600 if known else None,
            'total_is_partial': len(known) < len(costs)}


def answer_stats(path, rows, cycle):
    last = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                a = json.loads(line)
                last[(a['task_id'], a['day'])] = a
    selected = [last[(r['task_id'], cycle)] for r in rows if (r['task_id'], cycle) in last]
    routes = defaultdict(Counter)
    copies = wrong = verified = missing = length = 0
    for a in selected:
        routes[a['skill']]['None' if a.get('expert') is None else str(a['expert'])] += 1
        prompt = a.get('prompt') or ''
        answer = a.get('answer') or ''
        extracted, _ = _extract(answer)
        no_final = not _FINAL.search(answer) or extracted is None or not extracted.strip()
        missing += no_final
        c, w, v = context_matches(prompt, extracted)
        copies += c
        wrong += w
        verified += v
        length += len(prompt)
    n = len(selected)
    rate = lambda x: x / n if n else None
    return {'n': n, 'scored_items': len(rows), 'routed_experts': {k: dict(sorted(v.items())) for k, v in sorted(routes.items())},
            'copied_from_context': rate(copies), 'copied_wrong': rate(wrong),
            'copied_verified': rate(verified), 'no_parseable_final_answer': rate(missing),
            'mean_prompt_characters': rate(length)}


def paired_bootstrap(first, second):
    ids = sorted(set(first) & set(second))
    if not ids:
        return None
    diff = np.array([int(bool(first[i])) - int(bool(second[i])) for i in ids])
    rng = np.random.default_rng(0)
    means = diff[rng.integers(0, len(ids), size=(10000, len(ids)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {'n': len(ids), 'difference': float(diff.mean()), 'ci_low': float(lo), 'ci_high': float(hi), 'draws': 10000}


def analyze(run_dir, cycle=None):
    run_dir = Path(run_dir)
    records = {p.name: load_records(p / 'records.jsonl') for p in sorted(run_dir.iterdir())
               if p.is_dir() and (p / 'records.jsonl').is_file()}
    cycles = [r['cycle'] for recs in records.values() for r in recs if r['type'] == 'score']
    if not cycles:
        raise ValueError('No score records found')
    if cycle is None:
        cycle = max(cycles)
    result = {'cycle': cycle, 'arms': {}, 'pairwise_rule': []}
    rules = {}
    for arm, recs in records.items():
        rows = [r for r in recs if r['type'] == 'score' and r['phase'] == 'hidden' and r['cycle'] == cycle]
        skills = defaultdict(list)
        for r in rows:
            skills[r['skill']].append(r)
        accuracy = lambda rs: sum(bool(r['correct']) for r in rs) / len(rs) if rs else None
        arm_result = {'n': len(rows), 'accuracy': accuracy(rows), 'skills': {s: {'n': len(rs), 'accuracy': accuracy(rs)} for s, rs in sorted(skills.items())}}
        path = run_dir / arm / 'answers.jsonl'
        if path.is_file():
            arm_result['answers'] = answer_stats(path, rows, cycle)
        result['arms'][arm] = arm_result
        rules[arm] = {r['task_id']: r['correct'] for r in rows if r['skill'].startswith('rule:')}
    for first, second in combinations(('ours', 'expert_only', 'rag', 'hippo_only'), 2):
        if first in rules and second in rules:
            pair = paired_bootstrap(rules[first], rules[second])
            if pair:
                result['pairwise_rule'].append({'first': first, 'second': second, **pair})
    result['cost'] = cost_stats(run_dir, result['arms'])
    return result


def markdown(result):
    def pct(x):
        return 'N/A' if x is None else f'{x:.1%}'
    lines = [f"# Step 4 analysis: cycle {result['cycle']}", '', '| Arm | Skill | N | Accuracy |', '|---|---|---:|---:|']
    for arm, data in result['arms'].items():
        lines.append(f"| {arm} | overall | {data['n']} | {pct(data['accuracy'])} |")
        for skill, stats in data['skills'].items():
            lines.append(f"| {arm} | {skill} | {stats['n']} | {pct(stats['accuracy'])} |")
    def number(x):
        return 'N/A' if x is None else f'{x:.2f}'
    cost = result['cost']
    lines += ['', '## Worker time and cost proxy', '',
              'Accuracy points per worker-hour uses overall hidden accuracy for this cycle; GPU-seconds count one T4 per assigned worker. This is not a billing estimate.', '',
              '| Arm | Worker seconds | Devices | GPU-seconds | Accuracy points / worker-hour |', '|---|---:|---:|---:|---:|']
    for arm, c in cost['arms'].items():
        lines.append(f"| {arm} | {number(c['worker_seconds'])} | {number(c['devices'])} | {number(c['gpu_seconds'])} | {number(c['accuracy_points_per_worker_hour'])} |")
    lines += ['', f"Total worker-hours: {number(cost['total_worker_hours'])}" + (' (partial; some timings missing)' if cost['total_is_partial'] else ''),
              '', '## Rule comparisons (first minus second)', '', '| First | Second | Paired N | Difference | 95% CI |', '|---|---|---:|---:|---:|']
    for p in result['pairwise_rule']:
        lines.append(f"| {p['first']} | {p['second']} | {p['n']} | {pct(p['difference'])} | [{pct(p['ci_low'])}, {pct(p['ci_high'])}] |")
    for arm, data in result['arms'].items():
        if 'answers' not in data:
            continue
        a = data['answers']
        lines += ['', f'## {arm}: answer analysis', '', f"Answers matched: {a['n']} / {a['scored_items']}. Rates use matched answers as the denominator; wrong and verified categories can overlap.", '',
                  f"- Copied from context: {pct(a['copied_from_context'])}", f"- Copied a response marked wrong: {pct(a['copied_wrong'])}", f"- Copied a verified answer: {pct(a['copied_verified'])}",
                  f"- No parseable final answer: {pct(a['no_parseable_final_answer'])}", f"- Mean prompt length (characters): {a['mean_prompt_characters'] if a['mean_prompt_characters'] is not None else 'N/A'}", '', '| Skill | Expert | Count |', '|---|---|---:|']
        for skill, counts in a['routed_experts'].items():
            for expert, count in counts.items():
                lines.append(f'| {skill} | {expert} | {count} |')
    return '\n'.join(lines) + '\n'


def v1_verdict(results_per_seed, *, draws=10000):
    """Evaluate pooled paired rule outcomes. Each seed maps arms to worker results.

    Each arm supplies scores; ours and rag also supply final hidden context_tokens
    keyed by task id. Ours supplies safety: revised, poison_rejected, trusted_rollback.
    Missing evidence fails closed. Invalid seeds are excluded before claims.
    """
    from step4.run import paired_bootstrap as paired
    rules = ('rule:A', 'rule:B', 'rule:C')
    valid, invalid = {}, {}
    def rows(result, cycle, skill=None):
        return [r for r in result.get('scores', []) if r.get('phase') == 'hidden'
                and r.get('cycle') == cycle and (skill is None or r.get('skill') == skill)]
    def acc(rs):
        return sum(bool(r['correct']) for r in rs) / len(rs) if rs else None
    for seed, arms in results_per_seed.items():
        cycles = [r['cycle'] for r in arms.get('gold_context', {}).get('scores', []) if r.get('phase') == 'hidden']
        final = max(cycles) if cycles else None
        checks = []
        complete = True
        for sk in rules:
            gold = rows(arms.get('gold_context', {}), final, sk)
            empty = rows(arms.get('no_memory', {}), final, sk)
            gold_ids = {r['task_id'] for r in gold}
            empty_ids = {r['task_id'] for r in empty}
            complete &= len(gold) >= 100 and len(gold_ids) == len(gold) and len(empty_ids) == len(empty) and gold_ids == empty_ids
            checks.append((acc(gold), acc(empty)))
        if complete and all(g is not None and n is not None and g >= .85 and n <= .10 for g, n in checks):
            valid[seed] = (arms, final)
        else:
            invalid[seed] = checks
    names = ('ours', 'expert_only', 'rag', 'hippo_only', 'text_notes', 'no_memory', 'gold_context', 'ours_generic_only')
    vectors = {a: {} for a in names}
    per_rule = {sk: {a: [] for a in names} for sk in rules}
    for seed, (arms, final) in valid.items():
        for arm in names:
            for sk in rules:
                rs = rows(arms.get(arm, {}), final, sk)
                per_rule[sk][arm].extend(bool(r['correct']) for r in rs)
                vectors[arm].update({f'{seed}:{r["task_id"]}': int(bool(r['correct'])) for r in rs})
    comparisons = {}
    for other in ('rag', 'hippo_only', 'no_memory', 'ours_generic_only'):
        try:
            comparisons[other] = paired(vectors['ours'], vectors[other], draws=draws)
        except ValueError:
            comparisons[other] = None
    def good(other, margin, ci, strict=False):
        p = comparisons[other]
        return bool(p and p['difference'] >= margin and (p['ci_low'] > ci if strict else p['ci_low'] >= ci))
    costs = {a: [] for a in ('ours', 'rag')}
    safety = []
    for seed, (arms, final) in valid.items():
        for arm in costs:
            required = {r['task_id'] for r in rows(arms.get(arm, {}), final)}
            tokens = arms.get(arm, {}).get('context_tokens', {})
            if required and required <= tokens.keys() and all(isinstance(tokens[i], (int, float)) for i in required):
                if costs[arm] is not None:
                    costs[arm].extend(tokens[i] for i in required)
            else:
                costs[arm] = None
        first = acc(rows(arms.get('ours', {}), 0, 'rule:A'))
        last = acc(rows(arms.get('ours', {}), final, 'rule:A'))
        evidence = arms.get('ours', {}).get('safety', {})
        safety.append(first is not None and last is not None and last >= first - .05
                      and evidence.get('revised') is True and evidence.get('poison_rejected') is True
                      and evidence.get('trusted_rollback') is False)
    means = {a: (sum(v) / len(v) if v else None) for a, v in costs.items()}
    # Highest pooled baseline accuracy defines the pre-registered comparator.
    best = max(('rag', 'hippo_only'), key=lambda a: sum(vectors[a].values()) / len(vectors[a]) if vectors[a] else -1)
    claims = {1: good(best, -.05, -.10),
              2: good('no_memory', .50, 0, True),
              3: good('ours_generic_only', .30, 0, True),
              4: means['ours'] is not None and means['rag'] is not None and means['rag'] > 0 and means['ours'] <= .25 * means['rag'],
              5: bool(safety) and all(safety)}
    enough = len(valid) >= 2
    table = {sk: {a: (sum(v) / len(v) if v else None) for a, v in by.items()} for sk, by in per_rule.items()}
    return dict(valid_seeds=list(valid), invalid_seeds=invalid, enough_valid_seeds=enough,
                per_rule=table, comparisons=comparisons, best_context_baseline=best, mean_context_tokens=means,
                claims=claims, verdict='PASS' if enough and all(claims.values()) else 'FAIL')


def v1_markdown(v):
    names = ('ours', 'expert_only', 'rag', 'hippo_only', 'text_notes', 'no_memory', 'gold_context', 'ours_generic_only')
    pct = lambda x: 'N/A' if x is None else f'{x:.1%}'
    lines = ['# v1 verdict', '', f"Valid seeds: {v['valid_seeds']}; excluded: {list(v['invalid_seeds'])}.", '',
             '| Rule | ' + ' | '.join(names) + ' |', '|---|' + '---:|' * len(names)]
    for sk, data in v['per_rule'].items():
        lines.append('| ' + sk + ' | ' + ' | '.join(pct(data[a]) for a in names) + ' |')
    lines += ['', 'Rule C is the composition test.', f"Claim 1 comparator: {v['best_context_baseline']}.",
              'Context cost is measured on final hidden prompts, including the Context header.', '']
    for other, p in v['comparisons'].items():
        if p:
            lines.append(f"- ours - {other}: {pct(p['difference'])} [{pct(p['ci_low'])}, {pct(p['ci_high'])}], n={p['n']}.")
    lines += ['', f"Mean context tokens: {v['mean_context_tokens']}", '']
    labels = {1: 'non-inferiority', 2: 'rule learning', 3: 'not format learning', 4: 'cost', 5: 'retention and safety'}
    lines += [f"Claim {i} ({labels[i]}): {'PASS' if ok else 'FAIL'}" for i, ok in v['claims'].items()]
    lines += ['', f"At least two valid seeds: {'PASS' if v['enough_valid_seeds'] else 'FAIL'}", f"Overall verdict: {v['verdict']}"]
    return '\n'.join(lines) + '\n'


def load_v1_run(run_dir):
    from harness.step5_world import step5_stream
    from harness.runner import read_lines
    path = Path(run_dir)
    data = read_json(path / 'results.json')
    cfg = data['config']
    if cfg.get('world') != 'step5':
        raise ValueError('v1 requires step5 runs')
    stream = step5_stream(cfg['cycles'], cfg['world_seed'])
    final = cfg['cycles'] - 1
    ids = {i.key: i.id for i in stream.hidden_items(final) if i.skill == 'fact:smoke'}
    arms = data['results']
    for name, result in arms.items():
        answer_rows = read_lines(path / name / 'answers.jsonl')
        result['context_tokens'] = {r['task_id']: r.get('context_tokens') for r in answer_rows if r.get('day') == final}
    ours = arms.get('ours', {})
    scores = {r['task_id']: r for r in ours.get('scores', []) if r.get('phase') == 'hidden' and r.get('cycle') == final}
    revised, poison = scores.get(ids['fact|Alder']), scores.get(ids['fact|Birch'])
    # A gate rejection on the trusted revision cycle is conservatively treated as a rollback;
    # gate records do not establish causal attribution to a particular update.
    nights = [n for n in ours.get('sleeps', []) if n.get('cycle') == 1 and 'checks' in n]
    ours['safety'] = dict(revised=bool(revised and revised['correct']),
                          poison_rejected=bool(poison and poison['correct'] and not poison.get('absorbed')),
                          trusted_rollback=(not nights[0]['accepted']) if nights else None)
    return cfg['world_seed'], arms


def main():
    parser = argparse.ArgumentParser(description='Analyze a Step 4 run')
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--cycle', type=int)
    parser.add_argument('--v1-runs', nargs='+', type=Path, help='additional seed output directories for the v1 verdict')
    args = parser.parse_args()
    if args.v1_runs is not None:
        runs = [load_v1_run(p) for p in [args.run_dir] + args.v1_runs]
        seeds = dict(runs)
        if len(seeds) != len(runs):
            raise ValueError("duplicate world seeds")
        verdict = v1_verdict(seeds)
        report = v1_markdown(verdict)
        (args.run_dir / "v1-verdict.md").write_text(report, encoding="utf-8")
        (args.run_dir / "v1-verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        print(report, end="")
        return
    result = analyze(args.run_dir, args.cycle)
    report = markdown(result)
    (args.run_dir / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (args.run_dir / 'analysis.md').write_text(report, encoding='utf-8')
    print(report, end='')


if __name__ == '__main__':
    main()
