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
    return copied, wrong, verified


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
    return result


def markdown(result):
    def pct(x):
        return 'N/A' if x is None else f'{x:.1%}'
    lines = [f"# Step 4 analysis: cycle {result['cycle']}", '', '| Arm | Skill | N | Accuracy |', '|---|---|---:|---:|']
    for arm, data in result['arms'].items():
        lines.append(f"| {arm} | overall | {data['n']} | {pct(data['accuracy'])} |")
        for skill, stats in data['skills'].items():
            lines.append(f"| {arm} | {skill} | {stats['n']} | {pct(stats['accuracy'])} |")
    lines += ['', '## Rule comparisons (first minus second)', '', '| First | Second | Paired N | Difference | 95% CI |', '|---|---|---:|---:|---:|']
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


def main():
    parser = argparse.ArgumentParser(description='Analyze a Step 4 run')
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--cycle', type=int)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.cycle)
    report = markdown(result)
    (args.run_dir / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (args.run_dir / 'analysis.md').write_text(report, encoding='utf-8')
    print(report, end='')


if __name__ == '__main__':
    main()
