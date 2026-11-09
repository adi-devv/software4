"""Describe rule-answer miss types in step5 runs; no verdict is produced.

Expected answers are reconstructed from the same registered world as v2_analyze.
Percentages use all matched rule answers in each arm/rule/cycle as denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.grading import _extract, match
from harness.step5_world import step5_stream
from harness.tasks import grading_scope

CATEGORIES = ('letter_error', 'dropped_suffix', 'garbled_format', 'other')
RULES = ('rule:A', 'rule:B', 'rule:C')


def shape(text):
    return ''.join(('A' if c.isupper() else 'a') if c.isalpha()
                   else '0' if c.isdigit() else c for c in text)


def classify(answer, expected):
    """Return None for a graded correct response, otherwise one miss category."""
    if not isinstance(answer, str):
        return 'other'
    text, _ = _extract(answer)
    if text is None or not text.strip():
        return 'other'
    if match(expected, text):
        return None
    text, expected = text.strip().casefold(), expected.strip().casefold()
    if shape(text) == shape(expected):
        # Same shape can also arise from differing digits; reserve letter_error
        # for cases where all nonletters are identical and letters differ.
        if any(a != b and a.isalpha() and b.isalpha() for a, b in zip(text, expected)) and all(
                a == b or (a.isalpha() and b.isalpha()) for a, b in zip(text, expected)):
            return 'letter_error'
        return 'other'
    suffix = re.search(r'-\d+$', expected)
    if suffix and text == expected[:suffix.start()]:
        return 'dropped_suffix'
    return 'garbled_format'


def analyze(run_dirs):
    counts = defaultdict(Counter)
    issues = []
    for run in map(Path, run_dirs):
        try:
            config = json.loads((run / 'results.json').read_text(encoding='utf-8'))['config']
            if config.get('world') != 'step5':
                raise ValueError('requires a step5 world')
            cycles, seed = config['cycles'], config['world_seed']
            stream = step5_stream(cycles, seed)
            items = {c: {item.id: item for item in stream.hidden_items(c)} for c in range(cycles)}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f'{run}: cannot reconstruct expected answers: {exc}')
            continue
        files = sorted(run.glob('*/answers.jsonl'))
        if not files:
            issues.append(f'{run}: no arm answers.jsonl files')
        for path in files:
            try:
                with path.open(encoding='utf-8') as source:
                    for line_no, line in enumerate(source, 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            if not isinstance(row, dict):
                                raise ValueError('row is not an object')
                        except (ValueError, TypeError) as exc:
                            issues.append(f'{path}:{line_no}: {exc}')
                            continue
                        if row.get('skill') not in RULES:
                            continue
                        cycle = row.get('day')
                        item = items.get(cycle, {}).get(row.get('task_id'))
                        if item is None or item.skill != row['skill']:
                            issues.append(f'{path}:{line_no}: unknown hidden task or skill')
                            continue
                        with grading_scope():
                            expected = item.expected_at(cycle)
                        key = (path.parent.name, row['skill'], cycle)
                        category = classify(row.get('answer'), expected)
                        counts[key]['answers'] += 1
                        if category:
                            counts[key][category] += 1
            except OSError as exc:
                issues.append(f'{path}: {exc}')
    rows = []
    for (arm, rule, cycle), tally in sorted(counts.items()):
        n = tally['answers']
        rows.append(dict(arm=arm, rule=rule, cycle=cycle, answers=n,
                         counts={c: tally[c] for c in CATEGORIES},
                         percentages={c: 100 * tally[c] / n for c in CATEGORIES}))
    return {'rows': rows, 'issues': issues}


def markdown(data):
    lines = ['# Rule answer miss types', '',
             'Percentages use all matched rule answers in the arm, rule, and cycle. '
             'Runs are pooled; duplicate logged answers are counted.', '']
    arms = sorted({row['arm'] for row in data['rows']})
    for arm in arms:
        lines += [f'## {arm}', '', '| Rule | Cycle | Answers | ' + ' | '.join(CATEGORIES) + ' |',
                  '|---|---:|---:|' + '---:|' * len(CATEGORIES)]
        for row in data['rows']:
            if row['arm'] == arm:
                values = [f"{row['counts'][c]} ({row['percentages'][c]:.1f}%)" for c in CATEGORIES]
                lines.append(f"| {row['rule']} | {row['cycle']} | {row['answers']} | " + ' | '.join(values) + ' |')
        lines.append('')
    if data['issues']:
        lines += ['## Unavailable evidence', ''] + [f'- {issue}' for issue in data['issues']] + ['']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', nargs='+')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    data = analyze(args.run_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'v3-misstypes.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    (args.out / 'v3-misstypes.md').write_text(markdown(data), encoding='utf-8')


if __name__ == '__main__':
    main()
