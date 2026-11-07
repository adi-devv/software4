import json
import subprocess
import sys
from pathlib import Path


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def test_analysis_cli(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'analyze.py'
    for arm in ('ours', 'expert_only'):
        folder = tmp_path / arm
        folder.mkdir()
        scores = [dict(type='score', arm=arm, stream='s', cycle=1, attempt=1, phase='hidden',
                       task_id=f't{i}', family='rule' if i < 4 else 'fact',
                       skill='rule:A' if i < 2 else 'rule:B' if i < 4 else 'fact:smoke',
                       correct=(i < (3 if arm == 'ours' else 4))) for i in range(6)]
        scores += [dict(type='score', arm=arm, stream='s', cycle=1, attempt=2, phase='hidden',
                        task_id='discarded', family='rule', skill='rule:A', correct=False),
                   dict(type='cycle_done', arm=arm, stream='s', cycle=1, attempt=1)]
        write_jsonl(folder / 'records.jsonl', scores)
    prompts = ['Question: X\nResponse: Final answer: 12\nCorrect: False\nCorrection: 13',
               'Question: X\nVerified answer: blue',
               'Question: X\nResponse: Final answer: blue\nCorrect: True', '', '', '']
    answers = ['Final answer: 12', 'Final answer: blue', 'Final answer: blue',
               'single line', 'Final answer: yes\nFinal answer: no', 'Final answer: new']
    entries = [dict(arm='ours', task_id=f't{i}', day=1, family='rule',
                    skill='rule:A' if i < 2 else 'rule:B' if i < 4 else 'fact:smoke',
                    expert=None if i == 0 else 'expert-1', retrieved=[], prompt=prompts[i], answer=answers[i])
               for i in range(6)]
    old = dict(entries[-1], answer='Final answer: old', prompt='Verified answer: old')
    write_jsonl(tmp_path / 'ours' / 'answers.jsonl', [old, *entries])
    run = subprocess.run([sys.executable, '-B', str(script), str(tmp_path)], capture_output=True, text=True, check=True)
    data = json.loads((tmp_path / 'analysis.json').read_text(encoding='utf-8'))
    ours = data['arms']['ours']
    assert data['cycle'] == 1 and ours['n'] == 6 and ours['accuracy'] == 0.5
    assert ours['skills']['rule:A']['accuracy'] == 1
    assert ours['skills']['rule:B']['accuracy'] == 0.5
    stats = ours['answers']
    assert stats['n'] == 6 and stats['copied_from_context'] == 0.5
    assert stats['copied_wrong'] == 1 / 6 and stats['copied_verified'] == 1 / 6
    assert stats['no_parseable_final_answer'] == 2 / 6
    assert stats['mean_prompt_characters'] == sum(map(len, prompts)) / 6
    assert stats['routed_experts']['rule:A'] == {'None': 1, 'expert-1': 1}
    assert 'answers' not in data['arms']['expert_only']
    pair = data['pairwise_rule'][0]
    assert pair['n'] == 4 and pair['difference'] == -0.25 and pair['draws'] == 10000
    assert 'rule:B' in run.stdout and (tmp_path / 'analysis.md').is_file()
