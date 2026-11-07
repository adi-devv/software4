import json
import subprocess
import sys
from pathlib import Path
from step4.analyze import analyze, context_matches, markdown


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


def test_fact_copy_matches_whole_values():
    prompt = 'Context:\nFact: Field note: the leader of Alder is Blue.\n\nTask:\nBlue'
    assert context_matches(prompt, 'Blue') == (True, False, False)
    assert context_matches(prompt, 'Blu') == (False, False, False)
    assert context_matches('Task:\nFactless Blue', 'Blue') == (False, False, False)


def test_costs_from_artifacts_and_missing_values(tmp_path):
    for arm in ('ours', 'rag'):
        folder = tmp_path / arm
        folder.mkdir()
        write_jsonl(folder / 'records.jsonl', [dict(type='score', arm=arm, stream='s', cycle=0,
                    attempt=1, phase='hidden', task_id='x', skill='fact:smoke', correct=True),
                    dict(type='cycle_done', arm=arm, stream='s', cycle=0, attempt=1)])
    (tmp_path / 'timings.json').write_text(json.dumps({'arms_seconds': {'ours': 1800}}))
    (tmp_path / 'ours' / 'result.json').write_text(json.dumps({'worker_device': {'scheduled': {}}}))
    (tmp_path / 'rag' / 'result.json').write_text(json.dumps({'elapsed_seconds': 900}))
    (tmp_path / 'results.json').write_text(json.dumps({'config': {'backend': 'fake'}}))
    result = analyze(tmp_path)
    assert result['cost']['arms']['ours'] == {'worker_seconds': 1800, 'devices': 1, 'gpu_seconds': 1800, 'accuracy_points_per_worker_hour': 200}
    assert result['cost']['arms']['rag']['gpu_seconds'] == 0
    assert result['cost']['total_worker_hours'] == 0.75
    assert 'Accuracy points / worker-hour' in markdown(result)
    (tmp_path / 'rag' / 'result.json').unlink()
    assert analyze(tmp_path)['cost']['total_is_partial']
