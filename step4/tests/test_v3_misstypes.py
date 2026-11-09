import json
import subprocess
import sys
from pathlib import Path

from harness.step5_world import step5_stream
from harness.tasks import grading_scope
from step4.v3_misstypes import classify, shape

ROOT = Path(__file__).resolve().parents[2]


def test_classification_and_grading():
    assert shape('Ab-19!') == 'Aa-00!'
    assert classify('Final answer: XYZ-7', 'ABC-7') == 'letter_error'
    assert classify('Final answer: xyz-7', 'ABC-7') == 'letter_error'
    assert classify('Final answer: ABC', 'ABC-7') == 'dropped_suffix'
    assert classify('Final answer: abc', 'ABC-7') == 'dropped_suffix'
    assert classify('Final answer: ABC_7', 'ABC-7') == 'garbled_format'
    assert classify('Final answer: ABC-8', 'ABC-7') == 'other'
    assert classify('Final answer: abc-7', 'ABC-7') is None
    assert classify('Final answer: ABC\nFinal answer: XYZ', 'ABC-7') == 'other'


def test_cli_real_log_schema_and_missing_evidence(tmp_path):
    run = tmp_path / 'run'
    arm = run / 'ours'
    arm.mkdir(parents=True)
    (run / 'results.json').write_text(json.dumps({'config': {'world': 'step5', 'cycles': 3, 'world_seed': 48}}))
    stream = step5_stream(3, 48)
    items = [i for i in stream.hidden_items(2) if i.skill == 'rule:A'][:5]
    rows = []
    for index, item in enumerate(items):
        with grading_scope():
            expected = item.expected_at(2)
        answers = [expected, 'Z' * (len(expected) - 2) + '-7', expected[:-2], expected.replace('-', '_'), '']
        rows.append(dict(arm='ours', task_id=item.id, day=2, family='rule', skill=item.skill,
                         expert=None, retrieved=[], prompt='task line', answer='Final answer: ' + answers[index],
                         context_tokens=0))
    rows.append(dict(arm='ours', task_id='unknown', day=2, skill='rule:B', answer='x'))
    (arm / 'answers.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    out = tmp_path / 'out'
    subprocess.run([sys.executable, '-B', str(ROOT / 'step4/v3_misstypes.py'), str(run),
                    str(tmp_path / 'missing'), '--out', str(out)], check=True, cwd=ROOT)
    data = json.loads((out / 'v3-misstypes.json').read_text(encoding='utf-8'))
    row, = data['rows']
    assert (row['arm'], row['rule'], row['cycle'], row['answers']) == ('ours', 'rule:A', 2, 5)
    assert row['counts'] == dict.fromkeys(('letter_error', 'dropped_suffix', 'garbled_format', 'other'), 1)
    assert set(row['percentages'].values()) == {20.0}
    assert len(data['issues']) == 2
    assert '1 (20.0%)' in (out / 'v3-misstypes.md').read_text(encoding='utf-8')
