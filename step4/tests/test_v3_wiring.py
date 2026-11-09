"""Offline v3 registration, logging and control-validity checks."""
import json
from pathlib import Path

import pytest

from baselines.sleep import ExpertOnly, SleepArm
from baselines.v3 import FALLBACK_TRIGGER_K
from baselines.wrong_rule import WrongRuleSleepArm
from step4 import run
from step4.v3_control_validity import control_validity


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_flags(monkeypatch):
    for flag in ('FACT_FILTER', 'FALLBACK', 'WRONG_RULE'):
        monkeypatch.delenv('S4_V3_' + flag, raising=False)
    monkeypatch.setenv('S4_BACKEND', 'fake')
    monkeypatch.setenv('S4_WORLD', 'step5')
    monkeypatch.setenv('S4_WORLD_SEED', '48')
    monkeypatch.setenv('S4_ARMS', 'ours,expert_only')
    monkeypatch.setenv('S4_STEPS', '2')


def test_flags_keys_registration_and_factories(clean_flags, monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    cfg, arms, key, smoke_key = run.settings(ROOT)
    assert not any(k.startswith('v3_') for k in cfg)
    assert 'wrong_rule' not in run.available_arms(cfg)
    monkeypatch.setenv('S4_ARMS', 'wrong_rule')
    with pytest.raises(ValueError, match='S4_ARMS'):
        run.settings(ROOT)
    with pytest.raises(ValueError, match='v3_wrong_rule'):
        run.make_arm('wrong_rule', run.make_backend(cfg, None), cfg, None)
    for flag in ('FACT_FILTER', 'FALLBACK', 'WRONG_RULE'):
        monkeypatch.setenv('S4_V3_' + flag, '1')
    enabled, arms, new_key, new_smoke = run.settings(ROOT)
    assert arms == ['wrong_rule'] and 'wrong_rule' in run.available_arms(enabled)
    assert key != new_key and smoke_key != new_smoke
    assert run.ARMS == ('ours', 'expert_only', 'hippo_only', 'rag', 'text_notes', 'no_memory',
                        'rag_starved', 'gold_context')
    for name, cls in [('ours', SleepArm), ('expert_only', ExpertOnly), ('wrong_rule', WrongRuleSleepArm)]:
        arm = run.make_arm(name, run.make_backend(cfg, None), enabled, None)
        assert isinstance(arm, cls)
        assert arm.cfg.v3_fallback == (name == 'ours')
        assert arm.cfg.v3_fact_filter == (name != 'wrong_rule')
    assert run.retriever_config(arm)['enabled'] is False
    monkeypatch.setenv('S4_ARMS', 'ours,expert_only')
    assert run.settings(ROOT)[1] == ['ours', 'expert_only']  # enabling never auto-schedules control


def test_fallback_worker_logs_context_and_counter(clean_flags, monkeypatch, tmp_path):
    import tempfile
    from baselines.llm import FakeLLM
    from baselines.sleep_fake import FakeBackend
    from types import SimpleNamespace
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    cfg = run.settings(ROOT)[0]
    cfg['v3_fallback'] = True
    original_make = run.make_arm
    def make(name, backend, config, oracle, stream):
        arm = original_make(name, backend, config, oracle, stream)
        # Exercise the real worker/logging/checkpoint flow, with no model training.
        arm.sleep = lambda cycle: {}
        arm.route = lambda prompt: 'expert-test'
        text = 'Question: Demo\nVerified answer: ABC-7'
        def recall(query, k, **kwargs):
            assert k in (FALLBACK_TRIGGER_K, arm.k)
            return [SimpleNamespace(id='m0', text=text, kind='episode')]
        original_reset = arm.reset
        def reset():
            original_reset()
            arm.hippo.add(text, id='m0', kind='episode', source='system', t=0)
            arm.hippo.recall = recall
        arm.reset = reset
        arm.hippo.recall = recall
        return arm
    def backend(config, out):
        b = FakeBackend()
        b.count_tokens = FakeLLM().count_tokens
        b.generate = lambda prompt, expert=None: 'Final answer: ABC-7' if 'Verified answer:' in prompt else 'Final answer: ABC'
        return b
    monkeypatch.setattr(run, 'make_arm', make)
    monkeypatch.setattr(run, 'make_backend', backend)
    assert run.worker('ours', cfg, tmp_path, False) == 0
    result = json.loads((tmp_path / 'ours/result.json').read_text())
    rows = [json.loads(line) for line in (tmp_path / 'ours/answers.jsonl').read_text().splitlines()]
    assert rows and all(r['fallback'] for r in rows)
    b = backend(cfg, None)
    for row in rows:
        assert len(row['prompts']) == 2 and row['prompt'] == row['prompts'][-1]
        assert row['context_tokens'] == sum(run.context_block_tokens(b, p) for p in row['prompts'])
        assert row['context_tokens'] > run.context_block_tokens(b, row['prompt'])
    assert result['stats']['fallback_fired'] >= len(rows)
    assert result['retriever']['enabled'] is True


def rows(n=360, good=324):
    return [dict(arm='wrong_rule', family='rule', day=2, task_id=str(i), correct='B-ABC-7',
                 answer='Final answer: X-XYZ-1' if i < good else 'Final answer: XYZ') for i in range(n)]


def expected():
    return {(2, str(i)): 'B-ABC-7' for i in range(360)}


def test_validity_threshold_and_wrong_but_well_formed():
    result = control_validity(rows(), expected=expected())
    assert result == dict(arm='wrong_rule', rule_items=360, well_formed=324, well_formed_rate=0.9, valid=True)
    assert control_validity(rows(good=323), expected=expected())['valid'] is False
    extra = [dict(arm='ours', family='rule'), dict(arm='wrong_rule', family='fact')]
    assert control_validity(rows() + extra, expected=expected()) == result


def test_validity_raw_logs_need_targets_and_never_use_correct_bool():
    data = rows()
    for row in data:
        row['correct'] = False
    with pytest.raises(ValueError, match='expected answer text'):
        control_validity(data)
    assert control_validity(data, expected=expected())['valid'] is True
    assert control_validity(data, expected={str(i): 'B-ABC-7' for i in range(360)})['valid'] is True
    data[0]['answer'] = None
    data[0]['expected'] = 'XYZ'  # row metadata cannot override registered targets
    assert control_validity(data, expected=expected())['valid'] is False


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'unknown', 'wrong-cycle', 'empty', 'short-universe'])
def test_validity_requires_complete_universe(kind):
    data, targets = rows(good=360), expected()
    if kind == 'missing':
        data = data[:1]  # one good row must never establish 90% over 360
    elif kind == 'duplicate':
        data.append(dict(data[0]))
    elif kind == 'unknown':
        data.append({**data[0], 'task_id': 'unregistered'})
    elif kind == 'wrong-cycle':
        data[0]['day'] = 1
    elif kind == 'empty':
        data = []
    else:
        targets = {(2, '0'): 'B-ABC-7'}
    with pytest.raises(ValueError, match='missing|duplicate|unknown|360'):
        control_validity(data, expected=targets)


def test_validity_conflicting_final_answers_are_malformed():
    data = rows()
    data[0]['answer'] = 'Final answer: XYZ\nFinal answer: X-XYZ-1'
    assert not control_validity(data, expected=expected())['valid']
    data[0]['answer'] = 'Final answer: XYZ\nFinal answer: ABC-7'
    targets = expected()
    targets[(2, '0')] = 'ABC-7'
    assert not control_validity(data, expected=targets)['valid']


@pytest.mark.parametrize('name', ['expert_only', 'wrong_rule'])
def test_control_run_config_never_retries(clean_flags, monkeypatch, tmp_path, name):
    from baselines.sleep_fake import FakeBackend
    from harness.tasks import Task
    cfg = run.settings(ROOT)[0]
    cfg.update(v3_fallback=True, v3_wrong_rule=True)
    backend = FakeBackend()
    arm = run.make_arm(name, backend, cfg, None)
    arm.route = lambda prompt: 'expert'
    arm.hippo.recall = lambda *args, **kwargs: pytest.fail('control must not recall for fallback')
    backend.generate = lambda prompt, expert=None: 'Final answer: malformed'
    arm.answer(Task('q', 'question', 'rule', 'rule:test', mode='day'))
    assert not arm.cfg.v3_fallback and arm.fallback_fired == 0
    assert 'prompts' not in arm.answer_trace and arm.answer_trace['retrieved'] == []
    assert run.retriever_config(arm)['enabled'] is False


def test_validity_accepts_case_variations():
    data = rows(good=360)
    data[0]['answer'] = 'Final answer: x-xYz-1'
    assert control_validity(data, expected=expected())['well_formed'] == 360
