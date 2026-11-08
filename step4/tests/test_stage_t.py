import pytest
from harness.step5_world import Step5Source, step5_stream, RULES
from harness.step4_world import Step4Source, smoke_codes
from harness.tasks import grading_scope
from baselines.gold_context import GoldContext
from baselines.llm import FakeLLM
from step4.analyze import v1_verdict


def test_new_world_disjoint_and_last_cycle():
    forbidden = smoke_codes()
    for seed in (41, 42, 43):
        forbidden |= {i['input_key'] for i in Step4Source(seed).raw()[1] if i['skill'].startswith('rule:')}
    for seed in (44, 45, 46):
        source = Step5Source(seed)
        items = [i for i in source.raw()[1] if i['skill'].startswith('rule:')]
        assert not forbidden.intersection(i['input_key'] for i in items)
        assert len({i['input_key'] for i in items}) == len(items)
        stream = step5_stream(seed=seed)
        assert len([i for i in stream.train_items(2) if i.skill == 'rule:C']) == 24
        assert len([i for i in stream.gate_items(2) if i.skill == 'rule:C']) == 4
        assert len([i for i in stream.hidden_items(2) if i.skill == 'rule:C']) == 120
    with pytest.raises(ValueError):
        Step5Source(hidden=99)


def test_gold_train_only_and_budget():
    stream = step5_stream()
    arm = GoldContext(llm=FakeLLM(), stream=stream, context_tokens=1000)
    task = next(i for i in stream.hidden_items(2) if i.skill == 'rule:C').task('test', 2)
    text = arm.context(task)[0][1]
    assert all(RULES[k] in text for k in ('A', 'B', 'C'))
    assert text.count('Verified answer:') == 3
    train = [i for i in stream.train_items(2) if i.skill == 'rule:C']
    assert sum(i.prompt in text for i in train) == 3
    assert all(i.prompt not in text for i in stream.hidden_items(2) + stream.gate_items(2))
    arm.context_tokens = 1
    with pytest.raises(ValueError, match='context_tokens'):
        arm.context(task)


def sample():
    arms = {}
    for arm, count, tokens in [('ours', 90, 2), ('rag', 90, 20), ('hippo_only', 90, 20),
                               ('no_memory', 0, 0), ('ours_generic_only', 10, 0), ('gold_context', 95, 10)]:
        scores = [dict(phase='hidden', cycle=2, skill='rule:' + rule, task_id=f'{rule}{i}', correct=i < count)
                  for rule in 'ABC' for i in range(100)]
        scores += [dict(phase='hidden', cycle=0, skill='rule:A', task_id=f'A{i}', correct=i < 90) for i in range(100)]
        arms[arm] = dict(scores=scores, context_tokens={f'{r}{i}': tokens for r in 'ABC' for i in range(100)},
                         safety=dict(revised=True, poison_rejected=True, trusted_rollback=False))
    return arms


def test_v1_claims_and_invalid_seed():
    import copy
    base = sample()
    data = {44: base, 45: copy.deepcopy(base), 46: copy.deepcopy(base)}
    assert v1_verdict(data, draws=100)['verdict'] == 'PASS'
    for claim in range(1, 6):
        changed = copy.deepcopy(data)
        for arms in changed.values():
            if claim == 1:
                for r in arms['rag']['scores']: r['correct'] = True
            if claim == 2:
                for r in arms['ours']['scores']: r['correct'] = False
            if claim == 3:
                for r in arms['ours_generic_only']['scores']: r['correct'] = True
            if claim == 4: arms['ours']['context_tokens'] = {k: 20 for k in arms['ours']['context_tokens']}
            if claim == 5: arms['ours']['safety']['revised'] = False
        assert not v1_verdict(changed, draws=100)['claims'][claim]
    invalid = copy.deepcopy(data)
    for r in invalid[46]['gold_context']['scores']: r['correct'] = False
    verdict = v1_verdict(invalid, draws=100)
    assert verdict['verdict'] == 'PASS' and verdict['valid_seeds'] == [44, 45]
    assert v1_verdict({44: base}, draws=100)['verdict'] == 'FAIL'


def test_world_specific_smoke_digest(monkeypatch):
    from pathlib import Path
    from step4.run import settings
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv('S4_BACKEND', 'fake')
    monkeypatch.setenv('S4_WORLD', 'step5')
    monkeypatch.setenv('S4_WORLD_SEED', '44')
    cfg, _, _, smoke = settings(root)
    assert cfg['world'] == 'step5'
    monkeypatch.setenv('S4_WORLD_SEED', '45')
    assert settings(root)[3] == smoke
    monkeypatch.setenv('S4_WORLD', 'step4')
    assert settings(root)[3] != smoke
