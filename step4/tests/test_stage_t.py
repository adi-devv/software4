import copy
import json

import pytest
from harness.step5_world import Step5Source, step5_stream, RULES
from harness.step4_world import Step4Source, smoke_codes
from harness.tasks import grading_scope
from baselines.gold_context import GoldContext
from baselines.llm import FakeLLM
from step4.analyze import v1_verdict, v1_markdown, load_v1_run


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


ARM_SPEC = [('ours', 90, 2), ('rag', 90, 20), ('hippo_only', 90, 20), ('expert_only', 50, 0),
            ('text_notes', 5, 12), ('no_memory', 0, 0), ('ours_generic_only', 10, 0), ('gold_context', 100, 10)]


def sample():
    """All 8 arms, 120 hidden items per rule at the final cycle, plus cycle-0 rule A rows."""
    arms = {}
    for arm, pct, tokens in ARM_SPEC:
        scores = [dict(phase='hidden', cycle=2, skill='rule:' + rule, task_id=f'{rule}{i}', correct=i * 100 < pct * 120)
                  for rule in 'ABC' for i in range(120)]
        scores += [dict(phase='hidden', cycle=0, skill='rule:A', task_id=f'A{i}', correct=i * 100 < 90 * 120)
                   for i in range(120)]
        keys = {f'{r["cycle"]}:{r["task_id"]}': tokens for r in scores}
        arms[arm] = dict(scores=scores, context_tokens=keys,
                         safety=dict(revised=True, poison_rejected=True, trusted_rollback=False))
    return arms


def seeds(n=3):
    return {s: copy.deepcopy(sample()) for s in (44, 45, 46)[:n]}


def test_v1_claims_and_invalid_seed():
    base = sample()
    data = seeds()
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


def test_v1_requires_all_arms_and_items():
    data = seeds()
    for s in (44, 45):
        del data[s]['text_notes']
    out = v1_verdict(data, draws=100)
    assert out['valid_seeds'] == [46] and out['verdict'] == 'FAIL'
    short = seeds()
    short[44]['rag']['scores'] = [r for r in short[44]['rag']['scores'] if r['task_id'] != 'B0']
    short[45]['ours']['scores'][1]['task_id'] = 'renamed'
    out = v1_verdict(short, draws=100)
    assert out['valid_seeds'] == [46] and out['verdict'] == 'FAIL'


def test_v1_registered_seeds_only():
    base = sample()
    data = {41: base, 42: copy.deepcopy(base), 44: copy.deepcopy(base)}
    out = v1_verdict(data, draws=100)
    assert out['valid_seeds'] == [44] and out['invalid_seeds'][41] == 'unregistered seed'
    assert out['verdict'] == 'FAIL'


def test_v1_cost_covers_every_hidden_prompt():
    data = seeds()
    for arms in data.values():
        # A cheap final cycle must not hide an expensive early cycle.
        for k in arms['ours']['context_tokens']:
            if k.startswith('0:'):
                arms['ours']['context_tokens'][k] = 1000
    assert not v1_verdict(data, draws=100)['claims'][4]
    for arms in data.values():
        del arms['rag']['context_tokens']['0:A0']
    assert v1_verdict(data, draws=100)['mean_context_tokens']['rag'] is None
    none_case = seeds()
    for arms in none_case.values():
        arms['ours']['context_tokens']['2:A5'] = None
    out = v1_verdict(none_case, draws=100)
    assert out['mean_context_tokens']['ours'] is None and not out['claims'][4]


def test_v1_single_rule_qualification():
    data = seeds()
    for arms in data.values():
        for arm in ('ours', 'rag', 'hippo_only'):
            for r in arms[arm]['scores']:
                if r['cycle'] == 2:
                    r['correct'] = r['skill'] != 'rule:C'
    out = v1_verdict(data, draws=100)
    assert out['rules_learned'] == {'rule:A': True, 'rule:B': True, 'rule:C': False}
    assert 'single-rule only' in out['qualification']
    assert 'single-rule only' in v1_markdown(out)
    clean = v1_verdict(seeds(), draws=100)
    assert clean['qualification'] is None
    assert 'single-rule only' not in v1_markdown(clean)


def _run_dir(path, *, sleeps, seed=44, cycles=3, world='step5'):
    arms = {}
    for name in ('ours', 'rag'):
        d = path / name
        d.mkdir(parents=True)
        (d / 'answers.jsonl').write_text('', encoding='utf-8')
        arms[name] = dict(scores=[], sleeps=sleeps if name == 'ours' else [])
    for item in step5_stream(3, 44).hidden_items(2):
        if item.key in ('fact|Alder', 'fact|Birch'):
            arms['ours']['scores'].append(dict(phase='hidden', cycle=2, task_id=item.id, correct=True))
    (path / 'results.json').write_text(json.dumps(dict(config=dict(world=world, cycles=cycles, world_seed=seed),
                                                       results=arms)), encoding='utf-8')
    return path


def test_loader_rollback_any_cycle_and_fail_closed(tmp_path):
    ok = [dict(cycle=c, accepted=True, checks={}) for c in range(3)]
    _, arms = load_v1_run(_run_dir(tmp_path / 'ok', sleeps=ok))
    assert arms['ours']['safety']['trusted_rollback'] is False
    bad = ok[:2] + [dict(cycle=2, accepted=False, checks={})]
    _, arms = load_v1_run(_run_dir(tmp_path / 'bad', sleeps=bad))
    assert arms['ours']['safety']['trusted_rollback'] is True
    _, arms = load_v1_run(_run_dir(tmp_path / 'missing', sleeps=ok[:2]))
    assert arms['ours']['safety']['trusted_rollback'] is None
    _, arms = load_v1_run(_run_dir(tmp_path / 'nocheck', sleeps=[dict(cycle=c, accepted=True) for c in range(3)]))
    assert arms['ours']['safety']['trusted_rollback'] is None


def test_loader_rejects_unregistered_runs(tmp_path):
    ok = [dict(cycle=c, accepted=True, checks={}) for c in range(3)]
    with pytest.raises(ValueError, match='seeds'):
        load_v1_run(_run_dir(tmp_path / 'seed', sleeps=ok, seed=41))
    with pytest.raises(ValueError, match='cycles'):
        load_v1_run(_run_dir(tmp_path / 'cyc', sleeps=ok, cycles=4))
    with pytest.raises(ValueError, match='step5'):
        load_v1_run(_run_dir(tmp_path / 'world', sleeps=ok, world='step4'))


def test_context_block_tokens_never_counts_whole_prompt():
    from step4.run import context_block_tokens

    class B:
        count_tokens = staticmethod(lambda t: len(t.split()))
    assert context_block_tokens(B, 'a b c\n\nTask:\nx y z w') == 3
    assert context_block_tokens(B, 'no separator here') == 0
    assert context_block_tokens(B, None) == 0


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
