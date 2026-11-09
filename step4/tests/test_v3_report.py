"""Synthetic registered worlds and fake CPU workers; no model downloads."""
import ast
import base64
import copy
import io
import json
from pathlib import Path
import zipfile

import pytest

from baselines.v3 import shape_signature
from harness.step5_world import step5_stream
from harness.tasks import grading_scope
from step4 import run, v2_analyze as v2, v3_report as v3
from step4.prepare import prepare
from step4.tests.test_v2_analyze import sample

ROOT = Path(__file__).resolve().parents[2]


def synthetic():
    """Start with the frozen v2 fixture; the former control is deliberately good."""
    arms = sample()
    arms['wrong_rule'] = arms.pop('generic_forced_noret')
    arms['wrong_rule']['control_validity'] = dict(valid=True, rule_items=360,
                                                 well_formed=360, well_formed_rate=1.)
    arms['generic_forced_noret'] = copy.deepcopy(arms['expert_only'])
    arms['ours']['stats'] = dict(fallback_fired=7)
    return arms


@pytest.fixture
def data():
    return {s: synthetic() for s in v3.V3_SEEDS}


def test_frozen_claims_and_inference_identical_with_only_control_replaced(data):
    original = copy.deepcopy(data)
    report = v3.v3_verdict(data)
    adapted = {s - 5: {**a, 'generic_forced_noret': a['wrong_rule']} for s, a in data.items()}
    frozen = v2.v2_verdict(adapted)
    assert data == original  # adapter does not mutate inputs or frozen globals
    assert report['verdict'] == 'PASS' and report['procedure'] == 'READY'
    for claim in (1, 2, 4, 5):
        assert report['claims'][claim] == frozen['claims'][claim]
    assert report['claims'][3] is True  # using the old control would fail
    assert report['valid_seeds'] == [53, 54, 55]
    assert report['comparisons']['pooled']['wrong_rule'] == frozen['comparisons']['pooled']['generic_forced_noret']
    assert report['comparisons']['pooled']['wrong_rule']['pooled']['draws'] == 10000
    assert report['mean_context_tokens'] == frozen['mean_context_tokens']
    assert report['fallback_fired'][53]['ours'] == 7
    text = v3.v3_markdown(report)
    assert 'Claim 3: PASS' in text and 'expert_only minus wrong_rule' in text
    assert 'expert_only minus generic_forced_noret' not in text and 'fallback_fired' in text
    assert 'Mean context tokens' in text and '| 53 |' in text


@pytest.mark.parametrize('claim', [1, 2, 4, 5])
def test_unchanged_claim_failures_match_frozen_path(data, claim):
    for arms in data.values():
        if claim in (1, 2):
            for row in arms['ours']['scores']:
                row['correct'] = False
        elif claim == 4:
            arms['ours']['context_tokens'] = {k: 3 for k in arms['ours']['context_tokens']}
        else:
            arms['ours']['fact_answers'][0]['category'] = 'stale'
    adapted = {s - 5: {**a, 'generic_forced_noret': a['wrong_rule']} for s, a in data.items()}
    assert v3.v3_verdict(data)['claims'][claim] == v2.v2_verdict(adapted)['claims'][claim] == False


@pytest.mark.parametrize('failure', ['bad-shape', 'missing-gate', 'training', 'margin', 'duplicate-score'])
def test_claim3_fails_closed_without_authorizing_replacement(data, failure):
    control = data[54]['wrong_rule']
    if failure == 'bad-shape':
        control['control_validity'] = dict(valid=False, rule_items=360, well_formed_rate=.89)
    elif failure == 'missing-gate':
        del control['control_validity']
    elif failure == 'training':
        control['sleeps'][0]['steps'] += 1
    elif failure == 'margin':
        for arms in data.values():
            for row in arms['wrong_rule']['scores']:
                if row['cycle'] == 2:
                    row['correct'] = True
    else:
        control['scores'].append(dict(control['scores'][0]))
    report = v3.v3_verdict(data)
    assert not report['void_seeds'] and not report['invalid_seeds']
    assert report['claims'][3] == ('UNINFORMATIVE' if failure in ('bad-shape', 'missing-gate') else False)
    assert report['verdict'] != 'PASS'
    text = v3.v3_markdown(report)
    assert 'Claim 3: PASS' not in text
    if failure in ('bad-shape', 'missing-gate'):
        assert 'Claim 3: UNINFORMATIVE' in text


def void(arms):
    for row in arms['gold_context']['scores']:
        row['correct'] = False


def test_no_subset_verdicts_or_bootstrap_before_complete_seeds(data, monkeypatch):
    monkeypatch.setattr(v2, 'v2_verdict', lambda _: pytest.fail('no subset analysis'))
    del data[55]
    report = v3.v3_verdict(data)
    assert report['procedure'] == 'PENDING' and report['pending_seeds'] == [55]
    assert all(c is None for c in report['claims'].values())
    assert 'Claim 3: PASS' not in v3.v3_markdown(report)


def test_void_replacements_are_required_in_order(data):
    void(data[53])
    pending = v3.v3_verdict(data)
    assert pending['procedure'] == 'PENDING' and pending['pending_seeds'] == [56]
    skipped = {**data, 57: synthetic()}
    assert v3.v3_verdict(skipped)['procedure'] == 'STOP'
    data[56] = synthetic()
    assert v3.v3_verdict(data)['valid_seeds'] == [54, 55, 56]
    assert v3.v3_verdict(data)['verdict'] == 'PASS'
    void(data[56])
    assert v3.v3_verdict(data)['procedure'] == 'PENDING'
    data[57] = synthetic()
    assert v3.v3_verdict(data)['valid_seeds'] == [54, 55, 57]
    assert v3.v3_verdict(data)['verdict'] == 'PASS'


@pytest.mark.parametrize('difference,low,passes', [(.30, .01, True), (.29999, .01, False), (.30, 0., False)])
def test_claim3_registered_margin_and_strict_ci(data, monkeypatch, difference, low, passes):
    monkeypatch.setattr(v2, 'paired_item_bootstrap', lambda first, second:
                        dict(n=len(first), difference=difference, ci_low=low, ci_high=.5, draws=10000))
    assert v3.v3_verdict(data)['claims'][3] is passes


@pytest.mark.parametrize('valid_count', [0, 1, 2])
def test_exhausted_replacements_fewer_than_three_wording(data, valid_count):
    for seed in v3.V3_SEEDS:
        void(data[seed])
    data.update({56: synthetic(), 57: synthetic()})
    for seed in (56, 57)[:2 - valid_count]:
        void(data[seed])
    report = v3.v3_verdict(data)
    assert len(report['valid_seeds']) == valid_count
    assert report['procedure'] == 'READY'
    assert report['verdict'] == ('PASS' if valid_count == 2 else 'FAIL')
    text = v3.v3_markdown(report)
    for line in text.splitlines():
        if line.startswith(('Claim 1:', 'Overall verdict:')):
            assert 'Fewer than 3 valid seeds' in line and '53' in line and 'gold_context' in line


@pytest.mark.parametrize('failure', ['invalid-gate', 'unregistered', 'unauthorized-replacement', 'dead-worker'])
def test_invalid_seed_stops_before_any_verdict(data, failure, monkeypatch):
    if failure == 'invalid-gate':
        data[54]['gold_context']['scores'].pop(0)
    elif failure == 'unregistered':
        data[48] = synthetic()
    elif failure == 'dead-worker':
        data[54]['ours']['status'] = 'failed'
    else:
        data[56] = synthetic()
    monkeypatch.setattr(v2, 'v2_verdict', lambda _: pytest.fail('stop before verdict'))
    report = v3.v3_verdict(data)
    assert report['procedure'] == 'STOP'
    assert 'Stop on invalid seed' in v3.v3_markdown(report)


def test_duplicate_seeds_rejected(data):
    with pytest.raises(ValueError, match='duplicate world seeds'):
        v3.v3_verdict(list(data.items()) + [(53, synthetic())])


@pytest.mark.parametrize('failure', ['missing-arm', 'missing-status'])
def test_missing_worker_evidence_stops_before_verdict(data, failure, monkeypatch):
    if failure == 'missing-arm':
        del data[54]['ours']
    else:
        del data[54]['ours']['status']
    monkeypatch.setattr(v2, 'v2_verdict', lambda _: pytest.fail('stop before verdict'))
    report = v3.v3_verdict(data)
    assert all(gate['status'] == 'valid' for gate in report['gates'].values())
    assert report['procedure'] == report['verdict'] == 'STOP'
    assert 'ours' in report['invalid_seeds'][54]
    assert all(claim is None for claim in report['claims'].values())


def write_run(path, seed=53):
    """Fake generations on the actual registered world; no training/inference."""
    arms = {}
    stream = step5_stream(3, seed)
    for arm in v3.V3_ARMS:
        scores, answers = [], []
        for cycle in range(3):
            for item in stream.hidden_items(cycle):
                with grading_scope():
                    target = item.expected_at(cycle)
                output = target.lower() if arm == 'expert_only' else target
                if item.skill in v2.RULES:
                    if arm == 'no_memory':
                        output = 'unknown'
                    elif arm == 'wrong_rule':
                        output = shape_signature(target).replace('a', 'z').replace('0', '9')
                answers.append(dict(arm=arm, day=cycle, task_id=item.id, skill=item.skill,
                                    family='rule' if item.skill in v2.RULES else 'fact',
                                    answer='Final answer: ' + output, context_tokens=2 if arm == 'ours' else 8))
                scores.append(dict(phase='hidden', cycle=cycle, task_id=item.id, skill=item.skill,
                                   correct=False))  # loader must regrade, not trust this
        arms[arm] = dict(status='complete', scores=scores,
                         sleeps=[dict(cycle=c, checks={}, accepted=True, examples=8, steps=16) for c in range(3)])
        (path / arm).mkdir(parents=True)
        (path / arm / 'answers.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in answers), encoding='utf-8')
    (path / 'results.json').write_text(json.dumps(dict(config=dict(world='step5', cycles=3, world_seed=seed),
                                                     results=arms)), encoding='utf-8')


def test_loader_real_seed_casefold_cycle2_and_complete_gate(tmp_path):
    write_run(tmp_path)
    seed, arms = v3.load_v3_run(tmp_path)
    assert seed == 53
    assert v2._vector(arms['expert_only'], 'pooled') == v2._vector(arms['gold_context'], 'pooled')
    assert all(r['correct_exact'] is False for r in v2._rows(arms['expert_only'], 'pooled'))
    assert arms['wrong_rule']['control_validity']['valid']
    assert arms['wrong_rule']['control_validity']['rule_items'] == 360
    assert all(not a['missing'] and not a['duplicates'] for a in arms['ours']['answer_row_audit'].values())
    for other in (54, 55):
        write_run(tmp_path / str(other), other)
    report = v3.v3_verdict([(seed, arms)] + [v3.load_v3_run(tmp_path / str(s)) for s in (54, 55)])
    assert report['verdict'] == 'PASS' and report['claims'][3] is True


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'unknown', 'conflicting'])
def test_loader_control_gate_and_outside_final_cycle_audit(tmp_path, failure):
    write_run(tmp_path)
    path = tmp_path / 'wrong_rule/answers.jsonl'
    rows = [json.loads(r) for r in path.read_text().splitlines()]
    final = next(r for r in rows if r['day'] == 2 and r['family'] == 'rule')
    if failure == 'missing':
        rows.remove(final)
    elif failure == 'duplicate':
        rows.append(dict(final))
    elif failure == 'unknown':
        rows.append({**final, 'task_id': 'unknown'})
    else:
        for row in rows:
            if row['day'] == 2 and row['family'] == 'rule':
                row['answer'] = 'Final answer: XYZ\nFinal answer: ABC-7'
    early = next(r for r in rows if r['day'] == 0 and r['family'] == 'rule')
    rows.append(dict(early))
    missing = next(r for r in rows if r['day'] == 1 and r['family'] == 'rule')
    rows.remove(missing)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    _, arms = v3.load_v3_run(tmp_path)
    assert not arms['wrong_rule']['control_validity']['valid']
    report = v3.v3_verdict({53: arms, 54: synthetic(), 55: synthetic()})
    assert report['claims'][3] == 'UNINFORMATIVE'
    assert 'Claim 3: PASS' not in v3.v3_markdown(report)
    audit = arms['wrong_rule']['answer_row_audit']
    assert audit[0]['duplicates'][early['task_id']] == 2
    assert missing['task_id'] in audit[1]['missing']


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    for flag in ('FACT_FILTER', 'FALLBACK', 'WRONG_RULE'):
        monkeypatch.delenv('S4_V3_' + flag, raising=False)
    for key, value in dict(S4_BACKEND='fake', S4_WORLD='step5', S4_WORLD_SEED='53',
                           S4_ARMS='no_memory', S4_CYCLES='3', S4_STEPS='2').items():
        monkeypatch.setenv(key, value)
    return run.settings(ROOT)[0]


@pytest.mark.parametrize('seed', [53, 54, 55, 56, 57])
def test_seed_dispatch_never_reaches_v1_or_v2(seed, cfg, tmp_path):
    cfg['world_seed'] = seed
    assert run.is_v3_run(cfg)
    run.report(tmp_path, cfg, [], {}, False, 0)
    assert (tmp_path / 'v3-verdict.json').is_file()
    text = (tmp_path / 'RESULTS.md').read_text()
    assert 'Answer-row audit' in text and 'fallback_fired' in text and 'Mean context tokens' in text
    assert not (tmp_path / 'v1-verdict.json').exists()
    assert not (tmp_path / 'v2-verdict.json').exists()
    assert v3.load_v3_run(tmp_path) == (seed, {})


@pytest.mark.parametrize('flag', ['v3_fact_filter', 'v3_fallback', 'v3_wrong_rule'])
def test_flag_selects_v3_and_dev_seeds_never_get_confirmation_verdict(flag):
    assert run.is_v3_run(dict(world='step5', world_seed=48, **{flag: True}))
    assert not run.is_v3_run(dict(world='step5', world_seed=48, **{flag: False}))


def test_v3_settings_three_cycles_and_development_seed_rejection(cfg, monkeypatch, tmp_path):
    assert cfg['protocol'] == 'v3' and 'semantic_retriever' in cfg
    monkeypatch.setenv('S4_CYCLES', '4')
    with pytest.raises(ValueError, match='v3 requires exactly 3 cycles'):
        run.settings(ROOT)
    write_run(tmp_path)
    path = tmp_path / 'results.json'
    payload = json.loads(path.read_text())
    payload['config']['world_seed'] = 48
    payload['config']['v3_fallback'] = True
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='v3 seeds'):
        v3.load_v3_run(tmp_path)


@pytest.mark.parametrize('seed,flag', [(53, None), (57, None), (44, 'v3_fallback')])
def test_resumed_worker_rows_equal_uninterrupted(seed, flag, cfg, tmp_path):
    cfg['world_seed'] = seed
    if flag:
        cfg[flag] = True
    name = 'no_memory'
    assert run.worker(name, cfg, tmp_path, False) == 0
    path = tmp_path / name / 'answers.jsonl'
    original = path.read_bytes()
    records_path = tmp_path / name / 'records.jsonl'
    records = [json.loads(r) for r in records_path.read_text().splitlines()]
    records = [r for r in records if r['type'] != 'run_summary' and
               not (r['type'] == 'cycle_done' and r['cycle'] == 2)]
    records_path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    assert run.worker(name, cfg, tmp_path, False) == 0
    assert path.read_bytes() == original
    # Completed-cycle duplicates must remain visible to the validity audit.
    row = next(json.loads(r) for r in original.decode().splitlines() if json.loads(r)['day'] == 2)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    run.discard_unfinished_v2_answers(path.parent)
    assert len(path.read_text().splitlines()) == len(original.decode().splitlines()) + 1


def test_report_packaged_and_source_digest_covers_it(cfg, tmp_path):
    target = prepare(world='step5', world_seed=53, arms='no_memory', target=tmp_path / 'prepared')
    tree = ast.parse((target / 'run.py').read_text(encoding='utf-8'))
    packed = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'SOURCE_B64' for t in n.targets))
    unpacked = tmp_path / 'unpacked'
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(packed))) as archive:
        assert 'step4/v3_report.py' in archive.namelist()
        archive.extractall(unpacked)
    digest = run.source_digest(unpacked)
    assert digest == run.source_digest(ROOT)
    with (unpacked / 'step4/v3_report.py').open('a') as f:
        f.write('\n# test source coverage\n')
    assert run.source_digest(unpacked) != digest
