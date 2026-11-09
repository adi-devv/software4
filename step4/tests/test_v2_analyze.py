"""Local CPU fixtures for the preregistered v2 verdict. No model backends."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from harness.step5_world import step5_stream
from harness.tasks import grading_scope
from step4 import v2_analyze as v2


ROOT = Path(__file__).resolve().parents[2]


def sample():
    arms = {}
    for arm in v2.V2_ARMS:
        count = 0 if arm == 'no_memory' else 72 if arm == 'generic_forced_noret' else 108
        if arm == 'gold_context':
            count = 120
        scores = [dict(phase='hidden', cycle=2, skill=rule, task_id=f'{rule}:{i:03}',
                       correct=i < count, correct_exact=i < count)
                  for rule in v2.RULES for i in range(120)]
        scores += [dict(row, cycle=0) for row in scores[:120]]
        facts = [dict(cycle=2, task_id=f'fact:{entity}', entity=entity, category='correct', format_leaked=False)
                 for entity in ('Alder', 'Birch', 'Cedar', 'Dover', 'Elm', 'Fenn')]
        scores += [dict(phase='hidden', cycle=2, skill='fact:smoke', task_id=f['task_id'],
                        correct=True, correct_exact=True) for f in facts]
        token_count = 2 if arm == 'ours' else 8
        arms[arm] = dict(status='complete', scores=scores, fact_answers=facts,
                         context_tokens={f'{r["cycle"]}:{r["task_id"]}': token_count for r in scores},
                         sleeps=[dict(cycle=c, checks={}, accepted=True, examples=9, steps=7) for c in range(3)])
    return arms


def seeds(*which):
    return {s: sample() for s in (which or (48, 49))}


def set_accuracy(arms, arm, successes, rule=None):
    for row in arms[arm]['scores']:
        if row['cycle'] == 2 and row['skill'] in v2.RULES and (rule is None or row['skill'] == rule):
            row['correct'] = int(row['task_id'].rsplit(':', 1)[1]) < successes


def test_bootstrap_exact_reference_deterministic_sorted_and_unstratified():
    left = {(49, 'z'): 1, (48, 'x'): 0, (48, 'a'): 1, (49, 'a'): 0}
    right = {key: 0 for key in reversed(left)}
    got = v2.paired_item_bootstrap(left, right)
    diff = np.array([left[key] - right[key] for key in sorted(left)])
    rng = np.random.default_rng(0)
    means = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5], method='linear')
    assert got == dict(n=4, difference=.5, ci_low=low, ci_high=high, draws=10000)
    assert got == v2.paired_item_bootstrap(dict(reversed(list(left.items()))), right)
    with pytest.raises(ValueError, match='identical'):
        v2.paired_item_bootstrap(left, {})


def test_full_pass_report_and_one_seed_fails():
    verdict = v2.v2_verdict(seeds())
    assert verdict['verdict'] == 'PASS'
    assert all(verdict['claims'].values())
    assert verdict['comparisons']['pooled']['rag_semantic']['rule:A']['n'] == 240
    assert verdict['comparisons']['pooled']['rag_semantic']['pooled']['n'] == 720
    report = v2.v2_markdown(verdict)
    assert '| 48 |' in report and '| 49 |' in report and '| pooled |' in report
    assert 'Exact-case' in report and 'Claim 3: PASS' in report
    for arm in v2.V2_ARMS:
        for rule in v2.ENDPOINTS:
            assert f'## {arm}: {rule}' in report
    assert not any(v2.v2_verdict(seeds(48))['claims'].values())
    assert v2.v2_verdict({})['verdict'] == 'FAIL'


@pytest.mark.parametrize('reject_early', [True, False])
def test_claim3_identical_observations_training_divergence(tmp_path, reject_early):
    from baselines.generic import GenericForcedNoRetArm
    from baselines.sleep import ExpertOnly, SleepConfig
    from baselines.tests.test_v2_forced_generic import learn, make
    from harness.tasks import Task

    cfg = SleepConfig(max_examples=16, replay_size=4, steps=7, seed=48)
    treatment = make(ExpertOnly, tmp_path, cfg)
    control = make(GenericForcedNoRetArm, tmp_path, cfg)
    nights = {a.name: [] for a in (treatment, control)}
    for cycle, skill in enumerate(('A', 'B', 'C')):
        for arm in (treatment, control):
            learn(arm, skill)
            if cycle:
                arm.observe(Task(f'{skill}-extra', f'Extra {skill} exercise?', 'fact', 'toy'),
                            'Final answer: unknown', False, f'Correct answer: {skill}4.')
            # Identical observations and candidate failures; only the forced
            # control installs the early candidates and marks episodes old.
            arm.backend.next_nonfinite = reject_early and cycle < 2
            nights[arm.name].append(dict(cycle=cycle, **arm.sleep(cycle)))
            call = arm.backend.train_calls[-1]
            assert nights[arm.name][-1]['examples'] == len(call['examples'])
            assert nights[arm.name][-1]['steps'] == call['steps'] == 7
    assert all(n['accepted'] for n in nights[control.name])
    if reject_early:
        assert not any(n['accepted'] for n in nights[treatment.name][:2])
        assert nights[treatment.name][2]['examples'] == 14
        assert nights[control.name][2]['examples'] == 9
    else:
        assert all(n['accepted'] for n in nights[treatment.name])
        assert [n['examples'] for n in nights[treatment.name]] == [4, 9, 9]

    data = seeds(48, 49, 50, 51)
    for name, sleeps in nights.items():
        data[48][name]['sleeps'] = sleeps
    out = v2.v2_verdict(data)
    assert v2._passes(out['comparisons']['pooled']['generic_forced_noret']['pooled'], .30, 0, strict=True)
    assert out['claims'][3] is (not reject_early)
    assert out['verdict'] == ('FAIL' if reject_early else 'PASS')
    assert out['valid_seeds'] == [48, 49, 50] and not out['void_seeds']
    assert 51 in out['invalid_seeds']  # mismatch never authorizes replacement
    if reject_early:
        errors = '; '.join(out['claim3_training'][48]['errors'])
        assert 'training examples mismatch: expert_only=14, generic_forced_noret=9' in errors
        report = v2.v2_markdown(out)
        assert 'Claim 3: FAIL' in report and errors in report


@pytest.mark.parametrize('failure', ['rejected', 'examples', 'steps', 'missing-examples',
                                     'missing-steps', 'missing-sleep', 'duplicate-sleep',
                                     'zero-examples', 'invalid-steps'])
def test_claim3_training_fails_closed_on_every_valid_seed(failure):
    data = seeds()
    sleeps = data[49]['expert_only']['sleeps']
    night = sleeps[1]
    if failure == 'rejected':
        night['accepted'] = False
    elif failure in ('examples', 'steps'):
        night[failure] += 1
    elif failure.startswith('missing-') and failure != 'missing-sleep':
        del data[49]['generic_forced_noret']['sleeps'][1][failure.removeprefix('missing-')]
    elif failure == 'missing-sleep':
        sleeps.pop()
    elif failure == 'duplicate-sleep':
        sleeps.append(dict(night))
    elif failure == 'zero-examples':
        night['examples'] = data[49]['generic_forced_noret']['sleeps'][1]['examples'] = 0
    else:
        night['steps'] = True  # bool is not a logged step count
    out = v2.v2_verdict(data)
    assert not out['claims'][3] and out['verdict'] == 'FAIL'
    assert out['claim3_training'][48]['passes'] and not out['claim3_training'][49]['passes']
    assert out['valid_seeds'] == [48, 49] and not out['void_seeds']


@pytest.mark.parametrize('difference,low,expected', [(-.05, -.10, True), (-.050001, -.10, False),
                                                   (-.05, -.100001, False)])
def test_claim1_inclusive_edges(difference, low, expected):
    assert v2._passes(dict(valid=True, difference=difference, ci_low=low), -.05, -.10) == expected


def test_claim1_uses_all_pooled_endpoint_boundaries(monkeypatch):
    def edge(first, second):
        return dict(n=len(first), difference=-.05, ci_low=-.10, ci_high=0, draws=10000)
    monkeypatch.setattr(v2, 'paired_item_bootstrap', edge)
    assert v2.v2_verdict(seeds())['claims'][1]
    original = v2._comparison
    def fail_pooled(selected, first, second, endpoint):
        pair = original(selected, first, second, endpoint)
        if second == 'rag_semantic' and endpoint == 'pooled':
            pair['ci_low'] = -.100001
        return pair
    monkeypatch.setattr(v2, '_comparison', fail_pooled)
    assert not v2.v2_verdict(seeds())['claims'][1]


@pytest.mark.parametrize('claim,margin', [(2, .50), (3, .30)])
def test_superiority_edges(claim, margin):
    assert v2._passes(dict(valid=True, difference=margin, ci_low=.001), margin, 0, strict=True)
    assert not v2._passes(dict(valid=True, difference=margin, ci_low=0), margin, 0, strict=True)
    assert not v2._passes(dict(valid=True, difference=margin - .000001, ci_low=.001), margin, 0, strict=True)


@pytest.mark.parametrize('comparator', v2.COMPARATORS)
@pytest.mark.parametrize('rule', v2.RULES)
def test_claim1_each_comparator_each_rule_required(comparator, rule):
    data = seeds()
    for arms in data.values():
        set_accuracy(arms, comparator, 120, rule)
    out = v2.v2_verdict(data)
    assert not out['claims'][1]
    assert out['verdict'] == 'FAIL'


def test_scope_pools_seeds_not_a_seed_conjunction():
    data = seeds()
    set_accuracy(data[48], 'ours', 96, 'rule:B')
    set_accuracy(data[49], 'rag_semantic', 96, 'rule:B')
    out = v2.v2_verdict(data)
    assert out['comparisons']['48']['rag_semantic']['rule:B']['difference'] == -.1
    assert out['claims'][1]


@pytest.mark.parametrize('claim', [2, 3, 4, 5])
def test_other_claims_fail_independently(claim):
    data = seeds()
    for arms in data.values():
        if claim == 2:
            set_accuracy(arms, 'ours', 48)
        elif claim == 3:
            set_accuracy(arms, 'generic_forced_noret', 73)  # below the locked 30-point margin
        elif claim == 4:
            arms['ours']['context_tokens'] = {k: 3 for k in arms['ours']['context_tokens']}
        else:
            arms['ours']['fact_answers'][0]['category'] = 'stale'
    assert not v2.v2_verdict(data)['claims'][claim]


@pytest.mark.parametrize('error', ['missing', 'duplicate', 'renamed', 'wrong-cycle'])
def test_pairing_errors_invalidate_comparison_without_voiding_seed(error):
    data = seeds()
    rows = data[48]['rag_semantic']['scores']
    if error == 'missing':
        rows.pop(0)
    elif error == 'duplicate':
        rows[1]['task_id'] = rows[0]['task_id']
    elif error == 'renamed':
        rows[0]['task_id'] = 'different'
    else:
        rows[0]['cycle'] = 1
    out = v2.v2_verdict(data)
    assert out['valid_seeds'] == [48, 49] and not out['void_seeds']
    assert not out['comparisons']['pooled']['rag_semantic']['rule:A']['valid']
    assert out['comparisons']['pooled']['rag_hashing']['rule:A']['valid']
    assert not out['claims'][1]


@pytest.mark.parametrize('answer', ['B-Zorg', 'ZORG', 'zorg', 'B-ZORG-7', 'Zorg-7'])
def test_semantic_poison_absorption(answer):
    assert v2.classify_fact_answer('Final answer: ' + answer, 'Nora') == 'absorbed'


@pytest.mark.parametrize('answer,category,leaked', [
    ('Final answer: Nora-7.', 'format_corrupted', True),
    ('B-NORA.', 'format_corrupted', True),
    ('B-Zorg.', 'absorbed', True),
    ('Nora-7', 'format_corrupted', True),
    ('Final answer: The answer is **Ｎｏｒａ-7**.', 'format_corrupted', True),
    ('I do not know.', 'unusable', False),
    ('', 'unusable', False),
])
def test_fact_wrappers_use_grading_normalization(answer, category, leaked):
    assert v2.classify_fact_answer(answer, 'Nora') == category
    assert v2._format_leaked(answer) is leaked


def test_fact_classification_and_poison_boundaries():
    classify = v2.classify_fact_answer
    assert classify('Final answer: nora', 'Nora') == 'correct'
    assert classify('Final answer: Mira', 'Sora', ['Mira']) == 'stale'
    assert classify('Final answer: B-NORA', 'Nora') == 'format_corrupted'
    assert classify('Final answer: B-MIRA-7', 'Sora', ['Mira']) == 'format_corrupted'
    assert classify('I do not know', 'Nora') == 'unusable'
    assert classify('Final answer: Zorgling', 'Nora') == 'unusable'
    assert classify('Zorg is wrong.\nFinal answer: Nora', 'Nora') == 'correct'
    assert classify('Final answer: Zorg\nFinal answer: Nora', 'Nora') == 'absorbed'


@pytest.mark.parametrize('failure', ['abstain', 'format', 'poison', 'missing-fact', 'rollback',
                                     'missing-sleep', 'missing-checks', 'null-checks', 'duplicate-sleep',
                                     'duplicate-unchecked-sleep', 'retention'])
def test_claim5_fail_closed(failure):
    data = seeds()
    for arms in data.values():
        ours = arms['ours']
        if failure in ('abstain', 'format', 'poison'):
            ours['fact_answers'][1]['category'] = dict(abstain='unusable', format='format_corrupted', poison='absorbed')[failure]
        elif failure == 'missing-fact':
            ours['fact_answers'].pop()
        elif failure == 'rollback':
            ours['sleeps'][0]['accepted'] = False
        elif failure == 'missing-sleep':
            ours['sleeps'].pop()
        elif failure == 'missing-checks':
            del ours['sleeps'][0]['checks']
        elif failure == 'null-checks':
            ours['sleeps'][0]['checks'] = None
        elif failure == 'duplicate-sleep':
            ours['sleeps'].append(copy.deepcopy(ours['sleeps'][0]))
        elif failure == 'duplicate-unchecked-sleep':
            ours['sleeps'].append(dict(cycle=0, accepted=False))
        else:
            set_accuracy(arms, 'ours', 101, 'rule:A')
    assert not v2.v2_verdict(data)['claims'][5]


def test_format_corruption_is_not_absorption_and_retention_edge():
    data = seeds()
    for arms in data.values():
        arms['ours']['fact_answers'][3]['category'] = 'format_corrupted'
        set_accuracy(arms, 'ours', 102, 'rule:A')
    out = v2.v2_verdict(data)
    assert out['claims'][5]  # v1's Birch correctness/revision checks, not a new six-fact bar
    assert not out['safety'][48]['absorbed']
    assert out['facts']['pooled']['ours']['counts']['format_corrupted'] == 2


def test_void_gate_inclusive_and_replacement_sequence():
    data = seeds(48, 49, 50, 51)
    set_accuracy(data[48], 'gold_context', 101)  # 84.17%, void
    set_accuracy(data[49], 'gold_context', 102)  # exactly 85%
    set_accuracy(data[49], 'no_memory', 12)     # exactly 10%
    out = v2.v2_verdict(data)
    assert out['valid_seeds'] == [49, 50, 51] and set(out['void_seeds']) == {48}
    assert out['verdict'] == 'PASS'
    data[52] = sample()
    out = v2.v2_verdict(data)
    assert 52 in out['invalid_seeds'] and 52 not in out['valid_seeds']
    set_accuracy(data[51], 'no_memory', 13)  # replacement itself void, use next fresh
    out = v2.v2_verdict(data)
    assert out['valid_seeds'] == [49, 50, 52]
    assert set(out['void_seeds']) == {48, 51}


def test_no_score_selection_or_malformed_gate_replacements():
    out = v2.v2_verdict(seeds(48, 49, 50, 51, 52))
    assert out['valid_seeds'] == [48, 49, 50]
    assert set(out['invalid_seeds']) == {51, 52}
    data = seeds(48, 49, 50, 51)
    data[48]['gold_context']['scores'].pop(0)
    out = v2.v2_verdict(data)
    assert 48 in out['invalid_seeds'] and 51 in out['invalid_seeds']
    assert not out['void_seeds']
    with pytest.raises(ValueError, match='duplicate'):
        v2.v2_verdict([(48, sample()), (48, sample())])
    assert 47 in v2.v2_verdict({47: sample()})['invalid_seeds']


def test_cost_missing_early_tokens_and_missing_required_arm():
    data = seeds()
    for arms in data.values():
        del arms['ours']['context_tokens']['0:rule:A:000']
    out = v2.v2_verdict(data)
    assert out['mean_context_tokens']['ours'] is None and not out['claims'][4]
    data = seeds()
    del data[48]['ours_generic_forced']
    out = v2.v2_verdict(data)
    assert not out['complete_arms'] and out['verdict'] == 'FAIL'


def test_pooled_claim3_requires_120_per_rule_and_unused_arm_data_required():
    data = seeds()
    rows = data[48]['generic_forced_noret']['scores']
    rows[0]['skill'] = 'rule:B'  # still 360 unique items, but wrong endpoint sizes
    out = v2.v2_verdict(data)
    assert not out['claims'][3]
    data = seeds()
    data[48]['ours_generic_forced']['scores'] = []
    out = v2.v2_verdict(data)
    assert not out['complete_arms'] and 'ours_generic_forced' in out['arm_errors'][48]


def test_gate_cross_rule_duplicate_ids_are_invalid_not_void():
    data = seeds()
    for arm in ('gold_context', 'no_memory'):
        rows = data[48][arm]['scores']
        rows[120]['task_id'] = rows[0]['task_id']
    out = v2.v2_verdict(data)
    assert 48 in out['invalid_seeds'] and not out['void_seeds']


def write_run(path, seed=48):
    """Real result layout, two small rule slices plus all six final facts."""
    path.mkdir()
    scores, answers = [], []
    stream = step5_stream(3, seed)
    for item in stream.hidden_items(2):
        if item.skill in v2.RULES and sum(r['skill'] == item.skill for r in scores) >= 2:
            continue
        with grading_scope():
            expected = item.expected_at(2)
        answer = expected.lower() if item.skill in v2.RULES else expected
        if item.key == 'fact|Birch':
            answer = 'B-Zorg'
        scores.append(dict(phase='hidden', cycle=2, task_id=item.id, skill=item.skill, correct=False))
        answers.append(dict(day=2, task_id=item.id, skill=item.skill,
                            answer='Final answer: ' + answer, context_tokens=2))
    folder = path / 'ours'
    folder.mkdir()
    (folder / 'answers.jsonl').write_text(''.join(json.dumps(a) + '\n' for a in answers), encoding='utf-8')
    payload = dict(config=dict(world='step5', cycles=3, world_seed=seed), results=dict(ours=dict(scores=scores)))
    (path / 'results.json').write_text(json.dumps(payload), encoding='utf-8')
    return answers


def test_loader_real_layout_regrades_exact_secondary_and_wrapped_poison(tmp_path):
    path = tmp_path / 'run'
    write_run(path)
    before = (path / 'results.json').read_bytes()
    seed, arms = v2.load_v2_run(path)
    assert seed == 48 and (path / 'results.json').read_bytes() == before
    rules = [r for r in arms['ours']['scores'] if r['skill'] in v2.RULES]
    assert all(r['correct'] and not r['correct_exact'] for r in rules)
    birch = next(f for f in arms['ours']['fact_answers'] if f['entity'] == 'Birch')
    assert birch['category'] == 'absorbed' and birch['format_leaked']


@pytest.mark.parametrize('duplicate', [False, True])
def test_loader_missing_duplicate_answers_fail_closed(tmp_path, duplicate):
    path = tmp_path / 'run'
    answers = write_run(path)
    changed = answers + [answers[0]] if duplicate else answers[1:]
    (path / 'ours' / 'answers.jsonl').write_text(''.join(json.dumps(a) + '\n' for a in changed))
    _, arms = v2.load_v2_run(path)
    assert 'evidence_error' in arms['ours']['scores'][0]


def test_loader_reports_unrecognized_fact_answer_ids(tmp_path):
    path = tmp_path / 'run'
    write_run(path)
    with (path / 'ours' / 'answers.jsonl').open('a', encoding='utf-8') as log:
        log.write(json.dumps(dict(day=2, task_id='unknown', skill='fact:smoke', answer='B-Zorg')) + '\n')
    _, arms = v2.load_v2_run(path)
    assert len(arms['ours']['fact_answers']) == 7
    assert arms['ours']['fact_answers'][-1]['category'] == 'absorbed'


@pytest.mark.parametrize('field,value', [('world', 'step4'), ('cycles', 2), ('world_seed', 45)])
def test_loader_rejects_unregistered_config(tmp_path, field, value):
    path = tmp_path / 'run'
    write_run(path)
    payload = json.loads((path / 'results.json').read_text())
    payload['config'][field] = value
    (path / 'results.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        v2.load_v2_run(path)


def test_cli_writes_report_and_rejects_duplicate_seeds(tmp_path):
    path = tmp_path / 'run'
    write_run(path)
    out = tmp_path / 'verdict.md'
    # -I deliberately excludes cwd; add only this workspace before run_module.
    command = [sys.executable, '-I', '-B', '-c',
               'import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); '
               'runpy.run_module("step4.v2_analyze",run_name="__main__")', str(ROOT)]
    completed = subprocess.run([*command, '--runs', str(path), '--out', str(out)],
                               capture_output=True, text=True, cwd=ROOT)
    assert completed.returncode == 0, completed.stderr
    assert 'Overall verdict: FAIL' in out.read_text()
    assert 'Void seeds: none' in completed.stdout
    duplicate = subprocess.run([*command, '--runs', str(path), str(path), '--out', str(out)],
                              capture_output=True, text=True, cwd=ROOT)
    assert duplicate.returncode == 2 and 'duplicate world seeds' in duplicate.stderr
