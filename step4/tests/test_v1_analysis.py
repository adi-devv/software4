"""Synthetic checks of the registered V1 gates and paired inference."""

import copy
import os
import subprocess
import sys

import numpy as np
import pytest

from step4.v1_analysis import (
    BASELINES, METRICS, REGISTERED_CELLS, _cluster_layout, _cluster_weights,
    auroc, b2_auroc, bootstrap_compare, brier, ece, eligibility, eval_validity,
    fit_group_accuracy, fit_predict, partly_learned_groups, verdict,
    within_group_auroc,
)


CELLS = dict.fromkeys(REGISTERED_CELLS, True)


def world(kind='signal', n_fit=80, n_eval=120):
    rng = np.random.default_rng(123)
    datasets = []
    for split, n in [('fit', n_fit), ('eval', n_eval)]:
        rows = []
        for seed in (44, 45, 46):
            for cycle in range(3):
                for r in range(cycle + 1):
                    rate = (0.25, 0.5, 0.75)[r] if kind == 'rule' else 0.5
                    if kind == 'deterministic':
                        rate = 1.0 if r == 0 else 0.0
                    y = np.zeros(n, dtype=int)
                    y[:int(n * rate)] = 1
                    rng.shuffle(y)
                    lp = (2 * y - 1 + rng.normal(0, 0.1, n)
                          if kind == 'logprob' else rng.normal(-2, 1, n))
                    signal = (2 * y - 1) * 3 + rng.normal(0, 0.2, n)
                    if kind == 'rule':
                        signal = np.zeros(n)
                    elif kind == 'logprob':
                        signal = lp + rng.normal(0, 0.5, n)
                    X = np.c_[signal, np.tile(np.eye(3)[r], (n, 1))]
                    for i in range(n):
                        rows.append((seed, cycle, 'ABC'[r], split, f'{split}-{i}',
                                     y[i], lp[i], X[i]))
        datasets.append({k: np.array([row[j] for row in rows]) for j, k in enumerate(
            ('seed', 'cycle', 'rule', 'split', 'code', 'y', 'mean_logprob', 'X'))})
    return datasets


@pytest.fixture(scope='module')
def signal_result():
    f, e = world()
    pred = fit_predict(f, e, C=1.0, pca_dims=4)
    result = bootstrap_compare(e, pred, partly_learned_groups(f), n_draws=200, rng_seed=91)
    return f, e, pred, result


def test_planted_within_group_signal_passes(signal_result):
    f, e, pred, result = signal_result
    answer = verdict(f, e, result, cells=CELLS)
    assert answer['verdict'] == 'PASS', answer['reasons']
    for s in ('pooled', 'partly_learned'):
        for b in BASELINES:
            for m in METRICS:
                assert result[s][b][m]['ci_low'] > 0
                assert result[s][b][m]['skipped_draws'] == 0
    assert within_group_auroc(e, pred['V'], partly_learned_groups(f))[0] > 0.99


@pytest.mark.parametrize('kind,baseline', [('rule', 'B1'), ('logprob', 'B2')])
def test_rule_or_confidence_readout_fails(kind, baseline):
    f, e = world(kind)
    predictions = fit_predict(f, e, C=1.0)
    result = bootstrap_compare(e, predictions, partly_learned_groups(f),
                               n_draws=200, rng_seed=91)
    answer = verdict(f, e, result, cells=CELLS)
    assert answer['verdict'] == 'FAIL'
    for b in (baseline, 'B3'):
        assert any(b in reason for reason in answer['reasons'])
    if kind == 'rule':
        expected = (20 + 1) / (80 + 2)
        assert predictions['B1'][0] == expected


def test_near_deterministic_world_is_inconclusive():
    f, e = world('deterministic')
    # The gate must not inspect a critic result, even when none is supplied.
    answer = verdict(f, e, None, cells=CELLS)
    assert answer['verdict'] == 'INCONCLUSIVE'
    assert not answer['eligibility']['passed']
    assert any('fewer than 6' in r for r in answer['reasons'])


def test_missing_or_invalid_cells_override_great_metrics(signal_result):
    f, e, _, result = signal_result
    for flags in ({k: v for k, v in CELLS.items() if k != (45, 1)},
                  {**CELLS, (46, 2): False}):
        answer = verdict(f, e, result, cells=flags)
        assert answer['verdict'] == 'INVALID'
        assert any('validation cell' in r for r in answer['reasons'])
    keep = ~((e['seed'] == 44) & (e['cycle'] == 0))
    missing = {k: v[keep] for k, v in e.items()}
    assert verdict(f, missing, result, cells=CELLS)['verdict'] == 'INVALID'


def test_partly_learned_uses_fit_only_and_inclusive_bounds(signal_result):
    f, e, _, _ = signal_result
    before = eligibility(f)
    changed = {**e, 'y': np.ones(len(e['y']), dtype=int)}
    assert eligibility(f) == before
    assert partly_learned_groups(f) == before['partly_groups']
    assert not eval_validity(changed, before['partly_groups'])['passed']
    assert eval_validity(e, before['partly_groups'])['passed']
    tiny = {'seed': [44] * 10, 'rule': ['A'] * 5 + ['B'] * 5,
            'cycle': [0] * 10, 'y': [1, 0, 0, 0, 0, 1, 1, 1, 1, 0]}
    assert fit_group_accuracy(tiny) == {(44, 'A', 0): 0.2, (44, 'B', 0): 0.8}
    assert len(partly_learned_groups(tiny)) == 2
    with pytest.raises(ValueError, match='fit'):
        partly_learned_groups(e)


def test_tiny_metrics():
    assert brier([0, 1], [0.25, 0.75]) == 0.0625
    assert auroc([0, 1], [0, 1]) == 1
    assert auroc([0, 1], [1, 0]) == 0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert auroc([0, 0, 1, 1], [0, 0.5, 0.5, 1]) == 0.875
    assert np.isnan(auroc([1, 1], [0.2, 0.3]))
    assert np.isnan(auroc([], []))
    assert np.isnan(brier([], []))
    assert ece([0, 1], [0.25, 0.75]) == 0.25
    assert ece([0, 1], [0, 1]) == 0
    assert ece([0, 1], [0.5, 0.5]) == 0
    assert b2_auroc([0, 1], [0.5, 0.5], [0, 1]) == {
        'raw': 1.0, 'isotonic': 0.5, 'best': 1.0}


def test_within_group_exclusions_and_validity(signal_result):
    f, e, pred, _ = signal_result
    e = {k: v.copy() for k, v in e.items()}
    group = (44, 'A', 0)
    mask = (e['seed'] == 44) & (e['rule'] == 'A') & (e['cycle'] == 0)
    e['y'][mask] = 1
    partly = partly_learned_groups(f) | {(44, 'C', 0)}
    _, excluded = within_group_auroc(e, pred['V'], partly)
    assert excluded == [(44, 'A', 0), (44, 'C', 0)]
    validity = eval_validity(e, partly)
    assert validity['successes'] >= 100 and validity['failures'] >= 100
    assert not validity['passed']
    assert validity['excluded_groups'] == excluded
    assert verdict(f, e, None, cells=CELLS)['verdict'] == 'INCONCLUSIVE'


def test_bootstrap_determinism_and_joint_code_resampling(signal_result):
    f, e, pred, _ = signal_result
    partly = partly_learned_groups(f)
    a = bootstrap_compare(e, pred, partly, n_draws=20, rng_seed=17)
    b = bootstrap_compare(e, pred, partly, n_draws=20, rng_seed=17)
    assert a == b
    inverse, strata = _cluster_layout(e)
    weights = _cluster_weights(np.random.default_rng(77), inverse, strata, 200)
    # A code repeats across different cycles AND rules; every copy moves together.
    same = (e['seed'] == 44) & (e['code'] == 'eval-0')
    assert len(set(e['cycle'][same])) == 3
    assert len(set(e['rule'][same])) == 3
    assert np.all(weights[:, same] == weights[:, same][:, :1])
    other_seed = np.flatnonzero((e['seed'] == 45) & (e['code'] == 'eval-0'))[0]
    assert np.any(weights[:, same][:, 0] != weights[:, other_seed])
    for seed in (44, 45, 46):
        representatives = np.unique(inverse[e['seed'] == seed], return_index=True)[1]
        seed_weights = weights[:, e['seed'] == seed][:, representatives]
        assert np.all(seed_weights.sum(axis=1) == 120)


def test_difference_orientation(signal_result):
    f, e, pred, _ = signal_result
    partly = partly_learned_groups(f)
    good = {**pred, 'B1': 1 - pred['V']}
    result = bootstrap_compare(e, good, partly, n_draws=10, rng_seed=1)
    for m in METRICS:
        assert result['partly_learned']['B1'][m]['estimate'] > 0
    good['V'], good['B1'] = good['B1'], good['V']
    reverse = bootstrap_compare(e, good, partly, n_draws=10, rng_seed=1)
    for m in METRICS:
        assert reverse['partly_learned']['B1'][m]['estimate'] == pytest.approx(
            -result['partly_learned']['B1'][m]['estimate'])


def test_sign_gate_every_comparison_and_two_seed_case(signal_result):
    f, e, _, result = signal_result
    broken = copy.deepcopy(result)
    broken['partly_learned']['B3']['brier']['per_seed_signs'] = {44: 1, 45: 0, 46: -1}
    answer = verdict(f, e, broken, cells=CELLS)
    assert answer['verdict'] == 'FAIL'
    assert any('B3 brier: positive sign' in r for r in answer['reasons'])
    # Seed 46 has no partly-learned groups: both remaining seeds must agree.
    f = {k: v.copy() for k, v in f.items()}
    f['y'][f['seed'] == 46] = 1
    assert eligibility(f)['sign_agreement_seeds'] == [44, 45]
    assert verdict(f, e, result, cells=CELLS)['verdict'] == 'PASS'
    broken = copy.deepcopy(result)
    broken['partly_learned']['B1']['auroc']['per_seed_signs'][45] = -1
    assert verdict(f, e, broken, cells=CELLS)['verdict'] == 'FAIL'


def test_predictors_are_fit_only_and_cell_local(signal_result):
    f, e, pred, _ = signal_result
    changed = {**e, 'y': 1 - e['y']}
    check = fit_predict(f, changed, C=1.0, pca_dims=4)
    for k in pred:
        np.testing.assert_array_equal(pred[k], check[k])
    changed_fit = {k: v.copy() for k, v in f.items()}
    changed_fit['y'][(f['seed'] == 44) & (f['cycle'] == 0)] = 0
    changed_fit['X'][(f['seed'] == 44) & (f['cycle'] == 0)] += 100
    check = fit_predict(changed_fit, e, C=1.0, pca_dims=4)
    untouched = ~((e['seed'] == 44) & (e['cycle'] == 0))
    for k in pred:
        np.testing.assert_array_equal(pred[k][untouched], check[k][untouched])
    assert np.all(check['V'][~untouched] == 0)
    assert np.all(check['B3'][~untouched] == 0)
    assert np.all(check['B2'][~untouched] == 0)
    with pytest.raises(ValueError, match='pca_dims'):
        fit_predict(f, e, C=1, pca_dims=65)


def test_undefined_bootstrap_draws_are_counted():
    d = {'seed': np.array([44, 44]), 'cycle': np.array([0, 0]),
         'rule': np.array(['A', 'A']), 'code': np.array(['x', 'z']),
         'y': np.array([0, 1])}
    p = {k: np.array([0.2, 0.8]) for k in ('V', *BASELINES, 'B2_raw')}
    r = bootstrap_compare(d, p, {(44, 'A', 0)}, n_draws=200, rng_seed=3)
    for s in ('pooled', 'partly_learned'):
        for b in BASELINES:
            assert r[s][b]['brier']['skipped_draws'] == 0
            assert 0 < r[s][b]['auroc']['skipped_draws'] < 200
            assert r[s][b]['auroc']['skipped_draws'] == r[s][b]['within_group_auroc']['skipped_draws']


def test_b2_uses_raw_when_isotonic_ties_lose_ranking():
    d = {'seed': np.array([44] * 4), 'cycle': np.array([0] * 4),
         'rule': np.array(['A'] * 4), 'code': np.array(list('abcd')),
         'y': np.array([0, 1, 0, 1])}
    p = {k: np.array([0.1, 0.9, 0.2, 0.8]) for k in ('V', *BASELINES)}
    p['B2'] = np.full(4, 0.5)
    p['B2_raw'] = np.array([-4., -1., -3., -2.])
    result = bootstrap_compare(d, p, {(44, 'A', 0)}, n_draws=20, rng_seed=4)
    assert result['b2_auroc']['pooled'] == {'raw': 1., 'isotonic': 0.5, 'best': 1.}
    assert result['b2_within_group_auroc'] == {'raw': 1., 'isotonic': 0.5, 'best': 1.}
    for s in ('pooled', 'partly_learned'):
        for m in ('auroc', 'within_group_auroc'):
            assert result[s]['B2'][m]['estimate'] == 0
            assert result[s]['B2'][m]['ci_low'] == 0
        assert result[s]['B2']['brier']['baseline'] == 0.25


def test_import_does_not_load_torch():
    result = subprocess.run([sys.executable, '-c',
                             'import step4.v1_analysis; import sys; '
                             'assert "torch" not in sys.modules'],
                            capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr
