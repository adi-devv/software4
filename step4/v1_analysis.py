"""CPU-only, conditional-on-fitted-model analysis for V1.

Data are dicts of aligned numpy arrays: seed (int), cycle (int), rule
('A'/'B'/'C'), split ('fit'/'eval'), code (str task identity), y (0/1),
mean_logprob (float), X (n,d float). Utilities need only the columns they use.
Groups are (seed, rule, cycle); cells are (seed, cycle). No model selection is
performed here: C and optional PCA dimensions must be frozen before scoring.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REGISTERED_SEEDS = (44, 45, 46)
REGISTERED_CELLS = tuple((s, c) for s in REGISTERED_SEEDS for c in range(3))
BASELINES = ('B1', 'B2', 'B3')
METRICS = ('brier', 'auroc', 'within_group_auroc')


def _columns(data, split=None):
    d = {k: np.asarray(v) for k, v in data.items()}
    y = d['y']
    if y.ndim != 1 or not np.isin(y, (0, 1)).all():
        raise ValueError('y must be a one-dimensional binary array')
    if any(v.ndim == 0 or len(v) != len(y) for v in d.values()):
        raise ValueError('columns must have matching first dimensions')
    if split is not None and 'split' in d and np.any(d['split'] != split):
        raise ValueError(f'expected only {split} records')
    return d


def _groups(d):
    return [(int(s), str(r), int(c))
            for s, r, c in zip(d['seed'], d['rule'], d['cycle'])]


def _group_indices(d):
    groups = {}
    for i, g in enumerate(_groups(d)):
        groups.setdefault(g, []).append(i)
    return {g: np.asarray(ix, dtype=int) for g, ix in groups.items()}


def fit_group_accuracy(fit):
    """Observed accuracy for each fit group (never reads eval outcomes)."""
    d = _columns(fit, 'fit')
    return {g: float(d['y'][ix].mean()) for g, ix in _group_indices(d).items()}


def partly_learned_groups(fit, lo=0.2, hi=0.8):
    """Fit-only inclusive eligibility; return a set of (seed, rule, cycle)."""
    if not 0 <= lo <= hi <= 1:
        raise ValueError('require 0 <= lo <= hi <= 1')
    return {g for g, a in fit_group_accuracy(fit).items() if lo <= a <= hi}


def eligibility(fit):
    """Gate on the supplied fit records; seed 47 must be reported separately."""
    accuracy = fit_group_accuracy(fit)
    partly = partly_learned_groups(fit)
    seeds = sorted({g[0] for g in partly})
    return {'counts': {'groups': len(accuracy), 'partly_learned_groups': len(partly),
                       'sign_agreement_seeds': len(seeds),
                       'partly_groups_per_seed': {s: sum(g[0] == s for g in partly)
                                                 for s in sorted({g[0] for g in accuracy})}},
            'passed': len(partly) >= 6 and len(seeds) >= 2,
            'partly_groups': partly, 'sign_agreement_seeds': seeds,
            'excluded_seeds': sorted({g[0] for g in accuracy} - set(seeds))}


def eval_validity(eval_, partly_groups):
    """Check pooled counts and both classes in EVERY eligible group, even absent ones."""
    d = _columns(eval_, 'eval')
    indices = _group_indices(d)
    counts = {}
    for g in sorted(partly_groups):
        y = d['y'][indices.get(g, np.array([], dtype=int))]
        counts[g] = {'successes': int(y.sum()), 'failures': int(len(y) - y.sum())}
    successes = sum(c['successes'] for c in counts.values())
    failures = sum(c['failures'] for c in counts.values())
    excluded = [g for g, c in counts.items() if min(c.values()) == 0]
    return {'passed': successes >= 100 and failures >= 100 and not excluded,
            'successes': successes, 'failures': failures,
            'group_counts': counts, 'excluded_groups': excluded}


def _scores(y, score):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or not np.isin(y, (0, 1)).all():
        raise ValueError('expected aligned binary outcomes and one-dimensional scores')
    if not np.isfinite(score).all():
        raise ValueError('scores must be finite')
    return y, score


def brier(y, score):
    y, score = _scores(y, score)
    return float(np.mean((y - score) ** 2)) if len(y) else float('nan')


def auroc(y, score):
    """Tie-aware Mann-Whitney AUROC; one-class/empty samples return nan."""
    y, score = _scores(y, score)
    pos = y.sum()
    neg = len(y) - pos
    if not pos or not neg:
        return float('nan')
    order = np.argsort(score, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(score[order])) + 1]
    p = np.add.reduceat(y[order], starts)
    n = np.diff(np.r_[starts, len(y)]) - p
    return float(np.sum(p * (np.cumsum(n) - n / 2)) / (pos * neg))


def ece(y, score):
    """Reported only: 10 equal-width probability bins, including score 1."""
    y, score = _scores(y, score)
    if np.any((score < 0) | (score > 1)):
        raise ValueError('ECE requires probabilities in [0, 1]')
    if not len(y):
        return float('nan')
    bins = np.minimum((score * 10).astype(int), 9)
    return float(sum(abs(np.sum(y[bins == b] - score[bins == b]))
                     for b in range(10)) / len(y))


def within_group_auroc(eval_, score, partly_groups):
    """Unweighted mean across eligible mixed groups; return (mean, excluded list)."""
    d = _columns(eval_, 'eval')
    _, score = _scores(d['y'], score)
    indices = _group_indices(d)
    values, excluded = [], []
    for g in sorted(partly_groups):
        ix = indices.get(g, np.array([], dtype=int))
        value = auroc(d['y'][ix], score[ix])
        if np.isfinite(value):
            values.append(value)
        else:
            excluded.append(g)
    return (float(np.mean(values)) if values else float('nan'), excluded)


def b2_auroc(y, isotonic_score, mean_logprob):
    """Return raw, isotonic, and their better AUROC (the registered B2 value)."""
    iso, raw = auroc(y, isotonic_score), auroc(y, mean_logprob)
    return {'isotonic': iso, 'raw': raw, 'best': max(iso, raw)}


def _logistic(X, y, C, pca_dims, max_iter, random_state):
    if C <= 0 or not np.isfinite(C):
        raise ValueError('C must be finite and positive')
    steps = [StandardScaler()]
    if pca_dims is not None:
        if not isinstance(pca_dims, (int, np.integer)) or not 1 <= pca_dims <= 64:
            raise ValueError('pca_dims must be an integer in [1, 64] or None')
        steps.append(PCA(n_components=min(pca_dims, *X.shape), svd_solver='full'))
    steps.append(LogisticRegression(C=C, max_iter=max_iter, random_state=random_state))
    return make_pipeline(*steps).fit(X, y)


def fit_predict(fit, eval_, *, C, pca_dims=None, b3_C=None,
                max_iter=1000, random_state=0):
    """Fit once per (seed, cycle) on F_c only, predict aligned E_c arrays.

    Returns V/B1/B2/B3 probabilities plus B2_raw scores. C is required; b3_C
    defaults to the same frozen C. Single-class fits use their constant outcome
    for logistic models. Unseen eval rules/cells are errors, never eval-fitted.
    """
    f, e = _columns(fit, 'fit'), _columns(eval_, 'eval')
    for d in (f, e):
        if d['X'].ndim != 2 or not np.isfinite(d['X']).all():
            raise ValueError('X must be a finite n x d matrix')
        if not np.isfinite(d['mean_logprob']).all():
            raise ValueError('mean_logprob must be finite')
    if f['X'].shape[1] != e['X'].shape[1]:
        raise ValueError('fit/eval feature dimensions differ')
    if C <= 0 or not np.isfinite(C) or (b3_C is not None and
                                       (b3_C <= 0 or not np.isfinite(b3_C))):
        raise ValueError('C strengths must be finite and positive')
    if pca_dims is not None and (not isinstance(pca_dims, (int, np.integer)) or
                                 not 1 <= pca_dims <= 64):
        raise ValueError('pca_dims must be an integer in [1, 64] or None')
    out = {k: np.empty(len(e['y'])) for k in ('V', *BASELINES, 'B2_raw')}
    out['B2_raw'][:] = e['mean_logprob']
    cells = sorted(set(zip(e['seed'], e['cycle'])))
    for seed, cycle in cells:
        fi = (f['seed'] == seed) & (f['cycle'] == cycle)
        ei = (e['seed'] == seed) & (e['cycle'] == cycle)
        if not fi.any():
            raise ValueError(f'missing fit cell {(seed, cycle)}')
        y = f['y'][fi]
        rates = {r: (y[f['rule'][fi] == r].sum() + 1) /
                    (np.sum(f['rule'][fi] == r) + 2) for r in np.unique(f['rule'][fi])}
        if any(r not in rates for r in e['rule'][ei]):
            raise ValueError('eval rule absent from fit cell')
        fr = np.array([rates[r] for r in f['rule'][fi]])
        er = np.array([rates[r] for r in e['rule'][ei]])
        out['B1'][ei] = er
        out['B2'][ei] = IsotonicRegression(out_of_bounds='clip', increasing=True).fit(
            f['mean_logprob'][fi], y).predict(e['mean_logprob'][ei])
        designs = (('V', f['X'][fi], e['X'][ei], C, pca_dims),
                   ('B3', np.c_[fr, f['mean_logprob'][fi]],
                    np.c_[er, e['mean_logprob'][ei]], C if b3_C is None else b3_C, None))
        for name, train, test, strength, dims in designs:
            out[name][ei] = (float(y[0]) if len(np.unique(y)) == 1 else
                             _logistic(train, y, strength, dims, max_iter,
                                       random_state).predict_proba(test)[:, 1])
    return out


def _auc_plan(y, scores, ix):
    order = ix[np.argsort(scores[ix], kind='stable')]
    starts = (np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
              if len(order) else np.array([], dtype=int))
    return order, starts, y[order]


def _weighted_auc(weights, plan):
    order, starts, y = plan
    if not len(order):
        return np.full(len(weights), np.nan)
    w = weights[:, order]
    p = np.add.reduceat(w * y, starts, axis=1)
    n = np.add.reduceat(w * (1 - y), starts, axis=1)
    denominator = p.sum(axis=1) * n.sum(axis=1)
    numerator = np.sum(p * (np.cumsum(n, axis=1) - n / 2), axis=1)
    return np.divide(numerator, denominator, out=np.full(len(w), np.nan),
                     where=denominator > 0)


def _cluster_layout(d):
    """Internal layout is also exposed to tests auditing joint resampling."""
    clusters, inverse = np.unique(np.rec.fromarrays([d['seed'], d['code']]),
                                  return_inverse=True)
    strata = [np.flatnonzero(clusters['f0'] == seed) for seed in np.unique(d['seed'])]
    return inverse, strata


def _cluster_weights(rng, inverse, strata, draws):
    counts = np.zeros((draws, int(inverse.max()) + 1), dtype=float)
    for ix in strata:
        counts[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=draws)
    return counts[:, inverse]


def bootstrap_compare(eval_, predictions, partly_groups, *, n_draws=10000,
                      rng_seed=0, batch_size=128):
    """Paired 95% two-sided percentile intervals, conditional on fitted critics.

    Resample (seed, code) clusters jointly across ALL cycles and rules, with
    replacement within each seed. This deviates from stratifying by (seed,
    group): group sizes are preserved in expectation, not exactly. Attempts
    have equal weight; within-group AUROC is the registered mean of group
    AUROCs. Original one-class groups are excluded and listed. Draws may omit
    newly one-class groups; a draw with no usable group is undefined/skipped.
    B2 uses max(raw, isotonic) separately for each AUROC aggregate, seed, and
    draw; Brier/ECE always use isotonic probabilities. Within-group AUROC uses
    partly groups on BOTH strata, as registered. Signs are -1/0/+1 (nan when
    undefined), on the partly-learned stratum only.
    """
    d = _columns(eval_, 'eval')
    if not len(d['y']) or n_draws < 1 or batch_size < 1:
        raise ValueError('nonempty eval, positive n_draws and batch_size required')
    names = ('V', *BASELINES, 'B2_raw')
    scores = {k: _scores(d['y'], predictions[k])[1] for k in names}
    y = d['y']
    group_ix = _group_indices(d)
    partly_groups = set(partly_groups)
    mixed = {g: group_ix[g] for g in sorted(partly_groups) if g in group_ix and
             len(np.unique(y[group_ix[g]])) == 2}
    excluded = sorted(partly_groups - mixed.keys())
    partly_ix = np.array([i for i, g in enumerate(_groups(d)) if g in partly_groups], dtype=int)
    strata_ix = {'pooled': np.arange(len(y)), 'partly_learned': partly_ix}
    plans = {k: {s: _auc_plan(y, scores[k], ix) for s, ix in strata_ix.items()}
             for k in names}
    group_plans = {k: {g: _auc_plan(y, scores[k], ix) for g, ix in mixed.items()}
                   for k in names}
    errors = {k: (y - scores[k]) ** 2 for k in ('V', *BASELINES)}

    def evaluate(weights, seed=None):
        results = {s: {} for s in strata_ix}
        within = {}
        for k in names:
            arrays = [_weighted_auc(weights, plan) for g, plan in group_plans[k].items()
                      if seed is None or g[0] == seed]
            if arrays:
                a = np.array(arrays)
                count = np.isfinite(a).sum(axis=0)
                within[k] = np.divide(np.nansum(a, axis=0), count,
                                      out=np.full(len(weights), np.nan), where=count > 0)
            else:
                within[k] = np.full(len(weights), np.nan)
        for s, ix in strata_ix.items():
            total = weights[:, ix].sum(axis=1)
            for k in ('V', *BASELINES):
                br = np.sum(weights[:, ix] * errors[k][ix], axis=1)
                br = np.divide(br, total, out=np.full(len(weights), np.nan), where=total > 0)
                auc = _weighted_auc(weights, plans[k][s])
                wg = within[k]
                if k == 'B2':
                    auc = np.maximum(auc, _weighted_auc(weights, plans['B2_raw'][s]))
                    wg = np.maximum(wg, within['B2_raw'])
                results[s][k] = {'brier': br, 'auroc': auc, 'within_group_auroc': wg}
        return results

    def differences(values, s, b, m):
        v, base = values[s]['V'][m], values[s][b][m]
        return base - v if m == 'brier' else v - base

    point = evaluate(np.ones((1, len(y))))
    samples = {s: {b: {m: np.empty(n_draws) for m in METRICS} for b in BASELINES}
               for s in strata_ix}
    inverse, seed_clusters = _cluster_layout(d)
    rng = np.random.default_rng(rng_seed)
    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        values = evaluate(_cluster_weights(rng, inverse, seed_clusters, stop - start))
        for s in strata_ix:
            for b in BASELINES:
                for m in METRICS:
                    samples[s][b][m][start:stop] = differences(values, s, b, m)
    per_seed = {seed: evaluate((d['seed'] == seed)[None, :].astype(float), seed)
                for seed in sorted({g[0] for g in partly_groups})}
    b2_within = {k: within_group_auroc(d, scores[name], partly_groups)[0]
                 for k, name in (('isotonic', 'B2'), ('raw', 'B2_raw'))}
    b2_within['best'] = max(b2_within.values())
    out = {'n_draws': n_draws, 'rng_seed': rng_seed, 'excluded_groups': excluded,
           'b2_auroc': {s: b2_auroc(y[ix], scores['B2'][ix], scores['B2_raw'][ix])
                        for s, ix in strata_ix.items()},
           'b2_within_group_auroc': b2_within,
           'ece': {s: {k: ece(y[ix], scores[k][ix]) for k in ('V', *BASELINES)}
                   for s, ix in strata_ix.items()}}
    for s in strata_ix:
        out[s] = {}
        for b in BASELINES:
            out[s][b] = {}
            for m in METRICS:
                a = samples[s][b][m]
                valid = a[np.isfinite(a)]
                low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
                cell = {'estimate': float(differences(point, s, b, m)[0]),
                        'v': float(point[s]['V'][m][0]),
                        'baseline': float(point[s][b][m][0]),
                        'ci_low': float(low), 'ci_high': float(high),
                        'skipped_draws': int(n_draws - len(valid))}
                if s == 'partly_learned':
                    cell['per_seed_differences'] = {
                        seed: float(differences(v, s, b, m)[0]) for seed, v in per_seed.items()}
                    cell['per_seed_signs'] = {seed: float(np.sign(delta)) for seed, delta in
                                              cell['per_seed_differences'].items()}
                out[s][b][m] = cell
    return out


def verdict(fit, eval_, comparisons, *, cells):
    """Return verdict/reasons/gates. cells maps (seed, cycle) to validation bool.

    Missing or false registered flags, or cells absent from fit/eval, invalidate
    the run. Only registered seeds enter gates; seed 47 can never rescue them.
    Gates are computed independently of comparisons, before any metric is read.
    """
    f, e = _columns(fit, 'fit'), _columns(eval_, 'eval')
    f = {k: v[np.isin(f['seed'], REGISTERED_SEEDS)] for k, v in f.items()}
    e = {k: v[np.isin(e['seed'], REGISTERED_SEEDS)] for k, v in e.items()}
    gate = eligibility(f)
    valid = eval_validity(e, gate['partly_groups'])
    present_f, present_e = set(zip(f['seed'], f['cycle'])), set(zip(e['seed'], e['cycle']))
    reasons = []
    for cell in REGISTERED_CELLS:
        if cell not in cells:
            reasons.append(f'missing validation cell {cell}')
        elif not cells[cell]:
            reasons.append(f'invalid validation cell {cell}')
        if cell not in present_f:
            reasons.append(f'missing fit cell {cell}')
        if cell not in present_e:
            reasons.append(f'missing eval cell {cell}')
    if gate['counts']['partly_learned_groups'] < 6:
        reasons.append('eligibility: fewer than 6 partly-learned groups')
    if len(gate['sign_agreement_seeds']) < 2:
        reasons.append('eligibility: fewer than 2 sign-agreement seeds')
    if valid['successes'] < 100:
        reasons.append('evaluation validity: fewer than 100 partly-learned successes')
    if valid['failures'] < 100:
        reasons.append('evaluation validity: fewer than 100 partly-learned failures')
    reasons.extend(f'evaluation validity: group {g} lacks both outcomes'
                   for g in valid['excluded_groups'])
    invalid = any(c not in cells or not cells[c] or c not in present_f or c not in present_e
                  for c in REGISTERED_CELLS)
    result = {'verdict': 'INVALID' if invalid else 'INCONCLUSIVE', 'reasons': reasons,
              'eligibility': gate, 'eval_validity': valid}
    if invalid or not gate['passed'] or not valid['passed']:
        return result
    seeds = gate['sign_agreement_seeds']
    required = 2 if len(seeds) == 3 else len(seeds)
    for s in ('pooled', 'partly_learned'):
        for b in BASELINES:
            for m in METRICS:
                cell = comparisons.get(s, {}).get(b, {}).get(m, {})
                if not np.isfinite(cell.get('ci_low', np.nan)) or cell.get('ci_low', np.nan) <= 0:
                    reasons.append(f'{s} {b} {m}: CI low is not > 0')
                if s == 'partly_learned':
                    signs = cell.get('per_seed_signs', {})
                    if sum(signs.get(seed, np.nan) > 0 for seed in seeds) < required:
                        reasons.append(f'{s} {b} {m}: positive sign required in {required} seeds')
                    for seed in seeds:
                        if not np.isfinite(signs.get(seed, np.nan)):
                            reasons.append(f'{s} {b} {m}: seed {seed} lacks eligible metric data')
    result['verdict'] = 'FAIL' if reasons else 'PASS'
    return result
