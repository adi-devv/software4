"""Opt-in v3 report; frozen v2 claim calculations and inference stay unchanged.

The v2 verdict accepts only 48-52. A pure data adapter maps the five v3
registration slots to those slots and wrong_rule to its former control slot.
Actual world generation/grading always uses the real v3 seed. No frozen module
globals, source files, run artifacts, answers, or score values are rewritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import json
from pathlib import Path
import re

from harness.grading import _extract, match
from harness.runner import read_lines
from harness.step5_world import step5_stream
from harness.tasks import grading_scope
from step4 import v2_analyze as v2
from step4.v3_control_validity import control_validity

V3_SEEDS = (53, 54, 55)
V3_REPLACEMENTS = (56, 57)
V3_ARMS = tuple('wrong_rule' if a == 'generic_forced_noret' else a for a in v2.V2_ARMS)
_TO_V2 = dict(zip((*V3_SEEDS, *V3_REPLACEMENTS), (*v2.V2_SEEDS, *v2.V2_REPLACEMENTS)))


def load_v3_run(run_dir):
    """Grade the registered world, retaining all missing/duplicate evidence.

    Mirrors the frozen loader's evidence policy, using its fact classifiers.
    The separate audit also covers cycles 0/1 and unscored logged attempts.
    """
    path = Path(run_dir)
    data = v2.read_json(path / 'results.json')
    cfg = data.get('config', {})
    seed = cfg.get('world_seed')
    if cfg.get('world') != 'step5' or cfg.get('cycles') != 3:
        raise ValueError('v3 requires step5 and exactly 3 cycles')
    if type(seed) is not int or seed not in _TO_V2:
        raise ValueError('v3 seeds are 53, 54, 55 (conditional replacements: 56, 57)')
    arms = data.get('results')
    if not isinstance(arms, dict):
        raise ValueError('missing worker results')
    stream = step5_stream(3, seed)
    items = {c: {i.id: i for i in stream.hidden_items(c)} for c in range(3)}
    with grading_scope():
        targets = {(2, i.id): i.expected_at(2) for i in items[2].values() if i.skill in v2.RULES}
    for name, result in arms.items():
        answers = read_lines(path / name / 'answers.jsonl')
        indexed = defaultdict(list)
        for row in answers:
            indexed[(row.get('day'), row.get('task_id'))].append(row)
        result['context_tokens'], result['fact_answers'] = {}, []
        result['answer_row_audit'] = {
            c: dict(missing=sorted(i for i in expected if not indexed[(c, i)]),
                    duplicates={i: len(rows) for (day, i), rows in indexed.items()
                                if day == c and len(rows) > 1},
                    unknown=sorted(str(i) for (day, i) in indexed if day == c and i not in expected))
            for c, expected in items.items()}
        for row in result.get('scores', []):
            cycle, task_id = row.get('cycle'), row.get('task_id')
            if row.get('phase') != 'hidden' or cycle not in items:
                continue
            item, evidence = items[cycle].get(task_id), indexed[(cycle, task_id)]
            row['correct_exact'] = None
            if item is None or item.skill != row.get('skill') or len(evidence) != 1:
                row['evidence_error'] = 'unknown task/skill or missing/duplicate answer ID'
                continue
            answer = evidence[0].get('answer')
            if not isinstance(answer, str):
                row['evidence_error'] = 'missing answer text'
                continue
            text, _ = _extract(answer)
            with grading_scope():
                expected = item.expected_at(cycle)
            row['correct'] = bool(text is not None and match(expected, text))
            row['correct_exact'] = bool(text is not None and text.strip() == expected.strip())
            result['context_tokens'][f'{cycle}:{task_id}'] = evidence[0].get('context_tokens')
        for (cycle, task_id), evidence in indexed.items():
            item = items.get(cycle, {}).get(task_id)
            if item is None:
                fact_rows = [r for r in evidence if r.get('skill') == 'fact:smoke']
                expected, stale, entity = '', (), 'unknown'
            elif item.skill == 'fact:smoke':
                fact_rows = evidence
                with grading_scope():
                    expected, stale = item.expected_at(cycle), item.stale_at(cycle)
                entity = item.key.split('|', 1)[1]
            else:
                continue
            for answer in fact_rows:
                result['fact_answers'].append(dict(
                    cycle=cycle, task_id=task_id, entity=entity,
                    category=v2.classify_fact_answer(answer.get('answer'), expected, stale),
                    format_leaked=v2._format_leaked(answer.get('answer'))))
        if name == 'wrong_rule':
            try:
                result['control_validity'] = control_validity(answers, expected=targets)
            except ValueError as error:
                result['control_validity'] = dict(valid=False, rule_items=len(targets), error=str(error))
    return seed, arms


def _rename(value, keys, *, errors=False):
    """Translate only metadata keys/labels, never numeric measurements."""
    if isinstance(value, dict):
        return {keys.get(k, k): _rename(v, keys, errors=errors) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename(v, keys, errors=errors) for v in value]
    if isinstance(value, str):
        value = value.replace('generic_forced_noret', 'wrong_rule') if errors else value
        if errors:
            value = re.sub(r'^(48|49|50|51|52):', lambda m: str(int(m[1]) + 5) + ':', value)
        return value
    return value


def v3_verdict(runs):
    """Enforce Addendum 1 before emitting any claim verdict, including subsets."""
    pairs = list(runs.items() if isinstance(runs, Mapping) else runs)
    seeds = dict(pairs)
    if len(seeds) != len(pairs):
        raise ValueError('duplicate world seeds')
    gates = {s: v2._gate(a) for s, a in seeds.items() if type(s) is int and s in _TO_V2}
    invalid = {s: 'unregistered seed' for s in seeds if type(s) is not int or s not in _TO_V2}
    invalid.update({s: g['reason'] for s, g in gates.items() if g['status'] == 'invalid'})
    for seed, arms in seeds.items():
        unfinished = [a for a in V3_ARMS if a not in arms or arms[a].get('status') != 'complete']
        unfinished += [a for a, r in arms.items()
                       if a not in V3_ARMS and r.get('status', 'complete') != 'complete']
        if unfinished:
            invalid[seed] = f'workers not complete (no usable finished seed): {unfinished}'
    pending = [s for s in V3_SEEDS if s not in seeds]
    slots = sum(gates.get(s, {}).get('status') == 'void' for s in V3_SEEDS)
    for s in V3_REPLACEMENTS:
        if s in seeds:
            if pending or slots < 1:
                invalid[s] = 'replacement requires a void seed and all preceding registered gate results'
            elif gates[s]['status'] == 'valid':
                slots -= 1
        elif slots > 0 and not pending:
            pending.append(s)
    state = 'STOP' if invalid else 'PENDING' if pending else 'READY'
    extras = dict(procedure=state, pending_seeds=pending,
                  answer_row_audit={s: {a: r.get('answer_row_audit', {}) for a, r in arms.items()}
                                    for s, arms in seeds.items()},
                  fallback_fired={s: {a: r.get('stats', {}).get('fallback_fired', 0) for a, r in arms.items()}
                                  for s, arms in seeds.items()},
                  context_cost={s: {a: v2._context_mean({s: arms}, a) for a in arms}
                                for s, arms in seeds.items()})
    if state != 'READY':
        return dict(**extras, gates=gates, invalid_seeds=invalid,
                    void_seeds={s: g['reason'] for s, g in gates.items() if g['status'] == 'void'},
                    valid_seeds=[s for s, g in gates.items() if g['status'] == 'valid' and s not in invalid],
                    claims={i: None for i in range(1, 6)}, verdict=state,
                    reason='Stop on invalid seed; no replacement/rerun authorized.' if invalid else
                    'No subset verdicts; supply all confirmation seeds and required void replacements.')
    # Slot translation preserves ordering and within-seed pairing. The old
    # control, even if present, is excluded from the Claim 3 adapter input.
    adapted = {_TO_V2[s]: {**{a: r for a, r in arms.items() if a not in ('wrong_rule', 'generic_forced_noret')},
                           **({'generic_forced_noret': arms['wrong_rule']} if 'wrong_rule' in arms else {})}
               for s, arms in seeds.items()}
    frozen = v2.v2_verdict(adapted)
    names = {v: k for k, v in _TO_V2.items()}
    names.update({str(v): str(k) for k, v in _TO_V2.items()})
    names['generic_forced_noret'] = 'wrong_rule'
    report = _rename(frozen, names, errors=True)
    report['valid_seeds'] = [names[s] for s in frozen['valid_seeds']]
    valid = {s: seeds[s] for s in report['valid_seeds']}
    controls = {s: arms.get('wrong_rule', {}).get('control_validity', dict(valid=False, error='missing validity evidence'))
                for s, arms in valid.items()}
    informative = bool(controls) and all(c.get('valid') is True for c in controls.values())
    # Frozen Claim 3 already uses expert_only minus the adapted wrong_rule,
    # the registered 30-point margin, strict CI low > 0, and training checks.
    report['claims'][3] = report['claims'][3] if informative else 'UNINFORMATIVE'
    report['control_validity'] = controls
    report['verdict'] = ('PASS' if report['complete_arms'] and
                         all(c is True for c in report['claims'].values()) else
                         'UNINFORMATIVE' if not informative and all(report['claims'][i] for i in (1, 2, 4, 5))
                         and report['complete_arms'] else 'FAIL')
    return dict(report, **extras)


def v3_markdown(report):
    """Reuse frozen evidence tables; replace only v3 labels and Claim 3 status."""
    v = report
    qualifier = (f" Fewer than 3 valid seeds ({len(v['valid_seeds'])}); excluded/void: "
                 f"{v['invalid_seeds']}; {v['void_seeds']}." if len(v['valid_seeds']) < 3 else '')
    if v['procedure'] == 'READY':
        render = _rename(v, {'wrong_rule': 'generic_forced_noret'})
        text = v2.v2_markdown(render).replace('# v2 verdict', '# v3 verdict').replace('generic_forced_noret', 'wrong_rule')
        lines = []
        for line in text.splitlines():
            if line.startswith('Claim 3:'):
                status = 'PASS' if v['claims'][3] is True else 'UNINFORMATIVE' if v['claims'][3] == 'UNINFORMATIVE' else 'FAIL'
                line = f'Claim 3: {status} (expert_only - wrong_rule >= 30 points, CI low > 0; control >=90% well-formed on all final-cycle rule items; matched training).'
            if line.startswith('Claim ') and re.match(r'Claim [1-5]:', line):
                line += qualifier
            if line.startswith('Overall verdict:'):
                line = f"Overall verdict: {v['verdict']}" + qualifier
            lines.append(line)
    else:
        lines = ['# v3 report', '', f"Procedure: {v['procedure']}. {v['reason']}",
                 f"Pending seeds: {v['pending_seeds']}; invalid: {v['invalid_seeds']}; void: {v['void_seeds']}."]
    lines += ['', f"Wrong-rule control validity: {v.get('control_validity', 'not evaluated before complete seed set')}.",
              f"fallback_fired (per seed/arm): {v['fallback_fired']}.",
              f"Mean context tokens per hidden answer (per seed/arm, all cycles, including fallback cost): {v['context_cost']}.",
              'Answer-row audit (all hidden cycles, including zero missing/duplicates):',
              '```json', json.dumps(v['answer_row_audit'], indent=2), '```']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = v3_verdict([load_v3_run(p) for p in args.runs])
    except ValueError as error:
        parser.error(str(error))
    args.out.write_text(v3_markdown(report), encoding='utf-8')
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
