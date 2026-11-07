"""Run the real local three-cycle smoke. No downloads, no model grading.

HF_HUB_OFFLINE=1 S4_MODEL=Qwen/Qwen3-0.6B python scripts/smoke_sleep.py
S4_SMOKE=1 uses two optimizer steps instead of 64 (same world and checks).
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines import NoMemory, HippoOnly, SleepArm, SleepConfig
from baselines.arms import build_prompt
from harness.grading import grade
from harness.runner import read_lines, run_arm
from harness.smoke_world import smoke_stream, smoke_canaries, gold_context, REVISED, POISON
from harness.tasks import arm_scope, grading_scope


class Oracle(NoMemory):
    name = 'oracle'

    def context(self, task):
        return [(0, gold_context(task))] if gold_context(task) else []


class MeasuredSleep(SleepArm):
    def sleep(self, cycle):
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if cycle == 2:
            self.backend.next_broken = True
        steps = self.cfg.steps
        if cycle == 2:
            self.cfg.steps = 1  # fault injection needs no prolonged optimization
        try:
            info = super().sleep(cycle)
        finally:
            self.cfg.steps = steps
        torch.cuda.synchronize()
        info.update(wall_seconds=time.perf_counter() - start,
                    peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                    fault_injected=cycle == 2)
        print(f'Sleep {cycle + 1}: {info}', flush=True)
        return info


def digest_outputs(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def prompt_for(backend, task, context=()):
    return build_prompt(backend, task, list(context), 64)[0]


def measure(items, cycle, predict):
    # Arms receive tasks and produce strings before the trusted grader is opened.
    rows = []
    for it in items:
        with arm_scope():
            answer = predict(it.task('test', cycle))
        with grading_scope():
            result = grade(it, answer, cycle)
        rows.append(dict(task_id=it.id, skill=it.skill, prompt=it.prompt,
                         answer=answer, correct=int(result.correct)))
    return rows


def accuracy(rows, skill):
    selected = [r['correct'] for r in rows if r['skill'] == skill]
    return sum(selected) / len(selected) if selected else None


def summarize_scores(records):
    """Only held-out/canary scores belong in the retention table, never day scores."""
    result = {}
    for name in ['no_memory', 'hippo_only', 'ours', 'oracle']:
        result[name] = {}
        for cycle in range(-1, 3):
            rows = [r for r in records if r['type'] == 'score' and r['arm'] == name
                    and r['cycle'] == cycle and r['phase'] in ('hidden', 'canary')]
            result[name][str(cycle)] = {s: accuracy(rows, s) for s in
                                      ['rule:A', 'rule:B', 'fact:smoke', 'canary:smoke']}
    return result


def main():
    import torch
    from baselines.sleep_hf import HFBackend
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    small = os.environ.get('S4_SMOKE') == '1'
    root = Path(__file__).resolve().parents[1]
    out = args.output or root / 'results' / 'smoke_sleep'
    if small and args.output is None:
        out = out / 'small'
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / 'results.jsonl'
    if result_path.exists():
        raise RuntimeError(f'Use a fresh output directory; {result_path} already exists')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    start = time.perf_counter()
    stream, canary = smoke_stream(0), smoke_canaries()
    backend = HFBackend(directory=out / 'adapters')
    fingerprint = backend.frozen_fingerprint()
    print(f'{torch.cuda.get_device_name()} | torch {torch.__version__} | small={small}', flush=True)

    # Verify every canary on the actual base before permitting the run.
    verified = measure(canary.items(), -1, lambda t: backend.generate(prompt_for(backend, t)))
    if not all(r['correct'] for r in verified):
        (out / 'canary_failures.json').write_text(json.dumps(verified, indent=2), encoding='utf-8')
        raise RuntimeError('Base failed a canary: inspect canary_failures.json and replace it')
    (out / 'canary_verification.json').write_text(json.dumps(verified, indent=2), encoding='utf-8')
    print('Base canaries: 8/8', flush=True)

    all_items = stream.hidden_items(2) + canary.items()
    base_before = measure(all_items, 2, lambda t: backend.generate(prompt_for(backend, t)))
    ablations, routing, recall, rollback = {}, {}, {}, {}
    for arm in [NoMemory(llm=backend, context_tokens=64),
                HippoOnly(llm=backend, context_tokens=64),
                Oracle(llm=backend, context_tokens=64)]:
        print(f'Running {arm.name}', flush=True)
        run_arm(arm, stream, [canary], result_path, ckpt_dir=out / 'checkpoints')

    arm = MeasuredSleep(backend, SleepConfig(steps=2 if small else 64,
                                            context_tokens=64, seed=0), llm=backend)
    before_rollback, active_before, state_before, old_weights = [], [], {}, {}

    def hook(cycle, phase, n):
        nonlocal before_rollback, active_before, state_before, old_weights
        if phase == 'sleep':
            old_weights = {h: backend._state(h) for h in arm.active_experts}
            if cycle == 2:
                # Freeze the inputs too, to test complete deployed-output rollback.
                with arm_scope():
                    rollback['prompts'] = [(arm._prompt(it.task('test', cycle)),
                                             arm.route(it.prompt)) for it in all_items]
                before_rollback = [backend.generate(p, h) for p, h in rollback['prompts']]
                active_before = list(arm.active_experts)
                state_before = dict(centroids=copy.deepcopy(arm.centroids),
                                    none=copy.deepcopy(arm.none_centroid),
                                    events=copy.deepcopy(arm.hippo.log.events))
        if phase != 'hidden' or n != 0:
            return
        items = stream.hidden_items(cycle)
        ablations[str(cycle)] = {}
        for mode in ['on_no_hippo', 'off_no_hippo', 'off_with_hippo']:
            def predict(task):
                prompt = arm._prompt(task) if mode == 'off_with_hippo' else prompt_for(backend, task)
                expert = arm.route(task.prompt) if mode == 'on_no_hippo' else None
                return backend.generate(prompt, expert)
            ablations[str(cycle)][mode] = measure(items, cycle, predict)
        routing[str(cycle)] = [dict(task_id=it.id, skill=it.skill,
                                    expert=arm.route(it.prompt)) for it in items + canary.items()]
        recall[str(cycle)] = {}
        for it in items:
            if it.family == 'fact':
                hits = arm.context(it.task('test', cycle))
                recall[str(cycle)][it.input_key] = [text for _, text in hits]
        unchanged = all(torch.equal(p, backend._state(h)[name][k])
                        for h, weight in old_weights.items()
                        for name, pair in weight.items() for k, p in pair.items())
        ablations[str(cycle)]['old_adapters_unchanged'] = unchanged
        if cycle == 2:
            backend._cache.clear()
            after = [backend.generate(p, arm.route(it.prompt))
                     for (p, _), it in zip(rollback['prompts'], all_items)]
            rollback.update(before_hash=digest_outputs(before_rollback),
                            after_hash=digest_outputs(after), outputs_identical=after == before_rollback,
                            active_unchanged=active_before == arm.active_experts,
                            events_unchanged=state_before['events'] == arm.hippo.log.events,
                            router_unchanged=(list(state_before['centroids']) == list(arm.centroids)
                                and all((v == arm.centroids[h]).all()
                                        for h, v in state_before['centroids'].items())
                                and ((state_before['none'] is None and arm.none_centroid is None)
                                     or (state_before['none'] is not None and arm.none_centroid is not None
                                         and (state_before['none'] == arm.none_centroid).all()))))

    print('Running ours', flush=True)
    run_arm(arm, stream, [canary], result_path, ckpt_dir=out / 'checkpoints', fault_hook=hook)
    backend._cache.clear()
    base_after = measure(all_items, 2, lambda t: backend.generate(prompt_for(backend, t)))
    records = read_lines(result_path)
    sleeps = [r for r in records if r['type'] == 'cycle_done' and r['arm'] == 'ours' and r['cycle'] >= 0]
    per_arm = summarize_scores(records)
    short_ablations = {c: {m: {s: accuracy(rows, s) for s in ['rule:A', 'rule:B', 'fact:smoke']}
                           for m, rows in modes.items() if isinstance(rows, list)}
                       for c, modes in ablations.items()}
    first_a = short_ablations['0']['on_no_hippo']['rule:A']
    second_a = short_ablations['1']['on_no_hippo']['rule:A']
    off_a = short_ablations['0']['off_no_hippo']['rule:A']
    base_a = accuracy(base_before, 'rule:A')
    expected_handles = {f'rule:{r}': sleeps[i]['info'].get('candidate')
                        if sleeps[i]['info']['accepted'] else 'NOT_ACCEPTED'
                        for i, r in enumerate(('A', 'B'))}
    route_errors = [dict(cycle=int(c) + 1, **r) for c, rows in routing.items() for r in rows
                    if r['expert'] != expected_handles.get(r['skill'])]
    revised_hits = recall['1']['Alder|leader']
    poison_hits = [text for values in recall.values() for hits in values.values()
                   for text in hits if POISON in text]
    quarantine = [e for e in arm.hippo.log.events if e['type'] == 'add' and POISON in e['text']]
    train_prompts = {i.prompt for c in range(3) for i in stream.train_items(c) if i.kind == 'probe'}
    forbidden = {i.prompt for i in stream.hidden_items(2)} | {
        i.prompt for c in range(3) for i in stream.gate_items(c)} | {i.prompt for i in canary.items()}
    actual_prompts = {p.rsplit('\n\nTask:\n', 1)[-1]
                      for call in backend.train_calls for p, _ in call['examples']}
    finite = all(call['finite'] and call['skipped_steps'] == 0 for call in backend.train_calls)
    final_fingerprint = backend.frozen_fingerprint()
    checks = {}

    def check(letter, passed, evidence):
        checks[letter] = dict(status='pass' if passed else 'fail', evidence=evidence)

    check('a', first_a > base_a and off_a == base_a,
          dict(rule_A_on_no_hippo=first_a, experts_off=off_a, base=base_a,
               criterion='positive no-context gain; off equals base'))
    check('b', first_a > 0 and second_a >= first_a,
          dict(A_after_sleep1=first_a, A_after_sleep2=second_a,
               criterion='nonzero learning retained without context'))
    check('c', not route_errors, dict(errors=route_errors,
          total=sum(len(rows) for rows in routing.values()), expected=expected_handles))
    check('d', any(REVISED in t for t in revised_hits) and not any('Mira' in t for t in revised_hits)
          and not poison_hits and bool(quarantine) and all(e['quarantined'] for e in quarantine),
          dict(revised_recall=revised_hits, poison_in_recall=poison_hits,
               quarantined_adds=quarantine, criterion='current fact only; poison never recalled'))
    check('e', not sleeps[2]['info']['accepted'] and rollback['outputs_identical']
          and rollback['active_unchanged'] and rollback['router_unchanged'] and rollback['events_unchanged'],
          dict(decision=sleeps[2]['info'], **{k: v for k, v in rollback.items() if k != 'prompts'}))
    check('f', fingerprint == final_fingerprint and base_before == base_after
          and all(v['old_adapters_unchanged'] for v in ablations.values()),
          dict(before_fingerprint=fingerprint, after_fingerprint=final_fingerprint,
               base_before_hash=digest_outputs(base_before), base_after_hash=digest_outputs(base_after),
               old_adapters_unchanged=[v['old_adapters_unchanged'] for v in ablations.values()]))
    check('g', finite and not stream.audit() and actual_prompts <= train_prompts
          and not actual_prompts & forbidden,
          dict(split_audit=stream.audit(), trained_unique_prompts=len(actual_prompts),
               forbidden_overlap=sorted(actual_prompts & forbidden),
               all_training_prompts_from_train=actual_prompts <= train_prompts,
               training=[{k: v for k, v in c.items() if k != 'examples'} for c in backend.train_calls]))
    explanations = {
        'a': f'A without hippocampus: {first_a:.3f}; experts off/base: {off_a:.3f}/{base_a:.3f}.',
        'b': f'A without hippocampus after sleeps 1/2: {first_a:.3f}/{second_a:.3f}; '
             'requires nonzero learned accuracy and no drop.',
        'c': f'{len(route_errors)} routing errors across {sum(map(len, routing.values()))} test prompts.',
        'd': f'Current revision recalled: {any(REVISED in t for t in revised_hits)}; '
             f'obsolete original also recalled: {any("Mira" in t for t in revised_hits)}; '
             f'quarantined poison recalled: {len(poison_hits)}.',
        'e': f'Broken candidate accepted: {sleeps[2]["info"]["accepted"]}; '
             f'post-sleep outputs identical: {rollback["outputs_identical"]}. '
             'The unchanged gate rejects regressions beyond paired noise, not arbitrary bad weights.',
        'f': f'Frozen fingerprint equal: {fingerprint == final_fingerprint}; '
             f'uncached base outputs identical: {base_before == base_after}.',
        'g': f'Nonfinite/skipped training steps: {sum(c["skipped_steps"] for c in backend.train_calls)}; '
             f'split-audit violations: {len(stream.audit())}; '
             f'forbidden training prompt overlap: {len(actual_prompts & forbidden)}.',
    }
    for k, explanation in explanations.items():
        checks[k]['explanation'] = explanation
    payload = dict(small=small, model=os.environ.get('S4_MODEL', 'Qwen/Qwen3-0.6B'),
                   seed=0, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                   steps=2 if small else 64, lr=backend.lr,
                   elapsed_seconds=time.perf_counter() - start, checks=checks,
                   per_arm=per_arm, ablations=short_ablations,
                   sleeps=[r['info'] for r in sleeps], canaries_verified=verified)
    (out / 'CHECKLIST.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    (out / 'evidence.json').write_text(json.dumps(dict(ablations=ablations, routing=routing,
        recall=recall, base_before=base_before, base_after=base_after,
        training=backend.train_calls), indent=2), encoding='utf-8')
    lines = [f'# Local sleep smoke ({"small" if small else "full"})', '',
             f'Model: {payload["model"]}; seed 0; fp16 base, fp32 rank-4 q/v LoRA; '
             f'{payload["steps"]} optimizer steps, batch 4, lr 0.0002. '
             'Cycle 3: one step then finite random damage to the new adapter.', '',
             '| Arm | Cycle | A correct / 8 | B correct / 8 | Facts correct / 6 | Canaries correct / 8 |',
             '|---|---:|---:|---:|---:|---:|']
    def count(value, n):
        return '-' if value is None else str(round(value * n))
    for name, cycles in per_arm.items():
        for cycle, data in cycles.items():
            lines.append(f'| {name} | {int(cycle) + 1} | {count(data["rule:A"],8)} | '
                         f'{count(data["rule:B"],8)} | {count(data["fact:smoke"],6)} | '
                         f'{count(data["canary:smoke"],8)} |')
    lines += ['', '| Cycle | Ablation | A correct / 8 | B correct / 8 |', '|---:|---|---:|---:|']
    for c, modes in short_ablations.items():
        for mode, data in modes.items():
            lines.append(f'| {int(c)+1} | {mode} | {count(data["rule:A"],8)} | {count(data["rule:B"],8)} |')
    lines += ['', '| Sleep | Gate | Wall seconds | Peak allocated MiB | Peak reserved MiB |',
              '|---:|---|---:|---:|---:|']
    for i, r in enumerate(sleeps):
        info = r['info']
        lines.append(f'| {i+1} | {"accept" if info["accepted"] else "reject"} | '
                     f'{info["wall_seconds"]:.1f} | {info["peak_gpu_mib"]:.1f} | {info["peak_reserved_mib"]:.1f} |')
    lines += ['', 'Checklist: ' + ', '.join(f'{k}: {v["status"]}' for k, v in checks.items()) + '.', '',
              f'Total wall time: {payload["elapsed_seconds"]:.1f} seconds.', '',
              'Gate uses the unchanged harness paired-noise test (alpha 0.05). '
              'No learning-gain threshold is used for accepting candidates. '
              'Eight items cannot establish broad retention or statistical significance. '
              'All scores use harness exact-answer grading.', '',
              'Assumptions: hashing router and its default 0.2 threshold; 64-token context budget; '
              '112 input tokens plus at most 16 target tokens. The same world is used in both runs. '
              'A and B are opaque rule names; instructions are supplied only as train corrections '
              'or oracle context. Cycle 3 has eight fresh arithmetic control examples. '
              'No hippocampus, existing arm, gate, or grading code was changed. '
              'Failed checks and complete evidence are in CHECKLIST.json and evidence.json.']
    lines += ['', *[f'- {k}: **{v["status"]}**. {v["explanation"]}' for k, v in checks.items()]]
    report = out / 'SMOKE_RESULTS.md' if small or args.output else root / 'results' / 'SMOKE_RESULTS.md'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: v['status'] for k, v in checks.items()}), flush=True)
    print(f'Finished in {payload["elapsed_seconds"]:.1f}s. {report}', flush=True)


if __name__ == '__main__':
    main()
