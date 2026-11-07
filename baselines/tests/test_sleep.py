"""Offline sleep integration: real hippocampus, services, gate and runner."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from baselines import ExpertOnly, FakeBackend, HippoOnly, SleepArm, SleepConfig
from harness.canaries import ListCanarySet, SyntheticCanarySet
from harness.runner import Services, read_lines, run_arm, SimulatedCrash
from harness.streams import synthetic_stream
from harness.tasks import Item, Task, arm_scope, grading_scope
from harness.worlds import StreamConfig
from hippocampus import Hippocampus


PROMPTS = {
    'A': [f'Cobalt beacon signal {word}?' for word in ('alpha', 'bravo', 'charlie', 'delta')],
    'B': [f'Amber orchard harvest {word}?' for word in ('echo', 'foxtrot', 'golf', 'hotel')],
}


def task(prompt, *, id='q', mode='day', kind='probe', source='trusted'):
    return Task(id, prompt, 'fact', 'toy', mode=mode, kind=kind, source=source)


def held_out(prompt):
    # Distinct prompt strings and item ids; the fake's documented normalization
    # handles only this polite wrapper, not unseen facts or semantic reasoning.
    return 'Please ' + prompt.lower()


def item(prompt, answer, id, *, family='fact'):
    return Item(id=id, prompt=prompt, expected=answer, skill='toy', family=family,
                kind='probe', key=id, input_key=id, split='gate')


def services(arm, *, gate=None, canaries=None):
    gate = gate if gate is not None else [
        item(held_out(p), f'{skill}{i}', f'gate-{skill}-{i}')
        for skill, prompts in PROMPTS.items() for i, p in enumerate(prompts)]
    canaries = canaries if canaries is not None else [
        ListCanarySet('arithmetic', [(str(i), f'What is {i} + 20?', i + 20)
                                   for i in range(10, 30)])]
    stream = SimpleNamespace(stream_id='sleep-test', n_cycles=10, gate_items=lambda c: gate)
    service = Services(stream, canaries)
    arm.bind(service)
    return service


def learn(arm, skill):
    for i, prompt in enumerate(PROMPTS[skill]):
        t = task(prompt, id=f'{skill}-{i}')
        answer = arm.answer(t)
        arm.observe(t, answer, False, f'Correct answer: {skill}{i}. A training lesson.')


def answer_skill(arm, skill):
    return [arm.answer(task(held_out(p), id=f'test-{skill}-{i}', mode='test'))
            for i, p in enumerate(PROMPTS[skill])]


def test_day_matches_hippo_prompt_and_logs_only_train():
    backend, hippo = FakeBackend(), HippoOnly()
    arm = SleepArm(backend)
    fact = task('Field note: the leader of Alder is Mira.', kind='teach')
    for a in (arm, hippo):
        a.observe(fact, '', None, None)
    q = task('Who leads Alder?')
    answer = arm.answer(q)
    assert answer == hippo.answer(q) == 'Final answer: Mira'
    assert backend.generate_calls[-1] == (
        'Context:\nFact: Field note: the leader of Alder is Mira.\n\nTask:\nWho leads Alder?', None)
    arm.observe(q, answer, True, 'Correct answer: Mira.')
    hippo.observe(q, answer, True, 'Correct answer: Mira.')
    assert arm.hippo.log.events == hippo.hippo.log.events
    state = arm.save_state()
    frozen = task('Who leads Alder?', mode='test')
    assert arm.answer(frozen) == answer
    arm.observe(frozen, 'secret', False, 'secret correction')
    assert arm.save_state() == state
    outcome = arm.hippo.log.events[-1]
    assert outcome['verified'] and outcome['success'] and outcome['memory_ids']


def test_two_sleeps_replay_routing_and_memory_independent_answers():
    arm = SleepArm(FakeBackend())
    services(arm)
    learn(arm, 'A')
    with arm_scope():
        first = arm.sleep(0)
    assert first['accepted'] and first['replay'] == 0
    a = arm.active_experts[0]
    frozen_a = copy.deepcopy(arm.backend.experts[a])
    assert all(arm.route(held_out(p)) == a for p in PROMPTS['A'])
    memory = arm.hippo
    arm.hippo = Hippocampus()
    assert answer_skill(arm, 'A') == [f'Final answer: A{i}' for i in range(4)]
    for p in PROMPTS['A']:
        assert arm.backend.generate('Context:\n\n\nTask:\n' + held_out(p), None) == 'Final answer: unknown'
    arm.hippo = memory
    learn(arm, 'B')
    second = arm.sleep(1)
    assert second['accepted'] and second['replay'] == 4
    b = arm.active_experts[1]
    assert arm.backend.experts[a] == frozen_a
    assert all(arm.route(held_out(p)) == b for p in PROMPTS['B'])
    assert all(arm.route(held_out(p)) == a for p in PROMPTS['A'])
    assert arm.route('What is 15 + 20?') is None
    assert arm.route('unrelated zygomatic xylophone') is None
    assert arm.backend.train_calls[-1]['steps'] == arm.cfg.steps
    assert arm.backend.train_calls[-1]['seed'] == arm.cfg.seed + 1
    arm.hippo = Hippocampus()  # no retrieval crutch
    for skill in ('A', 'B'):
        assert answer_skill(arm, skill) == [f'Final answer: {skill}{i}' for i in range(4)]
        for p in PROMPTS[skill]:
            assert arm.backend.generate('Context:\n\n\nTask:\n' + held_out(p), None) == 'Final answer: unknown'
    assert [e['type'] for e in arm.save_state()['events']] == []


class CollisionEmbedder:
    """Some old prompts overlap the new skill; other canaries pull none away."""
    def embed(self, texts):
        return np.array([[1., 0., 0.] if ('danger' in p or 'What is 10 ' in p)
                         else [0., 1., 0.] if 'Cobalt' in p or 'cobalt' in p
                         else [0., 0., 1.] for p in texts])


@pytest.mark.parametrize('failure', ['norm', 'canary'])
def test_broken_candidate_pre_gate_rejection_preserves_deployment(failure):
    backend = FakeBackend()
    arm = SleepArm(backend, embedder=CollisionEmbedder())
    canaries = [ListCanarySet('collision', [
        (str(i), f'What is {i} + 20?', i + 20) for i in range(10, 30)])]
    services(arm, canaries=canaries)
    learn(arm, 'A')
    assert arm.sleep(0)['accepted']
    for i in range(8):
        arm.observe(task(f'danger exercise {i}', id=f'danger-{i}'),
                    'Final answer: danger', True, None)
    before_active = list(arm.active_experts)
    before_router = copy.deepcopy(arm.centroids)
    before_none = arm.none_centroid.copy()
    before_events = copy.deepcopy(arm.hippo.log.events)
    queries = [task('What is 10 + 20?', mode='test'), task('danger exercise', mode='test')]
    before = [arm.answer(t) for t in queries] + answer_skill(arm, 'A')
    fingerprint = backend.frozen_fingerprint()
    backend.next_broken = True
    if failure == 'canary':
        # Finite, small-norm damage must also be caught on absolute canaries.
        backend.adapter_delta_ratio = lambda h: 0.01
    with arm_scope():
        decision = arm.sleep(1)
    assert not decision['accepted']
    assert decision['reasons'] == (['adapter delta exceeds frozen projection norm']
                                  if failure == 'norm' else ['absolute canary output drift'])
    if failure == 'canary':
        assert len(decision['pre_gate']['changed_canaries']) == 1
    assert 'checks' not in decision  # statistical gate has not been scored
    assert backend.discarded == [decision['candidate']]
    assert decision['candidate'] not in backend.experts
    assert arm.active_experts == before_active
    assert arm.hippo.log.events == before_events
    for h in before_router:
        np.testing.assert_array_equal(arm.centroids[h], before_router[h])
    np.testing.assert_array_equal(arm.none_centroid, before_none)
    assert [arm.answer(t) for t in queries] + answer_skill(arm, 'A') == before
    assert backend.frozen_fingerprint() == fingerprint
    backend.next_broken = False
    # Rejected episodes stay new; a later finite candidate can retry them.
    assert arm.sleep(2)['examples'] == decision['examples']


def test_norm_pre_gate_rejects_before_session_even_when_router_hides_damage():
    arm = SleepArm(FakeBackend())
    svc = services(arm)
    learn(arm, 'A')
    before = arm.save_state()
    arm.backend.next_broken = True
    info = arm.sleep(0)
    assert not info['accepted'] and info['pre_gate']['max_delta_base_ratio'] > 1
    assert svc.gate_session(0).tasks
    after = arm.save_state()
    for key in ('active_experts', 'events', 'centroids', 'none_centroid', 'pending', 'tick'):
        assert after[key] == before[key]
    # Retry the same data with a healthy candidate: no learning-gain threshold.
    assert arm.sleep(1)['accepted']


def test_below_cap_router_hidden_damage_rejected_on_trusted_examples():
    from harness.smoke_world import smoke_stream, smoke_canaries
    from harness.grading import grade
    from harness.tasks import grading_scope
    arm = SleepArm(FakeBackend())
    stream, canary = smoke_stream(0), smoke_canaries()
    arm.bind(Services(stream, [canary]))
    for cycle in range(3):
        for it in stream.train_items(cycle):
            t = it.task('day', cycle)
            answer = arm.answer(t)
            if t.kind == 'teach':
                arm.observe(t, answer, None, None)
            else:
                with grading_scope():
                    result = grade(it, answer, cycle)
                    correction = stream.correction_for(it)
                arm.observe(t, answer, result.correct, correction)
        if cycle < 2:
            arm.sleep(cycle)
    before = arm.save_state()
    arm.backend.next_broken = True
    arm.backend.adapter_delta_ratio = lambda h: 0.99
    info = arm.sleep(2)
    assert not info['accepted']
    assert info['pre_gate']['changed_canaries'] == []
    assert info['pre_gate']['validation']['regressions']
    assert info['pre_gate']['max_delta_base_ratio'] == 0.99
    assert info['pre_gate']['changed_canaries'] == []
    assert 'checks' not in info
    after = arm.save_state()
    for key in ('events', 'active_experts', 'centroids', 'none_centroid', 'accepted_ids'):
        assert after[key] == before[key]


WORDS = ('alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel')
WIDE = {'A': [f'Cobalt beacon signal {w}?' for w in WORDS],
        'B': [f'Amber orchard harvest {w}?' for w in WORDS]}


def wide_arm(corrupt):
    """Two accepted-skill sleeps with a replay of 8; `corrupt` replay prompts flip in sleep 2."""
    arm = SleepArm(FakeBackend(), SleepConfig(replay_size=8))
    gate = [item(held_out(p), f'{skill}{i}', f'gate-{skill}-{i}')
            for skill, prompts in WIDE.items() for i, p in enumerate(prompts)]
    services(arm, gate=gate)
    def learn_wide(skill):
        for i, prompt in enumerate(WIDE[skill]):
            t = task(prompt, id=f'{skill}-{i}')
            arm.observe(t, arm.answer(t), False, f'Correct answer: {skill}{i}. A training lesson.')
    learn_wide('A')
    assert arm.sleep(0)['accepted']
    learn_wide('B')
    train = arm.backend.train_expert
    def damaged(examples, steps, seed):
        handle = train(examples, steps, seed)
        for prompt in WIDE['A'][:corrupt]:
            arm.backend.experts[handle]['table'].pop(FakeBackend._key(prompt))
        return handle
    arm.backend.train_expert = damaged
    return arm


def test_replay_tolerance_accepts_one_flip_in_eight():
    arm = wide_arm(corrupt=1)
    info = arm.sleep(1)
    validation = info['pre_gate']['validation']
    assert info['replay'] == 8 and validation['replay_flip_tolerance'] == 1
    assert len(validation['tolerated']) == 1 and validation['regressions'] == []
    assert 'trusted train/replay validation regressed' not in info['reasons']
    assert 'checks' in info  # reached the unchanged statistical gate


def test_replay_tolerance_rejects_two_flips_in_eight():
    arm = wide_arm(corrupt=2)
    before = arm.save_state()
    info = arm.sleep(1)
    validation = info['pre_gate']['validation']
    assert not info['accepted'] and info['reasons'] == ['trusted train/replay validation regressed']
    assert len(validation['tolerated']) == 1 and len(validation['regressions']) == 1
    assert arm.save_state()['active_experts'] == before['active_experts']
    assert 'checks' not in info


def test_replay_tolerance_never_applies_to_new_examples_or_exceeds_one_eighth():
    arm = wide_arm(corrupt=0)
    info = arm.sleep(1)
    assert info['pre_gate']['validation']['tolerated'] == []
    with pytest.raises(ValueError):
        SleepConfig(replay_flip_fraction=0.25)


def test_uncovered_healthy_candidate_is_deferred():
    arm = SleepArm(FakeBackend())
    services(arm, gate=[])
    learn(arm, 'A')
    info = arm.sleep(0)
    assert not info['accepted']
    assert 'uncovered candidate routes' in info['reasons']
    assert 'checks' not in info


@pytest.mark.parametrize('covered', [False, True])
def test_displaced_retained_facts_require_independent_gate_coverage(covered):
    class FactCollision:
        def embed(self, texts):
            return np.array([[0., 1.] if 'What is ' in p else [1., 0.] for p in texts])

    arm = SleepArm(FakeBackend(), embedder=FactCollision())
    retained = Task('retained', 'Field note: the leader of Alder is Mira.',
                    'fact', 'fact:retained', kind='teach', source='trusted')
    arm.observe(retained, '', None, None)
    gate = [item(held_out(p), f'A{i}', f'gate-A-{i}')
            for i, p in enumerate(PROMPTS['A'])]
    if covered:
        gate.append(Item(id='gate-retained', prompt='Who leads Alder?', expected='Mira',
                         family='fact', skill='fact:retained', kind='probe', split='gate',
                         key='gate-retained', input_key='leader|Alder'))
    services(arm, gate=gate)
    learn(arm, 'A')
    info = arm.sleep(0)
    assert info['accepted'] is covered
    missing = info['pre_gate']['coverage']['uncovered']
    if not covered:
        assert any(row.get('skill') == 'fact:retained' for row in missing)
        assert 'checks' not in info


def test_below_cap_hidden_damage_rejected_on_trusted_validation():
    from harness.smoke_world import smoke_stream, smoke_canaries
    stream = smoke_stream(0)
    arm = SleepArm(FakeBackend())
    arm.bind(Services(stream, [smoke_canaries()]))
    for cycle in range(3):
        for it in stream.train_items(cycle):
            t = it.task('day', cycle)
            if t.kind == 'teach':
                arm.observe(t, '', None, None)
            else:
                # Trusted training feedback, never hidden/gate answers.
                with grading_scope():
                    target = it.expected_at(cycle)
                arm.observe(t, arm.answer(t), False, 'Correct answer: ' + target + '.')
        if cycle < 2:
            assert arm.sleep(cycle)['accepted']
    arm.backend.next_broken = True
    arm.backend.adapter_delta_ratio = lambda h: 0.99
    before = arm.save_state()
    info = arm.sleep(2)
    assert not info['accepted']
    assert info['pre_gate']['max_delta_base_ratio'] == 0.99
    assert info['pre_gate']['changed_canaries'] == []
    assert info['pre_gate']['validation']['regressions']
    assert not arm.active_experts or arm.active_experts == before['active_experts']


def test_healthy_candidate_without_relevant_routes_is_deferred():
    arm = SleepArm(FakeBackend())
    services(arm, gate=[])
    learn(arm, 'A')
    info = arm.sleep(0)
    assert not info['accepted']
    assert 'uncovered candidate routes' in info['reasons']
    assert info['pre_gate']['validation']['candidate_correct'] == 4


def test_snapshot_bundle_relocates_without_original_directory(tmp_path):
    import shutil
    from pathlib import Path
    original, copied = tmp_path / 'original', tmp_path / 'copied'
    arm = SleepArm(FakeBackend(), snapshot_dir=original / 'backend_snapshots')
    services(arm)
    learn(arm, 'A')
    assert arm.sleep(0)['accepted']
    state = json.loads(json.dumps(arm.save_state()))
    assert not Path(state['backend_snapshot']).is_absolute()
    shutil.copytree(original, copied)
    shutil.rmtree(original)
    restored = SleepArm(FakeBackend(), snapshot_dir=copied / 'backend_snapshots')
    restored.load_state(state)
    assert answer_skill(restored, 'A') == answer_skill(arm, 'A')


def test_nonfinite_rejected_before_opening_or_running_gate():
    arm = SleepArm(FakeBackend())
    svc = services(arm)
    learn(arm, 'A')
    calls = len(arm.backend.generate_calls)
    arm.backend.next_nonfinite = True
    decision = arm.sleep(0)
    assert decision['reasons'] == ['non-finite candidate']
    assert not decision['accepted'] and not arm.active_experts
    assert len(arm.backend.generate_calls) == calls
    assert arm.backend.discarded == [decision['candidate']]
    assert svc.gate_session(0).tasks  # session was never opened
    assert not any(e['type'] == 'sleep' for e in arm.hippo.log.events)


@pytest.mark.parametrize('excluded', ['quarantine', 'untrusted', 'tombstone', 'unverified',
                                     'bad-context', 'no-target', 'teach'])
def test_ineligible_episodes_never_train(excluded):
    arm = SleepArm(FakeBackend())
    services(arm)
    learn(arm, 'A')
    poison = task('poison payload', id='poison',
                  source='untrusted' if excluded == 'quarantine' else 'trusted',
                  kind='teach' if excluded == 'teach' else 'probe')
    arm.observe(poison, 'Final answer: poison', True, None)
    mid = next(e['id'] for e in reversed(arm.hippo.log.events) if e['type'] == 'add')
    if excluded == 'untrusted':
        arm.hippo.set_trust(mid, 0.1, t=arm.tick)
    elif excluded == 'tombstone':
        arm.hippo.tombstone(mid, t=arm.tick)
    elif excluded == 'unverified':
        # Add an independently unverified outcome/episode through the public API.
        arm.hippo.tombstone(mid, t=arm.tick)
        mid = arm.hippo.add('Question: poison payload\nResponse: Final answer: poison\nCorrect: True',
                           source='system', kind='episode', t=arm.tick)
        arm.hippo.record_outcome('poison', success=True, verified=False, episode_id=mid, t=arm.tick)
    elif excluded == 'bad-context':
        arm.hippo.tombstone(mid, t=arm.tick)
        # Even a trusted episode is excluded if an involved memory is revoked.
        ep = arm.hippo.add('Question: poison payload\nResponse: Final answer: poison\nCorrect: True',
                          source='system', kind='episode', t=arm.tick)
        arm.hippo.record_outcome('poison', [mid], success=True, verified=True,
                                episode_id=ep, t=arm.tick)
    elif excluded == 'no-target':
        arm.hippo.tombstone(mid, t=arm.tick)
        arm.observe(task('poison payload', id='empty'), 'wrong', False, None)
    assert arm.sleep(0)['accepted']
    assert all('poison' not in q and 'poison' not in a
               for q, a in arm.backend.train_calls[-1]['examples'])


def test_revocation_applies_to_old_replay_too():
    arm = SleepArm(FakeBackend())
    services(arm)
    learn(arm, 'A')
    assert arm.sleep(0)['accepted']
    for i, mid in enumerate(sorted(arm._accepted_ids)):
        if i % 2:
            arm.hippo.tombstone(mid, t=arm.tick)
        else:
            arm.hippo.set_trust(mid, 0.1, t=arm.tick)
    learn(arm, 'B')
    assert arm.sleep(1)['replay'] == 0
    assert all('Cobalt' not in p for p, _ in arm.backend.train_calls[-1]['examples'])


def test_caps_targets_and_since_last_accepted_sleep():
    arm = SleepArm(FakeBackend(), SleepConfig(max_examples=4, replay_size=1, steps=3, seed=9))
    services(arm)
    learn(arm, 'A')
    assert arm.sleep(0)['examples'] == 4
    learn(arm, 'B')
    info = arm.sleep(1)
    assert info['examples'] == 4 and info['replay'] == 1
    examples = arm.backend.train_calls[-1]['examples']
    assert all(p.startswith('Context:\n\n\nTask:\n') and 'Response:' not in p for p, _ in examples)
    assert all(a.startswith('Final answer: ') and 'lesson' not in a for _, a in examples)
    calls = len(arm.backend.train_calls)
    assert arm.sleep(2)['reasons'] == ['no new verified examples']
    assert len(arm.backend.train_calls) == calls


@pytest.mark.parametrize('answer,result,correction,target', [
    ('reasoning\nFinal answer: 3.14', True, None, '3.14'),
    ('plain response', True, 'Correct answer: target.', 'target'),
    ('wrong', False, 'Correct answer: 3.14. Decimal lesson.', '3.14'),
    ('wrong', False, 'Final answer: several words', 'several words'),
    ('wrong', False, 'an explicit correction', 'an explicit correction'),
])
def test_correct_final_or_correction_becomes_target(answer, result, correction, target):
    arm = SleepArm(FakeBackend())
    services(arm, gate=[item(held_out('training query'), target, 'gate-target')])
    arm.observe(task('training query'), answer, result, correction)
    assert arm.sleep(0)['accepted']
    assert arm.backend.train_calls[-1]['examples'][0][1] == 'Final answer: ' + target


def test_save_load_immutable_snapshots_and_reset():
    arm = SleepArm(FakeBackend())
    services(arm)
    learn(arm, 'A')
    assert arm.sleep(0)['accepted']
    state_a = json.loads(json.dumps(arm.save_state()))
    learn(arm, 'B')
    assert arm.sleep(1)['accepted']
    state_ab = json.loads(json.dumps(arm.save_state()))
    restored = SleepArm(FakeBackend())
    restored.load_state(state_ab)
    assert restored.save_state() == state_ab
    assert restored.tick == arm.tick
    for skill in ('A', 'B'):
        assert answer_skill(restored, skill) == answer_skill(arm, skill)
    restored.load_state(state_a)  # later saves never overwrite earlier weights
    assert len(restored.active_experts) == 1
    assert answer_skill(restored, 'A') == [f'Final answer: A{i}' for i in range(4)]
    restored.hippo = Hippocampus()
    assert answer_skill(restored, 'B') == ['Final answer: unknown'] * 4
    restored.reset()
    assert not restored.active_experts and not restored.centroids and not len(restored.hippo)
    assert restored.tick == 0 and restored.none_centroid is None
    assert not restored.backend.experts


def test_run_arm_short_synthetic_stream_and_checkpoint_resume(tmp_path):
    cfg = StreamConfig(n_cycles=4, entities_per_domain=6, items_per_rule=24,
                       rule_kinds=('affine',))
    stream = synthetic_stream(7, cfg)
    canaries = [SyntheticCanarySet(seed=2, n=12)]
    arm = SleepArm(FakeBackend())
    path, checkpoints = tmp_path / 'run.jsonl', tmp_path / 'checkpoints'

    def crash(cycle, phase, n):
        if cycle == 1 and phase == 'sleep':
            raise SimulatedCrash()

    with pytest.raises(SimulatedCrash):
        run_arm(arm, stream, canaries, path, ckpt_dir=checkpoints, fault_hook=crash)
    resumed = SleepArm(FakeBackend())
    run_arm(resumed, stream, canaries, path, ckpt_dir=checkpoints)
    uninterrupted = SleepArm(FakeBackend())
    full_path = tmp_path / 'full.jsonl'
    run_arm(uninterrupted, stream, canaries, full_path)
    assert resumed.active_experts == uninterrupted.active_experts
    assert resumed.hippo.log.events == uninterrupted.hippo.log.events
    assert any(r['type'] == 'cycle_done' and r['cycle'] == 3 for r in read_lines(path))
    assert not any(r['type'] == 'violation' for r in read_lines(path))
    for it in stream.hidden_items(1):
        assert resumed.answer(it.task('test', 1)) == uninterrupted.answer(it.task('test', 1))


def test_expert_only_shows_an_empty_context_but_keeps_ours_sleep_and_routing():
    ours, expert = SleepArm(FakeBackend(), SleepConfig(steps=8)), ExpertOnly(FakeBackend(), SleepConfig(steps=8))
    for arm in (ours, expert):
        arm.reset()
        services(arm)
        learn(arm, 'A')
    probe = task(held_out(PROMPTS['A'][0]), id='probe', mode='test')
    assert 'Verified answer: A0' in ours._prompt(probe)  # ours still retrieves context
    assert expert._prompt(probe) == 'Context:\n\n\nTask:\n' + probe.prompt  # empty, as experts train
    assert expert._prompt(probe, remember=True) == expert._prompt(probe)
    assert expert.name == 'expert_only' and expert.route(probe.prompt) is None
    assert isinstance(expert, SleepArm)
