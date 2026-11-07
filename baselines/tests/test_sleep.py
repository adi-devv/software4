"""Offline sleep integration: real hippocampus, services, gate and runner."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from baselines import FakeBackend, HippoOnly, SleepArm, SleepConfig
from harness.canaries import ListCanarySet, SyntheticCanarySet
from harness.runner import Services, read_lines, run_arm, SimulatedCrash
from harness.streams import synthetic_stream
from harness.tasks import Item, Task, arm_scope
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
        return np.array([[1., 0., 0.] if ('danger' in p or 'What is 1' in p)
                         else [0., 1., 0.] if 'Cobalt' in p or 'cobalt' in p
                         else [0., 0., 1.] for p in texts])


def test_broken_candidate_gate_rejection_preserves_deployment():
    backend = FakeBackend()
    arm = SleepArm(backend, embedder=CollisionEmbedder())
    canaries = [ListCanarySet('collision', [
        (str(i), f'What is {i} + 20?', i + 20) for i in range(10, 30)])]
    services(arm, gate=[], canaries=canaries)
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
    with arm_scope():
        decision = arm.sleep(1)
    assert not decision['accepted'] and decision['checks']['canary']['regression']
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
    services(arm, gate=[])
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
    services(arm, gate=[])
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
    services(arm, gate=[])
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
