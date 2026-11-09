"""CPU-only v2 controls against the real selection, routing and gate machinery."""
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from baselines.generic import (
    GenericForcedNoRetArm, GenericForcedSleepArm, GenericOnlySleepArm,
)
from baselines.sleep import ExpertOnly, SleepArm, SleepConfig
from baselines.sleep_fake import FakeBackend
from harness.canaries import ListCanarySet
from harness.runner import Services
from harness.tasks import Item, Task, arm_scope


PROMPTS = {
    'A': [f'Cobalt beacon signal {w}?' for w in ('alpha', 'bravo', 'charlie', 'delta')],
    'B': [f'Amber orchard harvest {w}?' for w in ('echo', 'foxtrot', 'golf', 'hotel')],
    'C': [f'Violet mountain summit {w}?' for w in ('india', 'juliet', 'kilo', 'lima')],
}


def bind(arm, gate=None):
    if gate is None:
        gate = [Item(id=f'gate-{skill}-{i}', prompt='Please ' + p.lower(),
                     expected=f'{skill}{i}', skill='toy', family='fact', kind='probe',
                     key=f'{skill}-{i}', input_key=f'{skill}-{i}', split='gate')
                for skill, prompts in PROMPTS.items() for i, p in enumerate(prompts)]
    stream = SimpleNamespace(stream_id='v2-forced-test', n_cycles=3, gate_items=lambda c: gate)
    canaries = ListCanarySet('arithmetic', [(str(i), f'What is {i} + 20?', i + 20)
                                         for i in range(10, 30)])
    arm.bind(Services(stream, [canaries]))


def learn(arm, skill):
    for i, prompt in enumerate(PROMPTS[skill]):
        task = Task(f'{skill}-{i}', prompt, 'fact', 'toy')
        # Identical verified episodes across arms, independent of daytime answers.
        arm.observe(task, 'Final answer: unknown', False, f'Correct answer: {skill}{i}.')


def make(cls, tmp_path, cfg=None, **kwargs):
    arm = cls(FakeBackend(), cfg, snapshot_dir=tmp_path / cls.__name__, **kwargs)
    bind(arm)
    return arm


@pytest.mark.parametrize('failure', ['nonfinite', 'norm'])
def test_forced_installs_when_v1_rejects_and_logs_all_checks(tmp_path, failure):
    normal = make(GenericOnlySleepArm, tmp_path)
    forced = make(GenericForcedSleepArm, tmp_path)
    for arm in (normal, forced):
        learn(arm, 'A')
        if failure == 'nonfinite':
            arm.backend.next_nonfinite = True
        else:
            arm.backend.next_broken = True
    rejected = normal.sleep(0)
    with arm_scope():
        installed = forced.sleep(0)
    assert rejected['accepted'] is False
    assert rejected['reasons'] == (['non-finite candidate'] if failure == 'nonfinite'
                                   else ['adapter delta exceeds frozen projection norm'])
    assert 'gate_bypassed' not in rejected
    assert normal.active_experts == []
    assert normal.backend.discarded == [rejected['candidate']]
    assert installed['accepted'] is True and installed['reasons'] == []
    assert forced.active_experts == forced.backend.committed == [installed['candidate']]
    assert forced.backend.discarded == []
    assert forced.backend.train_calls == normal.backend.train_calls
    checks = {entry['check']: entry for entry in installed['gate_bypassed']}
    assert checks['finite_candidate' if failure == 'nonfinite'
                  else 'adapter_delta_norm']['would_have_failed']
    assert {'finite_candidate', 'adapter_delta_norm', 'absolute_canary_output',
            'trusted_train_replay', 'route_coverage', 'statistical_gate_decision',
            'statistical_gate', 'statistical_canary'} <= checks.keys()
    assert all(set(entry) == {'check', 'would_have_failed', 'detail'}
               for entry in installed['gate_bypassed'])
    assert installed['gate_decision']['accept'] == installed['gate_would_accept']
    # Nonfinite ratio values must be logged without NaN/Infinity.
    json.dumps(installed, allow_nan=False)


def test_healthy_generic_fit_uses_generic_targets_not_world_validation(tmp_path):
    arm = make(GenericForcedSleepArm, tmp_path, SleepConfig(seed=48, steps=7))
    learn(arm, 'A')
    info = arm.sleep(0)
    actual = arm.backend.train_calls[-1]['examples']
    assert info['generic_train_accuracy'] == info['generic_train_exact_accuracy'] == 1.0
    assert info['generic_train_correct'] == info['generic_train_exact_correct'] == len(actual)
    assert info['generic_train_examples'] == info['examples'] == len(actual) == 4
    assert info['pre_gate']['validation']['candidate_correct'] == 0
    assert all('Cobalt beacon' not in q for q, _ in actual)
    assert all((q, info['candidate']) in arm.backend.generate_calls for q, _ in actual)


def test_trusted_validation_and_statistical_rejections_are_bypassed(tmp_path):
    normal = make(GenericOnlySleepArm, tmp_path)
    forced = make(GenericForcedSleepArm, tmp_path)
    subjects = ('Alder', 'Birch', 'Dover', 'Elm')
    gate = [Item(id=f'fact-{i}', prompt=f'Who leads {subject}?', expected=f'Leader{i}',
                 skill='toy', family='fact', kind='probe', key=subject,
                 input_key=subject, split='gate') for i, subject in enumerate(subjects)]
    for arm in (normal, forced):
        bind(arm, gate)
        for i, subject in enumerate(subjects):
            arm.observe(Task(f'note-{i}', f'Field note: the leader of {subject} is Leader{i}.',
                             'fact', 'toy', kind='teach'), '', None, None)
            arm.observe(Task(f'fact-{i}', f'Who leads {subject}?', 'fact', 'toy'),
                        f'Final answer: Leader{i}', True, None)
    rejected, installed = normal.sleep(0), forced.sleep(0)
    assert rejected['reasons'] == ['trusted train/replay validation regressed']
    assert rejected['accepted'] is False and installed['accepted'] is True
    checks = {entry['check']: entry for entry in installed['gate_bypassed']}
    assert checks['trusted_train_replay']['would_have_failed']
    assert checks['statistical_gate']['would_have_failed']
    assert checks['statistical_gate_decision']['would_have_failed']
    assert installed['gate_would_accept'] is False
    assert installed['gate_decision']['reasons']
    assert installed['generic_train_accuracy'] == 1.0


def test_absolute_canary_rejection_is_bypassed_without_skipping_other_checks(tmp_path):
    class CollisionEmbedder:
        def embed(self, texts):
            return np.array([[1., 0., 0.] if ('danger' in p or 'What is 10 ' in p)
                             else [0., 1., 0.] if 'cobalt' in p.lower()
                             else [0., 0., 1.] for p in texts])

    arms = [make(cls, tmp_path, embedder=CollisionEmbedder())
            for cls in (GenericOnlySleepArm, GenericForcedSleepArm)]
    for arm in arms:
        learn(arm, 'A')
        assert arm.sleep(0)['accepted']
        for i in range(8):
            arm.observe(Task(f'danger-{i}', f'danger exercise {i}', 'fact', 'toy'),
                        'Final answer: danger', True, None)
        arm.backend.next_broken = True
        arm.backend.adapter_delta_ratio = lambda h: 0.01
    rejected, installed = [arm.sleep(1) for arm in arms]
    assert rejected['reasons'] == ['absolute canary output drift']
    assert rejected['accepted'] is False and installed['accepted'] is True
    checks = {entry['check']: entry for entry in installed['gate_bypassed']}
    assert checks['absolute_canary_output']['would_have_failed']
    assert len(checks['absolute_canary_output']['detail']['changed_canaries']) == 1
    assert checks['statistical_canary']['detail'] == installed['checks']['canary']
    assert 'validation' in installed['pre_gate'] and 'coverage' in installed['pre_gate']


def test_uncovered_expert_route_is_logged_and_installed(tmp_path):
    arm = make(GenericForcedNoRetArm, tmp_path)
    # All gate prompts are canaries; their none centroid prevents routing to the
    # candidate, so the usual independent-evidence check would reject it.
    bind(arm, gate=[])
    learn(arm, 'A')
    info = arm.sleep(0)
    assert info['accepted']
    checks = {entry['check']: entry for entry in info['gate_bypassed']}
    assert checks['route_coverage']['would_have_failed']
    assert {'expert': info['candidate'], 'reason': 'no routed gate/canary tasks'} in (
        checks['route_coverage']['detail']['uncovered'])


@pytest.mark.parametrize('cls', [GenericForcedSleepArm, GenericForcedNoRetArm])
@pytest.mark.parametrize('treatment', [SleepArm, ExpertOnly])
def test_matching_selection_training_replay_and_task_routing_across_cycles(tmp_path, cls, treatment):
    cfg = SleepConfig(max_examples=5, replay_size=1, steps=7, seed=48)
    ours = make(treatment, tmp_path, cfg)
    control = make(cls, tmp_path, cfg)
    rows = [json.loads(line) for line in Path('data/generic/pool.jsonl').read_text(
        encoding='utf-8').splitlines() if line]
    for cycle, skill in enumerate(('A', 'B', 'C')):
        for arm in (ours, control):
            learn(arm, skill)
        ours_new, ours_replay = ours._training_set(cycle)
        new, replay = control._training_set(cycle)
        # Memory ids are random; task ids and pinned target versions must match.
        signature = lambda pairs: [(item.task_id, example) for item, example in pairs]
        assert signature(new) == signature(ours_new)
        assert signature(replay) == signature(ours_replay)
        selected = new + replay
        shuffled = list(rows)
        random.Random(cfg.seed + cycle).shuffle(shuffled)
        expected = [('Context:\n\n\nTask:\n' + shuffled[i % len(rows)]['prompt'],
                     'Final answer: ' + shuffled[i % len(rows)]['answer'])
                    for i in range(len(selected))]
        # Explicitly verify the task-derived route evidence, never pool prompts.
        old_centroids = {h: v.copy() for h, v in control.centroids.items()}
        expected_centroid = ours._centroid([q for _, (q, _) in ours_new + ours_replay])
        expected_none = ours._centroid([f'What is {i} + 20?' for i in range(10, 30)]
                                      + [q for _, (q, _) in ours_replay])
        a, b = ours.sleep(cycle), control.sleep(cycle)
        assert a['accepted'] and b['accepted']
        first, second = ours.backend.train_calls[-1], control.backend.train_calls[-1]
        assert second['examples'] == expected
        assert len(first['examples']) == len(second['examples']) == len(selected)
        assert first['steps'] == second['steps'] == cfg.steps
        assert a['examples'] == b['examples'] == len(first['examples'])
        assert a['steps'] == b['steps'] == cfg.steps
        assert first['seed'] == second['seed'] == cfg.seed + cycle
        assert a['replay'] == b['replay'] == (0 if cycle == 0 else 1)
        np.testing.assert_array_equal(control.centroids[b['candidate']], expected_centroid)
        np.testing.assert_array_equal(control.none_centroid, expected_none)
        np.testing.assert_array_equal(ours.centroids[a['candidate']], expected_centroid)
        for handle, centroid in old_centroids.items():
            np.testing.assert_array_equal(control.centroids[handle], centroid)
        assert a['pre_gate']['coverage'] == b['pre_gate']['coverage']
    assert len(control.active_experts) == 3


def test_noret_never_calls_retrieval_in_answers_or_sleep(tmp_path, monkeypatch):
    arm = make(GenericForcedNoRetArm, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError('no-retrieval control called retrieval')

    monkeypatch.setattr(arm, 'context', forbidden)
    monkeypatch.setattr(arm.hippo, 'recall', forbidden)
    learn(arm, 'A')
    for cycle in (0, 1):
        if cycle:
            learn(arm, 'B')
        probe = Task(f'probe-{cycle}', 'unseen question?', 'fact', 'toy')
        arm.answer(probe)  # unrouted path, where SleepArm normally retrieves
        assert arm.answer_trace['expert'] is None
        assert arm.answer_trace['retrieved'] == []
        assert arm._pending[probe.id] == []
        info = arm.sleep(cycle)
        assert info['accepted']
        arm.answer(Task('routed', PROMPTS['A' if cycle == 0 else 'B'][0], 'fact', 'toy',
                        mode='test'))
        assert arm.answer_trace['retrieved'] == []
    assert all(prompt.startswith('Context:\n\n\nTask:\n')
               for prompt, _ in arm.backend.generate_calls)
    assert GenericForcedNoRetArm._prompt is ExpertOnly._prompt


@pytest.mark.parametrize('cls', [SleepArm, ExpertOnly, GenericForcedSleepArm, GenericForcedNoRetArm])
def test_no_new_data_does_not_train_or_open_gate(tmp_path, cls):
    arm = make(cls, tmp_path)
    info = arm.sleep(0)
    assert not info['accepted'] and info['reasons'] == ['no new verified examples']
    assert info['examples'] == info['steps'] == 0
    if cls in (GenericForcedSleepArm, GenericForcedNoRetArm):
        assert info['gate_bypassed'] == []
    assert arm.backend.train_calls == []


def test_pool_wraps_exactly_like_v1_for_each_seed_and_cycle(tmp_path):
    cfg = SleepConfig(seed=50)
    control = make(GenericForcedSleepArm, tmp_path, cfg)
    normal = make(GenericOnlySleepArm, tmp_path, cfg)
    count = len(Path('data/generic/pool.jsonl').read_text(encoding='utf-8').splitlines())
    for cycle in range(3):
        examples = control._expert_examples([None] * (count + 1), cycle)
        assert examples == normal._expert_examples([None] * (count + 1), cycle)
        assert examples[0] == examples[-1]


@pytest.mark.parametrize('cls', [GenericForcedSleepArm, GenericForcedNoRetArm])
def test_forced_checkpoint_resume_and_reset(tmp_path, cls):
    arm = make(cls, tmp_path / 'original')
    learn(arm, 'A')
    assert arm.sleep(0)['accepted']
    state = json.loads(json.dumps(arm.save_state()))
    restored = make(cls, tmp_path / 'original')
    restored.load_state(state)
    assert restored.save_state() == state
    assert restored.route(PROMPTS['A'][0]) == arm.active_experts[0]
    learn(restored, 'B')
    assert restored.sleep(1)['replay'] == 4
    assert len(restored.active_experts) == 2
    restored.reset()
    assert restored.active_experts == [] and restored.centroids == {}
    assert restored.backend.experts == {} and restored._accepted_ids == set()
