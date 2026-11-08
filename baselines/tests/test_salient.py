from collections import Counter

from baselines import FakeBackend, SleepConfig
from baselines.salient import SalientSleepArm
from harness.tasks import Task


def add(arm, skill, number, *, success=False, source='trusted'):
    task = Task(f'{skill}-{number}', f'Question {skill} {number}?', 'fact', skill,
                mode='day', kind='probe', source=source)
    arm.observe(task, 'Final answer: yes' if success else 'Final answer: no',
                success, None if success else 'Correct answer: yes.')


def selected(arm):
    new, replay = arm._training_set(0)
    return new + replay


def test_weighting_caps_and_determinism():
    arm = SalientSleepArm(FakeBackend(), SleepConfig(max_examples=20, replay_size=0))
    for skill in ('A', 'B'):
        add(arm, skill, 0)
        add(arm, skill, 1, success=True)
    first = selected(arm)
    counts = Counter(item.task_id for item, _ in first)
    assert counts == {'A-0': 2, 'A-1': 1, 'B-0': 2, 'B-1': 1}
    assert max(counts.values()) <= 3
    assert first == selected(arm)
    assert len(first) <= arm.cfg.max_examples


def test_skill_cap_and_admission():
    arm = SalientSleepArm(FakeBackend(), SleepConfig(max_examples=8, replay_size=0))
    for i in range(5):
        add(arm, 'A', i)
    add(arm, 'B', 0, success=True)
    add(arm, 'B', 1, source='untrusted')
    # A recorded but unverified outcome cannot enter the parent selection.
    episode = arm.hippo.add('Question: Unverified?\nResponse: Final answer: yes\nCorrect: True',
                            kind='episode', source='system', meta={'skill': 'B'})
    arm.hippo.record_outcome('unverified', success=True, verified=False, episode_id=episode)
    result = selected(arm)
    counts = Counter(arm.hippo.get(item.episode_id)['meta']['skill'] for item, _ in result)
    assert result and max(counts.values()) * 2 <= len(result)
    assert all(item.task_id not in ('B-1', 'unverified') for item, _ in result)
    assert len(result) <= arm.cfg.max_examples
    assert max(Counter(item.episode_id for item, _ in result).values()) <= 3


def test_failure_weighting_survives_full_budget():
    arm = SalientSleepArm(FakeBackend(), SleepConfig(max_examples=6, replay_size=0))
    for skill in ('A', 'B'):
        for i in range(3):
            add(arm, skill, i, success=True)
        add(arm, skill, 9)
    parent_new, parent_replay = super(SalientSleepArm, arm)._training_set(0)
    assert len(parent_new) + len(parent_replay) == arm.cfg.max_examples
    counts = Counter(item.task_id for item, _ in selected(arm))
    assert counts['A-9'] == 2 and counts['B-9'] == 2
    assert sum(counts.values()) <= arm.cfg.max_examples


def test_single_skill_keeps_examples():
    arm = SalientSleepArm(FakeBackend(), SleepConfig(max_examples=8, replay_size=0))
    for i in range(3):
        add(arm, 'A', i)
    result = selected(arm)
    assert {item.task_id for item, _ in result} == {'A-0', 'A-1', 'A-2'}
    assert len(result) <= arm.cfg.max_examples


def test_single_skill_trims_to_budget():
    arm = SalientSleepArm(FakeBackend(), SleepConfig(max_examples=4, replay_size=0))
    for i in range(6):
        add(arm, 'A', i)
    result = selected(arm)
    assert len(result) == arm.cfg.max_examples
    assert first_pass_deterministic(arm, result)


def first_pass_deterministic(arm, result):
    return result == selected(arm)


def test_default_arms_exclude_salient():
    from step4 import run
    assert 'ours_salient' not in run.ARMS and 'ours_salient' in run.AVAILABLE_ARMS
