from types import SimpleNamespace

from baselines.dream import DreamSleepArm
from baselines.sleep import SleepConfig
from baselines.sleep_fake import FakeBackend


def test_dream_variants_are_deterministic_verified_and_capped():
    arm = DreamSleepArm(FakeBackend(), SleepConfig(max_examples=3, replay_size=0))
    selected = [(SimpleNamespace(), ("What is 1 + 1?", "Final answer: 2")),
                (SimpleNamespace(), ("Another?", "Final answer: yes"))]
    first = arm._extra_examples(selected, 2)
    assert first == arm._extra_examples(selected, 2)
    assert len(first) == 1 and first[0][1] == selected[0][1][1]
    arm._variant = lambda q, a, rng: (q, "Final answer: wrong")
    assert arm._extra_examples(selected, 2) == [selected[0][1]]


def test_dream_selection_uses_only_parent_admitted(monkeypatch):
    from baselines.sleep import SleepArm
    trusted = (SimpleNamespace(), ("Trusted?", "Final answer: yes"))
    monkeypatch.setattr(SleepArm, '_training_set', lambda self, cycle: ([trusted], []))
    arm = DreamSleepArm(FakeBackend(), SleepConfig())
    selected, replay = arm._training_set(0)
    assert len(arm._extra_examples(selected + replay, 0)) == 1



def test_untrusted_and_unverified_episodes_never_enter_dreams():
    from harness.tasks import Task
    arm = DreamSleepArm(FakeBackend(), SleepConfig())
    for i, (source, correct, correction) in enumerate([
        ('trusted', False, 'Correct answer: safe.'),
        ('untrusted', False, 'Correct answer: poison.'),
        ('trusted', None, None),
    ]):
        task = Task(str(i), f'Question {i}?', 'fact', 'toy', source=source)
        arm.observe(task, 'Final answer: unknown', correct, correction)
    new, replay = arm._training_set(0)
    examples = arm._extra_examples(new + replay, 0)
    assert len(examples) == 1
    assert 'Question 0' in examples[0][0] or 'question 0' in examples[0][0]
    assert all('poison' not in answer for _, answer in examples)
