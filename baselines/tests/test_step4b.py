from baselines import FakeBackend, HippoOnly, Rag, SleepArm
from baselines.arms import render_entry
from harness.tasks import Task


def task(mode='day'):
    return Task('q', 'What is the answer?', 'fact', 'toy', mode=mode, kind='probe', source='trusted')


def test_render_entries():
    wrong = 'Question: Q\nResponse: WRONG\nCorrect: False\nCorrection: Correct answer: yes. lesson'
    assert render_entry(wrong) == 'Question: Q\nVerified answer: yes'
    assert render_entry(wrong.replace('False', 'None').replace('Correct answer: yes. lesson', 'Final answer: yes')) == 'Question: Q\nVerified answer: yes'
    assert render_entry('Question: Q\nResponse: reasoning\nFinal answer: yes\nCorrect: True\nCorrection: ') == 'Question: Q\nVerified answer: yes'
    assert render_entry('Fact: hello') == 'Fact: hello'
    assert render_entry('unparseable') == 'unparseable'


def test_memory_arms_render_and_trace():
    for cls in (HippoOnly, Rag):
        arm = cls()
        arm.observe(task(), 'WRONG', False, 'Correct answer: yes. lesson')
        arm.answer(task('test'))
        trace = arm.answer_trace
        assert 'WRONG' not in trace['prompt']
        assert 'Verified answer: yes' in trace['prompt']
        assert trace['expert'] is None and trace['retrieved']


def test_sleep_prompt_follows_route_and_trace():
    arm = SleepArm(FakeBackend())
    arm.observe(task(), 'WRONG', False, 'Correct answer: yes. lesson')
    arm.route = lambda prompt: None
    arm.answer(task())
    assert 'Verified answer: yes' in arm.answer_trace['prompt']
    assert arm.answer_trace['retrieved'] == arm._pending['q'] and arm._pending['q']
    arm.route = lambda prompt: 'expert'
    arm.backend.generate = lambda prompt, expert: 'response'
    arm.answer(task())
    assert arm.answer_trace == {'expert': 'expert', 'retrieved': [], 'prompt': 'Context:\n\n\nTask:\nWhat is the answer?'}
    assert arm._pending['q'] == []
