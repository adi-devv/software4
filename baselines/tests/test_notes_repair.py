from baselines import TextNotes, FakeLLM
from harness.tasks import Task

class Junk(FakeLLM):
    def generate(self, prompt, max_new_tokens=256):
        return 'Final answer: WHITE'

def test_junk_fallback_and_similarity():
    arm = TextNotes(Junk())
    for i, (q, a) in enumerate([('What is the color of snow?', 'WHITE'), ('Who leads Alder?', 'SORA')]):
        task = Task(str(i), q, 'fact', 'fact')
        arm.observe(task, 'wrong', False, 'Correct answer: ' + a + '.')
    arm.sleep(0)
    assert len(arm.notes) == 2
    assert all(n['text'].startswith('Example: ') for n in arm.notes)
    assert all(not n['text'].startswith('Final answer') for n in arm.notes)
    assert 'snow' in arm.retrieve('What is the color of snow?')[0][1]
    assert 'Alder' in arm.retrieve('Who leads Alder?')[0][1]

def test_no_lesson_attribute_access():
    class TaskWithoutLesson:
        mode = 'day'; kind = 'probe'; prompt = 'What is the color of snow?'; id = 'x'
        @property
        def lesson(self): raise AssertionError('lesson read')
    arm = TextNotes(Junk())
    arm.observe(TaskWithoutLesson(), '', False, 'Correct answer: WHITE.')
    arm.sleep(0)
    assert arm.notes[0]['text'].endswith('-> WHITE')
