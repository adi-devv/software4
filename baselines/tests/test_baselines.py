import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
from baselines import NoMemory, Rag, LongContext, TextNotes, HippoOnly, FakeLLM, HFLLM, build_prompt
from harness.tasks import Task
from harness.worlds import StreamConfig
from harness.streams import synthetic_stream
from harness.runner import run_arm
from harness.results import Results

CLASSES = [NoMemory, Rag, LongContext, TextNotes, HippoOnly]


def task(prompt, kind='probe', mode='day', id='q'):
    return Task(id, prompt, 'fact', 'fact', kind=kind, mode=mode, source='trusted')


def teach(arm, name='Alder', value='Mira'):
    arm.observe(task(f'Field note: the leader of {name} is {value}.', 'teach', id=name), '', None, None)


@pytest.mark.parametrize('thinking', [True, False])
def test_hf_chat_template_and_token_count(monkeypatch, thinking):
    class Tokenizer:
        chat_template = 'enable_thinking' if thinking else 'plain chat'

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            assert messages == [{'role': 'user', 'content': 'Who leads Alder?'}]
            assert tokenize is False and add_generation_prompt is True
            assert kwargs == ({'enable_thinking': False} if thinking else {})
            return '<user> Who leads Alder? </user> <assistant> <think> </think>'

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return text.split()

        def __call__(self, text, return_tensors, add_special_tokens=True):
            assert return_tensors == 'pt' and add_special_tokens is False
            sent.append(text)
            return Inputs(input_ids=SimpleNamespace(shape=(1, len(text.split()))))

        def decode(self, tokens, skip_special_tokens):
            assert tokens == ['answer'] and skip_special_tokens is True
            return 'Final answer: Dana'

    class Inputs(dict):
        def to(self, device):
            assert device == 'cpu'
            return self

    class Output:
        def __getitem__(self, index):
            assert index == (0, slice(8, None))
            return ['answer']

    def generate(**kwargs):
        assert kwargs['input_ids'].shape == (1, count)
        assert kwargs['max_new_tokens'] == 10 and kwargs['do_sample'] is False
        return Output()

    sent = []
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(no_grad=nullcontext))
    llm = HFLLM(device='cpu')
    llm._tokenizer, llm._model = Tokenizer(), SimpleNamespace(generate=generate)
    count = llm.count_tokens('Who leads Alder?')
    assert count == 8
    assert llm.generate('Who leads Alder?', max_new_tokens=10) == 'Final answer: Dana'
    assert sent == ['<user> Who leads Alder? </user> <assistant> <think> </think>']


def test_hf_count_tokens_loads_before_accessing_tokenizer(monkeypatch):
    llm = HFLLM()

    def load():
        llm._tokenizer = SimpleNamespace(chat_template='plain chat',
                                         apply_chat_template=lambda *args, **kwargs: 'user text assistant',
                                         encode=lambda text, **kwargs: text.split())

    monkeypatch.setattr(llm, '_load', load)
    assert llm.count_tokens('text') == 3


def test_text_notes_distillation_batches_fit_input_limit():
    class LimitedLLM(FakeLLM):
        input_limit = 128

        def generate(self, prompt, max_new_tokens=256):
            assert self.count_tokens(prompt) <= self.input_limit
            prompts.append(prompt)
            return super().generate(prompt, max_new_tokens)

    prompts = []
    arm = TextNotes(LimitedLLM(), context_tokens=16)
    for i in range(20): teach(arm, 'Place' + str(i), 'Person')
    episodes = list(arm.episodes)
    arm.sleep(0)
    assert len(prompts) > 1
    assert '\n'.join(p.split('\n', 2)[2] for p in prompts) == '\n'.join(episodes)
    assert len(arm.notes) == 20 and not arm.episodes


def test_text_notes_rejects_oversized_episode_without_losing_state():
    llm = FakeLLM()
    llm.input_limit = 16
    arm = TextNotes(llm)
    teach(arm)
    state = arm.save_state()
    with pytest.raises(ValueError, match='episode.*input_limit'):
        arm.sleep(0)
    assert arm.save_state() == state


def test_fake_llm_default_input_limit():
    assert FakeLLM().input_limit >= 32768


@pytest.mark.parametrize('cls', [Rag, TextNotes])
def test_chronological_prompt_preserves_selection_priority(cls):
    arm = cls(context_tokens=16)
    teach(arm, value='Mira')
    arm.sleep(0)
    teach(arm, value='Dana')
    arm.sleep(1)
    assert arm.answer(task('Who leads Alder?')) == 'Final answer: Dana'
    assert arm._pending['q'] == [1]


def test_build_prompt_chronology_returns_matching_indices():
    prompt, selected = build_prompt(FakeLLM(), task('Who leads Alder?'),
                                    ['The leader of Alder is Dana.', 'The leader of Alder is Mira.'],
                                    16, order=[1, 0])
    assert selected == [1, 0]
    assert prompt.index('Mira') < prompt.index('Dana')


@pytest.mark.parametrize('cls', CLASSES[1:])
def test_latest_fact_wins_in_chronological_prompt(cls):
    class RecordingLLM(FakeLLM):
        def generate(self, prompt, max_new_tokens=256):
            prompts.append(prompt)
            return super().generate(prompt, max_new_tokens)

    prompts = []
    arm = cls(RecordingLLM())
    teach(arm, value='Mira')
    arm.sleep(0)
    teach(arm, value='Dana')
    arm.sleep(1)
    assert arm.answer(task('Who leads Alder?')) == 'Final answer: Dana'
    assert prompts[-1].index('Mira') < prompts[-1].index('Dana')


@pytest.mark.parametrize('values', [('Mira', 'Dana'), ('Dana', 'Mira')])
def test_fake_llm_prefers_last_fact_position(values):
    context = '\n'.join(f'The leader of Alder is {value}.' for value in values)
    assert FakeLLM().generate('Context:\n' + context + '\n\nTask:\nWho leads Alder?') == 'Final answer: ' + values[-1]


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('budget', [0, 1, 16])
def test_all_arms_use_same_prompt_format(cls, budget):
    class RecordingLLM(FakeLLM):
        def generate(self, prompt, max_new_tokens=256):
            prompts.append(prompt)
            return super().generate(prompt, max_new_tokens)

    prompts = []
    arm = cls(RecordingLLM(), context_tokens=budget)
    q = task('Who leads Alder?')
    arm.answer(q)
    assert prompts[-1] == 'Context:\n\n\nTask:\nWho leads Alder?'
    teach(arm)
    arm.sleep(0)
    arm.answer(q)
    assert prompts[-1].startswith('Context:\n')
    assert prompts[-1].endswith('\n\nTask:\nWho leads Alder?')
    if cls is NoMemory or budget < 16:
        assert prompts[-1] == 'Context:\n\n\nTask:\nWho leads Alder?'


@pytest.mark.parametrize('cls', CLASSES)
def test_protocol_frozen_and_roundtrip(cls):
    arm = cls(FakeLLM())
    teach(arm)
    arm.sleep(0)
    q = task("Who leads Alder? End your reply with a line 'Final answer: <answer>'.", mode='test')
    state = json.loads(json.dumps(arm.save_state()))
    answer = arm.answer(q)
    if cls is not NoMemory:
        assert answer == 'Final answer: Mira'
    arm.observe(q, answer, True, 'secret')
    assert arm.save_state() == state
    other = cls(FakeLLM())
    other.load_state(state)
    assert other.answer(q) == answer
    assert isinstance(arm.sleep(1), dict)
    arm.reset()
    assert isinstance(arm.save_state(), dict)


@pytest.mark.parametrize('cls', CLASSES[1:])
def test_context_budget(cls):
    arm = cls(context_tokens=16)
    for i in range(10): teach(arm, 'Place' + str(i), 'Person')
    arm.sleep(0)
    q = task('Who leads Place9?')
    prompt, _ = build_prompt(arm.llm, q, [x[1] for x in arm.context(q)], arm.context_tokens)
    context = prompt.split('\n\nTask:\n')[0] if '\n\nTask:\n' in prompt else ''
    assert arm.llm.count_tokens(context) <= 16


def test_oldest_dropped():
    arm = LongContext(context_tokens=18)
    for i in range(4): teach(arm, 'Place' + str(i), 'Person')
    text = '\n'.join(x[1] for x in arm.context(task('query')))
    assert 'Place3' in text and 'Place0' not in text


def test_prune_notes():
    arm = TextNotes(prune_after=2)
    teach(arm)
    arm.sleep(0)
    for i in range(2):
        q = task('Who leads Alder?', id=str(i))
        arm.answer(q)
        arm.observe(q, 'wrong', False, None)
    arm.sleep(1)
    assert not arm.notes


def test_runner_all_arms(tmp_path):
    cfg = StreamConfig(n_cycles=8, entities_per_domain=8, items_per_rule=24)
    stream = synthetic_stream(3, cfg)
    path = tmp_path / 'results.jsonl'
    for cls in CLASSES:
        run_arm(cls(context_tokens=300, k=3) if cls in (Rag, HippoOnly) else cls(context_tokens=300), stream, [], path)
    results = Results.from_file(path)
    assert not results.violations
    scores = {cls.name: results.primary(cls.name, stream.stream_id, stream.n_cycles - 1) for cls in CLASSES}
    assert all(scores[cls.name] >= scores['no_memory'] for cls in CLASSES[1:])
    assert max(scores[cls.name] for cls in CLASSES[1:]) > scores['no_memory']
