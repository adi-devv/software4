"""CPU-only v3 behavior with tiny scripted FakeBackend generations."""
from types import SimpleNamespace

import pytest

from baselines.arms import HippoOnly, episode
from baselines.sleep import ExpertOnly, SleepArm, SleepConfig
from baselines.sleep_fake import FakeBackend
from baselines.v3 import FALLBACK_TRIGGER_K, demo_answer, shape_signature
from baselines.wrong_rule import WrongRuleSleepArm, wrong_rule_mapping
from harness.tasks import Task


def task(mode='test'):
    # Labels intentionally don't identify a rule: routing/fallback must not use them.
    return Task('q', 'Apply the beacon to code abc.', 'fact', 'unrelated', mode=mode)


def hit(mid, text, kind='episode'):
    return SimpleNamespace(id=mid, text=text, kind=kind)


def demo(mid, answer):
    text = episode(task(), 'Final answer: bad', False, 'Correct answer: ' + answer + '. A lesson.')
    return hit(mid, text)


class ScriptedBackend(FakeBackend):
    def __init__(self, first='Final answer: ABC', second='Final answer: XYZ-7'):
        super().__init__()
        self.first, self.second = first, second

    def generate(self, prompt, expert=None):
        self.generate_calls.append((prompt, expert))
        return self.second if 'Verified answer:' in prompt else self.first


def make(tmp_path, cls=SleepArm, **flags):
    backend = ScriptedBackend()
    arm = cls(backend, SleepConfig(**flags), snapshot_dir=tmp_path / cls.__name__)
    return arm, backend


@pytest.mark.parametrize('cls', [SleepArm, ExpertOnly])
@pytest.mark.parametrize('top_kind', ['fact', 'episode'])
def test_fact_padding(tmp_path, monkeypatch, cls, top_kind):
    arm, _ = make(tmp_path, cls, v3_fact_filter=True, k=3)
    hits = [hit('1', 'top', top_kind), hit('2', 'demo'), hit('3', 'other fact', 'fact')]
    calls = []
    def recall(query, k, **kwargs):
        calls.append((query, k, kwargs))
        return hits[:k]
    monkeypatch.setattr(arm.hippo, 'recall', recall)
    expected = [('1', 'top'), ('3', 'other fact')] if top_kind == 'fact' else [(h.id, h.text) for h in hits]
    assert arm.context(task()) == expected
    assert calls == [(task().prompt, 3, {'now': arm.tick})]


@pytest.mark.parametrize('text,expected', [
    ('B-ABC-7', 'a-aaa-0'), ('Abz09!?', 'aaa00!?'),
    (' x_Y\t3.5 ', ' a_a\t0.0 '), ('', ''), ('éÉ', 'aa'),
])
def test_signature(text, expected):
    assert shape_signature(text) == expected


def test_demo_answer_extracts_verified_not_failed_response():
    assert demo_answer(demo('1', 'B-ABC-7').text) == 'B-ABC-7'
    assert demo_answer(episode(task(), 'Final answer: ABC-7', True, None)) == 'ABC-7'
    assert demo_answer('Question: x\nVerified answer: X-7') == 'X-7'
    assert demo_answer('Correct answer: ABC-7. A lesson.') == 'ABC-7'
    assert demo_answer('Fact: unverified statement') is None
    assert demo_answer(episode(task(), 'Final answer: bad', False, None)) is None


@pytest.mark.parametrize('first,targets,fires', [
    ('Final answer: ABC', ['ABC-7', 'DEF-7'], True),
    ('Final answer: QRS-9', ['ABC-7', 'DEF-7'], False),
    ('Final answer: abc-7', ['ABC-7', 'DEF-7'], False),
    ('Final answer: B-ABC', ['ABC-7', 'B-XYZ'], False),
    ('Final answer: XYZ\nFinal answer: ABC-7', ['ABC-7', 'DEF-7'], True),
])
def test_fallback_only_on_shape_mismatch(tmp_path, monkeypatch, first, targets, fires):
    arm, backend = make(tmp_path, v3_fallback=True, v3_fact_filter=True, k=5)
    backend.first = first
    monkeypatch.setattr(arm, 'route', lambda prompt: 'expert-0')
    calls = []
    def recall(query, k, **kwargs):
        calls.append((query, k, kwargs))
        return [demo(str(i), a) for i, a in enumerate(targets + ['UNUSED'])][:k]
    monkeypatch.setattr(arm.hippo, 'recall', recall)
    assert arm.answer(task('day')) == (backend.second if fires else first)
    assert calls == [(task().prompt, FALLBACK_TRIGGER_K, {'now': arm.tick})] + ([(task().prompt, 5, {'now': arm.tick})] if fires else [])
    assert len(backend.generate_calls) == (2 if fires else 1)
    assert arm.logged_stats() == {'fallback_fired': int(fires)}
    if fires:
        prompt, expert = backend.generate_calls[-1]
        assert expert is None
        hippo = HippoOnly(k=arm.k, context_tokens=arm.context_tokens, llm=arm.llm)
        hippo.hippo, hippo.tick = arm.hippo, arm.tick
        hippo.answer(task('day'))
        assert prompt == hippo.answer_trace['prompt']
        assert prompt.count('Verified answer:') == 3
        assert 'UNUSED' in prompt and task().prompt in prompt
        assert arm.answer_trace['prompts'] == [p for p, _ in backend.generate_calls]
        assert arm.answer_trace['prompt'] == prompt
        assert arm._pending['q'] == arm.answer_trace['retrieved'] == ['0', '1', '2']


@pytest.mark.parametrize('hits', [[], [hit('f', 'Fact: x', 'fact')], [demo('1', 'ABC-7')]])
def test_fallback_empty_partial_and_raw_v2_recall(tmp_path, monkeypatch, hits):
    arm, backend = make(tmp_path, v3_fallback=True, v3_fact_filter=True)
    monkeypatch.setattr(arm, 'route', lambda prompt: 'expert-0')
    monkeypatch.setattr(arm.hippo, 'recall', lambda *a, **kw: hits)
    # Fallback never asks context(), even with the fact filter enabled.
    monkeypatch.setattr(arm, 'context', lambda task: pytest.fail('must use raw v2 recall'))
    arm.answer(task())
    usable = sum(demo_answer(h.text) is not None for h in hits)
    assert arm.fallback_fired == int(bool(usable))
    assert len(backend.generate_calls) == 1 + int(bool(usable))
    if usable:
        assert backend.generate_calls[-1][0].count('Verified answer:') == usable


def test_no_expert_does_not_fallback(tmp_path, monkeypatch):
    arm, backend = make(tmp_path, v3_fallback=True)
    calls = []
    monkeypatch.setattr(arm.hippo, 'recall', lambda q, k, **kw: calls.append(k) or [demo('1', 'ABC-7')])
    arm.answer(task())
    assert calls == [5] and len(backend.generate_calls) == 1 and arm.fallback_fired == 0


@pytest.mark.parametrize('cls', [SleepArm, ExpertOnly])
@pytest.mark.parametrize('expert', [None, 'expert-0'])
def test_flags_off_v2_prompt_and_answer_bytes(tmp_path, monkeypatch, cls, expert):
    arm, backend = make(tmp_path, cls, v3_fact_filter=False, v3_fallback=False)
    monkeypatch.setattr(arm, 'route', lambda prompt: expert)
    hits = [hit('0', 'Fact: Field note: the leader of Alder is Mira.', 'fact'), demo('1', 'ABC-7')]
    monkeypatch.setattr(arm.hippo, 'recall', lambda *a, **kw: hits)
    assert arm.context(task()) == HippoOnly.context(arm, task())
    context = ('Fact: Field note: the leader of Alder is Mira.\nQuestion: ' + task().prompt +
               '\nVerified answer: ABC-7') if cls is SleepArm and expert is None else ''
    prompt = 'Context:\n' + context + '\n\nTask:\n' + task().prompt
    expected = backend.second if context else backend.first
    assert arm.answer(task()).encode() == expected.encode()
    assert backend.generate_calls == [(prompt, expert)]
    assert arm.answer_trace == dict(expert=expert, retrieved=['0', '1'] if context else [], prompt=prompt)
    state = arm.save_state()
    assert not any(k.startswith('v3_') for k in state['config'])
    assert 'v3_stats' not in state and arm.logged_stats() == {}


def test_fallback_counter_checkpoint_and_reset(tmp_path, monkeypatch):
    arm, _ = make(tmp_path, v3_fallback=True)
    monkeypatch.setattr(arm, 'route', lambda prompt: 'expert-0')
    monkeypatch.setattr(arm.hippo, 'recall', lambda *a, **kw: [demo('1', 'ABC-7')])
    arm.answer(task())
    state = arm.save_state()
    arm.reset()
    assert arm.fallback_fired == 0
    arm.load_state(state)
    assert arm.cfg.v3_fallback and arm.fallback_fired == 1


def test_wrong_rule_mapping_deterministic_wrong_and_same_shape():
    answers = ['ABC-7', 'DEF-7', 'GHI-7', 'B-ABC', 'B-DEF', 'B-JKL-7', '123', 'abc-7', 'z', 'Z']
    mapping = wrong_rule_mapping(answers)
    assert mapping == wrong_rule_mapping(reversed(answers)) == wrong_rule_mapping(answers * 2)
    assert set(mapping) == set(answers)
    assert all(target.casefold() != wrong.casefold() and shape_signature(target) == shape_signature(wrong)
               for target, wrong in mapping.items())
    assert set(mapping[a] for a in answers[:3]) == set(answers[:3])


def test_wrong_rule_training_examples_use_true_prompts(tmp_path):
    backend = FakeBackend()
    with pytest.raises(ValueError, match='v3_wrong_rule'):
        WrongRuleSleepArm(backend, snapshot_dir=tmp_path)
    arm = WrongRuleSleepArm(backend, SleepConfig(v3_wrong_rule=True), snapshot_dir=tmp_path)
    for i, (family, value) in enumerate([('rule', 'ABC-7'), ('rule', 'DEF-7'), ('fact', 'Mira')]):
        t = Task(str(i), 'Question ' + str(i), family, family + ':test', source='trusted')
        arm.observe(t, 'Final answer: unknown', False, 'Correct answer: ' + value + '.')
    new, replay = arm._training_set(0)
    selected = new + replay
    assert len(selected) == 3
    assert sum(arm.hippo.get(item.episode_id)['meta']['family'] == 'rule'
               for item, _ in selected) == 2
    original = [example for _, example in selected]
    wrong = arm._expert_examples(selected, 0)
    assert wrong == arm._expert_examples(selected, 99)
    assert [q for q, _ in wrong] == [q for q, _ in original]
    for (item, (_, true)), (_, shuffled) in zip(selected, wrong):
        assert shape_signature(true) == shape_signature(shuffled)
        assert (true != shuffled) == (arm.hippo.get(item.episode_id)['meta']['family'] == 'rule')
    assert arm._prompt(task()) == ExpertOnly._prompt(arm, task())
    assert backend.train_calls == []  # inspect only; never launch the control


@pytest.mark.parametrize('cls', [ExpertOnly, WrongRuleSleepArm])
def test_control_never_falls_back_even_with_direct_flag(tmp_path, monkeypatch, cls):
    arm, backend = make(tmp_path, cls, v3_fallback=True, v3_wrong_rule=True)
    monkeypatch.setattr(arm, 'route', lambda prompt: 'expert')
    monkeypatch.setattr(arm.hippo, 'recall', lambda *a, **kw: pytest.fail('unexpected recall'))
    arm.answer(task('day'))
    assert len(backend.generate_calls) == 1
    assert arm.fallback_fired == 0 and 'prompts' not in arm.answer_trace


def test_retry_uses_unfiltered_hippo_context_and_budget(tmp_path, monkeypatch):
    arm, backend = make(tmp_path, v3_fallback=True, v3_fact_filter=True, k=5, context_tokens=30)
    monkeypatch.setattr(arm, 'route', lambda prompt: 'expert')
    hits = [hit('0', 'Fact: A short fact.', 'fact'), demo('1', 'ABC-7'),
            demo('2', 'DEF-7'), hit('3', 'Fact: A later fact.', 'fact'), demo('4', 'GHI-7')]
    monkeypatch.setattr(arm.hippo, 'recall', lambda q, k, **kw: hits[:k])
    arm.answer(task('day'))
    hippo = HippoOnly(llm=arm.llm, k=arm.k, context_tokens=arm.context_tokens)
    hippo.hippo, hippo.tick = arm.hippo, arm.tick
    hippo.answer(task('day'))
    assert backend.generate_calls[-1] == (hippo.answer_trace['prompt'], None)
    assert 'Verified answer:' in hippo.answer_trace['prompt']
    assert arm._pending['q'] == hippo._pending['q'] == arm.answer_trace['retrieved']
