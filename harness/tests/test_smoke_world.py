from collections import Counter

import pytest

from harness.smoke_world import smoke_stream, smoke_canaries, rule_answer, gold_context
from harness.tasks import grading_scope


@pytest.mark.parametrize('seed', [0, 7, 42])
def test_determinism_sizes_and_disjoint_splits(seed):
    s = smoke_stream(seed)
    assert s.digest() == smoke_stream(seed).digest()
    assert s.digest() != smoke_stream(seed + 1).digest()
    assert s.audit() == []
    for cycle, rule in [(0, 'A'), (1, 'B')]:
        train = [i for i in s.train_items(cycle) if i.family == 'rule']
        assert len(train) == 24
        assert all(i.skill == f'rule:{rule}' for i in train)
    assert len(s.train_items(2)) == 8
    assert Counter(i.skill for i in s.hidden_items(2)) == {
        'rule:A': 8, 'rule:B': 8, 'fact:smoke': 6}
    for cycle in range(3):
        assert Counter(i.skill for i in s.gate_items(cycle)) == (
            {'rule:A': 4} if cycle == 0 else {'rule:A': 4, 'rule:B': 4})
    train = {i.prompt for c in range(3) for i in s.train_items(c)}
    held = {i.prompt for i in s.hidden_items(2)}
    gate = [i.prompt for c in range(3) for i in s.gate_items(c)]
    assert not train & held and not train & set(gate) and not held & set(gate)
    assert len(gate) == len(set(gate))


def test_answers_format_revision_poison_and_oracle():
    s = smoke_stream()
    with grading_scope():
        for cycle in range(3):
            for it in s.train_items(cycle) + s.hidden_items(cycle) + s.gate_items(cycle):
                if it.kind == 'probe':
                    assert "Final answer:" in it.prompt
                    if it.family == 'rule':
                        code = it.input_key
                        assert it.expected_at(cycle) == rule_answer(it.skill[-1], code)
                        assert gold_context(it.task('test', cycle))
        alder = next(i for i in s.hidden_items(2) if i.input_key == 'Alder|leader')
        assert alder.expected_at(0) == 'Mira' and alder.expected_at(1) == 'Sora'
    poison = [i for i in s.train_items(1) if i.source == 'untrusted']
    assert len(poison) == 1 and 'Zorg' in poison[0].prompt
    assert len(smoke_canaries().items()) == 8
    assert all('Final answer:' in i.prompt for i in smoke_canaries().items())
