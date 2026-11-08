from types import SimpleNamespace

from baselines.oracle_trained import OracleTrainedSleepArm
from baselines.sleep import SleepArm, SleepConfig
from baselines.sleep_fake import FakeBackend
from harness.smoke_world import rule_prompt


def test_oracle_replaces_recorded_targets_with_gold(monkeypatch):
    item = SimpleNamespace()
    pairs = [(item, (rule_prompt('A', 'abcd'), 'Final answer: wrong')),
             (item, (rule_prompt('B', 'efgh'), 'Final answer: wrong')),
             (item, ('Unrelated?', 'Final answer: wrong'))]
    monkeypatch.setattr(SleepArm, '_training_set', lambda self, cycle: (pairs, []))
    arm = OracleTrainedSleepArm(FakeBackend(), SleepConfig())
    new, replay = arm._training_set(0)
    assert [example[1][1] for example in new] == ['Final answer: DCBA', 'Final answer: EFGH']
    assert not replay
