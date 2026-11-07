import json

import numpy as np
import pytest

from scripts.smoke_sleep import same_router, summarize_scores


def test_retention_table_excludes_day_scores():
    rows = [dict(type='score', arm='ours', cycle=0, phase='day', skill='rule:A', correct=0)
            for _ in range(24)]
    rows += [dict(type='score', arm='ours', cycle=0, phase='hidden', skill='rule:A', correct=1)
             for _ in range(8)]
    rows += [dict(type='score', arm='ours', cycle=0, phase='canary', skill='canary:smoke', correct=1)]
    result = summarize_scores(rows)['ours']['0']
    assert result['rule:A'] == 1.0 and result['canary:smoke'] == 1.0
    assert result['rule:B'] is None


@pytest.mark.parametrize('none,after_none,expected', [
    (None, None, True), (np.array([1., 0.]), np.array([1., 0.]), True),
    (np.array([1., 0.]), np.array([0., 1.]), False), (None, np.array([1., 0.]), False)])
def test_successful_router_rollback_evidence_serializes(none, after_none, expected):
    before = {'expert-0': np.array([1., 0.])}
    result = same_router(before, none, {'expert-0': np.array([1., 0.])}, after_none)
    assert result is expected
    assert json.loads(json.dumps({'router_unchanged': result}))['router_unchanged'] is expected
    assert not same_router(before, none, {}, after_none)


def test_recover_cli_from_copied_archive_without_original_snapshots(tmp_path, monkeypatch):
    import shutil
    import sys
    from types import SimpleNamespace
    from pathlib import Path
    from baselines import FakeBackend
    from baselines.llm import FakeLLM
    from harness.smoke_world import smoke_canaries
    from harness.tasks import grading_scope
    from scripts.smoke_sleep import main

    with grading_scope():
        canary_answers = {it.prompt: it.expected_at(0) for it in smoke_canaries().items()}

    instances = []

    class SmokeBackend(FakeBackend):
        input_limit = 1_000_000
        count_tokens = FakeLLM.count_tokens

        def __init__(self, *, directory, lr):
            super().__init__()
            self.directory, self.lr, self._cache = directory, lr, {}
            instances.append(self)

        def generate(self, prompt, expert=None):
            question = self._question(prompt)
            if question in canary_answers and (expert is None or not self.experts[expert]['broken']):
                return 'Final answer: ' + canary_answers[question]
            return super().generate(prompt, expert)

        def _state(self, handle):
            return {}

        def train_expert(self, examples, steps, seed):
            handle = super().train_expert(examples, steps, seed)
            metadata = dict(loss=0.0, skipped_steps=0, optimizer_steps=steps)
            self.experts[handle].update(metadata)
            self.train_calls[-1].update(metadata, finite=True, nonfinite_gradient_steps=0)
            return handle

    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0,
        get_device_name=lambda: 'test-double')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda, __version__='test',
        set_num_threads=lambda n: None, manual_seed=lambda n: None, equal=lambda a, b: a == b))
    monkeypatch.setitem(sys.modules, 'baselines.sleep_hf', SimpleNamespace(HFBackend=SmokeBackend))
    monkeypatch.delenv('S4_SMOKE', raising=False)
    original, copied = tmp_path / 'run', tmp_path / 'copied'
    monkeypatch.setattr(sys, 'argv', ['smoke_sleep', '--output', str(original)])
    main()
    expected = json.loads((original / 'CHECKLIST.json').read_text())
    states = [json.loads(p.read_text()) for p in (original / 'checkpoints').glob('ours*.json')]
    assert all(not Path(s['backend_snapshot']).is_absolute() for s in states)
    shutil.copytree(original, copied)
    shutil.rmtree(original)
    monkeypatch.setattr(sys, 'argv', ['smoke_sleep', '--output', str(copied), '--recover'])
    main()
    recovered = json.loads((copied / 'CHECKLIST.json').read_text())
    assert recovered['verification_recovered']
    assert recovered['per_arm'] == expected['per_arm']
    assert {k: v['status'] for k, v in recovered['checks'].items()} == {
        k: v['status'] for k, v in expected['checks'].items()}
    assert sum(not c.get('recovered_from_checkpoint') for c in instances[-1].train_calls) == 1


def test_copied_run_has_relative_self_contained_backend_snapshots(tmp_path):
    import shutil
    from pathlib import Path
    from baselines import FakeBackend, SleepArm
    original = tmp_path / 'original'
    arm = SleepArm(FakeBackend(), snapshot_dir=original / 'backend_snapshots')
    state = arm.save_state()
    assert not Path(state['backend_snapshot']).is_absolute()
    (original / 'checkpoint.json').write_text(json.dumps(state))
    archive = tmp_path / 'copied_archive'
    shutil.copytree(original, archive)
    shutil.rmtree(original)
    restored = SleepArm(FakeBackend(), snapshot_dir=archive / 'backend_snapshots')
    restored.load_state(json.loads((archive / 'checkpoint.json').read_text()))
    assert restored.save_state() == state
