import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from baselines.dream import DreamSleepArm
from baselines.generic import GenericSleepArm
from baselines.sleep import SleepConfig
from baselines.sleep_fake import FakeBackend
from harness.canaries import SyntheticCanarySet
from harness.smoke_world import smoke_canaries
from harness.step4_world import Step4Source, step4_stream
from harness.streams import synthetic_stream, _norm_prompt
from harness.worlds import StreamConfig
from harness.util import stable_hash


def test_generic_count_matches_dream_and_pool_is_disjoint():
    cfg = SleepConfig(max_examples=5, replay_size=0)
    selected = [(SimpleNamespace(), (f"Question {i}?", "Final answer: yes")) for i in range(3)]
    dream, generic = DreamSleepArm(FakeBackend(), cfg), GenericSleepArm(FakeBackend(), cfg)
    assert len(generic._extra_examples(selected, 0)) == len(dream._extra_examples(selected, 0)) == 2
    digest = lambda s: hashlib.sha256(s.strip().encode()).digest()
    pool = [json.loads(line)['prompt'] for line in Path('data/generic/pool.jsonl').read_text().splitlines()]
    _, items = Step4Source().raw()
    protected = [i['prompt'] for i in items if i.get('split') in ('hidden', 'gate', 'sealed')]
    protected += [i.prompt for i in smoke_canaries().items() + SyntheticCanarySet().items()]
    assert {digest(x) for x in pool}.isdisjoint({digest(x) for x in protected})

    for stream in (step4_stream(), synthetic_stream(0, StreamConfig())):
        sealed = {f.prompt_h for f in stream.fingerprints if f.split == 'sealed'}
        assert {stable_hash(_norm_prompt(p)) for p in pool}.isdisjoint(sealed)



def test_sleep_cycles_match_when_dream_verification_rejects_variants():
    from harness.tasks import Task

    cfg = SleepConfig(max_examples=5, replay_size=0)
    dream, generic = DreamSleepArm(FakeBackend(), cfg), GenericSleepArm(FakeBackend(), cfg)
    # Every candidate fails verification; admitted originals must fill the slots.
    dream._variant = lambda q, a, rng: (q, 'wrong')
    for arm in (dream, generic):
        arm.bind(object())
    for cycle, size in enumerate((1, 3, 4)):
        counts = []
        for arm in (dream, generic):
            for i in range(size):
                task = Task(f'{cycle}-{i}', f'Cycle {cycle} question {i}?', 'fact', 'toy')
                arm.observe(task, 'Final answer: unknown', False, 'Correct answer: yes.')
            new, replay = arm._training_set(cycle)
            selected = new + replay
            if arm is dream:
                assert arm._extra_examples(selected, cycle) == [
                    example for _, example in selected[:min(len(selected), cfg.max_examples - len(selected))]]
            # Stop after training so both arms retain the same admissions each cycle.
            arm.backend.next_nonfinite = True
            result = arm.sleep(cycle)
            counts.append(result['examples'] - len(selected))
            assert counts[-1] == min(len(selected), cfg.max_examples - len(selected))
        assert counts[0] == counts[1] > 0
