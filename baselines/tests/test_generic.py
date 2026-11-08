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

