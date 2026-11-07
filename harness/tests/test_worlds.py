import json

from harness.streams import build_stream, synthetic_stream
from harness.worlds import JsonlSource, StreamConfig, SyntheticSource, dump_jsonl, generate_world


def test_same_seed_identical_world_and_stream():
    cfg = StreamConfig()
    assert generate_world(5, cfg).canonical() == generate_world(5, cfg).canonical()
    assert synthetic_stream(5).digest() == synthetic_stream(5).digest()


def test_different_seeds_differ():
    cfg = StreamConfig()
    assert generate_world(5, cfg).canonical() != generate_world(6, cfg).canonical()
    assert synthetic_stream(5).digest() != synthetic_stream(6).digest()


def test_stream_shape(stream):
    assert stream.n_cycles >= 15
    fams = {m["family"] for m in stream.skills.values()}
    assert len(fams) >= 4
    assert len(stream.active_skills(stream.n_cycles - 1)) == len(stream.skills)
    assert len(stream.hidden_items(stream.n_cycles - 1)) > 100


def test_tasks_carry_fields_but_not_answers(stream):
    it = stream.hidden_items(stream.n_cycles - 1)[0]
    t = it.task("test", 3)
    assert t.id and t.prompt and t.family and t.skill
    assert "Final answer" in t.prompt
    assert not hasattr(t, "expected")


def test_jsonl_roundtrip(tmp_path):
    src = SyntheticSource(3, StreamConfig())
    p = tmp_path / "tasks.jsonl"
    dump_jsonl(src, p)
    for line in p.read_text(encoding="utf-8").splitlines():
        json.loads(line)
    a = build_stream(src, seed=3, n_cycles=16)
    b = build_stream(JsonlSource(p), seed=3, n_cycles=16)
    assert a.digest() == b.digest()
