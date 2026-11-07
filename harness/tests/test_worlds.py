import json

import pytest

from harness.grading import grade
from harness.streams import build_stream, synthetic_stream
from harness.tasks import grading_scope
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


def test_hidden_hop_uses_revised_ally_founding_year():
    st = synthetic_stream(1)
    item = next(it for it in st.hidden_items(8) if it.key == "hop|village|Zisex|founded")
    with grading_scope():
        assert item.expected_at(7) == "1618"
        assert item.expected_at(8) == "1282"
        assert grade(item, "Final answer: 1282", 8).correct
        old = grade(item, "Final answer: 1618", 8)
        assert not old.correct and old.stale
    assert st.sealed.read_count == 0


@pytest.mark.parametrize("seed", [1, 7])
def test_hop_answers_follow_trusted_revisions(seed):
    src = SyntheticSource(seed)
    skills, items = src.raw()
    intros = {s["name"]: s["intro_cycle"] for s in skills}
    updates = sorted((it for it in items if it["key"].startswith("update|")),
                     key=lambda it: it["cycle"])
    for hop in (it for it in items if it["skill"].startswith("hop:")):
        _, dom, entity, leaf = hop["key"].split("|")
        table = {e: dict(facts) for e, facts in src.world.domains[dom].items()}
        for cycle in range(src.cfg.n_cycles):
            for update in updates:
                if update["cycle"] != cycle:
                    continue
                _, e, attr, _ = update["key"].split("|")
                if e in table:
                    value = update["prompt"].split(" is now ")[1].removesuffix(".")
                    table[e][attr] = value.replace(",", "")
            if cycle < intros[hop["skill"]]:
                continue
            actual = hop["expected"]
            for start, value in hop.get("history", []):
                if start <= cycle:
                    actual = value
            assert actual == table[table[entity]["ally"]][leaf], (hop["key"], cycle)
