"""Behaviour tests for the hippocampus. Time is always driven by a ManualClock."""
from __future__ import annotations

import json
import random
import statistics
import time

import pytest

from hippocampus import Config, Hippocampus, ManualClock
from hippocampus.eventlog import EventLog

DAY = 86400.0


def make(path=None, **cfg):
    clock = ManualClock(1000.0)
    return Hippocampus(path, config=Config(**cfg), clock=clock), clock


def ids(hits):
    return [h.id for h in hits]


def scenario(hc, clock):
    """A little of everything: versions, tombstone, quarantine, trust, outcomes, naps."""
    a = hc.add("Mireva Tonks founded Quillon Labs in 2011.", id="a", entities=["mireva tonks", "quillon labs"])
    b = hc.add("Quillon Labs operates from the city of Pelmont.", id="b", entities=["quillon labs", "pelmont"])
    c = hc.add("Zebra apples are discussed in the Alpha topic.", id="c", kind="lesson")
    d = hc.add("Kiwi mango smoothie recipe for Beta customers.", id="d", kind="episode")
    e = hc.add("Wrong claim: Pelmont is on the moon.", id="e")
    f = hc.add("Untrusted web claim about Quillon Labs.", id="f", source="web")
    clock.advance(100)
    hc.update("c", "Zebra apples are discussed in the Alpha topic, revised.")
    hc.tombstone("e", reason="revoked")
    hc.release("f", trust=0.4)
    hc.set_trust("d", 0.9)
    clock.advance(100)
    hc.record_outcome("t1", ["a", "b"], success=True, verified=True)
    hc.record_outcome("t2", ["c", "d"], success=False, verified=True, correction="use kiwi")
    hc.record_outcome("t3", ["a", "c"], success=True, verified=False)
    clock.advance(100)
    hc.nap()
    hc.record_outcome("t4", ["a", "c", "d"], success=True, verified=True, episode_id="d")
    clock.advance(DAY)
    hc.nap()
    hc.mark_sleep()
    return hc


QUERIES = ["Where does Mireva Tonks work", "zebra apples", "kiwi mango", "Quillon Labs Pelmont", "moon"]


def snapshot(hc, now):
    out = {}
    for q in QUERIES:
        for deep in (True, None, False):
            out[(q, deep)] = [(h.id, h.version, h.text, h.score, h.via) for h in hc.recall(q, deep=deep, now=now)]
    return out


# ------------------------------------------------------------------ log / rebuild
def test_rebuild_gives_identical_recall_and_link_weights(tmp_path):
    path = tmp_path / "log.jsonl"
    hc, clock = make(path)
    scenario(hc, clock)
    now = clock.t + 5 * DAY
    before, links, raw = snapshot(hc, now), hc.link_weights(now), hc.raw_state()
    assert links and any(v > 0 for v in links.values())
    assert hc.rebuild() is hc
    assert snapshot(hc, now) == before
    assert hc.link_weights(now) == links and hc.raw_state() == raw
    fresh = Hippocampus(path, config=Config(), clock=ManualClock(0))   # a second process reading the same log
    assert snapshot(fresh, now) == before
    assert fresh.link_weights(now) == links and fresh.raw_state() == raw


def test_log_is_append_only_and_keeps_everything(tmp_path):
    path = tmp_path / "log.jsonl"
    hc, clock = make(path)
    hc.add("secret poison text", id="p")
    snap1 = path.read_bytes()
    hc.update("p", "clean text")
    hc.tombstone("p")
    snap2 = path.read_bytes()
    assert snap2.startswith(snap1)                          # earlier bytes are never touched
    types = [json.loads(l)["type"] for l in snap2.decode("utf-8").splitlines()]
    assert types == ["add", "update", "tombstone"]
    assert "secret poison text" in snap2.decode("utf-8")    # tombstoned content stays in the log


@pytest.mark.parametrize("disk_backed", [False, True])
@pytest.mark.parametrize("mutate_result", [False, True])
def test_eventlog_append_detaches_nested_data(tmp_path, disk_backed, mutate_result):
    log = EventLog(tmp_path / "log.jsonl" if disk_backed else None)
    event = {"type": "add", "meta": {"tags": []}}
    result = log.append(event)
    (result if mutate_result else event)["meta"]["tags"].append("changed")
    assert log.events[0]["meta"] == {"tags": []}
    log.reload()
    assert log.events[0]["meta"] == {"tags": []}


@pytest.mark.parametrize("disk_backed", [False, True])
def test_add_detaches_nested_metadata(tmp_path, disk_backed):
    hc, _ = make(tmp_path / "log.jsonl" if disk_backed else None)
    meta = {"tags": []}
    mid = hc.add("original memory", meta=meta)
    meta["tags"].append("changed")
    assert hc.get(mid)["meta"] == {"tags": []}
    assert hc.log.events[0]["meta"] == {"tags": []}
    hc.rebuild()
    assert hc.get(mid)["meta"] == {"tags": []}


@pytest.mark.parametrize("disk_backed", [False, True])
def test_get_detaches_nested_metadata_before_and_after_rebuild(tmp_path, disk_backed):
    hc, _ = make(tmp_path / "log.jsonl" if disk_backed else None)
    mid = hc.add("original memory", meta={"tags": []})
    for _ in range(2):
        hc.get(mid)["meta"]["tags"].append("changed")
        assert hc.get(mid)["meta"] == {"tags": []}
        assert hc.log.events[0]["meta"] == {"tags": []}
        hc.rebuild()
        assert hc.get(mid)["meta"] == {"tags": []}


def test_crash_truncated_log(tmp_path):
    path = tmp_path / "log.jsonl"
    hc, clock = make(path)
    for i in range(5):
        hc.add(f"memory number {i} about topic{i}", id=f"m{i}")
    hc.add("the sixth memory that gets cut off mid write", id="m5")
    path.write_bytes(path.read_bytes()[:-17])               # crash mid-line
    hc2 = Hippocampus(path, config=Config(), clock=ManualClock(2000))
    assert len(hc2) == 5 and hc2.log.corrupt_lines == 1
    assert ids(hc2.recall("memory number 3 topic3", deep=True))[0] == "m3"
    hc2.add("a new memory written after the crash", id="new")   # must not glue onto the fragment
    hc3 = Hippocampus(path, config=Config(), clock=ManualClock(2000))
    assert len(hc3) == 6 and "new" in hc3._mems and hc3.log.corrupt_lines == 1
    # a truncation that also cuts a multi-byte character
    hc3.add("café ☕ on the menu", id="u")
    path.write_bytes(path.read_bytes()[:-9])
    assert len(Hippocampus(path, config=Config(), clock=ManualClock(0))) == 6


def test_determinism_across_two_runs(tmp_path):
    runs = []
    for name in ("one", "two"):
        hc, clock = make(tmp_path / f"{name}.jsonl")
        scenario(hc, clock)
        for i in range(6):
            hc.record_outcome(f"x{i}", ["a", "c"], success=True, verified=True, episode_id="c", t=clock.t + i)
        hc.mark_sleep(clock.t + 10)
        hc.record_outcome("late", ["b"], success=True, verified=True, t=clock.t + 20)
        now = clock.t + DAY
        runs.append((snapshot(hc, now), hc.raw_state(), hc.replay_batch(4, seed=7, min_trust=0.3),
                     (tmp_path / f"{name}.jsonl").read_bytes()))
    assert runs[0] == runs[1]


# ------------------------------------------------------------ versions / tombstones
def test_versions_and_as_of():
    hc, clock = make()
    hc.add("The office is in Oslo.", id="o", t=10)
    hc.update("o", "The office is in Bergen.", t=20)
    assert hc.update("o", "The office is in Tromso.", t=30) == 3
    latest = hc.recall("office", deep=True, now=40)
    assert [(h.version, h.text) for h in latest] == [(3, "The office is in Tromso.")]
    assert hc.recall("office", now=40)[0].version == 3
    assert hc.recall("office", as_of=15, now=40)[0].text == "The office is in Oslo."
    assert hc.recall("office", as_of=25, now=40)[0].version == 2
    assert hc.recall("office", as_of=5, now=40) == []
    assert hc.get("o", version=1)["text"] == "The office is in Oslo."
    assert hc.get("o", as_of=25)["version"] == 2


def test_tombstone_removes_from_recall_and_replay_but_log_keeps_it(tmp_path):
    path = tmp_path / "log.jsonl"
    hc, clock = make(path)
    hc.add("The vault code is 4417.", id="v", t=10, entities=["vault code"])
    hc.add("Unrelated note about gardens and roses.", id="g", t=11)
    hc.record_outcome("t", ["v"], success=True, verified=True, t=12)
    hc.nap(now=13)
    assert ids(hc.recall("vault code", now=14))[0] == "v"
    assert hc.replay_batch(5)
    hc.tombstone("v", reason="revoked", t=20)
    for deep in (None, True, False):
        assert "v" not in ids(hc.recall("vault code 4417", deep=deep, now=21))
    assert hc.replay_batch(5) == []
    assert "v" not in hc.fast_ids()
    assert "4417" in path.read_text(encoding="utf-8")
    assert ids(hc.recall("vault code", as_of=15, now=21)) == ["v"]       # history is not rewritten
    with pytest.raises(ValueError):
        hc.update("v", "new text")
    hc.rebuild()
    assert "v" not in ids(hc.recall("vault code 4417", deep=True, now=21))


# ------------------------------------------------------------ quarantine / trust
def test_quarantine_excluded_until_released():
    hc, clock = make()
    hc.add("Official note: the deploy key rotates monthly.", id="ok", source="user")
    hc.add("Web claim: the deploy key rotates hourly.", id="web", source="blog.example")
    hc.record_outcome("t", ["web"], success=True, verified=True)
    hc.nap()
    for deep in (None, True, False):
        got = ids(hc.recall("deploy key rotates", deep=deep))
        assert "ok" in got and "web" not in got
    assert hc.strength("web") == 0.0                          # quarantined memories gain nothing
    assert hc.replay_batch(5, min_trust=0.0) == []
    assert "web" not in ids(hc.vector_search("deploy key rotates"))
    hc.release("web", trust=0.8)
    assert "web" in ids(hc.recall("deploy key rotates", deep=True))
    assert "web" in ids(hc.recall("deploy key rotates", deep=False))


@pytest.mark.parametrize("disk_backed", [False, True])
@pytest.mark.parametrize("source,quarantine", [("forum", False), ("user", True)])
def test_release_preserves_historical_quarantine(tmp_path, disk_backed, source, quarantine):
    hc, _ = make(tmp_path / "log.jsonl" if disk_backed else None)
    hc.add("deploy key rotates monthly", id="q", source=source, quarantine=quarantine, t=10)
    hc.add("deploy key rotates yearly", id="ok", t=10)
    assert ids(hc.recall("deploy key rotates", as_of=15)) == ["ok"]
    hc.release("q", trust=0.9, t=20)
    for _ in range(2):
        assert ids(hc.recall("deploy key rotates", as_of=15)) == ["ok"]
        assert set(ids(hc.recall("deploy key rotates", as_of=20))) == {"q", "ok"}
        assert set(ids(hc.recall("deploy key rotates"))) == {"q", "ok"}
        hc.rebuild()


def test_min_trust_filter():
    hc, _ = make()
    hc.add("Policy alpha says refunds take 5 days.", id="hi", trust=0.9)
    hc.add("Policy beta says refunds take 9 days.", id="lo", trust=0.3)
    for deep in (None, True, False):
        assert set(ids(hc.recall("policy refunds days", deep=deep))) == {"hi", "lo"}
        assert ids(hc.recall("policy refunds days", deep=deep, min_trust=0.5)) == ["hi"]
    hc.set_trust("lo", 0.95)
    assert "lo" in ids(hc.recall("policy refunds days", min_trust=0.5))


# ------------------------------------------------------------ plasticity
def _pair_setup(**cfg):
    hc, clock = make(**cfg)
    hc.add("Zebra apples are discussed in the Alpha topic.", id="q", entities=["zebra apples"])
    hc.add("Kiwi mango smoothie recipe for Beta customers.", id="y", entities=["kiwi mango"])
    for i in range(6):
        hc.add(f"Zebra crossing rules and apples number {i} for Person{i} Smith.", id=f"d{i}")
    return hc, clock


def test_strengthening_only_on_verified_success():
    hc, clock = _pair_setup()
    hc.nap()
    base = hc.raw_state()
    hc.record_outcome("f1", ["q", "y"], success=False, verified=True, correction="x")
    hc.record_outcome("f2", ["q", "y"], success=False, verified=False)
    hc.record_outcome("u1", ["q", "y"], success=True, verified=False)
    for _ in range(20):                                       # recall must not strengthen anything either
        hc.recall("zebra apples", deep=True)
        hc.recall("zebra apples")
    hc.nap()
    assert hc.raw_state() == base and hc.link_weight("q", "y") == 0.0 and hc.strength("q") == 0.0
    hc.record_outcome("s1", ["q", "y"], success=True, verified=True)
    assert hc.link_weight("q", "y") == 0.0                    # logged now, applied only at nap
    assert hc.nap() == {"pending": 1, "applied": 1}
    assert hc.link_weight("q", "y") == pytest.approx(0.3)
    assert hc.strength("q") == pytest.approx(0.3) and hc.strength("y") == pytest.approx(0.3)
    assert hc.strength("d0") == 0.0


def test_fading_and_saturation():
    hc, clock = make(half_life=100.0)
    hc.add("alpha memory", id="a")
    hc.add("beta memory", id="b")
    prev = 0.0
    for i in range(40):
        hc.record_outcome(f"t{i}", ["a", "b"], success=True, verified=True)
        hc.nap()
        s = hc.strength("a")
        assert prev < s <= 1.0                                # rises, never exceeds the bound
        assert hc.link_weight("a", "b") <= 1.0
        prev = s
    assert prev > 0.999
    t0 = clock.t
    assert hc.strength("a", now=t0 + 100) == pytest.approx(prev / 2)
    assert hc.link_weight("a", "b", now=t0 + 200) == pytest.approx(hc.link_weight("a", "b", now=t0) / 4)
    assert hc.strength("a", now=t0 + 100 * 40) < 1e-9
    assert hc.strength("a", now=t0 - 50) == pytest.approx(prev)    # never "unfades" backwards


def test_faded_memory_still_found_and_recovers_after_use():
    hc, clock = make(half_life=100.0, fast_size=1, fast_recent_cap=0, recency_weight=0.0)
    hc.add("Old skill: rotate the quartz gasket before winter.", id="old", entities=["quartz gasket"])
    hc.add("Other skill: bake the sourdough starter slowly.", id="other", entities=["sourdough starter"])
    for i in range(3):
        hc.record_outcome(f"o{i}", ["old"], success=True, verified=True)
    hc.nap()
    assert hc.fast_ids() == ["old"] and hc.strength("old") > 0.6
    clock.advance(100 * 50)                                   # 50 half-lives of disuse
    for i in range(2):
        hc.record_outcome(f"n{i}", ["other"], success=True, verified=True)
    hc.nap()
    assert hc.strength("old") < 1e-9 and hc.fast_ids() == ["other"]   # faded out of the fast index
    hits = hc.recall("quartz gasket winter")
    assert hits[0].id == "old" and hits[0].via == "deep"      # still findable, just not on the fast path
    for i in range(2):
        hc.record_outcome(f"r{i}", ["old"], success=True, verified=True)
    hc.nap()
    assert hc.strength("old") > 0.5 and "old" in hc.fast_ids()   # one use made it strong again
    hits = hc.recall("quartz gasket winter")
    assert hits[0].id == "old" and hits[0].via == "fast"


def test_strengthened_links_change_deep_ranking():
    hc, clock = _pair_setup()
    hc.nap()
    before = hc.recall("zebra apples", deep=True, k=8, entities=["zebra apples"])
    assert "y" not in ids(before)                             # y shares nothing with the query
    for i in range(3):
        hc.record_outcome(f"s{i}", ["q", "y"], success=True, verified=True)
    hc.nap()
    after = ids(hc.recall("zebra apples", deep=True, k=8, entities=["zebra apples"]))
    assert after.index("q") == 0 and "y" in after and after.index("y") <= 2


def test_strong_link_is_followed_in_fast_pass():
    hc, clock = _pair_setup(confidence=0.2)
    before = hc.recall("zebra apples")
    assert "y" not in ids(before)
    for i in range(3):
        hc.record_outcome(f"s{i}", ["q", "y"], success=True, verified=True)
    hc.nap()
    after = hc.recall("zebra apples")
    assert ids(after)[0] == "q" and "y" in ids(after)
    assert all(h.via == "fast" for h in after) and not hc.last_trace["deep_ran"]
    # failures never create the link, so nothing extra is pulled in
    hc2, _ = _pair_setup(confidence=0.2)
    for i in range(3):
        hc2.record_outcome(f"f{i}", ["q", "y"], success=False, verified=True, correction="c")
    hc2.nap()
    assert "y" not in ids(hc2.recall("zebra apples"))


# ------------------------------------------------------------ fast / deep
def test_fast_returns_without_deep_when_confident_and_falls_through_when_not():
    hc, clock = make(fast_size=2, fast_recent_cap=0, recency_weight=0.0)
    texts = {"a": "Gearbox calibration procedure for the Tesla bench.",
             "b": "Ledger reconciliation steps for Fridays.",
             "c": "Antenna alignment checklist for Harbor station.",
             "d": "Compost schedule for the Orchard committee."}
    for k, v in texts.items():
        hc.add(v, id=k)
    for i in range(3):
        hc.record_outcome(f"t{i}", ["a", "b"], success=True, verified=True)
    hc.nap()
    assert hc.fast_ids() == ["a", "b"]
    d0 = hc.stats["deep_calls"]
    hits = hc.recall("gearbox calibration procedure")
    assert hits[0].id == "a" and hits[0].via == "fast"
    assert hc.stats["deep_calls"] == d0 and hc.last_trace["deep_ran"] is False
    hits = hc.recall("antenna alignment checklist harbor")       # lives outside the fast index
    assert hc.stats["deep_calls"] == d0 + 1 and hc.last_trace["deep_ran"] is True
    assert hits[0].id == "c" and hits[0].via == "deep"
    forced = hc.recall("gearbox calibration procedure", deep=True)
    assert forced[0].via == "deep" and hc.stats["deep_calls"] == d0 + 2
    only_fast = hc.recall("antenna alignment checklist harbor", deep=False)
    assert "c" not in ids(only_fast) and hc.stats["deep_calls"] == d0 + 2
    for h in hits:
        assert h.version == 1 and h.text and isinstance(h.score, float)


def test_multi_hop_deep_finds_what_plain_vector_search_misses():
    hc, clock = make()
    hc.add("Mireva Tonks founded Quillon Labs in 2011.", id="A")
    hc.add("Quillon Labs operates from the city of Pelmont.", id="B")
    for i in range(30):
        hc.add(f"Person{i} Smith founded a company that does work in field {i}.", id=f"x{i}")
    question = "Where does the company of Mireva Tonks work?"
    assert "B" not in ids(hc.vector_search(question, k=5))    # B shares no words with the question
    deep = ids(hc.recall(question, deep=True, k=3))
    assert "A" in deep and "B" in deep                         # needs the A -> Quillon Labs -> B hop
    assert deep[0] == "A" and deep.index("B") <= 2


def test_synonym_edges_connect_similar_phrases():
    hc, _ = make(syn_threshold=0.5)
    hc.add("Dr Rowan studied the Quillon Labs archive.", id="a", entities=["rowan", "quillon labs"])
    hc.add("Notes on the Quillon Lab roof.", id="b", entities=["quillon lab", "roof"])
    assert hc._syn                                              # "quillon labs" ~ "quillon lab"
    assert "b" in ids(hc.recall("what did Rowan study", deep=True, k=3, entities=["rowan"]))


# ------------------------------------------------------------ replay supply
@pytest.mark.parametrize("disk_backed", [False, True])
@pytest.mark.parametrize("success", [False, True])
def test_replay_keeps_episode_version_at_verification(tmp_path, disk_backed, success):
    hc, _ = make(tmp_path / "log.jsonl" if disk_backed else None)
    hc.add("2+2=4", id="e", kind="episode", t=10)
    correction = None if success else "use addition"
    hc.record_outcome("original", success=success, verified=True, correction=correction,
                      episode_id="e", t=10)
    hc.update("e", "2+2=5", t=10)                           # event order matters even at the same time
    batch = hc.replay_batch(1)
    assert [b.text for b in batch] == ["2+2=4"]
    assert batch[0].correction == correction
    hc.record_outcome("updated", success=True, verified=True, episode_id="e", t=20)
    both = hc.replay_batch(2)
    assert {b.task_id: b.text for b in both} == {"original": "2+2=4", "updated": "2+2=5"}
    hc.rebuild()
    assert hc.replay_batch(2) == both


def test_replay_batch_filters_and_mixes_new_and_old():
    hc, clock = make()
    for i in range(20):
        hc.add(f"episode {i} about topic{i}", id=f"e{i}", kind="episode", t=1.0)
    hc.add("untrusted episode", id="q", kind="episode", source="forum", t=1.0)
    hc.add("low trust episode", id="lt", kind="episode", trust=0.1, t=1.0)
    hc.add("episode to revoke", id="rv", kind="episode", t=1.0)
    for i in range(10):
        hc.record_outcome(f"old{i}", [], success=True, verified=True, episode_id=f"e{i}", t=10.0 + i)
    hc.record_outcome("oldfix", [], success=False, verified=True, correction="do it this way",
                      episode_id="e10", t=21.0)
    hc.mark_sleep(t=50.0)
    for i in range(11, 17):
        hc.record_outcome(f"new{i}", [], success=True, verified=True, episode_id=f"e{i}", t=60.0 + i)
    hc.record_outcome("nofix", [], success=False, verified=True, episode_id="e17", t=80.0)
    hc.record_outcome("unver", [], success=True, verified=False, episode_id="e18", t=81.0)
    hc.record_outcome("fail-unver", [], success=False, verified=False, correction="c", episode_id="e18", t=81.0)
    hc.record_outcome("quar", [], success=True, verified=True, episode_id="q", t=82.0) if False else None
    hc.release("q", trust=0.9, t=1.0)
    hc.record_outcome("lowt", [], success=True, verified=True, episode_id="lt", t=83.0)
    hc.record_outcome("revoked", [], success=True, verified=True, episode_id="rv", t=84.0)
    hc.tombstone("rv", t=90.0)
    hc.set_trust("q", 0.9)
    batch = hc.replay_batch(8, seed=3, new_fraction=0.5, min_trust=0.5)
    tasks = {b.task_id for b in batch}
    assert len(batch) == 8
    assert sum(b.new for b in batch) == 4 and sum(not b.new for b in batch) == 4
    assert not tasks & {"nofix", "unver", "fail-unver", "lowt", "revoked"}
    assert all(b.new == b.task_id.startswith("new") for b in batch)
    everything = hc.replay_batch(100, seed=0, min_trust=0.5)
    assert {b.task_id for b in everything} == {f"old{i}" for i in range(10)} | {"oldfix"} | \
        {f"new{i}" for i in range(11, 17)}
    fix = [b for b in everything if b.task_id == "oldfix"][0]
    assert fix.kind == "corrected_failure" and fix.correction == "do it this way" and fix.text.startswith("episode 10")
    assert hc.replay_batch(8, seed=3) == batch                 # same seed, same batch
    assert hc.replay_batch(8, seed=4) != batch                 # seed matters
    # when one side is short, the other fills in
    small = hc.replay_batch(15, seed=1, new_fraction=0.9, min_trust=0.5)
    assert len(small) == 15 and sum(b.new for b in small) == 6


def test_replay_excludes_quarantined_episode():
    hc, _ = make()
    hc.add("good episode", id="g", kind="episode")
    hc.add("planted episode", id="p", kind="episode", source="forum")
    for e in ("g", "p"):
        hc.record_outcome(f"t-{e}", [], success=True, verified=True, episode_id=e)
    assert [b.task_id for b in hc.replay_batch(5, min_trust=0.0)] == ["t-g"]
    hc.release("p", trust=0.9)
    assert {b.task_id for b in hc.replay_batch(5, min_trust=0.5)} == {"t-g", "t-p"}


# ------------------------------------------------------------ scale
def test_recall_over_10k_memories_is_fast(capsys):
    rng = random.Random(0)
    words = [f"w{i}" for i in range(4000)]
    ents = [f"ent{i}" for i in range(3000)]
    hc = Hippocampus(None, config=Config(), clock=ManualClock(0.0))
    t0 = time.perf_counter()
    docs = []
    for i in range(10_000):
        e = rng.sample(ents, 3)
        text = f"doc{i} " + " ".join(rng.sample(words, 8))
        docs.append((text, e))
        hc.add(text, entities=e, t=float(i))
    t_build = time.perf_counter() - t0
    for j in range(300):
        group = rng.sample(range(10_000), 3)
        hc.record_outcome(f"t{j}", [f"m{g:06d}" for g in group], success=True, verified=True, t=10_000.0 + j)
    hc.nap(now=11_000.0)
    queries = [docs[rng.randrange(10_000)] for _ in range(20)]
    t0 = time.perf_counter()
    first = hc.recall(queries[0][0], deep=True, entities=queries[0][1], now=11_001.0)
    t_first = time.perf_counter() - t0
    deep_times, auto_times = [], []
    for text, e in queries:
        t0 = time.perf_counter()
        hits = hc.recall(text, deep=True, entities=e, now=11_001.0)
        deep_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        hc.recall(text, entities=e, now=11_001.0)
        auto_times.append(time.perf_counter() - t0)
        assert hits and hits[0].text == text                   # the exact memory is found first
    with capsys.disabled():
        print(f"\n[10k] build {t_build:.2f}s | first deep recall (builds graph) {t_first:.3f}s | "
              f"deep median {statistics.median(deep_times):.3f}s max {max(deep_times):.3f}s | "
              f"auto median {statistics.median(auto_times):.3f}s")
    assert first and max(deep_times) < 3.0 and t_first < 5.0
