"""Regression tests for cache coverage and automatic graph dispatch, offline."""
import numpy as np
import pytest

from hippocampus import Config, Hippocampus, ManualClock


class TableEmbedder:
    """Controlled cosine scores, without depending on feature-hash collisions."""

    def __init__(self, scores):
        self.scores = scores

    def embed(self, texts):
        return np.array([[s, np.sqrt(1 - s * s)] for s in
                         (self.scores.get(text, 1.0) for text in texts)], dtype=np.float32)


def controlled(scores, **config):
    return Hippocampus(embedder=TableEmbedder(scores), config=Config(**config),
                       clock=ManualClock(1000))


def ids(hits):
    return [h.id for h in hits]


@pytest.mark.parametrize("k", [2, 5, 20])
def test_confident_top_k_matches_full_vector_ranking_not_cache(k):
    scores = {f"anchor {i}": .6 + i * .05 for i in range(7)}
    hc = controlled(scores, fast_recent_cap=1)
    for text in scores:
        hc.add(text, id=text, entities=[])
    # Only the best hit is cached; every runner-up lives outside the fast set.
    assert ids(hc.recall("anchor", k=k, deep=False)) == ["anchor 6"]
    result = hc.recall("anchor", k=k)
    assert ids(result) == ids(hc.vector_search("anchor", k=k))
    assert len(result) == min(k, len(scores))
    assert hc.last_trace["vector_checked"]
    assert hc.last_trace["fast_consistent"]
    assert not hc.last_trace["deep_ran"]


@pytest.mark.parametrize("k", [1, 5])
def test_confident_wrong_cached_winner_is_checked_against_full_store(k):
    hc = controlled({"anchor correct": .95, "anchor wrong": .7}, fast_recent_cap=1)
    hc.add("anchor correct", id="correct", entities=[])
    hc.add("anchor wrong", id="wrong", entities=[])
    assert ids(hc.recall("anchor", k=k, deep=False)) == ["wrong"]
    assert hc.last_trace["fast_best"] >= hc.cfg.confidence
    assert hc.recall("anchor", k=k)[0].id == "correct"
    assert hc.last_trace["vector_checked"]
    assert not hc.last_trace["fast_consistent"]
    assert hc.last_trace["deep_ran"]


def test_confident_complete_single_hit_still_skips_ppr():
    hc = controlled({"anchor correct": .95, "anchor other": .7}, fast_recent_cap=1)
    hc.add("anchor other", id="other", entities=[])
    hc.add("anchor correct", id="correct", entities=[])
    assert ids(hc.recall("anchor", k=1)) == ["correct"]
    assert hc.stats == {"fast_calls": 1, "deep_calls": 0}
    assert hc.last_trace["vector_checked"]


@pytest.mark.parametrize("k", [1, 5])
def test_compound_question_reaches_ppr_despite_confident_first_fact(k):
    hc = Hippocampus(config=Config(confidence=.1, fast_recent_cap=8), clock=ManualClock(1000))
    hc.add("Willow manages deliveries.", id="second")
    for i in range(12):
        hc.add(f"Harbor{i} manages the supplier contracts of Granite{i}.", id=f"d{i}")
    hc.add("The supplier of Ardent is Willow.", id="first")
    query = "Who manages the supplier of Ardent?"
    fast = hc.recall(query, k=5, deep=False)
    assert fast[0].id == "first"
    assert fast[0].score >= hc.cfg.confidence
    assert "second" not in ids(fast)
    assert "second" not in ids(hc.vector_search(query, k=5))
    result = hc.recall(query, k=k)
    assert hc.last_trace["deep_ran"]
    assert "manages" in hc.last_trace["uncovered_terms"]
    assert result[0].id == "first"
    if k > 1:
        assert {"first", "second"} <= set(ids(result))


@pytest.mark.parametrize("deep", [None, True, False])
def test_expanded_candidates_keep_lifecycle_and_trust_filters(deep):
    scores = {"anchor old": .6, "anchor revised": .85,
              **{f"anchor {status}": .99 for status in
                 ("quarantined", "untrusted", "tombstoned", "low", "released")},
              "anchor recent": .9}
    hc = controlled(scores, fast_recent_cap=1)
    hc.add("anchor old", id="revised", entities=[])
    hc.update("revised", "anchor revised", entities=[])
    hc.add("anchor quarantined", id="quarantined", quarantine=True, entities=[])
    hc.add("anchor untrusted", id="untrusted", source="forum", entities=[])
    hc.add("anchor tombstoned", id="tombstoned", entities=[])
    hc.tombstone("tombstoned")
    hc.add("anchor low", id="low", trust=.1, entities=[])
    hc.add("anchor released", id="released", source="forum", entities=[])
    hc.add("anchor recent", id="recent", entities=[])
    before = hc.recall("anchor", k=20, min_trust=.5, deep=deep)
    assert set(ids(before)) <= {"revised", "recent"}
    assert all(h.version == (2 if h.id == "revised" else 1) for h in before)
    hc.release("released", trust=.9)
    after = hc.recall("anchor", k=20, min_trust=.5, deep=deep)
    assert set(ids(after)) <= {"revised", "recent", "released"}
    assert "released" in ids(after)
    if deep is None:
        assert {"revised", "recent", "released"} == set(ids(after))


def test_full_store_cosine_ties_use_same_stable_order_as_vector_search():
    hc = controlled({"anchor first": .8, "anchor last": .8}, fast_recent_cap=1)
    hc.add("anchor first", id="first", entities=[])
    hc.add("anchor last", id="last", entities=[])
    assert hc.recall("anchor", k=1, deep=False)[0].id == "last"
    assert ids(hc.recall("anchor", k=2)) == ids(hc.vector_search("anchor", k=2))


def test_empty_recall_and_zero_cutoff():
    hc = Hippocampus()
    assert hc.recall("empty", k=5) == []
    hc.add("anchor")
    assert hc.recall("anchor", k=0) == []
