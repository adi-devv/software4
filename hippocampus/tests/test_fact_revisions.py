"""Explicit fact notices share a versioned identity, including through the arm."""
import pytest

from baselines import FakeBackend, SleepArm
from harness.tasks import Task
from hippocampus import Hippocampus


ORIGINAL = 'Fact: Field note: the leader of Alder is Mira.'
REVISION = 'Fact: Update notice: the leader of Alder is now Sora.'


@pytest.mark.parametrize('deep', [None, False, True])
def test_default_recall_excludes_superseded_fact_but_keeps_history(deep):
    hippo = Hippocampus()
    mid = hippo.add(ORIGINAL, source='system', t=10)
    assert hippo.add(REVISION, source='system', t=20) == mid
    assert hippo.get(mid)['version'] == 2
    for rebuild in (False, True):
        if rebuild:
            hippo.rebuild()
        hits = hippo.recall('Who leads Alder?', k=5, deep=deep, now=30)
        assert [(h.id, h.version, h.text) for h in hits] == [(mid, 2, REVISION)]
        assert hippo.vector_search('leader Alder', k=5)[0].text == REVISION
        assert hippo.recall('leader Alder', as_of=15)[0].text == ORIGINAL
        assert hippo.get(mid, version=1)['text'] == ORIGINAL


@pytest.mark.parametrize('source,trust,quarantine', [
    ('untrusted', None, False), ('system', None, True), ('system', 0.2, False)])
def test_poison_never_supersedes_trusted_identity(source, trust, quarantine):
    hippo = Hippocampus()
    mid = hippo.add(ORIGINAL, source='system', t=1)
    poison = hippo.add(REVISION.replace('Sora', 'Zorg'), source=source,
                       trust=trust, quarantine=quarantine, t=2)
    assert poison != mid and hippo.get(mid)['version'] == 1
    if source == 'untrusted' or quarantine:
        assert all('Zorg' not in h.text for h in hippo.recall('leader Alder', deep=True))
    assert hippo.add(REVISION, source='system', t=3) == mid


def test_fake_backend_arm_recalls_only_current_fact_after_restore():
    arm = SleepArm(FakeBackend())
    for i, text in enumerate((ORIGINAL[6:], REVISION[6:])):
        arm.observe(Task(str(i), text, 'fact', 'toy', kind='teach', source='trusted'),
                    '', None, None)
    query = Task('query', 'Who leads Alder?', 'fact', 'toy', mode='test')
    assert arm.context(query) == [('m000000', REVISION)]
    assert arm.answer(query) == 'Final answer: Sora'
    restored = SleepArm(FakeBackend())
    restored.load_state(arm.save_state())
    assert restored.context(query) == arm.context(query)
    assert restored.answer(query) == arm.answer(query)


def test_identity_includes_relation_and_subject_and_requires_explicit_notice():
    hippo = Hippocampus()
    mid = hippo.add(ORIGINAL)
    for text in ('Fact: Update notice: the ally of Alder is now Sora.',
                 'Fact: Update notice: the leader of Birch is now Sora.',
                 'Alder has a leader named Sora.'):
        assert hippo.add(text) != mid
    assert hippo.get(mid)['version'] == 1


def test_identity_index_follows_explicit_update_and_tombstone():
    hippo = Hippocampus()
    mid = hippo.add(ORIGINAL)
    hippo.update(mid, ORIGINAL.replace('Alder', 'Birch'))
    replacement = hippo.add(REVISION)
    assert replacement != mid  # old key no longer points at a different subject
    hippo.tombstone(replacement)
    assert hippo.add(REVISION) != replacement


def test_historical_revision_authority_and_metadata_survive_rebuild():
    arm = SleepArm(FakeBackend())
    h = arm.hippo
    mid = h.add(ORIGINAL, source='user', trust=0.2, meta={'origin': 'first'}, t=1)
    assert h.add(REVISION, source='system', trust=1.0,
                 meta={'origin': 'signed revision'}, t=2) == mid
    for current in (h, h.rebuild()):
        assert current.get(mid)['trust'] == 1.0
        assert current.get(mid)['source'] == 'system'
        assert current.get(mid)['meta'] == {'origin': 'signed revision'}
        assert current.get(mid, version=1)['trust'] == 0.2
        assert current.get(mid, version=1)['meta'] == {'origin': 'first'}
    restored = SleepArm(FakeBackend())
    restored.load_state(arm.save_state())
    for current in (h, restored.hippo):
        assert current.add(REVISION.replace('Sora', 'Weak'), source='system',
                           trust=0.3, t=3) != mid
        assert current.get(mid)['text'] == REVISION
        assert current.get(mid)['trust'] == 1.0


@pytest.mark.parametrize('rebuild', [False, True])
def test_released_fact_receives_notice_on_same_identity(rebuild):
    h = Hippocampus()
    mid = h.add(ORIGINAL, source='system', quarantine=True, t=1)
    h.release(mid, trust=0.8, t=2)
    if rebuild:
        h.rebuild()
    assert h.add(REVISION, source='system', trust=0.9, t=3) == mid
    assert [(x.id, x.text) for x in h.recall('leader Alder', deep=True)] == [(mid, REVISION)]


@pytest.mark.parametrize('transition', ['trust', 'tombstone', 'key'])
def test_competing_identity_winner_reselected_live_and_replayed(transition):
    h = Hippocampus()
    first = h.add(ORIGINAL, source='system', trust=0.6, t=1)
    second = h.add(ORIGINAL.replace('Mira', 'Other'), source='system', trust=0.9, t=2)
    if transition == 'trust':
        h.set_trust(second, 0.1, t=3)
    elif transition == 'tombstone':
        h.tombstone(second, t=3)
    else:
        h.update(second, ORIGINAL.replace('Alder', 'Birch'), t=3)
    for current in (h, h.rebuild()):
        assert current._fact_ids[('leader', 'alder')] == first
    assert h.add(REVISION, source='system', trust=0.7, t=4) == first
    h.tombstone(first, t=5)
    h.rebuild()
    assert h._fact_ids.get(('leader', 'alder')) == (second if transition == 'trust' else None)


def test_revision_authority_and_provenance_survive_rebuild_and_restore():
    arm = SleepArm(FakeBackend())
    hippo = arm.hippo
    mid = hippo.add(ORIGINAL, source='user', trust=0.2, meta={'receipt': 'old'}, t=1)
    assert hippo.add(REVISION, source='system', trust=1.0,
                     meta={'receipt': 'verified'}, t=2) == mid
    event = hippo.log.events[-1]
    assert (event['trust'], event['source'], event['meta']) == (1.0, 'system', {'receipt': 'verified'})
    restored = SleepArm(FakeBackend())
    restored.load_state(arm.save_state())
    for store in (hippo.rebuild(), restored.hippo):
        assert store.add(REVISION.replace('Sora', 'Zorg'), source='user', trust=0.3, t=3) != mid
        current = store.get(mid)
        assert (current['text'], current['trust'], current['source'], current['meta']) == (
            REVISION, 1.0, 'system', {'receipt': 'verified'})
        original = store.get(mid, version=1)
        assert (original['trust'], original['source'], original['meta']) == (0.2, 'user', {'receipt': 'old'})


@pytest.mark.parametrize('rebuild', [False, True])
def test_released_fact_receives_notice_in_same_identity(rebuild):
    hippo = Hippocampus()
    mid = hippo.add(ORIGINAL, source='system', quarantine=True, t=1)
    hippo.release(mid, trust=0.8, t=2)
    if rebuild:
        hippo.rebuild()
    assert hippo.add(REVISION, source='system', t=3) == mid
    assert [h.text for h in hippo.recall('leader Alder', deep=True)] == [REVISION]


@pytest.mark.parametrize('transition', ['trust', 'tombstone', 'key'])
@pytest.mark.parametrize('rebuild', [False, True])
def test_competing_identity_reselects_eligible_authority(transition, rebuild):
    hippo = Hippocampus()
    first = hippo.add(ORIGINAL, id='first', trust=0.7, t=1)
    second = hippo.add(ORIGINAL.replace('Mira', 'Nora'), id='second', trust=0.9, t=2)
    if transition == 'trust':
        hippo.set_trust(second, 0.1, t=3)
    elif transition == 'tombstone':
        hippo.tombstone(second, t=3)
    else:
        hippo.update(second, ORIGINAL.replace('Alder', 'Birch'), t=3)
    if rebuild:
        hippo.rebuild()
    assert hippo.add(REVISION, trust=0.8, t=4) == first
    hippo.set_trust(first, 0.05, t=5)
    if transition == 'trust':
        assert hippo.add(REVISION.replace('Sora', 'Tali'), trust=0.2, t=6) == second
