"""CPU/stub-only semantic retrieval checks; never fetch a model."""
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from baselines.arms import Rag
from baselines.semantic import SemanticEmbedder, StubEncoder
from baselines import semantic
from harness.tasks import Task
from hippocampus import HashingEmbedder


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    weights = b"local fake weights, not a real checkpoint"
    (tmp_path / semantic.WEIGHTS_FILENAME).write_bytes(weights)
    monkeypatch.setattr(semantic, "WEIGHTS_SHA256", hashlib.sha256(weights).hexdigest())
    return tmp_path


@pytest.mark.parametrize("texts", [[], [""], ["Alder leads", " A B\n", "same", "same"]])
@pytest.mark.parametrize("factory", [
    lambda: HashingEmbedder(dim=384),
    lambda: SemanticEmbedder(encoder=StubEncoder(), allow_unpinned=True),
])
def test_interface_parity(factory, texts):
    embedder = factory()
    vectors = embedder.embed(texts)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (len(texts), embedder.dim)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.all(np.isclose(norms, 0) | np.isclose(norms, 1))
    np.testing.assert_array_equal(vectors, embedder.embed(tuple(texts)))


@pytest.mark.parametrize("pin", ["MODEL_REVISION", "WEIGHTS_SHA256"])
def test_placeholder_refused_before_any_loading(monkeypatch, pin):
    monkeypatch.setattr(semantic, pin, "TODO_PIN_not_known")
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="TODO_PIN"):
        SemanticEmbedder()


def test_stub_requires_explicit_override():
    with pytest.raises(ValueError, match="allow_unpinned=True"):
        SemanticEmbedder(encoder=StubEncoder())


@pytest.mark.parametrize("allow_unpinned", [False, True])
def test_hash_mismatch_always_refused(snapshot, monkeypatch, allow_unpinned):
    monkeypatch.setattr(semantic, "WEIGHTS_SHA256", "0" * 64)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        SemanticEmbedder(snapshot, allow_unpinned=allow_unpinned)


def test_pinned_local_constructor_and_empty_embed_are_lazy(snapshot, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    embedder = SemanticEmbedder(snapshot)
    assert embedder.model_path == snapshot.resolve()
    assert embedder.revision == semantic.MODEL_REVISION
    assert embedder.embed([]).shape == (0, 384)


def test_explicit_unpinned_development_override(snapshot, monkeypatch):
    monkeypatch.setattr(semantic, "WEIGHTS_SHA256", "TODO_PIN_weights")
    monkeypatch.setattr(semantic, "MODEL_REVISION", "TODO_PIN_revision")
    assert SemanticEmbedder(snapshot, allow_unpinned=True)._model is None


def test_local_environment_path_and_argument_precedence(snapshot, monkeypatch):
    monkeypatch.setenv(semantic.MODEL_PATH_ENV, str(snapshot))
    assert SemanticEmbedder().model_path == snapshot.resolve()
    monkeypatch.setenv(semantic.MODEL_PATH_ENV, str(snapshot / "missing"))
    assert SemanticEmbedder(snapshot).model_path == snapshot.resolve()


def test_missing_local_path_and_weights_fail_closed(snapshot, monkeypatch):
    monkeypatch.delenv(semantic.MODEL_PATH_ENV, raising=False)
    with pytest.raises(ValueError, match="downloads are disabled"):
        SemanticEmbedder()
    with pytest.raises(ValueError, match="directory does not exist"):
        SemanticEmbedder(snapshot / "not-a-directory")
    (snapshot / semantic.WEIGHTS_FILENAME).unlink()
    with pytest.raises(ValueError, match="missing local weights"):
        SemanticEmbedder(snapshot)


def test_recheck_weights_before_lazy_load(snapshot, monkeypatch):
    embedder = SemanticEmbedder(snapshot)
    (snapshot / semantic.WEIGHTS_FILENAME).write_bytes(b"changed after construction")
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        embedder.embed(["query"])


def test_stub_determinism_across_calls_and_instances():
    first = SemanticEmbedder(encoder=StubEncoder(), allow_unpinned=True)
    second = SemanticEmbedder(encoder=StubEncoder(), allow_unpinned=True)
    texts = ["Stored text\nEXACT case", "query", ""]
    expected = first.embed(texts)
    first.embed(["unrelated call"])
    np.testing.assert_array_equal(expected, first.embed(texts))
    np.testing.assert_array_equal(expected, second.embed(texts))


@pytest.mark.parametrize("budget", [0, 16, 300])
def test_rag_end_to_end_stub_preserves_text_and_policy(budget):
    class RecordingEncoder(StubEncoder):
        def encode(self, texts):
            self.seen = list(texts)
            return super().encode(texts)

    encoder = RecordingEncoder()
    arm = Rag(k=1, context_tokens=budget,
              embedder=SemanticEmbedder(encoder=encoder, allow_unpinned=True))
    for i, value in enumerate(["Mira", "Dana"]):
        task = Task(str(i), f"Field note: the leader of Alder is {value}.",
                    "hidden-family", "hidden-skill", kind="teach", source="trusted")
        arm.observe(task, "", None, None)
    query = Task("query", "Who leads Alder?", "unrelated-family", "unrelated-skill", mode="test")
    assert arm.retrieve(query.prompt)[0][0] == 1  # identical scores: newest wins
    assert encoder.seen == [query.prompt] + [entry["text"] for entry in arm.episodes]
    assert all("hidden-skill" not in text and "hidden-family" not in text for text in encoder.seen)
    state = arm.save_state()
    assert arm.answer(query) == "Final answer: " + ("Dana" if budget else "unknown")
    assert arm.answer_trace["retrieved"] == ([1] if budget else [])
    assert arm.llm.count_tokens(arm.answer_trace["prompt"].split("\n\nTask:\n")[0]) <= max(budget, 1)
    arm.observe(query, "ignored", True, "secret")
    assert arm.save_state() == state
    restored = Rag(k=1, context_tokens=budget, embedder=arm.embedder)
    restored.load_state(state)
    assert restored.answer(query) == arm.answer(query)


def test_local_transformers_loading_masked_pooling_and_cpu(snapshot, monkeypatch):
    torch = pytest.importorskip("torch")
    calls, tokenized = [], []

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == dict(padding=True, truncation=True, max_length=256, return_tensors="pt")
            tokenized.extend(texts)
            return {"input_ids": torch.ones((len(texts), 3), dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 0]] * len(texts))}

    class Model:
        config = SimpleNamespace(hidden_size=384)
        training = True

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            self.training = False
            return self

        def __call__(self, **inputs):
            assert not self.training and not torch.is_grad_enabled()
            assert all(value.device.type == "cpu" for value in inputs.values())
            hidden = torch.zeros((len(inputs["input_ids"]), 3, 384))
            hidden[:, 0, :2] = torch.tensor([2., 6.])
            hidden[:, 1, :2] = torch.tensor([4., 2.])
            hidden[:, 2, :] = 1000  # padding must not influence the mean
            return SimpleNamespace(last_hidden_state=hidden)

    def tokenizer_loader(path, **kwargs):
        calls.append(("tokenizer", path, kwargs))
        return Tokenizer()

    def model_loader(path, **kwargs):
        calls.append(("model", path, kwargs))
        return Model()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        AutoModel=SimpleNamespace(from_pretrained=model_loader)))
    embedder = SemanticEmbedder(snapshot, batch_size=2)
    texts = ["  Preserve CASE\n", "second", "third"]
    expected = np.zeros((3, 384), dtype=np.float32)
    expected[:, :2] = [.6, .8]
    np.testing.assert_allclose(embedder.embed(texts), expected, atol=1e-7)
    np.testing.assert_array_equal(embedder.embed(texts), embedder.embed(texts))
    assert tokenized[:3] == texts
    assert len(calls) == 2  # loaded once, despite repeated calls
    for kind, path, kwargs in calls:
        assert path == str(snapshot.resolve())
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        if kind == "model":
            assert kwargs["use_safetensors"] is True
            assert kwargs["dtype"] == torch.float32
            assert kwargs["attn_implementation"] == "eager"


@pytest.mark.parametrize("vectors", [np.zeros((1, 2)), np.full((1, 384), np.nan)])
def test_bad_encoder_output_refused(vectors):
    encoder = SimpleNamespace(dim=384, encode=lambda texts: vectors)
    with pytest.raises(ValueError, match="invalid embedding"):
        SemanticEmbedder(encoder=encoder, allow_unpinned=True).embed(["query"])
