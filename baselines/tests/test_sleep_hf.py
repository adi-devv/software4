"""No downloads: a tiny locally initialized Qwen and a character tokenizer."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from baselines.sleep_hf import HFBackend


class CharTokenizer:
    eos_token_id = 0
    chat_template = None

    def encode(self, text, **kwargs):
        return [1 + ord(c) % 127 for c in text]

    def decode(self, ids, **kwargs):
        return ','.join(str(int(i)) for i in ids if int(i))


@pytest.fixture
def backend(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(12)
    config = transformers.Qwen3Config(vocab_size=128, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, max_position_embeddings=256,
        bos_token_id=1, eos_token_id=0, pad_token_id=0, attention_dropout=0.0)
    return HFBackend(transformers.Qwen3ForCausalLM(config), CharTokenizer(),
                     device='cpu', directory=tmp_path / 'working', lr=0.02)


def test_train_changes_generation_base_identical_and_old_expert_untouched(backend):
    b = backend
    prompt = 'Context:\n\n\nTask:\nx'
    before = b.generate(prompt)
    ids = torch.tensor([b._ids(prompt)])
    with torch.no_grad():
        base_logits = b.model(ids).logits.clone()
    fingerprint = b.frozen_fingerprint()
    a = b.train_expert([(prompt, 'ZZZZ')], steps=16, seed=3)
    assert b.is_finite(a)
    assert b.generate(prompt, a) != before
    b._cache.clear()  # demand a real forward, never accept cache as invariance evidence
    assert b.generate(prompt) == before
    old = b._state(a)
    b.commit_expert(a)
    c = b.train_expert([(prompt, 'BBBB')], steps=2, seed=4)
    b._cache.clear()
    assert b.generate(prompt) == before
    with torch.no_grad():
        assert torch.equal(b.model(ids).logits, base_logits)
    assert b.frozen_fingerprint() == fingerprint
    for name in old:
        for k in old[name]:
            assert torch.equal(old[name][k], b._state(a)[name][k])
    assert all(p.dtype == torch.float32 for p in b._parameters(a))
    assert all(not p.requires_grad for n, p in b.model.named_parameters() if '.adapters.' not in n)
    b.discard_expert(c)
    assert c not in b.experts and all(c not in l.adapters for l in b.layers.values())
    assert not (b.directory / (c + '.pt')).exists()
    with pytest.raises(KeyError):
        b.generate(prompt, c)


def test_save_load_reset_and_finite_detection(backend, tmp_path):
    b = backend
    b.save(tmp_path / 'empty')
    h = b.train_expert([('x', 'Z')], steps=1, seed=1)
    b.commit_expert(h)
    expected = b.generate('x', h)
    b.save(tmp_path / 'snapshot')
    with torch.no_grad():
        b._parameters(h)[0].fill_(float('inf'))
    assert not b.is_finite(h)
    b.load(tmp_path / 'snapshot')
    assert b.is_finite(h) and b.generate('x', h) == expected
    b.load(tmp_path / 'empty')
    assert not b.experts


def test_nonfinite_loss_skips_step_and_marks_candidate(backend, monkeypatch):
    b = backend
    original = b.model.forward
    seen = []

    def bad_forward(*args, **kwargs):
        seen.append(kwargs)
        output = original(*args, **kwargs)
        return SimpleNamespace(loss=output.loss * float('nan'))

    monkeypatch.setattr(b.model, 'forward', bad_forward)
    h = b.train_expert([('x' * 300, 'Z' * 100)], steps=2, seed=1)
    assert not b.is_finite(h)
    assert b.train_calls[-1]['optimizer_steps'] == 0
    assert b.train_calls[-1]['skipped_steps'] == 2
    labels = seen[0]['labels'][0]
    assert len(labels) <= 128
    assert (labels == -100).sum() == 112
    assert (labels != -100).sum() == 16
    assert all(torch.isfinite(p).all() for p in b._parameters(h))


def test_fingerprint_large_tensor_endpoint(backend):
    # Above float32's exact-integer range, matching the real embedding failure.
    large = torch.nn.Parameter(torch.zeros(16_777_220, dtype=torch.float16), requires_grad=False)
    backend.model.register_parameter('large_frozen_tensor', large)
    before = backend.frozen_fingerprint()
    assert before == backend.frozen_fingerprint()
    with torch.no_grad():
        large[-1] = 1
    assert before != backend.frozen_fingerprint()


def test_effective_delta_norm_detects_finite_damage_and_is_factor_invariant(backend):
    b = backend
    h = b.train_expert([('x', 'Z')], steps=1, seed=1)
    ratio = b.adapter_delta_ratio(h)
    with torch.no_grad():
        dense = max(float((2 * layer.adapters[h]['b'] @ layer.adapters[h]['a']).norm()
                          / layer.base.weight.float().norm()) for layer in b.layers.values())
        assert ratio == pytest.approx(dense, rel=1e-5)
        for layer in b.layers.values():
            layer.adapters[h]['a'].mul_(2)
            layer.adapters[h]['b'].div_(2)
    assert b.adapter_delta_ratio(h) == pytest.approx(ratio)
    b.next_broken = True
    broken = b.train_expert([('x', 'Z')], steps=1, seed=2)
    assert b.is_finite(broken) and b.adapter_delta_ratio(broken) > 1


def test_inference_cap_is_wider_than_training_cap(backend):
    b = backend
    long = 'z' * 300
    assert len(b._ids(long)) == 112 == b.train_cap  # training/replay truncation is unchanged
    assert len(b._ids(long, b.infer_cap)) == 300  # a fair 512-token context is not cut at 112
    assert b.generate('x', max_new_tokens=2) is not None
    assert ('x', None, 2) in b._cache and ('x', None, 16) not in b._cache


def test_answer_prefix_is_prefilled_and_returned(backend):
    seen = []
    real = backend.model.generate

    def spy(input_ids, **kwargs):
        seen.append(input_ids[0].tolist())
        return real(input_ids=input_ids, **kwargs)

    backend.model.generate = spy
    plain = backend.generate('Q?')
    backend.answer_prefix = 'Final answer:'
    backend._cache.clear()
    prefixed = backend.generate('Q?')
    prefix_ids = backend.tokenizer.encode('Final answer:')
    assert seen[1][-len(prefix_ids):] == prefix_ids
    assert seen[1][:-len(prefix_ids)] == seen[0]
    assert prefixed.startswith('Final answer:') and not plain.startswith('Final answer:')
