"""Offline HF backend: frozen fp16 core and independent fp32 rank-four adapters."""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.active = None

    def add(self, name):
        pair = nn.ParameterDict({
            'a': nn.Parameter(torch.empty(4, self.base.in_features,
                                          device=self.base.weight.device, dtype=torch.float32)),
            'b': nn.Parameter(torch.zeros(self.base.out_features, 4,
                                          device=self.base.weight.device, dtype=torch.float32))})
        nn.init.kaiming_uniform_(pair['a'], a=math.sqrt(5))
        self.adapters[name] = pair

    def forward(self, x):
        y = self.base(x)
        if self.active is not None:
            p = self.adapters[self.active]
            # Keep the trainable branch in fp32, even inside the base autocast.
            with torch.autocast(x.device.type, enabled=False):
                delta = F.linear(F.linear(x.float(), p['a']), p['b']) * 2.0
            y = y + delta.to(y.dtype)
        return y


class HFBackend:
    input_limit = 128
    train_cap = 112       # left-truncation for training prompts (unchanged)
    infer_cap = 1024      # left-truncation at inference, so 512-token contexts fit
    # Optional assistant-turn prefill, identical for every arm. Matches the "Final answer: <x>" training
    # target, so a chatty base model cannot run out of its 16 tokens before writing the gradable line.
    answer_prefix = ""

    def __init__(self, model=None, tokenizer=None, *, device='cuda', directory=None, lr=2e-4):
        self.device = torch.device(device)
        self.lr = lr
        self.directory = Path(directory or tempfile.mkdtemp(prefix='hf-experts-'))
        self.directory.mkdir(parents=True, exist_ok=True)
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            name = os.environ.get('S4_MODEL', 'Qwen/Qwen3-0.6B')
            tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                name, local_files_only=True, torch_dtype=torch.float16,
                attn_implementation='eager')
        self.tokenizer = tokenizer
        self.model = model.to(device=self.device, dtype=torch.float16).eval()
        self.model.requires_grad_(False)
        self.layers = {}
        for name, module in list(self.model.named_modules()):
            if name.rsplit('.', 1)[-1] in ('q_proj', 'v_proj') and isinstance(module, nn.Linear):
                parent, _, attr = name.rpartition('.')
                wrapper = LoRALinear(module)
                setattr(self.model.get_submodule(parent), attr, wrapper)
                self.layers[name] = wrapper
        if not self.layers:
            raise ValueError('model has no q_proj/v_proj linear layers')
        self.experts = {}
        self.train_calls = []
        self._next = 0
        self.next_broken = False
        self._cache = {}  # exact prompt + immutable adapter handle; never caches training

    def _activate(self, handle):
        if handle is not None and handle not in self.experts:
            raise KeyError(handle)
        for layer in self.layers.values():
            layer.active = handle

    def _format(self, prompt):
        template = getattr(self.tokenizer, 'chat_template', '') or ''
        if not template:
            return prompt
        options = {'enable_thinking': False} if 'enable_thinking' in template else {}
        return self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True, **options)

    def _ids(self, prompt, cap=None):
        return self.tokenizer.encode(self._format(prompt), add_special_tokens=False)[-(cap or self.train_cap):]

    def count_tokens(self, text):
        return len(self.tokenizer.encode(self._format(text), add_special_tokens=False))

    def _autocast(self):
        return torch.autocast('cuda', dtype=torch.float16) if self.device.type == 'cuda' else nullcontext()

    def generate(self, prompt, expert=None, max_new_tokens=16):
        key = (prompt, expert, max_new_tokens)
        if expert is not None and expert not in self.experts:
            raise KeyError(expert)
        if key in self._cache:
            return self._cache[key]
        self._activate(expert)
        self.model.eval()
        prompt_ids = self._ids(prompt, self.infer_cap)
        if self.answer_prefix:
            prompt_ids = prompt_ids + self.tokenizer.encode(self.answer_prefix, add_special_tokens=False)
        ids = torch.tensor([prompt_ids], device=self.device)
        try:
            with torch.inference_mode(), self._autocast():
                output = self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id)
            answer = self.answer_prefix + self.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
            self._cache[key] = answer
            return answer
        finally:
            self._activate(None)

    def _parameters(self, handle):
        return [p for layer in self.layers.values() for p in layer.adapters[handle].values()]

    def _state(self, handle):
        return {name: {k: p.detach().cpu().clone() for k, p in layer.adapters[handle].items()}
                for name, layer in self.layers.items()}

    def _write(self, handle):
        torch.save(self._state(handle), self.directory / (handle + '.pt'))

    def train_expert(self, examples, steps, seed):
        if not examples or steps < 1:
            raise ValueError('nonempty examples and positive steps required')
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        rng = random.Random(seed)
        handle = f'expert-{self._next}'
        self._next += 1
        self.experts[handle] = {'finite': True, 'committed': False}
        for layer in self.layers.values():
            layer.add(handle)
            for name, pair in layer.adapters.items():
                pair.requires_grad_(name == handle)
        self._activate(handle)
        params = self._parameters(handle)
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == 'cuda', init_scale=256)
        encoded = []
        for prompt, target in examples:
            q = self._ids(prompt)
            a = self.tokenizer.encode(target, add_special_tokens=False)
            a = (a + [self.tokenizer.eos_token_id])[:16]
            encoded.append((torch.tensor([q + a], device=self.device),
                            torch.tensor([[-100] * len(q) + a], device=self.device)))
        start = time.perf_counter()
        losses, skipped, completed, bad_gradients = [], 0, 0, 0
        try:
            for _ in range(steps):
                batch = [encoded[rng.randrange(len(encoded))] for _ in range(4)]
                # fp16 loss-scale calibration: a nonfinite gradient at the current scale is
                # normal AMP behaviour. Halve the scale and redo the same batch; only a
                # nonfinite loss, or no finite gradient after every backoff, is a skipped step.
                done = False
                for _attempt in range(9 if scaler.is_enabled() else 1):
                    optimizer.zero_grad(set_to_none=True)
                    loss_ok = True
                    for ids, labels in batch:
                        with self._autocast():
                            loss = self.model(input_ids=ids, labels=labels, use_cache=False).loss
                        if not torch.isfinite(loss):
                            loss_ok = False
                            self.experts[handle]['finite'] = False
                            break
                        losses.append(float(loss.detach()))
                        scaler.scale(loss / 4).backward()
                    if not loss_ok:
                        break
                    scaler.unscale_(optimizer)
                    grads_ok = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
                    if grads_ok:
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        completed += 1
                        done = True
                        break
                    bad_gradients += 1
                    if scaler.is_enabled():
                        scaler.step(optimizer)  # records the skip so update() halves the scale
                        scaler.update()
                skipped += int(not done)
            if self.next_broken:
                # Finite fault injection: destroy only the new adapter, never the base.
                with torch.no_grad():
                    for layer in self.layers.values():
                        layer.adapters[handle]['b'].normal_(0, 100)
                self.next_broken = False
            self.experts[handle].update(loss=losses[-1] if losses else None,
                                        skipped_steps=skipped, optimizer_steps=completed)
            self._write(handle)
            self.train_calls.append({'handle': handle, 'examples': list(examples),
                                     'steps': steps, 'seed': seed, 'loss': losses[-1] if losses else None,
                                     'skipped_steps': skipped, 'optimizer_steps': completed,
                                     'nonfinite_gradient_steps': bad_gradients,
                                     'finite': self.is_finite(handle),
                                     'wall_seconds': time.perf_counter() - start})
            return handle
        finally:
            self._activate(None)
            for p in params:
                p.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            del optimizer, encoded

    def commit_expert(self, handle):
        self.experts[handle]['committed'] = True
        self._write(handle)

    def discard_expert(self, handle):
        self._activate(None)
        for layer in self.layers.values():
            del layer.adapters[handle]
        del self.experts[handle]
        (self.directory / (handle + '.pt')).unlink(missing_ok=True)
        self._cache = {k: v for k, v in self._cache.items() if k[1] != handle}
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def is_finite(self, handle):
        return self.experts[handle]['finite'] and all(
            bool(torch.isfinite(p).all()) for p in self._parameters(handle))

    def adapter_delta_ratio(self, handle):
        """Largest ||2BA||_F / ||W||_F, computed with rank-four Gram matrices.

        Unlike separate factor norms this is invariant to A/B rescaling. No
        dense delta or extra model activations are allocated on the GPU.
        """
        ratios = []
        with torch.no_grad():
            for layer in self.layers.values():
                a, b = (layer.adapters[handle][k] for k in ('a', 'b'))
                squared = ((a @ a.T) * (b.T @ b)).sum().clamp_min(0)
                delta = 2 * squared.sqrt()
                base = layer.base.weight.float().norm().clamp_min(1e-12)
                ratios.append(float(delta / base))
        return max(ratios)

    def frozen_fingerprint(self):
        h = hashlib.sha256()
        for name, tensor in self.model.named_parameters():
            if '.adapters.' in name:
                continue
            flat = tensor.detach().flatten()
            n = min(16, flat.numel())
            # Integer arithmetic: float32 linspace rounds large embedding endpoints
            # up to numel(), producing an out-of-bounds CUDA index.
            index = torch.arange(n, device=flat.device) * (flat.numel() - 1) // max(1, n - 1)
            h.update(name.encode())
            h.update(flat[index].cpu().numpy().tobytes())
        return h.hexdigest()

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for handle in self.experts:
            torch.save(self._state(handle), path / (handle + '.pt'))
        (path / 'backend.json').write_text(json.dumps({
            'experts': self.experts, 'next': self._next,
            'fingerprint': self.frozen_fingerprint()}), encoding='utf-8')

    def load(self, path):
        path = Path(path)
        state = json.loads((path / 'backend.json').read_text(encoding='utf-8'))
        if state['fingerprint'] != self.frozen_fingerprint():
            raise ValueError('frozen model fingerprint differs')
        # Read first: loading one's own working directory must also be safe.
        weights = {h: torch.load(path / (h + '.pt'), map_location='cpu', weights_only=True)
                   for h in state['experts']}
        for handle in list(self.experts):
            self.discard_expert(handle)
        self.experts, self._next = state['experts'], state['next']
        with torch.no_grad():
            for handle, weight in weights.items():
                for name, layer in self.layers.items():
                    layer.add(handle)
                    for k, p in layer.adapters[handle].items():
                        p.copy_(weight[name][k])
                        p.requires_grad_(False)
                self._write(handle)
        self._cache.clear()
