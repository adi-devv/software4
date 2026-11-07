# Qwen3.5-2B compatibility finding and proposed upstream patch

Config-only hub fetch: [config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json).
The exact response and its SHA-256/revision are in `qwen35-config*.json`. No weights were downloaded locally.

The 24 text layers repeat **linear, linear, linear, full** six times. Zero-based full-attention
indices are **3, 7, 11, 15, 19, 23**; all other indices are Gated DeltaNet linear attention.

Names are verified against [Transformers v5.3.0 source](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py), not inferred from config keys:

| Block | Linear modules below the decoder layer | Lean adapter targets |
|---|---|---|
| Full attention | `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` | `self_attn.q_proj`, `self_attn.v_proj` |
| Gated DeltaNet | `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` | `linear_attn.in_proj_qkv`, `linear_attn.out_proj` |
| Both FFNs | `mlp.gate_proj`, `up_proj`, `down_proj` | none in this lean run |

Paths begin `model.language_model.layers.<i>.` with the multimodal loader, or `model.layers.<i>.`
with the text-only class. `linear_attn.conv1d` is a depthwise convolution, not an `nn.Linear`.
The full-attention `q_proj` also emits its output gate; the fused DeltaNet QKV projection
adapts Q, K and V together. These targets are a small, explicit design choice, not a claim of optimality.

**Existing HFBackend partially works:** its `q_proj`/`v_proj` selector finds 12 projections in the
six full-attention layers, but leaves all 18 DeltaNet blocks unadapted. For this hybrid-target run,
add 36 projections, giving **48 rank-four LoRA wrappers**, with the existing fp32 adapter branch.
`step4/hf.py` makes this extension at runtime using imported `HFBackend` and `LoRALinear`.
The pinned [Auto mapping](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/auto/modeling_auto.py)
registers `qwen3_5` for causal-LM compatibility; its factory extracts `text_config`, so the existing
`AutoModelForCausalLM` loader **does work** in v5.3.0. The package uses this text-only path to avoid
loading the vision tower. HFBackend's offline cache requirement also remains intact.
No change to `baselines/sleep_hf.py` is needed to execute the prepared package.

The following is the proposed upstream equivalent, **for review only; not applied**. Rebase
against sleepC's final file before applying. Offline loading remains intentional: download into
the HF cache during the Kaggle setup stage, before importing the smoke helpers.

```diff
--- a/baselines/sleep_hf.py
+++ b/baselines/sleep_hf.py
@@ -60,14 +60,20 @@
         self.model = model.to(device=self.device, dtype=torch.float16).eval()
         self.model.requires_grad_(False)
         self.layers = {}
+        text_config = getattr(self.model.config, 'text_config', self.model.config)
+        hybrid = getattr(text_config, 'model_type', '') == 'qwen3_5_text'
         for name, module in list(self.model.named_modules()):
-            if name.rsplit('.', 1)[-1] in ('q_proj', 'v_proj') and isinstance(module, nn.Linear):
+            leaf = name.rsplit('.', 1)[-1]
+            target = leaf in ('q_proj', 'v_proj')
+            if hybrid and '.linear_attn.' in name:
+                target = leaf in ('in_proj_qkv', 'out_proj')
+            if target and isinstance(module, nn.Linear):
                 parent, _, attr = name.rpartition('.')
                 wrapper = LoRALinear(module)
                 setattr(self.model.get_submodule(parent), attr, wrapper)
                 self.layers[name] = wrapper
         if not self.layers:
-            raise ValueError('model has no q_proj/v_proj linear layers')
+            raise ValueError('model has no selected LoRA linear layers')
         self.experts = {}
         self.train_calls = []
         self._next = 0
```

Thinking: the [model card](https://huggingface.co/Qwen/Qwen3.5-2B#quickstart) says 2B defaults
to non-thinking. Keep **`enable_thinking=False`** in `apply_chat_template` explicitly when supported;
HFBackend already detects that option. Thinking can otherwise consume its 16-token answer cap
and invalidate exact-answer comparisons. Log/inspect a formatted prompt in the GPU review.

Precision: the config declares **bfloat16**, whereas T4 uses **fp16** here. fp16's finite range
ends at 65504; overflow can happen in base activations, recurrent updates or adapter gradients.
FP32 LoRA, fp32 reference DeltaNet math and GradScaler help, but GradScaler cannot repair an
already nonfinite forward. The package checks logits/loss, adapter finiteness and skipped steps,
and fails the arm on any violation. No fp16 stability claim is made before the actual T4 smoke.
Eager selects full attention only. `step4/hf.py` separately refuses installed/importable FLA,
causal-conv1d and hub-kernels libraries, records their detected versions, and asserts the
5.3.0 DeltaNet reference functions/norm. Newer global dispatch selects the original Torch
function bodies. See the README for the version-independent refusal tests and CPU checks.
Reference-path speed still needs measurement.
