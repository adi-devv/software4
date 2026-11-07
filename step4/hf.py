"""Small compatibility layer; HFBackend remains the training/generation implementation."""
import os
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path


def reference_kernel_environment():
    """Version-independent refusal policy, including libraries used by newer hub dispatch."""
    distributions = {
        "fla": ("flash-linear-attention", "fla-core", "fla"),
        "causal_conv1d": ("causal-conv1d",),
        "kernels": ("kernels",),
    }
    libraries = {}
    for module, names in distributions.items():
        versions = {}
        for name in names:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        libraries[module] = dict(importable=importlib.util.find_spec(module) is not None,
                                 distributions=versions)
    return dict(policy="torch_reference_only", transformers=importlib.metadata.version("transformers"),
                optional_libraries=libraries,
                status="refused" if any(v["importable"] or v["distributions"]
                                        for v in libraries.values()) else "verified")


def require_reference_kernels(out):
    environment = reference_kernel_environment()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel_environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    if environment["status"] != "verified":
        raise RuntimeError("reference DeltaNet requires no FLA, causal-conv1d or hub kernels; "
                           "optional libraries detected; inspect kernel_environment.json and use a clean environment")
    return environment


def assert_reference_deltanet(model, *, qwen=None):
    """Assert instance dispatch, or select the undecorated Torch global functions."""
    if qwen is None:
        import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    selected = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, qwen.Qwen3_5GatedDeltaNet):
            continue
        if type(layer.norm) is not qwen.Qwen3_5RMSNormGated:
            raise RuntimeError("DeltaNet gated norm is not the Transformers reference")
        # v5.3.0 binds these on each instance. Newer releases use module globals.
        if hasattr(layer, "chunk_gated_delta_rule"):
            expected = dict(causal_conv1d_fn=None,
                            causal_conv1d_update=qwen.torch_causal_conv1d_update,
                            chunk_gated_delta_rule=qwen.torch_chunk_gated_delta_rule,
                            recurrent_gated_delta_rule=qwen.torch_recurrent_gated_delta_rule)
            if any(getattr(layer, key, object()) is not value for key, value in expected.items()):
                raise RuntimeError("DeltaNet instance selected a non-reference function")
            selected[name] = {key: None if value is None else value.__module__ + "." + value.__name__
                              for key, value in expected.items()}
        else:
            # Newer releases decorate globals with optional-package/hub dispatch.
            # Select their original Torch bodies, even when a decorator preserves
            # the reference function's name/module through functools.wraps.
            functions = {}
            for key in ("causal_conv1d_fn", "causal_conv1d_update",
                        "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
                reference = inspect.unwrap(getattr(qwen, key))
                if (reference.__module__ != qwen.__name__ or reference.__name__ != key or
                    getattr(reference, "__code__", None) is None or
                    Path(reference.__code__.co_filename).resolve() != Path(qwen.__file__).resolve()):
                    raise RuntimeError(f"DeltaNet global {key} is not the Transformers reference")
                functions[key] = reference
            for key, reference in functions.items():
                setattr(qwen, key, reference)
            selected[name] = {key: value.__module__ + "." + value.__name__
                              for key, value in functions.items()}
        selected[name]["norm"] = type(layer.norm).__module__ + "." + type(layer.norm).__name__
    if not selected:
        raise RuntimeError("hybrid model has no reference DeltaNet layers")
    return selected


def make_hf_backend(cfg, out, *, model=None, tokenizer=None):
    environment = require_reference_kernels(out)
    import torch
    from torch import nn
    from baselines.sleep_hf import HFBackend, LoRALinear

    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = "cpu" if cfg["backend"] == "tiny" else "cuda"
    if device == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("each HF worker must see exactly one GPU")

    if cfg["backend"] == "tiny":
        if model is None:
            from transformers import LlamaConfig, LlamaForCausalLM
            model = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=16,
                        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2, max_position_embeddings=256,
                        bos_token_id=1, eos_token_id=2, pad_token_id=2))
        class ByteTokenizer:
            chat_template = ""
            eos_token_id = 2
            def encode(self, text, add_special_tokens=False):
                return [3 + b % 253 for b in text.encode("utf-8")]
            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(int(i)) for i in ids if int(i) > 2)
        tokenizer = tokenizer or ByteTokenizer()
    else:
        # Imports occur only after setup_runtime pinned the environment and cached weights.
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        os.environ["HF_HUB_OFFLINE"] = "1"
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"], local_files_only=True)
        config = AutoConfig.from_pretrained(cfg["model"], revision=cfg["revision"], local_files_only=True)
        # v5.3.0 explicitly registers qwen3_5 as a VLM compatibility entry here.
        # AutoFactory extracts text_config and loads the text-only causal class.
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], config=config,
                    revision=cfg["revision"], local_files_only=True,
                    dtype=torch.float16, attn_implementation="eager")

    class CheckedHFBackend(HFBackend):
        def train_expert(self, examples, steps, seed):
            handle = super().train_expert(examples, steps, seed)
            call = self.train_calls[-1]
            if not call["finite"] or call["skipped_steps"] or call["optimizer_steps"] != steps:
                detail = {k: call[k] for k in ("loss", "skipped_steps", "optimizer_steps",
                                               "nonfinite_gradient_steps", "finite", "steps")}
                raise RuntimeError("nonfinite/skipped optimizer step; dependent runs blocked: "
                                   + json.dumps(detail, default=str))
            return handle

    backend = CheckedHFBackend(model=model, tokenizer=tokenizer, device=device,
                               directory=out / "adapters", lr=cfg["lr"])
    backend.kernel_environment = environment
    if cfg["backend"] == "hf":
        backend.answer_prefix = "Final answer:"  # same prefill for every arm; see HFBackend.answer_prefix
    text_cfg = getattr(model.config, "text_config", model.config)
    if getattr(text_cfg, "model_type", "") == "qwen3_5_text":
        environment["deltanet_functions"] = assert_reference_deltanet(model)
        (out / "kernel_environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
        # Existing q_proj/v_proj wrappers cover full attention. Add two linear
        # projections per DeltaNet, using precisely the SAME fp32 LoRA class.
        for name, module in list(model.named_modules()):
            if ".linear_attn." in name and name.rsplit(".", 1)[-1] in ("in_proj_qkv", "out_proj"):
                if isinstance(module, nn.Linear):
                    parent, _, attr = name.rpartition(".")
                    wrapper = LoRALinear(module)
                    setattr(model.get_submodule(parent), attr, wrapper)
                    backend.layers[name] = wrapper
        expected = sum(2 for kind in text_cfg.layer_types if kind in ("linear_attention", "full_attention"))
        if len(backend.layers) != expected:
            raise RuntimeError(f"hybrid LoRA target mismatch: expected {expected}, got {list(backend.layers)}")

    def check_output(module, args, output):
        for name in ("logits", "loss"):
            value = getattr(output, name, None)
            if value is not None and not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"nonfinite {name} in fp16 forward; stop and review precision")
    model.register_forward_hook(check_output)
    return backend
