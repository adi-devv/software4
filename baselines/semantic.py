"""Local-only sentence embeddings for the v2 semantic RAG comparator.

Only the stored text/query text supplied by Rag is encoded, unchanged: no skill,
family, source, gold labels, query prefixes, or other metadata are added. Normal
model tokenization and truncation at 256 tokens still apply. Rag retains ownership
of top-k, context budget, tie-breaking, and storage policy.

Runtime dependencies: numpy, torch, transformers (repo version: 5.3.0), safetensors.
sentence_transformers is not required. Heavy imports happen on first nonempty
embed(), never on module import or when using StubEncoder. Inference is float32,
CPU-only, eval/inference mode, eager attention, masked mean pooling, then L2
normalization. Repeated calls are deterministic on a fixed CPU/library stack;
cross-platform bitwise identity is not promised.

Pins verified from publisher metadata (no weights downloaded during development):
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/model.safetensors

Future Kaggle setup (external preparation, never executed by this module):
  python -m pip install 'numpy>=1.26,<3' 'transformers==5.3.0' torch safetensors huggingface_hub
  hf download sentence-transformers/all-MiniLM-L6-v2 --revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf --include config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt --local-dir /kaggle/working/semantic-model
Upload that directory as <SEMANTIC_MODEL_DATASET>, attach it to the experiment,
and set S4_SEMANTIC_MODEL_PATH=/kaggle/input/<SEMANTIC_MODEL_DATASET>/semantic-model
(adjust for the dataset's actual directory layout). The loader verifies the
weights, but config/tokenizer provenance relies on this pinned snapshot setup.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from typing import Sequence

import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
WEIGHTS_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
WEIGHTS_FILENAME = "model.safetensors"
MODEL_PATH_ENV = "S4_SEMANTIC_MODEL_PATH"
EMBEDDING_DIM = 384
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt")


class StubEncoder:
    """Deterministic, nonsemantic test double; never use for reported v2 scores.

    Implements encode(texts) -> float32 array and exposes dim, the minimal
    injection protocol accepted by SemanticEmbedder's explicit test path.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        from hippocampus import HashingEmbedder

        return HashingEmbedder(dim=self.dim).embed(texts)


class SemanticEmbedder:
    """HashingEmbedder-compatible embed(texts) -> normalized float32 (n, dim).

    model_path must be a local snapshot directory, or supplied via
    S4_SEMANTIC_MODEL_PATH. Hub identifiers are never resolved/downloaded.
    allow_unpinned permits TODO_PIN/absent pins for development only; a known
    hash is ALWAYS checked, even with that override. An injected encoder also
    requires this explicit override and skips all real-model loading/checks.
    """

    def __init__(self, model_path: str | os.PathLike | None = None, *,
                 allow_unpinned: bool = False, encoder=None,
                 batch_size: int = 32, max_length: int = 256):
        if batch_size <= 0 or not 1 <= max_length <= 256:
            raise ValueError("batch_size must be positive and max_length in [1, 256]")
        self.batch_size, self.max_length = batch_size, max_length
        self._encoder = encoder
        self._model = self._tokenizer = None
        self.model_name, self.revision = MODEL_NAME, MODEL_REVISION
        self.weights_sha256 = WEIGHTS_SHA256
        self.dim = EMBEDDING_DIM
        self.model_path = None
        if encoder is not None:
            if not allow_unpinned:
                raise ValueError("test encoder requires allow_unpinned=True")
            self.dim = encoder.dim
            if not isinstance(self.dim, int) or self.dim <= 0:
                raise ValueError("encoder.dim must be a positive integer")
            return
        pinned = (re.fullmatch(r"[0-9a-f]{40}", self.revision or "") and
                  re.fullmatch(r"[0-9a-f]{64}", self.weights_sha256 or ""))
        if not pinned and not allow_unpinned:
            raise ValueError("TODO_PIN or missing model pins; use allow_unpinned=True only for development")
        location = model_path if model_path is not None else os.environ.get(MODEL_PATH_ENV)
        if not location:
            raise ValueError(f"provide local model_path or {MODEL_PATH_ENV}; downloads are disabled")
        self.model_path = Path(location).expanduser().resolve()
        if not self.model_path.is_dir():
            raise ValueError(f"local model directory does not exist: {self.model_path}")
        self._verify_weights()

    def _verify_weights(self):
        weights = self.model_path / WEIGHTS_FILENAME
        if not weights.is_file():
            raise ValueError(f"missing local weights file: {weights}")
        if re.fullmatch(r"[0-9a-f]{64}", self.weights_sha256 or ""):
            digest = hashlib.sha256()
            with weights.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != self.weights_sha256:
                raise ValueError(f"SHA256 mismatch for {weights}; expected {self.weights_sha256}")

    def _load(self):
        if self._model is not None:
            return
        # Recheck immediately before loading, in case the snapshot changed after
        # construction. Selecting safetensors excludes alternate .bin/shard files.
        self._verify_weights()
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = str(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False)
        model = AutoModel.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32, attn_implementation="eager")
        model = model.to("cpu").eval()
        if model.config.hidden_size != self.dim:
            raise ValueError(f"expected embedding dimension {self.dim}, got {model.config.hidden_size}")
        self._tokenizer, self._model = tokenizer, model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("embed accepts a sequence of strings")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._encoder is not None:
            vectors = self._encoder.encode(texts)
        else:
            self._load()
            import torch

            batches = []
            with torch.inference_mode():
                for start in range(0, len(texts), self.batch_size):
                    inputs = self._tokenizer(
                        texts[start:start + self.batch_size], padding=True,
                        truncation=True, max_length=self.max_length, return_tensors="pt")
                    inputs = {key: value.to("cpu") for key, value in inputs.items()}
                    hidden = self._model(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    batches.append(pooled.cpu().numpy())
            vectors = np.concatenate(batches, axis=0)
        vectors = np.array(vectors, dtype=np.float32, copy=True)
        if vectors.shape != (len(texts), self.dim) or not np.isfinite(vectors).all():
            raise ValueError("encoder returned invalid embedding shape or nonfinite values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=vectors, where=norms > 0)
