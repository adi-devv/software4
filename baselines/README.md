# Text-memory baselines

These arms implement build step 4 of the evaluation harness.
Import `NoMemory`, `Rag`, `LongContext`, `TextNotes`, or `HippoOnly` from `baselines`.
Construct an arm with an LLM and optional `context_tokens` and `seed`.
The default context budget is 3000 tokens, measured by the selected LLM.
Pass the arm to `harness.runner.run_arm` with a stream and result path.
`FakeLLM` is a deterministic offline prompt reader for tests.
`HFLLM` lazily loads Qwen/Qwen3.5-2B with transformers and fp16.
HF prompts and token counts use the tokenizer's chat template and assistant prefix,
with thinking disabled when the template supports it.
`input_limit` bounds formatted distillation inputs (HF default: 32768; Fake: 1000000).
Example: `arm = Rag(HFLLM(), context_tokens=3000, k=5)`.
Run `python -m pytest baselines/tests -q` for the offline tests.
Run `python -m pytest harness/tests/test_runner.py -q` for harness regression tests.
Teach statements and day observations are the only learning inputs.
Test answers and observations do not update the saved state.
RAG uses the offline hashing embedder by default, not semantic embeddings.
Long context retains the newest contiguous sequence of complete episodes.
Notes are generated at sleep, deduplicated, capped, and pruned by outcomes.
Distillation batches whole episodes to fit `input_limit`; an oversized single episode
raises `ValueError` before changing state.
All arms use the same Context/Task format, including an empty context when needed.
Selected memories appear chronologically, with the newest evidence last.
Hippo-only uses the in-memory hippocampus log and naps; it trains no weights.
Checkpoints contain memory state, not model weights or LLM configuration.
HF inference and note quality have not been evaluated on a real model or GPU.
This package does not implement experts, training, gates, or the other arms.
