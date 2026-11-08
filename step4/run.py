"""Lean step-4 driver. No Kaggle API/CLI calls. Default: smoke, never full.

Run prepare.py to freeze repository imports into a single uploadable script.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

# prepare.py replaces only these two literals in its NEW upload artifact.
SOURCE_B64 = ""
DEFAULT_SMOKE = "1"
DEFAULT_ARMS = ""   # "" keeps the default ARMS; prepare.py --arms a,b bakes a list in
DEFAULT_WORLD_SEED = ""   # "" keeps the harness default (41); prepare.py --world-seed N bakes one in
# `ours` first, then the other long-running weight arm; the inference-only baselines follow.
# Nothing waits for anything else: `ours` is the long pole and must start immediately.
ARMS = ("ours", "expert_only", "hippo_only", "rag", "text_notes", "no_memory", "rag_starved", "oracle")
AVAILABLE_ARMS = ARMS + ("ours_salient",)
FAIR_CONTEXT_TOKENS = 512    # every context-using arm; `rag_starved` alone keeps the old 64
STARVED_CONTEXT_TOKENS = 64
FIT_THRESHOLD = 0.9          # forced-expert train accuracy below this = optimisation failure
RULE_MARGIN = 0.15           # pre-registered stop rule, see step4/README.md
A_DROP_TOLERANCE = 0.05
QWEN35_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def source_root():
    if SOURCE_B64:
        data = base64.b64decode(SOURCE_B64)
        key = hashlib.sha256(data).hexdigest()[:16]
        root = Path(os.environ.get("S4_SOURCE_CACHE", "/kaggle/working/step4-source")) / key
        if not (root / "READY").exists():
            root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                archive.extractall(root)  # archive is generated from an explicit local allowlist
            (root / "READY").write_text(key, encoding="utf-8")
        return root
    return Path(__file__).resolve().parents[1]


def source_digest(root):
    h = hashlib.sha256()
    paths = [p for folder in ("baselines", "harness", "hippocampus")
             for p in (root / folder).glob("*.py")]
    paths += [root / "scripts/smoke_sleep.py", root / "step4/hf.py", root / "step4/run.py"]
    for path in sorted(paths):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def settings(root):
    backend = os.environ.get("S4_BACKEND", "hf")
    arms = os.environ.get("S4_ARMS", DEFAULT_ARMS or ",".join(ARMS)).split(",")
    arms = [a.strip() for a in arms if a.strip()]
    cycles = int(os.environ.get("S4_CYCLES", "3"))
    steps = int(os.environ.get("S4_STEPS", "64"))
    if backend not in ("fake", "tiny", "hf") or not arms or len(set(arms)) != len(arms):
        raise ValueError("invalid backend or empty/duplicate S4_ARMS")
    if set(arms) - set(AVAILABLE_ARMS) or cycles < 3 or steps < 1:
        raise ValueError("S4_ARMS must use " + ",".join(AVAILABLE_ARMS) + "; cycles >= 3; steps >= 1")
    model = os.environ.get("S4_MODEL", "Qwen/Qwen3.5-2B")
    revision = os.environ.get("S4_REVISION", QWEN35_REVISION if model == "Qwen/Qwen3.5-2B" else "main")
    from harness.step4_world import WORLD_SEED
    world_seed = int(os.environ.get("S4_WORLD_SEED", DEFAULT_WORLD_SEED) or WORLD_SEED)
    cfg = dict(backend=backend, model=model, revision=revision,
               cycles=cycles, steps=steps, seed=0, context_tokens=FAIR_CONTEXT_TOKENS,
               starved_tokens=STARVED_CONTEXT_TOKENS, lr=2e-4, world="step4", world_seed=world_seed,
               source_sha256=source_digest(root), schema=3,
               hardware_mode=("single_device_dev" if os.environ.get("S4_SINGLE_DEVICE_DEV") == "1"
                              else "dual_t4") if backend == "hf" else "cpu")
    # Arms and mode are deliberately excluded: independent arm outputs can be reused.
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    return cfg, arms, key


def validate_topology(hardware, *, allow_dev=False):
    mode = hardware.get("mode")
    devices = hardware.get("devices", [])
    expected = 1 if allow_dev and mode == "single_device_dev" else 2
    if mode not in (("dual_t4", "single_device_dev") if allow_dev else ("dual_t4",)):
        raise RuntimeError("two distinct T4 devices required; development receipts are ineligible")
    if len(devices) != expected or len({d["uuid"] for d in devices}) != expected:
        raise RuntimeError("two distinct T4 devices required (one only with S4_SINGLE_DEVICE_DEV=1)")
    if any(d["name"] not in ("Tesla T4", "NVIDIA T4", "T4") or
           not d["uuid"].startswith("GPU-") for d in devices):
        raise RuntimeError("selected devices must be physical NVIDIA T4 GPUs")


def detect_hardware(cfg):
    """Use the driver inventory before pip/downloads; workers receive UUIDs, never ambiguous ordinals."""
    if cfg["backend"] != "hf":
        return dict(mode="cpu", devices=[], inventory=[])
    proc = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,memory.total",
                           "--format=csv,noheader,nounits"], check=True, capture_output=True,
                          text=True, timeout=15)
    inventory = []
    for row in csv.reader(io.StringIO(proc.stdout), skipinitialspace=True):
        index, uuid, name, bus, memory = (cell.strip() for cell in row)
        inventory.append(dict(index=index, uuid=uuid, name=name, pci_bus_id=bus,
                              memory_mib=int(memory)))
    default = "0" if cfg["hardware_mode"] == "single_device_dev" else "0,1"
    selectors = [s.strip() for s in os.environ.get("CUDA_VISIBLE_DEVICES", default).split(",")]
    selected = []
    for selector in selectors:
        matches = [d for d in inventory if selector in (d["index"], d["uuid"])]
        if len(matches) != 1:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES selector {selector!r} does not identify one physical GPU")
        selected.append(matches[0])
    hardware = dict(mode=cfg["hardware_mode"], devices=selected, inventory=inventory)
    validate_topology(hardware, allow_dev=True)
    return hardware


def verify_worker_device(expected, cuda):
    if cuda.device_count() != 1:
        raise RuntimeError("each HF worker must see exactly one GPU")
    props = cuda.get_device_properties(0)
    observed = dict(uuid=str(getattr(props, "uuid", "")), name=props.name,
                    compute_capability=[props.major, props.minor], logical_device=0)
    assignment = dict(scheduled=expected, observed=observed,
                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    validate_assignment(assignment, [expected])
    return assignment


def bare_uuid(value):
    """nvidia-smi prints GPU-<uuid>; torch's device properties print the bare <uuid>."""
    value = str(value).lower()
    return value[4:] if value.startswith("gpu-") else value


def validate_assignment(assignment, devices):
    expected = assignment.get("scheduled", {})
    observed = assignment.get("observed", {})
    if (expected not in devices or assignment.get("cuda_visible_devices") != expected.get("uuid") or
        bare_uuid(observed.get("uuid", "")) != bare_uuid(expected.get("uuid", "!")) or
        observed.get("name") != expected.get("name") or
        observed.get("compute_capability") != [7, 5] or observed.get("logical_device") != 0):
        raise RuntimeError("worker device identity does not match the validated T4 assignment: "
                           + json.dumps(assignment, sort_keys=True))


def validate_execution(results, hardware, arms):
    validate_topology(hardware)
    used = set()
    for name in arms:
        result = results.get(name, {})
        if result.get("status") != "complete":
            raise RuntimeError("smoke is missing/failed for required arms; full run blocked")
        assignment = result.get("worker_device", {})
        validate_assignment(assignment, hardware["devices"])
        environment = result.get("kernel_environment", {})
        libraries = environment.get("optional_libraries", {})
        if (environment.get("policy") != "torch_reference_only" or environment.get("status") != "verified" or
            environment.get("transformers") != "5.3.0" or set(libraries) != {"fla", "causal_conv1d", "kernels"} or
            any(v.get("importable") is not False or v.get("distributions") != {} for v in libraries.values())):
            raise RuntimeError("receipt lacks verified pinned reference-kernel environment")
        used.add(assignment["scheduled"]["uuid"])
    if used != {d["uuid"] for d in hardware["devices"]}:
        raise RuntimeError("receipt must establish actual worker execution on both T4 devices")


def hardware_profile(hardware):
    return (hardware.get("mode"),
            sorted((d["name"], d["memory_mib"]) for d in hardware.get("devices", [])))


def validate_smoke_receipt(previous, cfg, arms, hardware):
    if previous.get("config") != cfg or previous.get("smoke") is not True:
        raise RuntimeError("smoke source/model/config mismatch; full run blocked")
    validate_topology(previous.get("hardware", {}))
    # Kaggle hands every session a fresh machine, so UUIDs never repeat. Match the hardware
    # profile (mode, count, model, memory); this run's own distinct-UUID topology is checked
    # by detect_hardware and every worker re-asserts its UUID.
    if hardware_profile(previous["hardware"]) != hardware_profile(hardware):
        raise RuntimeError("smoke hardware does not match current topology; rerun smoke on this hardware")
    validate_execution(previous.get("results", {}), previous["hardware"], set(arms) | {"ours"})


def setup_runtime(base, cfg, run_dir):
    """Only invoked by the real GPU driver on Kaggle, never by CPU tests."""
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("real HF run is Kaggle-only; use S4_BACKEND=fake or tiny on CPU")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "transformers==5.3.0", "accelerate==1.13.0", "numpy>=1.26,<3"], check=True)
    from step4.hf import require_reference_kernels
    environment = require_reference_kernels(run_dir)  # refuse before downloading weights
    # No PEFT or compiled FLA dependency: HFBackend owns the fp32 LoRA branch.
    # Outside /kaggle/working so the multi-GB weights are not saved as kernel output.
    os.environ.setdefault("HF_HOME", "/tmp/s4-hf-cache")
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download
    snapshot_download(cfg["model"], revision=cfg["revision"],
                      allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "*.safetensors"])
    return environment


def import_world(cfg):
    # This module forces HF_HUB_OFFLINE=1, intentionally: setup downloaded first.
    from scripts.smoke_sleep import Oracle, measure, prompt_for
    from harness.smoke_world import smoke_canaries
    from harness.step4_world import step4_stream
    stream = step4_stream(cfg["cycles"], cfg["world_seed"])  # fresh codes, >=100 hidden per rule
    stream.validate()
    return stream, smoke_canaries(), Oracle, measure, prompt_for


def make_backend(cfg, out):
    if cfg["backend"] == "fake":
        from baselines import FakeBackend
        from baselines.llm import FakeLLM
        backend = FakeBackend()
        # FakeBackend intentionally has no tokenizer; reuse FakeLLM's counter.
        backend.count_tokens = FakeLLM().count_tokens
        return backend
    from step4.hf import make_hf_backend
    return make_hf_backend(cfg, out)


class BackendLLM:
    """Distillation view of a sleep backend for TextNotes.

    HFBackend's own 128-token input_limit is its training cap; inference accepts far more, and a
    lesson needs more than the default 16 generated tokens. FakeBackend has neither knob.
    """
    distill_tokens = 64

    def __init__(self, backend, input_limit=480):
        import inspect
        self.backend, self.input_limit = backend, input_limit
        self._can_cap = "max_new_tokens" in inspect.signature(backend.generate).parameters

    def count_tokens(self, text):
        return self.backend.count_tokens(text)

    def generate(self, prompt, max_new_tokens=None):
        if self._can_cap and prompt.startswith("Distill lessons:"):
            return self.backend.generate(prompt, max_new_tokens=self.distill_tokens)
        return self.backend.generate(prompt)


def make_arm(name, backend, cfg, Oracle):
    from baselines import ExpertOnly, HippoOnly, NoMemory, Rag, SleepArm, SleepConfig, TextNotes
    budget = cfg["context_tokens"]
    common = dict(llm=backend, context_tokens=budget, seed=0)
    sleep_cfg = SleepConfig(steps=cfg["steps"], seed=0, context_tokens=budget)
    if name == "ours":
        return SleepArm(backend, sleep_cfg, llm=backend)
    if name == "ours_salient":
        from baselines.salient import SalientSleepArm
        return SalientSleepArm(backend, sleep_cfg, llm=backend)
    if name == "expert_only":
        return ExpertOnly(backend, sleep_cfg, llm=backend)
    if name == "rag_starved":
        return Rag(**{**common, "context_tokens": cfg["starved_tokens"]})
    if name == "text_notes":
        return TextNotes(**{**common, "llm": BackendLLM(backend)})
    return {"no_memory": NoMemory, "rag": Rag, "hippo_only": HippoOnly, "oracle": Oracle}[name](**common)


def longest_example(stream, backend, prompt_for):
    """Scan ONLY train items, using the backend's actual truncation and target cap."""
    from harness.tasks import grading_scope
    examples = []
    for cycle in range(stream.n_cycles):
        for item in stream.train_items(cycle):
            if item.kind != "probe":
                continue
            with grading_scope():
                target = "Final answer: " + str(item.expected_at(cycle))
            prompt = prompt_for(backend, item.task("day", cycle))
            if hasattr(backend, "_ids"):
                n = len(backend._ids(prompt)) + min(16, len(backend.tokenizer.encode(
                    target, add_special_tokens=False)) + 1)
            else:
                n = backend.count_tokens(prompt) + backend.count_tokens(target)
            examples.append((n, prompt, target, item.id))
    return max(examples, key=lambda x: x[0])


def smoke_arm(name, arm, backend, cfg, stream, canary, measure, prompt_for, out):
    from harness.tasks import arm_scope, grading_scope
    from harness.grading import grade
    # Full base-canary verification is small (8 items), and blocks the real run.
    verified = measure(canary.items(), -1, lambda t: backend.generate(prompt_for(backend, t)))
    write_json(out / "canary_verification.json", verified)
    if cfg["backend"] == "hf" and not all(r["correct"] for r in verified):
        wrong = [(r["task_id"], r["answer"][:200]) for r in verified if not r["correct"]]
        raise RuntimeError("base failed canaries; inspect this model/world before any full run: "
                           + json.dumps(wrong, ensure_ascii=True))
    longest = None
    expert = None
    if name in ("ours", "ours_salient", "expert_only"):
        n, prompt, target, tid = longest_example(stream, backend, prompt_for)
        # HFBackend accumulates FOUR microbatches per optimizer step. Repeating
        # the single longest actual example prevents random shorter selections.
        expert = backend.train_expert([(prompt, target)] * 4, steps=2, seed=0)
        if not backend.is_finite(expert):
            raise RuntimeError("smoke produced non-finite adapter")
        backend.save(Path(tempfile.tempdir) / "smoke-candidate")
        longest = dict(task_id=tid, encoded_tokens=n, optimizer_steps=2,
                       microbatch_size=1, accumulation=4, repeated_longest=True)
    else:
        for item in stream.train_items(0):
            if item.kind == "teach":
                with arm_scope():
                    arm.observe(item.task("day", 0), "", None, None)
        for item in [i for i in stream.train_items(0) if i.kind == "probe"][:2]:
            task = item.task("day", 0)
            with arm_scope():
                answer = arm.answer(task)
            with grading_scope():
                correct = bool(grade(item, answer, 0).correct)
            correction = stream.correction_for(item)
            with arm_scope():
                arm.observe(task, answer, correct, correction)
    rows = measure(stream.hidden_items(0)[:2], 0, lambda t:
                   backend.generate(prompt_for(backend, t), expert) if expert else arm.answer(t))
    return dict(smoke_only=True, longest_batch=longest, short_eval=rows,
                canary_verification=verified, comparison=None)


def routing_confusion(arm, stream, cycle):
    """Skill -> routed handle (or "none") counts over the hidden items of `cycle`, after its night.

    Routing reads only the task prompt (never the context), so this is exactly what hidden tests
    of `ours` and `expert_only` see. Arms without a router return None.
    """
    if not hasattr(arm, "route") or not hasattr(arm, "centroids"):
        return None
    table = {}
    for item in stream.hidden_items(cycle):
        handle = arm.route(item.task("test", cycle).prompt) or "none"
        row = table.setdefault(item.skill, {})
        row[handle] = row.get(handle, 0) + 1
    return table


def forced_train_accuracy(info):
    """Fraction of the night's train/replay examples the forced candidate expert answers right.

    None when the candidate never reached trusted validation (no example, norm cap or canary
    drift rejection). Below FIT_THRESHOLD this is an optimisation failure, not a design verdict.
    """
    validation = (info.get("pre_gate") or {}).get("validation")
    if not validation or not validation["examples"]:
        return None
    return validation["candidate_correct"] / len(validation["examples"])


def worker(name, cfg, run_dir, smoke):
    out = run_dir / name
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = dict(arm=name, status="failed", started_utc=datetime.now(timezone.utc).isoformat(),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), answer_log_failures=0)
    try:
        if cfg["backend"] == "hf":
            import torch
            result["worker_device"] = verify_worker_device(
                json.loads(os.environ["S4_WORKER_DEVICE"]), torch.cuda)
        # SleepArm's snapshots contain absolute paths. Keep them under working,
        # together with the JSON checkpoints, so kernel_sources can restore them.
        snapshots = out / "snapshots"
        snapshots.mkdir(exist_ok=True)
        tempfile.tempdir = str(snapshots.resolve())
        backend = make_backend(cfg, out)
        if hasattr(backend, "kernel_environment"):
            result["kernel_environment"] = backend.kernel_environment
        fingerprint = backend.frozen_fingerprint()
        history_path = out / "training.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        original_train = backend.train_expert
        def logged_train(examples, steps, seed):
            before = len(backend.train_calls)
            try:
                return original_train(examples, steps, seed)
            finally:
                history.extend(backend.train_calls[before:])
                write_json(history_path, history)
        backend.train_expert = logged_train
        stream, canary, Oracle, measure, prompt_for = import_world(cfg)
        arm = make_arm(name, backend, cfg, Oracle)
        if smoke:
            result.update(smoke_arm(name, arm, backend, cfg, stream, canary, measure, prompt_for, out))
        else:
            from harness.runner import run_arm
            from harness.results import load_records
            from harness.grading import _FINAL
            sleeps = []
            parse_path = out / "parse.json"
            parse = ({int(k): v for k, v in json.loads(parse_path.read_text(encoding="utf-8")).items()}
                     if parse_path.exists() else {})
            original_sleep, original_answer = arm.sleep, arm.answer
            canary_ids = {item.id for item in canary.items()}
            def counted_answer(task):
                response = original_answer(task)
                if task.mode == "test":
                    trace = getattr(arm, "answer_trace", {})
                    row = dict(arm=name, task_id=task.id, day=task.day, family=task.family,
                               skill=task.skill, expert=trace.get("expert"),
                               retrieved=trace.get("retrieved", []), prompt=trace.get("prompt"),
                               answer=response)
                    try:
                        serialized = json.dumps(row) + "\n"
                        with (out / "answers.jsonl").open("a", encoding="utf-8") as log:
                            log.write(serialized)
                            log.flush()
                    except (OSError, TypeError, ValueError, OverflowError):
                        result["answer_log_failures"] += 1
                if task.mode == "test" and task.day >= 0 and task.id not in canary_ids:
                    tally = parse.setdefault(task.day, dict(answers=0, parse_failures=0))
                    tally["answers"] += 1
                    tally["parse_failures"] += _FINAL.search(str(response)) is None
                return response
            arm.answer = counted_answer
            def timed_sleep(cycle):
                t = time.perf_counter()
                # Hidden tests for `cycle` run AFTER this night, so day `cycle` starts a fresh tally
                # here (a restore replays them too, and must not double count). Earlier days are done.
                parse[cycle] = dict(answers=0, parse_failures=0)
                write_json(parse_path, parse)
                info = original_sleep(cycle)
                info["wall_seconds"] = time.perf_counter() - t
                info["routing"] = routing_confusion(arm, stream, cycle)  # the router hidden(cycle) uses
                info["forced_train_accuracy"] = forced_train_accuracy(info)
                sleeps.append(dict(cycle=cycle, **info))
                return info
            arm.sleep = timed_sleep
            run_arm(arm, stream, [canary], out / "records.jsonl", arm_name=name,
                    ckpt_dir=out / "checkpoints")
            records = load_records(out / "records.jsonl")
            scores = [r for r in records if r["type"] == "score" and r["phase"] in ("hidden", "canary")]
            # Include restored cycles' timings as well as this invocation's work.
            sleeps = [dict(cycle=r["cycle"], **r["info"]) for r in records
                      if r["type"] == "cycle_done" and r["cycle"] >= 0]
            write_json(parse_path, parse)
            result.update(scores=scores, sleeps=sleeps, parse_by_cycle=parse,
                          sealed_reads=stream.sealed.read_count, split_audit=stream.audit())
        backend.save(out / "backend")
        if backend.frozen_fingerprint() != fingerprint:
            raise RuntimeError("frozen core fingerprint changed")
        result.update(status="complete", training=history, frozen_fingerprint=fingerprint)
        if cfg["backend"] != "fake":
            import torch
            result.update(torch=torch.__version__, device=str(backend.device),
                          lora_modules=list(backend.layers),
                          peak_gpu_mib=(torch.cuda.max_memory_allocated() / 2**20
                                        if backend.device.type == "cuda" else None))
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        print(result["error"], file=sys.stderr, flush=True)
        if (out / "training.json").exists():
            result["training"] = json.loads((out / "training.json").read_text(encoding="utf-8"))
        if (out / "kernel_environment.json").exists():
            result["kernel_environment"] = json.loads((out / "kernel_environment.json").read_text(encoding="utf-8"))
    result["elapsed_seconds"] = time.perf_counter() - started
    write_json(out / "result.json", result)
    return 0 if result["status"] == "complete" else 1


def restore_inputs(run_dir):
    if run_dir.exists() or not Path("/kaggle/input").exists():
        return
    for manifest in Path("/kaggle/input").rglob("run.json"):
        if manifest.parent.name == run_dir.name:
            shutil.copytree(manifest.parent, run_dir)
            return


def schedule(arms, cfg, run_dir, smoke, devices, hardware=None):
    """One child per visible GPU; no torch import or model in the driver."""
    pending, active, results = list(arms), {}, {}
    # 285 s per wave of arms: the budget is per worker slot, not for the whole eight-arm schedule.
    waves = -(-len(arms) // max(1, len(devices)))
    deadline = time.perf_counter() + 285 * waves if smoke else float("inf")
    # No arm waits for another: `ours` is first in the order and starts immediately.
    dependencies = {a: [] for a in arms}
    uses = {d: 0 for d in devices}
    for name in list(pending):
        path = run_dir / name / "result.json"
        if path.exists():
            results[name] = json.loads(path.read_text(encoding="utf-8"))
            if cfg["backend"] == "hf" and results[name]["status"] == "complete":
                validate_assignment(results[name].get("worker_device", {}), hardware["devices"])
                uses[results[name]["worker_device"]["scheduled"]["uuid"]] += 1
            pending.remove(name)  # both successes and failures are retained
    while pending or active:
        for name in list(pending):
            deps = dependencies[name]
            if any(d in results and results[d]["status"] != "complete" for d in deps):
                results[name] = dict(arm=name, status="skipped", reason="failed dependency", elapsed_seconds=0)
                write_json(run_dir / name / "result.json", results[name])
                pending.remove(name)
                continue
            if not all(d in results for d in deps):
                continue
            if time.perf_counter() >= deadline:
                results[name] = dict(arm=name, status="skipped", reason="smoke deadline", elapsed_seconds=0)
                write_json(run_dir / name / "result.json", results[name])
                pending.remove(name)
                continue
            free = min((d for d in devices if d not in active), key=uses.get, default=None)
            if free is None:
                break
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=free, PYTHONDONTWRITEBYTECODE="1",
                       S4_WORKER_CONFIG=json.dumps(cfg), S4_WORKER_DIR=str(run_dir),
                       S4_SMOKE="1" if smoke else "0", HF_HUB_OFFLINE="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            if cfg["backend"] == "hf":
                env["S4_WORKER_DEVICE"] = json.dumps(next(d for d in hardware["devices"] if d["uuid"] == free))
            log = (run_dir / (name + ".log")).open("w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                                     "--worker", name], env=env, stdout=log, stderr=subprocess.STDOUT)
            active[free] = (name, proc, log, time.perf_counter())
            uses[free] += 1
            pending.remove(name)
        for device, (name, proc, log, t) in list(active.items()):
            timed_out = time.perf_counter() >= deadline
            if proc.poll() is None and not timed_out:
                continue
            if timed_out and proc.poll() is None:
                proc.kill()
                proc.wait()
            log.close()
            path = run_dir / name / "result.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
            else:
                result = dict(arm=name, status="failed", error="smoke timeout" if timed_out else
                              f"worker exit {proc.returncode}; inspect {name}.log",
                              elapsed_seconds=time.perf_counter() - t)
            if proc.returncode != 0:
                result["status"] = "failed"
            write_json(path, result)
            results[name] = result
            print(f"{name}: {result['status']} ({result['elapsed_seconds']:.2f}s)", flush=True)
            if result["status"] != "complete":
                # Surface the cause in the kernel log so it is readable without the output download.
                print(f"  error: {str(result.get('error', ''))[:1500]}", flush=True)
                try:
                    tail = (run_dir / (name + ".log")).read_text(encoding="utf-8", errors="replace")[-2500:]
                    print(f"  --- {name}.log tail ---\n{tail}", flush=True)
                except OSError:
                    pass
            del active[device]
        if active:
            time.sleep(0.05)
    return results


SKILLS = ("rule:A", "rule:B", "fact:smoke")


def _hidden(result, cycle):
    return [x for x in result.get("scores", []) if x["phase"] == "hidden" and x["cycle"] == cycle]


def paired_bootstrap(first, second, draws=10000, seed=0):
    """Paired percentile bootstrap over items for mean(first) - mean(second).

    `first` and `second` map task id -> 0/1 correct and must cover the same ids. Items are
    resampled with replacement, jointly, so each draw keeps the pairing.
    """
    import numpy as np
    if set(first) != set(second) or not first:
        raise ValueError("paired bootstrap needs the same non-empty item ids for both arms")
    ids = sorted(first)
    diff = np.array([float(first[i]) - float(second[i]) for i in ids])
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(ids), size=(draws, len(ids)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(difference=float(diff.mean()), ci_low=float(lo), ci_high=float(hi), n=len(ids),
                draws=draws, excludes_zero=bool(lo > 0 or hi < 0))


def _rule_correct(result, cycle):
    return {x["task_id"]: int(bool(x["correct"])) for x in _hidden(result, cycle)
            if x["skill"].startswith("rule:")}


def _skill_accuracy(result, cycle, skill):
    rows = [x for x in _hidden(result, cycle) if x["skill"] == skill]
    return sum(bool(x["correct"]) for x in rows) / len(rows) if rows else None


def stop_rule(results, last_cycle, draws=10000):
    """The pre-registered rule from step4/README.md, evaluated on the final cycle's hidden rules.

    Continue only if ours beats BOTH fair RAG (512 tokens) and hippo_only on rule items by at
    least RULE_MARGIN with a paired-bootstrap 95% CI that excludes 0, rule B was accepted on its
    introduction night, and rule A did not drop. Descriptive only: this function decides nothing
    and is never tuned.
    """
    ours = results["ours"]
    verdict = dict(rule="ours - fair rag and ours - hippo_only >= 15 points on rule items, paired-bootstrap "
                        "95% CI excludes 0, rule B accepted, rule A not dropped",
                   margin=RULE_MARGIN, comparisons={})
    ours_rules = _rule_correct(ours, last_cycle)
    for other in ("rag", "hippo_only"):
        pair = paired_bootstrap(ours_rules, _rule_correct(results[other], last_cycle), draws)
        pair["passes"] = pair["difference"] >= RULE_MARGIN and pair["ci_low"] > 0
        verdict["comparisons"][other] = pair
    sleeps = {x["cycle"]: x for x in ours.get("sleeps", [])}
    verdict["rule_b_accepted"] = bool(sleeps.get(1, {}).get("accepted"))
    first, last = _skill_accuracy(ours, 0, "rule:A"), _skill_accuracy(ours, last_cycle, "rule:A")
    verdict["rule_a_first"], verdict["rule_a_last"] = first, last
    verdict["rule_a_not_dropped"] = (first is not None and last is not None
                                     and last >= first - A_DROP_TOLERANCE)
    fits = [x["forced_train_accuracy"] for x in sleeps.values()
            if x.get("forced_train_accuracy") is not None]
    verdict["forced_train_accuracy"] = fits
    # An expert that did not fit its own training set is an optimisation failure to fix, not a
    # reason to rethink the design; the verdict then reads "inconclusive", never "stop".
    verdict["optimisation_failure"] = bool(fits) and min(fits) < FIT_THRESHOLD
    passes = (all(v["passes"] for v in verdict["comparisons"].values())
              and verdict["rule_b_accepted"] and verdict["rule_a_not_dropped"])
    verdict["continue"] = passes
    verdict["verdict"] = ("continue" if passes else
                          "inconclusive: fix expert optimisation, rerun" if verdict["optimisation_failure"]
                          else "stop")
    return verdict


def parse_rate(result):
    tallies = (result.get("parse_by_cycle") or {}).values()
    answers = sum(t["answers"] for t in tallies)
    return sum(t["parse_failures"] for t in tallies) / answers if answers else None


def report(run_dir, cfg, arms, results, smoke, wall, hardware=None):
    eligible = False
    eligibility_error = None
    if cfg["backend"] == "hf":
        try:
            validate_execution(results, hardware, set(arms) | ({"ours"} if smoke else set()))
            eligible = True
        except RuntimeError as error:
            eligibility_error = str(error)
    table = ["# Step 4 results", "", f"Mode: {'smoke (plumbing only)' if smoke else 'full'}; "
             f"backend: {cfg['backend']}; model: {cfg['model']}; seed: 0; world seed: {cfg.get('world_seed')}; "
             f"context budget: {cfg.get('context_tokens')} tokens ({cfg.get('starved_tokens')} for rag_starved).", "",
             "| Arm | Status | Cycle | rule:A | rule:B | fact:smoke | Hidden all | Canary | Worker s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    final, final_by_skill = {}, {}

    def frac(rows):
        return f"{sum(bool(x['correct']) for x in rows)} / {len(rows)}" if rows else "-"

    for name in arms:
        r = results[name]
        rows = r.get("scores", [])
        for cycle in sorted({row["cycle"] for row in rows}) or [None]:
            hidden = [x for x in rows if x["cycle"] == cycle and x["phase"] == "hidden"]
            canary = [x for x in rows if x["cycle"] == cycle and x["phase"] == "canary"]
            by_skill = {k: [x for x in hidden if x["skill"] == k] for k in SKILLS}
            if cycle == cfg["cycles"] - 1 and hidden:
                final[name] = sum(bool(x["correct"]) for x in hidden) / len(hidden)
                final_by_skill[name] = {k: (sum(bool(x["correct"]) for x in v) / len(v) if v else None)
                                        for k, v in by_skill.items()}
            table.append(f"| {name} | {r['status']} | {cycle if cycle is not None else '-'} | "
                         + " | ".join(frac(by_skill[k]) for k in SKILLS)
                         + f" | {frac(hidden)} | {frac(canary)} | {r['elapsed_seconds']:.2f} |")
    # Diagnostics the stop rule needs: parse failures, routing confusion, expert fit.
    table += ["", "## Parse failures (hidden answers with no final-answer line)", "",
              "| Arm | Rate | By cycle (failures / answers) |", "|---|---:|---|"]
    for name in arms:
        rate = parse_rate(results[name])
        by_cycle = results[name].get("parse_by_cycle") or {}
        table.append(f"| {name} | {'-' if rate is None else format(rate, '.3f')} | " + ", ".join(
            f"{c}: {t['parse_failures']} / {t['answers']}" for c, t in sorted(by_cycle.items())) + " |")
    table += ["", "## Sleep diagnostics (ours and expert_only)", ""]
    for name in [a for a in ("ours", "expert_only") if a in results]:
        for night in results[name].get("sleeps", []):
            fit = night.get("forced_train_accuracy")
            table.append(f"- {name} cycle {night['cycle']}: accepted={night.get('accepted')}, forced-expert "
                         f"train accuracy={'-' if fit is None else format(fit, '.3f')}, "
                         f"reasons={night.get('reasons')}, "
                         f"routing (skill -> handle: n)={json.dumps(night.get('routing'), sort_keys=True)}")
    required = ("ours", "rag", "hippo_only")
    valid = not smoke and eligible and all(
        a in final and results[a]["status"] == "complete" for a in required)
    comparison = dict(status="descriptive" if valid else "unavailable", ours_beats_no_memory=None,
                      ours_at_least_rag=None, final_hidden_accuracy=final,
                      final_by_skill=final_by_skill, stop_rule=None)
    if valid:
        if "no_memory" in final:
            comparison["ours_beats_no_memory"] = final["ours"] > final["no_memory"]
        comparison["ours_at_least_rag"] = final["ours"] >= final["rag"]
        comparison["stop_rule"] = stop_rule(results, cfg["cycles"] - 1)
        table += ["", "## Pre-registered stop rule", "", "```json",
                  json.dumps(comparison["stop_rule"], indent=2), "```"]
    worker_hours = sum(r["elapsed_seconds"] for r in results.values()) / 3600
    table += ["", f"Measured driver wall: {wall:.2f}s; summed worker hours: {worker_hours:.6f}.",
              "GPU billing and weekly account usage are unknown; update the run ledger after completion.",
              "", f"Comparison: {json.dumps(comparison)}", "",
              "One seed and this small world provide a trend only. The last cycle adds no exposures; "
              "rejected candidates may retry. No deliberate fault injection. "
              "Smoke/fake/tiny scores do not establish model quality."]
    (run_dir / "RESULTS.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    payload = dict(config=cfg, smoke=smoke, results=results, comparison=comparison,
                   hardware=hardware, kaggle_full_run_eligible=eligible,
                   eligibility_error=eligibility_error,
                   wall_seconds=wall, summed_worker_hours=worker_hours,
                   gpu_hours=(worker_hours if cfg["backend"] == "hf" else 0),
                   weekly_gpu_hours=None, finished_utc=datetime.now(timezone.utc).isoformat())
    setup = json.loads((run_dir / "setup.json").read_text(encoding="utf-8")) if (run_dir / "setup.json").exists() else None
    payload["setup"] = setup
    write_json(run_dir / "results.json", payload)
    write_json(run_dir / "timings.json", {"wall_seconds": wall, "arms_seconds": {
        a: r["elapsed_seconds"] for a, r in results.items()}, "setup_excluded": True, "setup": setup})
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=AVAILABLE_ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = source_root()
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    smoke = os.environ.get("S4_SMOKE", DEFAULT_SMOKE) == "1"
    if args.worker:
        return worker(args.worker, json.loads(os.environ["S4_WORKER_CONFIG"]),
                      Path(os.environ["S4_WORKER_DIR"]), smoke)
    cfg, arms, key = settings(root)
    base = Path(os.environ.get("S4_OUT", "/kaggle/working/step4" if Path("/kaggle/working").is_dir()
                               else str(root / "step4/output"))).resolve()
    run_dir = base / (f"smoke-{key}" if smoke else f"full-{key}")
    restore_inputs(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg["backend"] == "hf" and cfg["hardware_mode"] == "single_device_dev" and not smoke:
        raise RuntimeError("S4_SINGLE_DEVICE_DEV=1 is smoke-only and ineligible for Kaggle full runs")
    if not smoke and cfg["backend"] == "hf":
        receipt_dir = base / f"smoke-{key}"
        restore_inputs(receipt_dir)
        receipt = receipt_dir / "results.json"
        if not receipt.exists():
            raise RuntimeError("run matching S4_SMOKE=1 first (or attach it through kernel_sources)")
        previous = json.loads(receipt.read_text(encoding="utf-8"))
    hardware = detect_hardware(cfg)
    write_json(run_dir / "hardware.json", hardware)
    write_json(run_dir / "run.json", dict(config=cfg, key=key, smoke=smoke, hardware=hardware))
    if cfg["backend"] == "hf":
        if hardware["mode"] == "dual_t4" and len(arms) < 2:
            raise RuntimeError("select at least two arms to exercise both T4 devices")
        if not smoke:
            validate_smoke_receipt(previous, cfg, arms, hardware)
    # Reusing completed outputs does not install packages or download again.
    needs_work = any(not (run_dir / a / "result.json").exists() for a in arms)
    if cfg["backend"] == "hf" and needs_work:
        setup_start = time.perf_counter()
        try:
            environment = setup_runtime(base, cfg, run_dir)
        except Exception:
            error = traceback.format_exc()
            environment_path = run_dir / "kernel_environment.json"
            write_json(run_dir / "setup.json", dict(status="failed", error=error,
                       kernel_environment=json.loads(environment_path.read_text(encoding="utf-8"))
                       if environment_path.exists() else None,
                       elapsed_seconds=time.perf_counter() - setup_start))
            results = {a: dict(arm=a, status="skipped", reason="setup failed", elapsed_seconds=0) for a in arms}
            report(run_dir, cfg, arms, results, smoke, 0, hardware)
            print(error, file=sys.stderr)
            return 1
        write_json(run_dir / "setup.json", dict(status="complete", kernel_environment=environment,
                   elapsed_seconds=time.perf_counter() - setup_start))
    devices = [d["uuid"] for d in hardware["devices"]] if cfg["backend"] == "hf" else [""]
    start = time.perf_counter()
    results = schedule(arms, cfg, run_dir, smoke, devices, hardware)
    payload = report(run_dir, cfg, arms, results, smoke, time.perf_counter() - start, hardware)
    print(f"Artifacts: {run_dir}", flush=True)
    success = all(r["status"] == "complete" for r in results.values())
    if cfg["backend"] == "hf" and hardware["mode"] == "dual_t4":
        success = success and payload["kaggle_full_run_eligible"]
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
