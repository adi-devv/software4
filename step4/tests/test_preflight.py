"""Hardware/optional-library gates must run without torch or Transformers installed."""
import copy
import functools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import pytest

from step4 import hf, run

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def work_dir():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".run-preflight-") as folder:
        yield Path(folder)


def topology():
    return dict(mode="dual_t4", inventory=[], devices=[
        dict(index=str(i), uuid=f"GPU-test-{i}", name="Tesla T4",
             pci_bus_id=f"00000000:{i:02}:00.0", memory_mib=15360) for i in range(2)])


def assignment(device):
    return dict(scheduled=device, cuda_visible_devices=device["uuid"],
                observed=dict(uuid=device["uuid"], name=device["name"],
                              compute_capability=[7, 5], logical_device=0))


def environment():
    return dict(policy="torch_reference_only", status="verified", transformers="5.3.0",
                optional_libraries={n: dict(importable=False, distributions={})
                                    for n in ("fla", "causal_conv1d", "kernels")})


def smoke_receipt(cfg, hardware):
    return dict(config=cfg, smoke=True, hardware=copy.deepcopy(hardware), results={
        name: dict(status="complete", worker_device=assignment(hardware["devices"][i]),
                   kernel_environment=environment()) for i, name in enumerate(("ours", "rag"))})


def mock_inventory(monkeypatch, hardware):
    output = "\n".join(f"{d['index']}, {d['uuid']}, {d['name']}, {d['pci_bus_id']}, {d['memory_mib']}"
                       for d in hardware["devices"])
    def probe(command, **kwargs):
        assert command[0] == "nvidia-smi"  # no pip, Python worker or download
        return SimpleNamespace(stdout=output)
    monkeypatch.setattr(run.subprocess, "run", probe)


@pytest.mark.parametrize("visible", ["0,1", "GPU-test-0,GPU-test-1"])
def test_two_t4_inventory_and_uuid_selection(monkeypatch, visible):
    hardware = topology()
    mock_inventory(monkeypatch, hardware)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    detected = run.detect_hardware(dict(backend="hf", hardware_mode="dual_t4"))
    assert detected["devices"] == hardware["devices"]
    assert detected["inventory"] == hardware["devices"]


@pytest.mark.parametrize("visible", ["0", "0,0", "0,GPU-test-0", "", "0,7", "0,1,2", "MIG-test"])
def test_bad_allocation_refused(monkeypatch, visible):
    mock_inventory(monkeypatch, topology())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(RuntimeError):
        run.detect_hardware(dict(backend="hf", hardware_mode="dual_t4"))


@pytest.mark.parametrize("smoke", [True, False])
@pytest.mark.parametrize("defect", ["single", "alias", "wrong_model"])
def test_allocation_rejected_before_setup_or_workers(monkeypatch, work_dir, smoke, defect):
    hardware = topology()
    if defect == "wrong_model":
        hardware["devices"][1]["name"] = "NVIDIA A100"
    mock_inventory(monkeypatch, hardware)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", {"single": "0", "alias": "0,GPU-test-0"}.get(defect, "0,1"))
    monkeypatch.setenv("S4_BACKEND", "hf")
    monkeypatch.setenv("S4_SMOKE", "1" if smoke else "0")
    monkeypatch.setenv("S4_OUT", str(work_dir))
    monkeypatch.delenv("S4_SINGLE_DEVICE_DEV", raising=False)
    monkeypatch.setattr(sys, "argv", ["run.py"])
    cfg, _, key = run.settings(ROOT)
    if not smoke:
        run.write_json(work_dir / f"smoke-{key}/results.json", smoke_receipt(cfg, hardware))
    def forbidden(*args, **kwargs):
        pytest.fail("expensive work started before hardware validation")
    monkeypatch.setattr(run, "setup_runtime", forbidden)
    monkeypatch.setattr(run, "schedule", forbidden)
    with pytest.raises(RuntimeError):
        run.main()


def test_single_device_dev_receipt_is_ineligible(monkeypatch, work_dir):
    hardware = topology()
    mock_inventory(monkeypatch, hardware)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    detected = run.detect_hardware(dict(backend="hf", hardware_mode="single_device_dev"))
    assert len(detected["devices"]) == 1
    run.validate_topology(detected, allow_dev=True)
    with pytest.raises(RuntimeError, match="development receipts are ineligible"):
        run.validate_topology(detected)
    cfg = dict(backend="hf", model="test", cycles=3)
    results = {"ours": dict(status="complete", elapsed_seconds=0,
                            worker_device=assignment(detected["devices"][0]))}
    payload = run.report(work_dir, cfg, ["ours"], results, True, 0, detected)
    assert not payload["kaggle_full_run_eligible"]
    assert payload["hardware"] == detected


def test_single_device_dev_full_refused_before_setup(monkeypatch, work_dir):
    monkeypatch.setenv("S4_BACKEND", "hf")
    monkeypatch.setenv("S4_SMOKE", "0")
    monkeypatch.setenv("S4_SINGLE_DEVICE_DEV", "1")
    monkeypatch.setenv("S4_OUT", str(work_dir))
    monkeypatch.setattr(sys, "argv", ["run.py"])
    with pytest.raises(RuntimeError, match="smoke-only"):
        run.main()


def test_mismatching_smoke_hardware_refused_before_setup(monkeypatch, work_dir):
    hardware = topology()
    monkeypatch.setenv("S4_BACKEND", "hf")
    monkeypatch.setenv("S4_SMOKE", "0")
    monkeypatch.delenv("S4_SINGLE_DEVICE_DEV", raising=False)
    monkeypatch.setenv("S4_OUT", str(work_dir))
    monkeypatch.setenv("S4_ARMS", "ours,rag")
    monkeypatch.setattr(sys, "argv", ["run.py"])
    cfg, _, key = run.settings(ROOT)
    previous = smoke_receipt(cfg, hardware)
    previous["hardware"]["devices"][1]["memory_mib"] = 8192
    run.write_json(work_dir / f"smoke-{key}/results.json", previous)
    monkeypatch.setattr(run, "detect_hardware", lambda cfg: hardware)
    def forbidden(*args, **kwargs):
        pytest.fail("setup/workers must not start with mismatching smoke hardware")
    monkeypatch.setattr(run, "setup_runtime", forbidden)
    monkeypatch.setattr(run, "schedule", forbidden)
    with pytest.raises(RuntimeError, match="does not match current topology"):
        run.main()


def test_worker_asserts_actual_cuda_uuid(monkeypatch):
    expected = topology()["devices"][0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", expected["uuid"])
    props = SimpleNamespace(uuid=expected["uuid"], name=expected["name"], major=7, minor=5)
    cuda = SimpleNamespace(device_count=lambda: 1, get_device_properties=lambda index: props)
    assert run.verify_worker_device(expected, cuda) == assignment(expected)
    props.uuid = "GPU-other"
    with pytest.raises(RuntimeError, match="identity"):
        run.verify_worker_device(expected, cuda)
    props.uuid = expected["uuid"]
    props.major = 8
    with pytest.raises(RuntimeError, match="identity"):
        run.verify_worker_device(expected, cuda)
    cuda.device_count = lambda: 2
    with pytest.raises(RuntimeError, match="exactly one GPU"):
        run.verify_worker_device(expected, cuda)


def test_matching_smoke_hardware_and_worker_assignments(work_dir):
    hardware = topology()
    cfg = dict(backend="hf", model="test", cycles=3)
    previous = smoke_receipt(cfg, hardware)
    run.validate_smoke_receipt(previous, cfg, ["rag", "ours"], hardware)
    for result in previous["results"].values():
        result["elapsed_seconds"] = 0
    payload = run.report(work_dir, cfg, ["rag", "ours"], previous["results"], True, 0, hardware)
    assert payload["hardware"] == hardware
    assert payload["kaggle_full_run_eligible"]
    assert json.loads((work_dir / "results.json").read_text())["results"]["ours"]["worker_device"]


def test_smoke_without_training_arm_is_ineligible(work_dir):
    hardware = topology()
    previous = smoke_receipt({}, hardware)
    results = {"oracle" if name == "ours" else name: dict(result, elapsed_seconds=0)
               for name, result in previous["results"].items()}
    payload = run.report(work_dir, dict(backend="hf", model="test", cycles=3),
                         ["rag", "oracle"], results, True, 0, hardware)
    assert not payload["kaggle_full_run_eligible"]


@pytest.mark.parametrize("reuse", [False, True])
def test_scheduler_exercises_both_uuids_including_after_reuse(monkeypatch, work_dir, reuse):
    hardware = topology()
    devices = [d["uuid"] for d in hardware["devices"]]
    if reuse:
        run.write_json(work_dir / "rag/result.json", dict(status="complete",
            worker_device=assignment(hardware["devices"][0]), kernel_environment=environment()))
    launched = []
    def launch(command, *, env, **kwargs):
        device = json.loads(env["S4_WORKER_DEVICE"])
        assert env["CUDA_VISIBLE_DEVICES"] == device["uuid"]
        launched.append((command[-1], device["uuid"]))
        run.write_json(work_dir / command[-1] / "result.json", dict(status="complete",
            elapsed_seconds=0, worker_device=assignment(device), kernel_environment=environment()))
        return SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(run.subprocess, "Popen", launch)
    results = run.schedule(["rag", "ours"], dict(backend="hf"), work_dir, False, devices, hardware)
    run.validate_execution(results, hardware, ["rag", "ours"])
    assert launched == ([("ours", devices[1])] if reuse else
                        [("rag", devices[0]), ("ours", devices[1])])


@pytest.mark.parametrize("defect", ["missing_hardware", "single_dev", "other_gpu", "other_name",
    "failed_arm", "missing_assignment", "same_gpu", "wrong_actual_uuid", "wrong_mask",
    "missing_kernel_record", "optional_lib", "other_runtime", "not_smoke", "config"])
def test_bad_smoke_receipt_refused(defect):
    hardware = topology()
    cfg = dict(backend="hf")
    previous = smoke_receipt(cfg, hardware)
    ours = previous["results"]["ours"]
    if defect == "missing_hardware":
        del previous["hardware"]
    elif defect == "single_dev":
        previous["hardware"]["mode"] = "single_device_dev"
    elif defect == "other_gpu":
        previous["hardware"]["devices"][1]["uuid"] = "GPU-different-allocation"
    elif defect == "other_name":
        previous["hardware"]["devices"][1]["name"] = "NVIDIA A100"
    elif defect == "failed_arm":
        ours["status"] = "failed"
    elif defect == "missing_assignment":
        del ours["worker_device"]
    elif defect == "same_gpu":
        ours["worker_device"] = previous["results"]["rag"]["worker_device"]
    elif defect == "wrong_actual_uuid":
        ours["worker_device"]["observed"]["uuid"] = "GPU-wrong"
    elif defect == "wrong_mask":
        ours["worker_device"]["cuda_visible_devices"] = "0,1"
    elif defect == "missing_kernel_record":
        del ours["kernel_environment"]
    elif defect == "optional_lib":
        ours["kernel_environment"]["optional_libraries"]["fla"]["importable"] = True
    elif defect == "other_runtime":
        ours["kernel_environment"]["transformers"] = "5.22.2"
    elif defect == "not_smoke":
        previous["smoke"] = False
    elif defect == "config":
        previous["config"] = dict(backend="different")
    with pytest.raises(RuntimeError):
        run.validate_smoke_receipt(previous, cfg, ["rag", "ours"], hardware)


@pytest.mark.parametrize("version", ["5.3.0", "5.22.2"])
@pytest.mark.parametrize("module,distribution", [("fla", "flash-linear-attention"),
    ("fla", "fla-core"), ("causal_conv1d", "causal-conv1d"), ("kernels", "kernels")])
@pytest.mark.parametrize("detection", ["importable", "metadata_only"])
def test_optional_kernels_refused_independent_of_transformers(monkeypatch, work_dir, version,
                                                            module, distribution, detection):
    def package_version(name):
        if name == "transformers":
            return version
        if name == distribution and detection == "metadata_only":
            return "9.9-test"
        raise hf.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(hf.importlib.metadata, "version", package_version)
    monkeypatch.setattr(hf.importlib.util, "find_spec", lambda name:
                        object() if name == module and detection == "importable" else None)
    # The same test executes on Python without torch/Transformers: refusal must
    # precede model imports, not depend on one release's dispatch implementation.
    with pytest.raises(RuntimeError, match="optional libraries detected"):
        hf.make_hf_backend(dict(backend="hf"), work_dir)
    record = json.loads((work_dir / "kernel_environment.json").read_text())
    assert record["status"] == "refused"
    assert record["transformers"] == version
    library = record["optional_libraries"][module]
    assert library["importable"] == (detection == "importable")
    assert library["distributions"] == ({distribution: "9.9-test"} if detection == "metadata_only" else {})


def test_clean_reference_environment(monkeypatch, work_dir):
    def package_version(name):
        if name == "transformers":
            return "5.3.0"
        raise hf.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(hf.importlib.metadata, "version", package_version)
    monkeypatch.setattr(hf.importlib.util, "find_spec", lambda name: None)
    assert hf.require_reference_kernels(work_dir) == environment()


def test_pinned_instance_dispatch_contract_without_transformers():
    # Model the v5.3.0 instance contract directly so the check stays covered even
    # when the optional CPU integration environment uses newer global dispatch.
    class Norm:
        pass
    class Delta:
        pass
    def reference(*args):
        pass
    qwen = SimpleNamespace(Qwen3_5GatedDeltaNet=Delta, Qwen3_5RMSNormGated=Norm,
        torch_causal_conv1d_update=reference, torch_chunk_gated_delta_rule=reference,
        torch_recurrent_gated_delta_rule=reference)
    layer = Delta()
    layer.norm = Norm()
    layer.causal_conv1d_fn = None
    layer.causal_conv1d_update = reference
    layer.chunk_gated_delta_rule = reference
    layer.recurrent_gated_delta_rule = reference
    model = SimpleNamespace(named_modules=lambda: [("linear_attn", layer)])
    assert hf.assert_reference_deltanet(model, qwen=qwen)["linear_attn"]["causal_conv1d_fn"] is None
    for attr in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
        before = getattr(layer, attr)
        setattr(layer, attr, lambda *args: None)  # pretend optional compiled kernel
        with pytest.raises(RuntimeError, match="non-reference function"):
            hf.assert_reference_deltanet(model, qwen=qwen)
        setattr(layer, attr, before)
    layer.norm = object()
    with pytest.raises(RuntimeError, match="gated norm"):
        hf.assert_reference_deltanet(model, qwen=qwen)


def test_global_dispatch_selects_original_functions_without_transformers():
    # Exercise the newer contract without requiring either Transformers release.
    class Norm:
        pass
    class Delta:
        pass
    def causal_conv1d_fn():
        return "reference"
    def causal_conv1d_update():
        return "reference"
    def torch_chunk_gated_delta_rule():
        return "reference"
    def torch_recurrent_gated_delta_rule():
        return "reference"
    references = {f.__name__: f for f in (causal_conv1d_fn, causal_conv1d_update,
                  torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)}
    def dispatch(reference):
        @functools.wraps(reference)
        def wrapped():
            pytest.fail("optional kernel dispatch must be bypassed")
        return wrapped
    qwen = SimpleNamespace(__name__=__name__, __file__=__file__, Qwen3_5GatedDeltaNet=Delta,
                           Qwen3_5RMSNormGated=Norm,
                           **{key: dispatch(value) for key, value in references.items()})
    layer = Delta()
    layer.norm = Norm()
    model = SimpleNamespace(named_modules=lambda: [("linear_attn", layer)])
    record = hf.assert_reference_deltanet(model, qwen=qwen)
    for key, reference in references.items():
        assert getattr(qwen, key) is reference
        assert getattr(qwen, key)() == "reference"
        assert record["linear_attn"][key] == __name__ + "." + key
        setattr(qwen, key, lambda: None)
        with pytest.raises(RuntimeError, match="not the Transformers reference"):
            hf.assert_reference_deltanet(model, qwen=qwen)
        setattr(qwen, key, reference)


def test_smoke_receipt_accepts_new_allocation_same_profile():
    smoke_hw = topology()
    now_hw = topology()
    for i, d in enumerate(now_hw["devices"]):
        d["uuid"] = f"GPU-new-allocation-{i}"
        d["pci_bus_id"] = f"00000000:{i + 7:02}:00.0"
    previous = smoke_receipt(dict(backend="hf"), smoke_hw)
    run.validate_smoke_receipt(previous, dict(backend="hf"), ["ours", "rag"], now_hw)


def test_worker_accepts_torch_bare_uuid_form():
    expected = topology()["devices"][0]
    run.validate_assignment(dict(scheduled=expected, cuda_visible_devices=expected["uuid"],
        observed=dict(uuid=expected["uuid"][4:].upper(), name=expected["name"],
                      compute_capability=[7, 5], logical_device=0)), [expected])
