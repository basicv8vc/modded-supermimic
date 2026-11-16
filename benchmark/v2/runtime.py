"""CUDA-only runtime enforcement, hardware provenance and telemetry for V2.

Benchmark V2's *formal* evaluator must run the official mjlab / MuJoCo Warp
simulation on a CUDA device.  This module is the single place that decides
which device is used.  There is deliberately **no CPU fallback**: if CUDA,
Warp-CUDA or the MuJoCo Warp backend is missing, ``CudaRuntimeUnavailable`` is
raised and the evaluation aborts.

It does not touch physics, the robot model, the V2 scoring protocol, V1, or the
policy API.  It only:

* probes / validates the CUDA runtime (:func:`probe_cuda`, :func:`require_cuda_device`),
* proves that a constructed mjlab env really holds GPU simulation state and
  CUDA step graphs (:func:`inspect_runtime`, :func:`verify_cuda_kernel_execution`),
* collects provenance from *runtime detection* (:func:`collect_provenance`),
* samples GPU / CPU utilisation while a rollout runs (:class:`GpuTelemetry`).

Frozen target stack (task prompt §2), recorded rather than silently changed:
mujoco 3.11.0 / mujoco-warp 3.11.0 / warp-lang 1.16.0 / mjlab 1.6.0 @
0fb8a681136be94ffc636a3dd423cabb97d91f10.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Callable

from . import spec as S

TARGET_VERSIONS: dict[str, str] = {
    "mujoco": "3.11.0",
    "mujoco_warp": "3.11.0",
    "warp_lang": "1.16.0",
    "mjlab": "1.6.0",
}
TARGET_MJLAB_COMMIT = "0fb8a681136be94ffc636a3dd423cabb97d91f10"
TARGET_GPU_NAME = "RTX PRO 6000"
TARGET_COMPUTE_CAPABILITY = (12, 0)  # sm_120 / Blackwell
DEFAULT_DEVICE = "cuda:0"


class CudaRuntimeUnavailable(RuntimeError):
    """Raised when CUDA / Warp-CUDA / MuJoCo Warp GPU execution is unavailable.

    The evaluator must never silently degrade to the CPU simulation path.
    """


# --------------------------------------------------------------------------
# Version / driver helpers
# --------------------------------------------------------------------------

def package_versions() -> dict[str, str]:
    out = {}
    for key, dist in (("mujoco", "mujoco"), ("mujoco_warp", "mujoco-warp"),
                      ("warp_lang", "warp-lang"), ("mjlab", "mjlab"),
                      ("torch", "torch")):
        try:
            out[key] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[key] = "missing"
    return out


def version_drift(versions: dict[str, str] | None = None) -> dict[str, str]:
    """Return ``{package: "actual (target expected)"}`` for every mismatch."""
    v = versions or package_versions()
    return {k: f"{v.get(k, 'missing')} (target {want})"
            for k, want in TARGET_VERSIONS.items() if v.get(k) != want}


def source_fingerprint() -> dict[str, str]:
    """SHA256 of the V2 implementation files, so a stale report is detectable.

    A GPU report is only meaningful for the exact code that produced it.  The
    earlier preflight run was generated minutes before its own modules were
    last edited, which is invisible in the JSON without this.  ``combined``
    covers every listed file; ``evaluator_git_sha`` alone is not enough on a
    Colab VM whose workspace is an uploaded tarball with no ``.git``.
    """
    import hashlib  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[2]
    rels = ("benchmark/v2/spec.py", "benchmark/v2/config.py",
            "benchmark/v2/runtime.py", "benchmark/v2/parity.py",
            "benchmark/v2/runner.py", "scripts/v2/run_eval.py",
            "scripts/v2/gpu_runtime_check.py")
    out: dict[str, str] = {}
    for rel in rels:
        p = root / rel
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"
    h = hashlib.sha256()
    for rel in sorted(out):
        h.update(f"{rel}:{out[rel]}\n".encode())
    out["combined"] = h.hexdigest()
    return out


def _nvidia_smi(fields: str) -> str:
    if not shutil.which("nvidia-smi"):
        return "unavailable (nvidia-smi not found)"
    try:
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        return f"unavailable ({type(exc).__name__})"


# --------------------------------------------------------------------------
# CUDA probe + hard device selection
# --------------------------------------------------------------------------

def probe_cuda(device: str = DEFAULT_DEVICE) -> dict[str, Any]:
    """Detect the real CUDA runtime.  Raises instead of falling back to CPU."""
    import torch  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    if not device.startswith("cuda"):
        raise CudaRuntimeUnavailable(
            f"V2 requires a CUDA device, got {device!r}; refusing a CPU simulation")
    if not torch.cuda.is_available():
        raise CudaRuntimeUnavailable(
            "torch.cuda.is_available() == False: no CUDA driver/runtime visible. "
            "V2 refuses to fall back to the CPU MuJoCo Warp backend.")
    if torch.cuda.device_count() < 1:
        raise CudaRuntimeUnavailable("torch.cuda.device_count() == 0")

    try:
        wp.init()
    except Exception as exc:  # pragma: no cover - Warp build specific
        raise CudaRuntimeUnavailable(f"warp.init() failed: {exc}") from exc
    cuda_devices = [str(d) for d in wp.get_cuda_devices()]
    if not cuda_devices:
        raise CudaRuntimeUnavailable(
            "warp.get_cuda_devices() is empty: Warp was built without CUDA "
            "support or the driver is not usable (no CPU fallback allowed).")
    try:
        wp_dev = wp.get_device(device)
    except Exception as exc:
        raise CudaRuntimeUnavailable(f"warp.get_device({device!r}) failed: {exc}") from exc
    if not bool(getattr(wp_dev, "is_cuda", False)):
        raise CudaRuntimeUnavailable(
            f"warp device {wp_dev!r} is not a CUDA device; refusing CPU fallback")

    idx = torch.cuda.current_device() if device == "cuda" else 0
    capability = tuple(int(x) for x in torch.cuda.get_device_capability(idx))
    name = torch.cuda.get_device_name(idx)
    versions = package_versions()
    info: dict[str, Any] = {
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "torch_device": str(device),
        "gpu_name": name,
        "gpu_count": torch.cuda.device_count(),
        "compute_capability": capability,
        "compute_capability_string": f"sm_{capability[0]}{capability[1]}",
        "warp_cuda_devices": cuda_devices,
        "warp_device": str(wp_dev),
        "warp_device_is_cuda": bool(wp_dev.is_cuda),
        "warp_device_arch": getattr(wp_dev, "arch", None),
        "cuda_runtime": torch.version.cuda or "unknown",
        "driver_version": _nvidia_smi("driver_version"),
        "mjlab_runtime_device": str(device),
        "simulation_backend": "mujoco_warp (mjlab.sim.sim.Simulation)",
        "versions": versions,
        "version_drift": version_drift(versions),
        "mjlab_commit_expected": TARGET_MJLAB_COMMIT,
        "target_gpu_name": TARGET_GPU_NAME,
        "target_gpu_match": TARGET_GPU_NAME in name,
        "sm_120_confirmed": capability == TARGET_COMPUTE_CAPABILITY,
    }
    return info


def require_cuda_device(device: str = DEFAULT_DEVICE) -> str:
    """Validate and return the CUDA device string, or raise.  No fallback."""
    probe_cuda(device)
    return device


def format_hardware_report(probe: dict[str, Any], runtime: dict[str, Any] | None = None) -> str:
    """The exact hardware lines required by task prompt §3."""
    rt = runtime or {}
    lines = [
        f"torch.cuda.is_available              : {probe['torch_cuda_is_available']}",
        f"torch.cuda.get_device_name(0)        : {probe['gpu_name']}",
        f"torch.cuda.get_device_capability(0)  : {probe['compute_capability']} "
        f"({probe['compute_capability_string']})",
        f"warp CUDA devices                    : {probe['warp_cuda_devices']}",
        f"mjlab runtime device                 : {rt.get('env_device', probe['mjlab_runtime_device'])}",
        f"MuJoCo Warp simulation backend       : {rt.get('simulation_backend', probe['simulation_backend'])}",
        f"warp device / arch                   : {probe['warp_device']} / {probe['warp_device_arch']}",
        f"CUDA runtime / driver                : {probe['cuda_runtime']} / {probe['driver_version']}",
        f"CUDA graph step captured             : {rt.get('step_graph_captured', 'n/a')}",
        f"physics state array device           : {rt.get('wp_data_qpos_device', 'n/a')}",
        f"target GPU RTX PRO 6000 / sm_120     : {probe['target_gpu_match']} / {probe['sm_120_confirmed']}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Post-construction verification: prove physics is on the GPU
# --------------------------------------------------------------------------

def inspect_runtime(env) -> dict[str, Any]:
    """Inspect a *constructed* mjlab env, abort on any CPU fallback.

    Checks the env device, the Warp device backing ``Simulation``, the
    ``mujoco_warp.Data`` state arrays, and the captured CUDA step graph.  A
    device string alone is not proof: ``nvidia-smi`` memory usage can come
    from an idle context while physics still runs on the CPU.
    """
    sim = getattr(env, "sim", None)
    if sim is None:
        raise CudaRuntimeUnavailable("env has no .sim (not a mjlab Simulation)")
    env_device = str(getattr(env, "device", sim.device))
    wp_dev = sim.wp_device
    wp_data = sim.wp_data
    qpos_arr = wp_data.qpos
    qpos_dev = getattr(qpos_arr, "device", None)
    qpos_is_cuda = bool(getattr(qpos_dev, "is_cuda", False))
    step_graph = getattr(sim, "step_graph", None)
    use_graph = bool(getattr(sim, "use_cuda_graph", False))
    graph_captured = step_graph is not None
    # mjlab disables CUDA graphs on its own when the Warp mempool is off or the
    # driver is older than 12.4 (sim.py:_should_use_cuda_graph).  That is a
    # *performance* decision, not a CPU fallback -- the physics kernels still
    # run on the GPU.  Record it as a separate fact instead of treating a
    # missing graph as proof of CPU execution.
    graph_status = ("captured" if graph_captured
                    else ("disabled_by_runtime" if not use_graph else "enabled_but_uncaptured"))

    info = {
        "env_device": env_device,
        "sim_device": str(sim.device),
        "simulation_class": f"{type(sim).__module__}.{type(sim).__name__}",
        "simulation_backend": "mujoco_warp",
        "mujoco_warp_data_class": f"{type(wp_data).__module__}.{type(wp_data).__name__}",
        "warp_device": str(wp_dev),
        "warp_device_is_cuda": bool(getattr(wp_dev, "is_cuda", False)),
        "warp_device_arch": getattr(wp_dev, "arch", None),
        "wp_data_qpos_device": str(qpos_dev),
        "wp_data_qpos_is_cuda": qpos_is_cuda,
        # Duck-typed so the CPU-fallback rejection test runs without mjlab
        # installed; the real class is mujoco_warp._src.*.Data.
        "wp_data_is_mjwarp": type(wp_data).__module__.startswith("mujoco_warp"),
        "use_cuda_graph": use_graph,
        "step_graph_captured": graph_captured,
        "graph_capture_status": graph_status,
        "no_cpu_fallback": True,
    }
    problems = []
    if not env_device.startswith("cuda"):
        problems.append(f"env.device={env_device!r} is not CUDA")
    if not str(sim.device).startswith("cuda"):
        problems.append(f"sim.device={sim.device!r} is not CUDA")
    if not info["warp_device_is_cuda"]:
        problems.append(f"sim.wp_device={wp_dev!r} is CPU")
    if not qpos_is_cuda:
        problems.append("sim.wp_data.qpos is not a CUDA Warp array")
    if not info["wp_data_is_mjwarp"]:
        problems.append("simulation state is not a mujoco_warp.Data")
    if problems:
        info["no_cpu_fallback"] = False
        raise CudaRuntimeUnavailable(
            "CPU simulation fallback detected: " + "; ".join(problems))
    # A missing CUDA graph degrades throughput but does not mean CPU physics;
    # surface it as a warning so a slow run has a visible cause.
    info["warnings"] = ([] if graph_captured
                        else [f"CUDA graph {graph_status}: physics is still on the "
                              f"GPU, but per-step launch overhead will dominate"])
    return info


def verify_cuda_kernel_execution(env, action) -> dict[str, Any]:
    """Step the env once and prove the GPU physics kernels actually ran.

    Reads ``sim.wp_data.qpos`` (a Warp CUDA array) before and after a step,
    synchronises both CUDA runtimes, and requires a finite, non-zero change.
    This is a kernel-execution check, not a mere ``nvidia-smi`` memory check.
    """
    import torch  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    sim = env.sim
    before = sim.wp_data.qpos.numpy().copy()
    wp.synchronize_device(sim.wp_device)
    torch.cuda.synchronize(sim.device)
    t0 = time.perf_counter()
    env.step(action)
    wp.synchronize_device(sim.wp_device)
    torch.cuda.synchronize(sim.device)
    elapsed = time.perf_counter() - t0
    after = sim.wp_data.qpos.numpy().copy()
    delta = float(abs(after - before).max())
    finite = bool((after == after).all())  # NaN check without numpy import
    info = {
        "kernel_execution_wall_s": elapsed,
        "qpos_max_abs_delta": delta,
        "qpos_finite_after_step": finite,
        "state_advanced_by_gpu_kernel": bool(delta > 0.0 and finite),
        "torch_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    if not info["state_advanced_by_gpu_kernel"]:
        raise CudaRuntimeUnavailable(
            f"GPU physics step did not advance the CUDA state (delta={delta}, finite={finite})")
    return info


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def collect_provenance(*, env=None, probe: dict[str, Any] | None = None,
                       runtime: dict[str, Any] | None = None,
                       kernel: dict[str, Any] | None = None,
                       telemetry: dict[str, Any] | None = None,
                       performance: dict[str, Any] | None = None,
                       extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance from *runtime detection*, never from a CLI ``--gpu-name``."""
    from ..evaluator.manifest import load_manifest  # noqa: PLC0415
    from ..evaluator.motion import CONVERTER_VERSION  # noqa: PLC0415
    from ..evaluator.runner import dependency_versions, source_git_sha  # noqa: PLC0415
    from ..evaluator import constants as C  # noqa: PLC0415

    p = probe or probe_cuda()
    manifest = load_manifest()
    # §9 "Simulation FPS" must be the *measured* steady-state CUDA rollout
    # rate, not the nominal 1/dt control frequency.  Fall back explicitly.
    measured_fps = None
    if performance:
        measured_fps = performance.get("steady_simulation_fps")
        if not isinstance(measured_fps, (int, float)) or measured_fps <= 0:
            vals = [v.get("steady_simulation_fps")
                    for v in (performance.get("per_clip") or {}).values()
                    if isinstance(v, dict)]
            vals = [v for v in vals if isinstance(v, (int, float)) and v > 0]
            measured_fps = sum(vals) / len(vals) if vals else None
    out: dict[str, Any] = {
        "gpu_name": p["gpu_name"],
        "gpu_compute_capability": p["compute_capability_string"],
        "gpu_count": p["gpu_count"],
        "actual_torch_device": p["torch_device"],
        "actual_warp_device": p["warp_device"],
        "warp_cuda_devices": p["warp_cuda_devices"],
        "actual_simulation_backend": (runtime or {}).get("simulation_backend", p["simulation_backend"]),
        "simulation_state_device": (runtime or {}).get("wp_data_qpos_device", "unknown"),
        "cuda_graph_step_captured": (runtime or {}).get("step_graph_captured", False),
        "cuda_runtime": p["cuda_runtime"],
        "driver_version": p["driver_version"],
        "mujoco_version": p["versions"]["mujoco"],
        "mujoco_warp_version": p["versions"]["mujoco_warp"],
        "warp_version": p["versions"]["warp_lang"],
        "mjlab_version": p["versions"]["mjlab"],
        "mjlab_commit": TARGET_MJLAB_COMMIT,
        "version_drift": p["version_drift"],
        "target_gpu_match": p["target_gpu_match"],
        "sm_120_confirmed": p["sm_120_confirmed"],
        # On a Colab VM the workspace may be an uploaded tarball without .git;
        # allow the caller to declare the exact commit that was uploaded (the
        # hardware fields above are still runtime-detected, never declared).
        "evaluator_git_sha": (os.environ.get("V2_EVALUATOR_GIT_SHA")
                              or source_git_sha()),
        "v2_protocol_sha256": S.protocol_sha256(),
        "v2_protocol_id": S.PROTOCOL_ID,
        "robot_model_sha256": manifest["robot_model"]["sha256"],
        "dataset_manifest_sha256": manifest["dataset"]["manifest_sha256"],
        "motion_converter_version": CONVERTER_VERSION,
        "simulation_fps": float(measured_fps) if measured_fps else 1.0 / C.STEP_DT,
        "simulation_fps_source": ("measured steady-state CUDA rollout" if measured_fps
                                  else "nominal control frequency (no measurement supplied)"),
        "nominal_control_fps": 1.0 / C.STEP_DT,
        "runtime": runtime or {},
        "kernel_execution": kernel or {},
        "telemetry": telemetry or {},
        "performance": performance or {},
        "collected_at_unix": time.time(),
        # posix.uname_result has .release, not .sysrelease.
        "platform": (os.uname().release if hasattr(os, "uname")
                     else "unknown"),
    }
    out["dependency_versions"] = dependency_versions()
    if extra:
        out.update(extra)
    return out


def write_provenance(path, provenance: dict[str, Any]) -> None:
    from pathlib import Path  # noqa: PLC0415
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Telemetry (sampled during a real rollout)
# --------------------------------------------------------------------------

class GpuTelemetry:
    """Sample GPU utilisation / memory and CPU utilisation in a daemon thread."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, float]] = []

    def start(self) -> "GpuTelemetry":
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        try:
            import psutil  # noqa: PLC0415
        except ImportError:
            psutil = None  # type: ignore[assignment]
        while not self._stop.is_set():
            gpu_util = _nvidia_smi("utilization.gpu")
            mem_used = _nvidia_smi("memory.used")
            try:
                gu = float(str(gpu_util).split(",")[0])
            except ValueError:
                gu = float("nan")
            try:
                mu = float(str(mem_used).split(",")[0])
            except ValueError:
                mu = float("nan")
            cpu = float(psutil.cpu_percent(interval=None)) if psutil else float("nan")
            self.samples.append({"gpu_utilization_pct": gu, "gpu_memory_used_mib": mu,
                                 "cpu_utilization_pct": cpu, "t": time.perf_counter()})
            self._stop.wait(self.interval)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        def stat(key: str) -> dict[str, float]:
            vals = [s[key] for s in self.samples if s[key] == s[key]]
            return ({"mean": float(sum(vals) / len(vals)), "max": float(max(vals)),
                     "min": float(min(vals))} if vals else {})
        return {"samples": len(self.samples), "interval_s": self.interval,
                "gpu_utilization_pct": stat("gpu_utilization_pct"),
                "gpu_memory_used_mib": stat("gpu_memory_used_mib"),
                "cpu_utilization_pct": stat("cpu_utilization_pct")}


def synchronized_time(fn: Callable[[], Any], device: str = DEFAULT_DEVICE) -> tuple[Any, float]:
    """Wall time of ``fn`` with both CUDA runtimes synchronised."""
    import torch  # noqa: PLC0415
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize(device)
    return out, time.perf_counter() - t0
