"""Synthetic fixtures + REAL upstream source oracles for V2 parity tests.

The reward / termination parity tests do not compare V2 with a hand-copied
formula: they import the actual source files of the two pinned upstream
implementations and call their own ``reward`` / ``termination`` methods on an
identical synthetic robot state, reference state, joint state and action pair.

    supermimic/mdp.py:108-152        @ 8bfc1158bd968f729fcc65d5e897a05b05ea3d36
    src/nablamimic/env.py:404-433    @ bfbb2a6773b4776c17402a34a93ec7589d9c7c18

Upstream checkouts are located via:
    V2_SUPERMIMIC_ROOT   /   V2_NABLAMIMIC_ROOT
or the defaults below, or ``research/modded-supermimic/.tmp/v2_upstream``
(populated by ``scripts/v2/fetch_upstream.py``).  If they are missing the
dynamic parity tests skip, but the frozen oracle snapshot tests always run.

Neither checkout is bundled into this repository; a manifest of their source
file SHA256 values is checked whenever they are present, so a wrong revision
cannot silently pass.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import sys
import types
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parent
sys.path.insert(0, str(REPO))

from benchmark.v2 import spec as S  # noqa: E402

SUPERMIMIC_COMMIT = "8bfc1158bd968f729fcc65d5e897a05b05ea3d36"
NABLAMIMIC_COMMIT = "bfbb2a6773b4776c17402a34a93ec7589d9c7c18"

# SHA256 of the upstream files at the two pinned commits (measured, see
# docs/V2_SOURCE_COMPARISON.md).  Wrong revision => test failure, not skip.
ORACLE_SHA256 = {
    "supermimic/mdp.py": "19662edc4d994553ebb4cf587f3080ca95b407e53806f7cd28939937e135754f",
    "supermimic/rotations.py": "f0d75dd6d8398220aff7f2910e7ef4a559bf33368fa3324a6d405ee2304be8de",
    "supermimic/evaluate.py": "5b8923a8c46dcbf2b19e048729ab0039b9ffe1a45777f83eb355df14b245bb36",
    "nablamimic/env.py": "6ce93ebcc29baeb9f095cf077866760f439f85ba0e91e2a79c5f614f73c14f68",
    "nablamimic/evaluate.py": "18f2d65b1411621415a2276911d6a18168dbd27324e3603a56ab21d7fbbf9d45",
}

B = 14   # tracked bodies
J = 29   # DoF


# --------------------------------------------------------------------------
# Source location / integrity
# --------------------------------------------------------------------------

def _first_existing(candidates) -> Path | None:
    for c in candidates:
        if c and Path(c).is_dir():
            return Path(c)
    return None


def supermimic_root() -> Path | None:
    return _first_existing([
        os.environ.get("V2_SUPERMIMIC_ROOT"),
        REPO / ".tmp/v2_upstream/supermimic_vibed",
        Path.home() / "Downloads/supermimic_vibed",
        Path("/tmp/supermimic_vibed_v2"),
    ])


def nablamimic_root() -> Path | None:
    return _first_existing([
        os.environ.get("V2_NABLAMIMIC_ROOT"),
        REPO / ".tmp/v2_upstream/NablaMimic",
        Path.home() / "Downloads/NablaMimic",
    ])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_integrity() -> dict:
    """Verify every upstream source file that is present (no silent wrong SHA)."""
    checks = {
        "supermimic/mdp.py": supermimic_root() / "supermimic/mdp.py" if supermimic_root() else None,
        "supermimic/rotations.py": supermimic_root() / "supermimic/rotations.py" if supermimic_root() else None,
        "supermimic/evaluate.py": supermimic_root() / "supermimic/evaluate.py" if supermimic_root() else None,
        "nablamimic/env.py": nablamimic_root() / "src/nablamimic/env.py" if nablamimic_root() else None,
        "nablamimic/evaluate.py": nablamimic_root() / "src/nablamimic/evaluate.py" if nablamimic_root() else None,
    }
    out = {}
    for key, path in checks.items():
        if path is None or not path.exists():
            out[key] = "MISSING"
        else:
            actual = _sha256(path)
            out[key] = "OK" if actual == ORACLE_SHA256[key] else f"MISMATCH:{actual}"
    return out


def sources_available() -> bool:
    return all(v == "OK" for v in source_integrity().values())


def heavy_deps_available() -> bool:
    try:
        import torch  # noqa: F401
        import jax  # noqa: F401
        import jax.numpy  # noqa: F401
        import mujoco  # noqa: F401
        import mujoco.mjx  # noqa: F401
        return True
    except Exception:
        return False


def real_oracle_available() -> bool:
    return sources_available() and heavy_deps_available()


# --------------------------------------------------------------------------
# Real-source loaders
# --------------------------------------------------------------------------

def _install_isaaclab_stubs() -> None:
    """Only ``supermimic/robot.py``'s module-level Cfg construction needs these.

    ``robot.py`` does ``import isaaclab.sim as sim_utils`` and builds a few Cfg
    objects at import time; no Isaac Lab functionality is used by mdp.py's
    reward/termination.  Nothing is emulated in the maths under test.
    """
    if "isaaclab" in sys.modules:
        return

    class AnyCfg:
        def __init__(self, *a, **k):
            pass

    class ArticulationCfg:
        InitialStateCfg = AnyCfg

        def __init__(self, *a, **k):
            pass

    sim = types.ModuleType("isaaclab.sim")
    for n in ("UsdFileCfg", "GroundPlaneCfg", "RigidBodyPropertiesCfg",
              "ArticulationRootPropertiesCfg", "SimulationCfg", "SimulationContext",
              "AssetBaseCfg", "InteractiveSceneCfg"):
        setattr(sim, n, AnyCfg)
    sim.sim_utils = sim
    actuators = types.ModuleType("isaaclab.actuators")
    actuators.ImplicitActuatorCfg = AnyCfg
    assets = types.ModuleType("isaaclab.assets")
    assets.ArticulationCfg = ArticulationCfg
    assets.AssetBaseCfg = AnyCfg
    pkg = types.ModuleType("isaaclab")
    pkg.sim, pkg.actuators, pkg.assets = sim, actuators, assets
    sys.modules.update({
        "isaaclab": pkg, "isaaclab.sim": sim,
        "isaaclab.actuators": actuators, "isaaclab.assets": assets,
    })


def load_supermimic():
    """Import the real ``supermimic.mdp`` (and rotations) at the pinned commit."""
    root = supermimic_root()
    if root is None:
        raise RuntimeError("SuperMimic source not found")
    _install_isaaclab_stubs()
    sys.path.insert(0, str(root))
    import supermimic.mdp as mdp  # noqa: PLC0415
    return mdp


def load_nablamimic():
    """Import the real ``nablamimic.env`` at the pinned commit."""
    root = nablamimic_root()
    if root is None:
        raise RuntimeError("NablaMimic source not found")
    menagerie = root / "audit/upstream/menagerie"
    os.environ.setdefault("MENAGERIE_DIR", str(menagerie if menagerie.exists() else root / "third_party/mujoco_menagerie"))
    sys.path.insert(0, str(root / "src"))
    import nablamimic.env as nenv  # noqa: PLC0415
    return nenv


# --------------------------------------------------------------------------
# Deterministic synthetic fixtures
# --------------------------------------------------------------------------

def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def make_case(kind: str = "drift", seed: int = 20261112) -> dict:
    """One fully specified control step, shared by all three implementations.

    ``kind``:
      ``perfect``          robot state == reference state
      ``coincident_anchor``  torso_link coincides with the pelvis, so SuperMimic's
                           pelvis anchor and NablaMimic/mjlab's torso anchor are
                           the same body -> full 8-term parity is expected
      ``drift``            general G1-like state with a distinct torso (exposes
                           the documented anchor divergence)
    """
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, 0.12, (B, 3))
    offsets[:, 2] += 0.25
    offsets[0] = 0.0                       # body 0 (pelvis) IS the root
    local_q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (B, 1))
    for i in range(1, B):
        q = rng.normal(0.0, 0.15, 4)
        local_q[i] = q / np.linalg.norm(q)
    if kind == "coincident_anchor":
        offsets[S.ANCHOR_INDEX] = 0.0
        local_q[S.ANCHOR_INDEX] = np.array([1.0, 0.0, 0.0, 0.0])

    ref_p = np.array([0.02, -0.01, 0.78])
    ref_q = quat_from_euler(0.03, -0.05, 0.31)
    if kind == "perfect":
        rob_p, rob_q = ref_p.copy(), ref_q.copy()
    else:
        rob_p = ref_p + np.array([0.06, -0.04, -0.035])
        rob_q = quat_from_euler(0.09, 0.12, 0.55)

    def build(root_p, root_q):
        R = S.quat_to_mat(root_q)
        return (root_p[None, :] + offsets @ R.T,
                S.quat_mul(root_q[None, :], local_q))

    rbp, rbq = build(rob_p, rob_q)
    refbp, refbq = build(ref_p, ref_q)
    flin = rng.normal(0.0, 0.3, (B, 3))
    fang = rng.normal(0.0, 0.8, (B, 3))
    if kind == "perfect":
        rlin, rang = flin.copy(), fang.copy()
    else:
        rlin, rang = rng.normal(0.0, 0.3, (B, 3)), rng.normal(0.0, 0.8, (B, 3))

    soft_lo = -0.5 - 0.02 * np.arange(J)
    soft_hi = 0.5 + 0.02 * np.arange(J)
    mid, half = (soft_lo + soft_hi) / 2, (soft_hi - soft_lo) / (2 * S.SOFT_LIMIT_FACTOR)
    return {
        "kind": kind,
        "robot_body_pos": rbp, "robot_body_quat": rbq,
        "robot_body_lin_vel": rlin, "robot_body_ang_vel": rang,
        "ref_body_pos": refbp, "ref_body_quat": refbq,
        "ref_body_lin_vel": flin, "ref_body_ang_vel": fang,
        "robot_root_z": float(rob_p[2]), "robot_root_quat": rob_q,
        "ref_root_z": float(ref_p[2]), "ref_root_quat": ref_q,
        "joint_pos": np.linspace(-0.7, 0.9, J),
        "action": rng.normal(0.0, 0.4, J),
        "prev_action": rng.normal(0.0, 0.4, J),
        "soft_lo": soft_lo, "soft_hi": soft_hi,
        "q_lo": mid - half, "q_hi": mid + half,
    }


def as_state(d: dict) -> S.V2State:
    keys = set(S.V2State.__dataclass_fields__) - {"qpos", "qvel"}
    return S.V2State(**{k: d[k] for k in keys})


# --------------------------------------------------------------------------
# Oracle calls
# --------------------------------------------------------------------------

NM_TO_V2 = {
    "body_pos": "motion_body_pos", "body_ori": "motion_body_ori",
    "body_lin_vel": "motion_body_lin_vel", "body_ang_vel": "motion_body_ang_vel",
    "root_pos": "motion_global_root_pos", "root_ori": "motion_global_root_ori",
    "action_rate": "action_rate_l2", "joint_limit": "joint_limit",
}


def v2_reward_terms(d: dict) -> dict[str, float]:
    return S.reward_terms(as_state(d))


def sm_reward(d: dict, mdp=None) -> tuple[float, dict[str, float]]:
    """Real ``supermimic.mdp.MDP.reward`` on the shared state."""
    import torch  # noqa: PLC0415
    mdp = mdp or load_supermimic()
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)

    class Kin:
        q_lo = t(d["q_lo"])
        q_hi = t(d["q_hi"])

        def forward(self, p, R, v, w, q, qd):
            return (t(d["robot_body_pos"]),
                    t(np.stack([S.quat_to_mat(q) for q in d["robot_body_quat"]])),
                    t(d["robot_body_lin_vel"]), t(d["robot_body_ang_vel"]))

    class Motion:
        def __init__(self):
            self.root_p = t(d["ref_body_pos"][0])[None]
            self.root_R = t(S.quat_to_mat(d["ref_body_quat"][0]))[None]
            self.body_p = t(d["ref_body_pos"])[None]
            self.body_R = t(np.stack([S.quat_to_mat(q) for q in d["ref_body_quat"]]))[None]
            self.body_v = t(d["ref_body_lin_vel"])[None]
            self.body_w = t(d["ref_body_ang_vel"])[None]

    head = mdp.MDP(Kin(), Motion(), t(np.zeros(J)))
    state = (t(d["robot_body_pos"][0]), t(S.quat_to_mat(d["robot_body_quat"][0])),
             t(np.zeros(3)), t(np.zeros(3)), t(d["joint_pos"]), t(np.zeros(J)))
    r, terms = head.reward(state, 0, t(d["action"]), t(d["prev_action"]), terms=True)
    return float(r), {k: float(v) for k, v in terms.items()}


def nm_reward(d: dict, nenv=None) -> tuple[float, dict[str, float]]:
    """Real ``nablamimic.env.TrackingEnv.reward`` on the shared state."""
    import jax.numpy as jnp  # noqa: PLC0415
    nenv = nenv or load_nablamimic()
    j = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float32)

    e = nenv.TrackingEnv.__new__(nenv.TrackingEnv)   # skip model build (needs scene XML)
    e.cfg = nenv.EnvConfig(residual_action=False, action_rate_space="setpoint", dr=False)
    e.body_ids = jnp.arange(B)
    e.anchor_k = S.ANCHOR_INDEX
    e.root_com_id = 0
    e.nu = J
    e.soft_lo, e.soft_hi = j(d["soft_lo"]), j(d["soft_hi"])
    e.ref_body_pos = j(d["ref_body_pos"])[None]
    e.ref_body_quat = j(d["ref_body_quat"])[None]
    e.ref_body_lin = j(d["ref_body_lin_vel"])[None]
    e.ref_body_ang = j(d["ref_body_ang_vel"])[None]

    xpos, xquat = j(d["robot_body_pos"]), j(d["robot_body_quat"])
    ang, lin = j(d["robot_body_ang_vel"]), j(d["robot_body_lin_vel"])
    com = jnp.zeros((1, 3))
    cvel = jnp.concatenate([ang, lin + jnp.cross(xpos - com[0], ang)], axis=-1)
    qpos = jnp.concatenate([j(np.array([0.0, 0.0, d["robot_root_z"]])),
                            j(d["robot_root_quat"]), j(d["joint_pos"])])
    dx = types.SimpleNamespace(xpos=xpos, xquat=xquat, cvel=cvel,
                               subtree_com=com, qpos=qpos, qvel=jnp.zeros(35))
    state = types.SimpleNamespace(dx=dx, frame=jnp.int32(0))
    total, terms = e.reward(state, j(d["action"]), j(d["prev_action"]), prev=None)
    return float(total), {k: float(v) for k, v in terms.items()}


# --------------------------------------------------------------------------
# Termination oracles
# --------------------------------------------------------------------------

def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def sm_termination(d: dict, *, dz: float | None = None, tilt: float | None = None,
                   nan: bool = False, mdp=None, head=None) -> bool:
    import torch  # noqa: PLC0415
    mdp = mdp or load_supermimic()
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
    if head is None:
        class Kin:
            q_lo = t(d["q_lo"])
            q_hi = t(d["q_hi"])

            def forward(self, *a):
                return (t(d["robot_body_pos"]), t(np.stack([S.quat_to_mat(q) for q in d["robot_body_quat"]])),
                        t(d["robot_body_lin_vel"]), t(d["robot_body_ang_vel"]))

        class Motion:
            root_p = t(d["ref_body_pos"][0])[None]
            root_R = t(S.quat_to_mat(d["ref_body_quat"][0]))[None]

        head = mdp.MDP(Kin(), Motion(), t(np.zeros(J)))
    p = d["robot_body_pos"][0].copy()
    p[2] = d["ref_root_z"] + (dz if dz is not None else d["robot_root_z"] - d["ref_root_z"])
    R = S.quat_to_mat(d["robot_root_quat"])
    if tilt is not None:
        R = R @ _rotation_x(tilt)
    if nan:
        R[0, 0] = float("nan")
    state = (t(p), t(R), t(np.zeros(3)), t(np.zeros(3)), t(d["joint_pos"]), t(np.zeros(J)))
    return bool(head.termination(state, 0).item())


def nm_termination(d: dict, *, dz: float | None = None, tilt: float | None = None,
                   nan: bool = False, nenv=None, e=None) -> bool:
    import jax.numpy as jnp  # noqa: PLC0415
    nenv = nenv or load_nablamimic()
    j = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float32)
    if e is None:
        e = nenv.TrackingEnv.__new__(nenv.TrackingEnv)
        e.cfg = nenv.EnvConfig(residual_action=False, action_rate_space="setpoint", dr=False)
    z = d["ref_root_z"] + (dz if dz is not None else d["robot_root_z"] - d["ref_root_z"])
    q = d["robot_root_quat"]
    if tilt is not None:
        q = S.mat_to_quat(S.quat_to_mat(d["robot_root_quat"]) @ _rotation_x(tilt))
    qpos = jnp.concatenate([j(np.array([0.0, 0.0, z])), j(q), j(d["joint_pos"])])
    ref_qpos = jnp.concatenate([j(np.array([0.0, 0.0, d["ref_root_z"]])),
                                j(d["ref_root_quat"]), j(d["joint_pos"])])[None]
    e.ref_qpos = ref_qpos
    if nan:
        qpos = qpos.at[0].set(jnp.nan)
    dx = types.SimpleNamespace(qpos=qpos, qvel=jnp.zeros(35))
    return bool(e.terminated(types.SimpleNamespace(dx=dx, frame=jnp.int32(0))))


def v2_termination(d: dict, *, dz: float | None = None, tilt: float | None = None,
                   nan: bool = False, ee_deviation: float = 0.0) -> S.TerminationResult:
    st = as_state(d)
    z = d["ref_root_z"] + (dz if dz is not None else d["robot_root_z"] - d["ref_root_z"])
    q = d["robot_root_quat"]
    if tilt is not None:
        q = S.mat_to_quat(S.quat_to_mat(d["robot_root_quat"]) @ _rotation_x(tilt))
    pos = st.robot_body_pos.copy()
    if ee_deviation:
        for k in (3, 6, 10, 13):       # ankles / wrists
            pos[k] += np.array([0.0, 0.0, ee_deviation])
    st = dataclasses.replace(st, robot_root_z=z, robot_root_quat=q, robot_body_pos=pos,
                             qpos=np.array([0.0, 0.0, float("nan")]) if nan else None)
    return S.termination(st)


# --------------------------------------------------------------------------
# Frozen oracle snapshot (measured from the real sources, for GPU-free CI)
# --------------------------------------------------------------------------

SNAPSHOT_PATH = TESTS_DIR / "v2_oracle_snapshot.json"


def build_snapshot() -> dict:
    """Measure the two real oracles + V2 on the fixed fixtures.

    Run once via ``scripts/v2/verify_v2.py --write-snapshot``; the JSON is
    committed so the parity assertions run even where the upstream checkouts
    (or jax / torch) are unavailable.
    """
    mdp, nenv = load_supermimic(), load_nablamimic()
    out = {"supermimic_commit": SUPERMIMIC_COMMIT,
           "nablamimic_commit": NABLAMIMIC_COMMIT,
           "oracle_sha256": ORACLE_SHA256, "cases": {}}
    for kind in ("perfect", "coincident_anchor", "drift"):
        d = make_case(kind)
        sm_total, sm_terms = sm_reward(d, mdp)
        nm_total, nm_terms = nm_reward(d, nenv)
        v2 = v2_reward_terms(d)
        out["cases"][kind] = {
            "supermimic": {"total": sm_total, "terms": sm_terms},
            "nablamimic": {"total": nm_total, "terms": nm_terms},
            "v2": {"total": float(sum(v2.values())), "terms": v2},
        }
    return out


def load_snapshot() -> dict | None:
    import json  # noqa: PLC0415
    if not SNAPSHOT_PATH.exists():
        return None
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
