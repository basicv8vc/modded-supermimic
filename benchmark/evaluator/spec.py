"""Build the canonical G1 tracking model from the official mjlab MJCF.

Mirrors what mjlab v1.6.0 does when it builds ``Mjlab-Tracking-Flat-Unitree-G1``:

1. ``MjSpec.from_file(g1.xml)`` (official ``unitree_g1/xmls/g1.xml``).
2. ``FULL_COLLISION``: every ``*_collision`` geom collides with itself and with
   the ground; feet get ``condim=3``/``priority=1``/``friction=0.6``, every other
   colliding geom gets ``condim=1``/``priority=0``; non-collision (visual) geoms
   are switched off.
3. One MuJoCo ``<position>`` actuator per joint (``kp``/``kv``/``forcerange``),
   added group by group in ``G1_ARTICULATION`` order; ``armature`` written onto
   the joint. ``ctrllimited=False`` so the policy may command beyond the joint
   range, ``forcerange`` still bounds the torque.
4. A flat ground plane (``mjlab TerrainEntityCfg(terrain_type="plane")``).
5. ``MujocoCfg`` solver/integrator settings applied to the compiled model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import constants as C


@dataclass
class ModelInfo:
    """Resolved indices/parameters of the canonical model."""

    model: mujoco.MjModel
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_ids: np.ndarray  # model joint id per entry of JOINT_ORDER
    qpos_adr: np.ndarray  # qpos address per joint
    dof_adr: np.ndarray  # dof address per joint
    body_ids: np.ndarray  # model body id per TRACKED_BODIES
    anchor_body_id: int
    ee_body_ids: np.ndarray
    actuator_ids: np.ndarray  # actuator id per entry of JOINT_ORDER
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    joint_pos_limits: np.ndarray  # (29, 2) hard limits
    soft_joint_pos_limits: np.ndarray  # (29, 2)
    imu_ang_vel_adr: int
    imu_lin_vel_adr: int
    terrain_geom_id: int
    robot_geom_ids: frozenset[int] = field(default_factory=frozenset)

    @property
    def njoint(self) -> int:
        return len(self.joint_ids)


def _matching_joint_names(spec: mujoco.MjSpec) -> list[str]:
    return [j.name for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]


def _resolve_default_joint_pos(joint_names: tuple[str, ...]) -> np.ndarray:
    out = np.zeros(len(joint_names), dtype=np.float64)
    for i, name in enumerate(joint_names):
        for pattern, value in C._KNEES_BENT.items():
            if re.fullmatch(pattern, name):
                out[i] = value
                break
    return out


def build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(C.ASSET_DIR / "g1.xml"))

    joint_names = _matching_joint_names(spec)
    if tuple(joint_names) != C.JOINT_ORDER:
        raise RuntimeError(
            "g1.xml joint order does not match the frozen JOINT_ORDER; refusing to "
            f"evaluate.\n  xml: {joint_names}\n  exp: {list(C.JOINT_ORDER)}"
        )

    foot_re = re.compile(C.FOOT_GEOM_REGEX)
    coll_re = re.compile(C.COLLISION_GEOM_REGEX)
    for geom in spec.geoms:
        name = geom.name or ""
        if coll_re.fullmatch(name):
            geom.contype = 1
            geom.conaffinity = 1
            if foot_re.fullmatch(name):
                geom.condim = 3
                geom.priority = 1
                geom.friction[0] = C.FOOT_FRICTION
            else:
                geom.condim = 1
                geom.priority = 0
        else:
            # mjlab CollisionCfg(disable_other_geoms=True)
            geom.contype = 0
            geom.conaffinity = 0

    # Flat ground plane, exactly as mjlab's TerrainEntity builds it (no class ->
    # MuJoCo defaults: contype=conaffinity=1, condim=3, friction=1 0.005 0.0001).
    spec.worldbody.add_geom(
        name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0.0, 0.0, 0.01)
    )

    # Position actuators, group by group, in mjlab's order.
    for group in C.ACTUATOR_GROUPS:
        patterns = [re.compile(p) for p in group.patterns]
        for joint_name in joint_names:
            if not any(p.fullmatch(joint_name) for p in patterns):
                continue
            joint = spec.joint(joint_name)
            joint.armature = group.armature

            act = spec.add_actuator(name=joint_name, target=joint_name)
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.dyntype = mujoco.mjtDyn.mjDYN_NONE
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm[0] = group.stiffness
            act.biasprm[1] = -group.stiffness
            act.biasprm[2] = -group.damping
            # Allow setpoints beyond the joint range; force is bounded separately.
            act.inheritrange = 0.0
            act.ctrllimited = False
            act.forcelimited = True
            act.forcerange[0] = -group.effort_limit
            act.forcerange[1] = group.effort_limit
            delta = group.effort_limit / group.stiffness
            lo, hi = joint.range[0], joint.range[1]
            act.ctrlrange[0] = lo - delta
            act.ctrlrange[1] = hi + delta

    return spec


def build_model() -> ModelInfo:
    spec = build_spec()
    model = spec.compile()

    # mjlab MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20) + defaults.
    model.opt.timestep = C.TIMESTEP
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = C.SOLVER_ITERATIONS
    model.opt.ls_iterations = C.SOLVER_LS_ITERATIONS
    model.opt.tolerance = C.SOLVER_TOLERANCE
    model.opt.ls_tolerance = C.SOLVER_LS_TOLERANCE
    model.opt.ccd_iterations = C.CCD_ITERATIONS
    model.opt.impratio = C.IMPRATIO
    model.opt.gravity[:] = C.GRAVITY

    joint_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(1, model.njnt)
    )
    if joint_names != C.JOINT_ORDER:
        raise RuntimeError("compiled joint order mismatch")

    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.JOINT_ORDER]
    )
    qpos_adr = model.jnt_qposadr[joint_ids].copy()
    dof_adr = model.jnt_dofadr[joint_ids].copy()

    body_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, model.nbody)
    )
    body_id = {n: i for i, n in enumerate(body_names, start=1)}

    actuator_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in C.JOINT_ORDER
        ]
    )
    if np.any(actuator_ids < 0):
        raise RuntimeError("missing actuator for a joint")

    default_joint_pos = _resolve_default_joint_pos(C.JOINT_ORDER)
    # G1_ACTION_SCALE = 0.25 * effort_limit / stiffness, per actuator group.
    action_scale = np.zeros(C.ROBOT_DOF, dtype=np.float64)
    for group in C.ACTUATOR_GROUPS:
        patterns = [re.compile(p) for p in group.patterns]
        for i, name in enumerate(C.JOINT_ORDER):
            if any(p.fullmatch(name) for p in patterns):
                action_scale[i] = C.ACTION_SCALE * group.effort_limit / group.stiffness

    limits = np.stack(
        [model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1]], axis=-1
    )
    mean = limits.mean(axis=-1, keepdims=True)
    half = 0.5 * (limits[:, 1:] - limits[:, 0:1])
    soft = np.concatenate(
        [mean - half * C.SOFT_JOINT_LIMIT_FACTOR, mean + half * C.SOFT_JOINT_LIMIT_FACTOR],
        axis=-1,
    )
    limited = model.jnt_limited[joint_ids].astype(bool)
    limits[~limited] = (-np.inf, np.inf)
    soft[~limited] = (-np.inf, np.inf)

    def sensor_adr(name: str) -> int:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f"sensor {name} missing from g1.xml")
        return int(model.sensor_adr[sid])

    terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    robot_geoms = frozenset(
        g
        for g in range(model.ngeom)
        if g != terrain_id and model.geom_bodyid[g] != 0
    )

    return ModelInfo(
        model=model,
        joint_names=joint_names,
        body_names=body_names,
        joint_ids=joint_ids,
        qpos_adr=qpos_adr,
        dof_adr=dof_adr,
        body_ids=np.array([body_id[n] for n in C.TRACKED_BODIES]),
        anchor_body_id=body_id[C.ANCHOR_BODY],
        ee_body_ids=np.array([body_id[n] for n in C.EE_BODIES]),
        actuator_ids=actuator_ids,
        default_joint_pos=default_joint_pos,
        action_scale=action_scale,
        joint_pos_limits=limits,
        soft_joint_pos_limits=soft,
        imu_ang_vel_adr=sensor_adr("imu_ang_vel"),
        imu_lin_vel_adr=sensor_adr("imu_lin_vel"),
        terrain_geom_id=terrain_id,
        robot_geom_ids=robot_geoms,
    )


def fingerprint(info: ModelInfo) -> dict[str, int]:
    """Structural fingerprint of the compiled model (for conformance tests)."""
    m = info.model
    return {
        "nbody": int(m.nbody),
        "njnt": int(m.njnt),
        "ngeom": int(m.ngeom),
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nu": int(m.nu),
        "ngeom_robot": int(len(info.robot_geom_ids)),
    }
