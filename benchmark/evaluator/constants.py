"""Frozen constants for ``canonical-mjlab-g1-v1``.

Every number here is copied verbatim from mjlab v1.6.0
(commit ``0fb8a681136be94ffc636a3dd423cabb97d91f10``):

  src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py
  src/mjlab/tasks/tracking/tracking_env_cfg.py
  src/mjlab/tasks/tracking/config/g1/env_cfgs.py
  src/mjlab/tasks/tracking/config/g1/rl_cfg.py
  src/mjlab/sim/sim.py                       (``MujocoCfg`` defaults)

Nothing in this module may be tuned without changing ``protocol.json`` and the
hashes in ``manifest.json``.
"""

from __future__ import annotations

from pathlib import Path

EVALUATOR_IMPLEMENTATION = "canonical-mjlab-g1-v1"

MJLAB_REPO = "mujocolab/mjlab"
MJLAB_VERSION = "1.6.0"
# Tag ``v1.6.0`` of mujocolab/mjlab (the commit PyPI's mjlab==1.6.0 was built from).
MJLAB_COMMIT = "0fb8a681136be94ffc636a3dd423cabb97d91f10"

# LAFAN1 retargeted G1 dataset revision (HuggingFace
# ``lvhaidong/LAFAN1_Retargeting_Dataset``), plus the 40 clips of the protocol.
LAFAN1_HF_REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
LAFAN1_HF_REVISION = "07ff8845a94d59d4195a8ac04fe918d4e6338239"
LAFAN1_INPUT_FPS = 30.0

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "assets" / "g1"
DATA_DIR = REPO_ROOT / "data" / "lafan1_g1"
MANIFEST_PATH = Path(__file__).with_name("manifest.json")

# --------------------------------------------------------------------------
# Robot model: 29 DoF, full self-collision (mjlab ``FULL_COLLISION``).
# --------------------------------------------------------------------------

ROBOT_DOF = 29
FREE_JOINT_NAME = "floating_base_joint"

#: Control/observation joint order == mjlab "natural" joint order, which for the
#: G1 MJCF is also the LAFAN1 CSV column order (asserted at model build time).
JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

#: Bodies tracked by the reward/termination terms (mjlab
#: ``unitree_g1_flat_tracking_env_cfg``), anchor first-order irrelevant.
ANCHOR_BODY = "torso_link"
TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
#: Feet used by the ``ee_body_pos`` termination.
EE_BODIES: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
FOOT_GEOM_REGEX = r"^(left|right)_foot[1-7]_collision$"
COLLISION_GEOM_REGEX = r".*_collision$"

# mjlab ``EntityCfg.InitialStateCfg`` KNEES_BENT_KEYFRAME -> ``default_joint_pos``.
# Unlisted joints default to 0.0 (``resolve_expr(..., default=0.0)``).
_KNEES_BENT = {
    r".*_hip_pitch_joint": -0.312,
    r".*_knee_joint": 0.669,
    r".*_ankle_pitch_joint": -0.363,
    r".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

#: mjlab ``G1_ARTICULATION.soft_joint_pos_limit_factor``.
SOFT_JOINT_LIMIT_FACTOR = 0.9

# --------------------------------------------------------------------------
# Actuators (mjlab ``BuiltinPositionActuatorCfg`` groups, in spec order).
# --------------------------------------------------------------------------
# Shared motor data (Unitree).
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def _reflected_inertia(rotor_inertia, gear_ratio) -> float:
    """``mjlab.utils.actuator.reflected_inertia_from_two_stage_planetary``."""
    assert gear_ratio[0] == 1
    r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
    r2 = rotor_inertia[1] * gear_ratio[2] ** 2
    r3 = rotor_inertia[2]
    return r1 + r2 + r3


ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))
ARMATURE_5020 = _reflected_inertia(ROTOR_INERTIAS_5020, GEARS_5020)

ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))
ARMATURE_7520_14 = _reflected_inertia(ROTOR_INERTIAS_7520_14, GEARS_7520_14)

ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)
ARMATURE_7520_22 = _reflected_inertia(ROTOR_INERTIAS_7520_22, GEARS_7520_22)

ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)
ARMATURE_4010 = _reflected_inertia(ROTOR_INERTIAS_4010, GEARS_4010)


class ActuatorGroup:
    __slots__ = ("name", "patterns", "stiffness", "damping", "effort_limit", "armature")

    def __init__(self, name, patterns, stiffness, damping, effort_limit, armature):
        self.name = name
        self.patterns = patterns
        self.stiffness = stiffness
        self.damping = damping
        self.effort_limit = effort_limit
        self.armature = armature


def _group(name, patterns, armature, effort_limit):
    return ActuatorGroup(
        name=name,
        patterns=patterns,
        stiffness=armature * NATURAL_FREQ**2,
        damping=2.0 * DAMPING_RATIO * armature * NATURAL_FREQ,
        effort_limit=effort_limit,
        armature=armature,
    )


#: Order matters: it fixes the ``ctrl`` layout (mjlab adds actuators group by group).
ACTUATOR_GROUPS: tuple[ActuatorGroup, ...] = (
    _group(
        "5020",
        (
            r".*_elbow_joint",
            r".*_shoulder_pitch_joint",
            r".*_shoulder_roll_joint",
            r".*_shoulder_yaw_joint",
            r".*_wrist_roll_joint",
        ),
        armature=ARMATURE_5020,
        effort_limit=25.0,
    ),
    _group(
        "7520_14",
        (r".*_hip_pitch_joint", r".*_hip_yaw_joint", r"waist_yaw_joint"),
        armature=ARMATURE_7520_14,
        effort_limit=88.0,
    ),
    _group(
        "7520_22",
        (r".*_hip_roll_joint", r".*_knee_joint"),
        armature=ARMATURE_7520_22,
        effort_limit=139.0,
    ),
    _group(
        "4010",
        (r".*_wrist_pitch_joint", r".*_wrist_yaw_joint"),
        armature=ARMATURE_4010,
        effort_limit=5.0,
    ),
    # Waist pitch/roll and ankles are 4-bar linkages driven by two 5020 motors.
    _group(
        "waist_5020x2",
        (r"waist_pitch_joint", r"waist_roll_joint"),
        armature=2.0 * ARMATURE_5020,
        effort_limit=2.0 * 25.0,
    ),
    _group(
        "ankle_5020x2",
        (r".*_ankle_pitch_joint", r".*_ankle_roll_joint"),
        armature=2.0 * ARMATURE_5020,
        effort_limit=2.0 * 25.0,
    ),
)

#: mjlab ``G1_ACTION_SCALE[joint] = 0.25 * effort_limit / stiffness``.
ACTION_SCALE = 0.25


# --------------------------------------------------------------------------
# Simulation / control loop (mjlab ``SimulationCfg`` + ``MujocoCfg``).
# --------------------------------------------------------------------------

TIMESTEP = 0.005
DECIMATION = 4
STEP_DT = TIMESTEP * DECIMATION  # 0.02 s -> 50 Hz policy / control rate
SOLVER_ITERATIONS = 10
SOLVER_LS_ITERATIONS = 20
SOLVER_TOLERANCE = 1e-8
SOLVER_LS_TOLERANCE = 0.01
CCD_ITERATIONS = 50
IMPRATIO = 1.0
CONE = "pyramidal"
INTEGRATOR = "implicitfast"
SOLVER = "newton"
GRAVITY = (0.0, 0.0, -9.81)
GROUND_FRICTION = (1.0, 0.005, 0.0001)  # MuJoCo default for the mjlab plane
FOOT_FRICTION = 0.6  # mjlab FULL_COLLISION
SELF_COLLISION_FORCE_THRESHOLD = 10.0  # N, mjlab self_collisions term

# --------------------------------------------------------------------------
# Observation / action semantics (actor group, ``play=True``).
# --------------------------------------------------------------------------
# concatenated in this exact order; total 160 dims.
OBS_TERMS: tuple[tuple[str, int], ...] = (
    ("command", 58),  # reference joint_pos (29) ++ joint_vel (29)
    ("motion_anchor_pos_b", 3),
    ("motion_anchor_ori_b", 6),
    ("base_lin_vel", 3),  # robot/imu_lin_vel
    ("base_ang_vel", 3),  # robot/imu_ang_vel
    ("joint_pos", 29),  # joint_pos_rel, biased=True (encoder_bias frozen to 0)
    ("joint_vel", 29),
    ("actions", 29),  # raw previous action
)
OBS_DIM = sum(dim for _, dim in OBS_TERMS)  # 160
ACTION_DIM = ROBOT_DOF  # 29

#: rsl_rl actor architecture (``rl_cfg.py``); obs normalization enabled.
ACTOR_HIDDEN_DIMS = (512, 256, 128)
ACTOR_ACTIVATION = "elu"
OBS_NORMALIZATION_EPS = 1e-2  # rsl_rl EmpiricalNormalization

# --------------------------------------------------------------------------
# Rewards (``tracking_env_cfg.rewards``, ``scale_rewards_by_dt=True``).
# --------------------------------------------------------------------------

REWARDS = {
    "motion_global_root_pos": {"weight": 0.5, "std": 0.3},
    "motion_global_root_ori": {"weight": 0.5, "std": 0.4},
    "motion_body_pos": {"weight": 1.0, "std": 0.3},
    "motion_body_ori": {"weight": 1.0, "std": 0.4},
    "motion_body_lin_vel": {"weight": 1.0, "std": 1.0},
    "motion_body_ang_vel": {"weight": 1.0, "std": 3.14},
    "action_rate_l2": {"weight": -0.1},
    "joint_limit": {"weight": -10.0},
    "self_collisions": {"weight": -10.0},
}

#: Max per-step *positive* reward used to normalize the return. Identical to the
#: denominator implicit in the published baselines (sum of positive weights x dt).
MAX_STEP_REWARD = sum(v["weight"] for v in REWARDS.values() if v["weight"] > 0) * STEP_DT

# --------------------------------------------------------------------------
# Terminations (``tracking_env_cfg.terminations``).
# --------------------------------------------------------------------------

TERM_ANCHOR_POS_THRESHOLD = 0.25  # m, |z error| of torso_link
TERM_ANCHOR_ORI_THRESHOLD = 0.8  # 1 - z of relative projected gravity
TERM_EE_BODY_POS_THRESHOLD = 0.25  # m, per-body z-only error
