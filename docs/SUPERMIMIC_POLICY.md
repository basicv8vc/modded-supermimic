# SuperMimic native inference on the canonical evaluator

First real, measured run of a trained SuperMimic policy on the frozen
`canonical-mjlab-g1-v1` evaluator. This document records what was run, how the
native inference was rebuilt, how it was verified, and what the numbers mean —
including why they are far below the 0.82 gate.

**Headline:** mean normalized return **0.2221**, **32 / 40 failed clips**,
validator `INVALID` for one reason only — the run happened on a CPU machine, not
a Colab G4. The 0.84165 from the training stack is *not* a canonical score and
is not recorded as one.

---

## 1. What was run

| Item | Value |
| --- | --- |
| Checkpoint | `ckpt_015000.pt`, 71.8 MB |
| SHA256 | `93bc5c43185d21f4ca9be7cde526d313c0cf71fcee4ef46c26ff6e29c23955e7` |
| Checkpoint metadata | iteration 15000, `wall_time = 5223.30 s`, `saved_at = 2026-11-03 06:13:58` |
| Training code | `zita-ch/supermimic_vibed` @ `8bfc1158bd968f729fcc65d5e897a05b05ea3d36` |
| Training simulator | IsaacLab 3 (kit-less) + Newton/MuJoCo-Warp, 4096 envs |
| Trained `args` | `goal_frames=10`, `hist=10`, `amp=bf16`, `seed=0`, `num_envs=4096` |
| Evaluation simulator | this repo, MuJoCo 3.11.0 CPU, full self-collision |
| Evaluation host | macOS arm64, CPU only (no CUDA), NumPy 2.5.4, Python 3.13.12 |
| Policy implementation | `benchmark/policies/supermimic.py` (NumPy, no torch) |

The checkpoint is **not** modified, re-trained, fine-tuned or converted: the
`pi.enc` / `pi.dec` tensors are read out of the `torch.save` archive with the
repo's existing torch-free reader and executed with their own float32 values.

---

## 2. What the original implementation actually does

Read from the sources at the training commit, not inferred from tensor shapes.

| Upstream | Native behaviour | Where it lives here |
| --- | --- | --- |
| `supermimic/mdp.py` `MDP.obs` | 99-dim single frame: `q - default_q - dq0` (29), `qd` (29), `R^T w` (3), `-R[2,:]` (3), `a_prev` (29), `mat_to_6d(R^T R_ref[f])` (6) | `SuperMimicPolicy.observation` |
| `supermimic/motion.py` `MotionLib.goal` | 730-dim = 10 frames x (dp 3, 6D rot 6, root v 3, root w 3, q 29, qd*0.1 29), frames `f+1..f+10` clamped to the clip end with zeroed velocity beyond it | `SuperMimicPolicy.goal` |
| `supermimic/motion.py` `_gradient`, root angular velocity | central differences, one-sided at the first/last frame; `so3_log(R[i+1] R[i-1]^T) / 2dt`, then rotated into the root frame | `reference_derivatives` |
| `supermimic/networks.py` `Policy` | `z = tanh(enc(goal_norm(goal)))`, `a = dec(cat(z, obs_norm(hist).flatten()))`, ELU, deterministic | `SuperMimicPolicy.forward` |
| `supermimic/networks.py` `RunningNorm` | `(x - mean) / sqrt(var + eps^2)`, `eps = 1e-2`, frozen buffers | inline in `forward` |
| `supermimic/evaluate.py` | history initialised by repeating the first observation 10 times, then `hist[1:] + new` (newest last) | `act` |
| `supermimic/env.py` | `target = default_q + dq0 + alpha * a`, `dq0 = 0` when domain randomisation is off (eval mode) | not needed — identical to the evaluator's contract (§4) |
| `supermimic/robot.py` | joint order, default pose, `alpha = 0.25 * effort_limit / stiffness` | `JOINT_ORDER`, `default_joint_pos_reference`, `action_scale_reference` |

**No recurrent, world-model or critic state is restored, because none is used.**
`supermimic/evaluate.py` calls exactly one function per step — `pi(hist, goal)` —
and never touches `wm`. The checkpoint holds 46 tensors: 26 `pi.*` (the actor and
its two normalisers) and 20 `wm.*` (20 of them the training-only world model,
which this implementation ignores).

Network shape, from the weights: encoder `730 -> 2048 -> 1024 -> 512 -> 256 -> 32`
(then `tanh`), decoder `1022 -> 2048 -> 1024 -> 512 -> 256 -> 29`. 9,118,135
numbers in the actor path (≈4.2 M encoder, ≈4.9 M decoder, plus the two
normalisers).

### The three inputs that are not handed over directly

The evaluator exposes physical quantities, not SuperMimic's internal state. Three
of the six observation blocks are exactly what it reads (`joint_pos`,
`joint_vel`, IMU `base_ang_vel`, plus the previous action). Two are not, because
SuperMimic reads the **pelvis** and the evaluator tracks a **torso** anchor:

```
R_torso = R_pelvis @ A(q),   A = Rz(waist_yaw) Rx(waist_roll) Ry(waist_pitch)
```

`assets/g1/g1.xml` puts those three joints between `pelvis` and `torso_link`
with rotation-free body offsets, so with `M = R_torso^T R_ref_torso` (the
evaluator's `anchor_ori_b`, 6D):

```
R_pelvis^T R_ref_pelvis = A(q_robot) M A(q_ref)^T          -> observation[93:99]
projected gravity       = A(q_robot) M A(q_ref)^T g_ref    -> observation[61:64]
```

No simulator handle is used, nothing is guessed, and both identities are checked
against `MjData` every step of a rollout (§3.2). The goal needs nothing extra:
`root_pos_w`, `root_quat_w` and `joint_pos` of the reference plus SuperMimic's
own derivative formulas reproduce it exactly.

One more gap worth stating: the goal uses the reference root's linear and
angular velocity, which `ReferenceClip` does not expose. They are recomputed from
the exposed positions and orientations with `MotionLib`'s own formulas rather
than taken from the evaluator's `joint_vel` array (whose edge frames use
`np.gradient`'s second-order one-sided stencil, unlike SuperMimic's first-order
one).

---

## 3. Verification

### 3.1 Weights and shapes

* SHA256 of the file matches the value in the task brief, byte for byte.
* All 26 `pi.*` tensors load; a missing one raises instead of silently zero-filling.
* `hist` and `goal_frames` are *derived from the weight widths* (`dec` input
  `1022 = 32 + 10*99`, `enc` input `730 = 10*73`) and must equal 10/10, so a
  differently-configured checkpoint is rejected rather than mis-fed.

### 3.2 The 99-dim observation equals what the original would compute

`tests/test_supermimic_policy.py::ObservationReconstructionTest` drives the real
`TrackingEnv` for 40 steps and compares, every step, the vector the policy builds
from `StepInput` against the vector built directly from `MjData` with
`supermimic.mdp.MDP.obs`'s own expression:

```
max |obs_policy - obs_from_MjData| = 1.0e-07   (float32 reference precision)
```

An approximation would be O(1) — the first version of this check was off by 1.0
because the 6D block was decoded as columns instead of `mat[..., :2].reshape(-1)`.

### 3.3 Goal, reference derivatives and network against the original code

`scripts/check_supermimic_parity.py` imports the upstream modules (with FK
stubbed, since IsaacLab is unavailable) and compares tensor by tensor:

| Comparison | `fallAndGetUp3_subject1` | `walk4_subject1` |
| --- | --- | --- |
| reference `q` (rad) | 7.6e-05 | 2.2e-04 |
| reference `qd` (rad/s) | 2.3e-03 | 2.2e-02 |
| reference root `v`, `w` in root frame | 3.0e-04 / 8.8e-04 | 4.6e-04 / 9.9e-04 |
| **goal (730-d), 13 frames** | **8.3e-05** | 1.6e-01 (last 10 frames only) |
| **action (29-d), 200 steps, fp32** | **5.2e-06** | **5.2e-06** |
| action, bf16 weights (the GPU path) | 4.0e-02 | 4.0e-02 |
| closed loop, numpy vs torch vs torch-bf16 | identical (`0.062031`, 1239/5108, `ee_body_pos`) | identical (`0.000000`, 8194/8194) |

Reading of the table:

* reference differences are the two 50 Hz resamplers disagreeing in the last
  bits (≤2.2e-4 rad = 0.01°), not a formula difference;
* the walk4 goal difference is confined to the final 10 frames, where the two
  clip lengths differ by one frame (`walk4` canonical 8195 vs upstream 8196) and
  the clamp therefore targets a different frame;
* the action agreement of 5.2e-06 (mean 5.0e-07) on actions of magnitude ~6-9 is
  float32 GEMM ordering, and it is *consistent*: three independent
  implementations of the same network produce the same closed-loop outcome on two
  different clips.

The original evaluation ran its matmuls under `torch.autocast("cuda", bf16)`.
A CPU cannot reproduce that (torch disables CUDA autocast), so the bf16 row is
the same network with every operand and weight stored in bfloat16 — the
narrowest version of what the GPU computed. At 4e-2 max / 5e-3 mean action
deviation it is 4 orders of magnitude above the fp32 agreement and is the
largest unquantified numerical gap in this work. It does not change the
closed-loop outcome on either clip tested.

### 3.4 Contract checks (run on every `reset`)

`reset` refuses to run if the evaluator's contract has drifted from what the
checkpoint was trained with: joint order must equal `G1_JOINT_NAMES`,
`action_scale` must equal `0.25 * effort_limit / stiffness` to 1e-8, and
`default_joint_pos` must equal SuperMimic's keyframe to 1e-12. On this build all
three hold — the action-scale residual is 2.2e-9 on the hip joints, because
`supermimic/robot.py` writes its armature constants to 9 decimals while the
evaluator derives them from rotor inertias.

**Consequence: the 29 numbers SuperMimic emits are used unchanged.** Both stacks
compute `target = default_joint_pos + action_scale * action` with the same joint
order, the same default pose and the same scale, so there is no unit, sign,
ordering or PD-setpoint conversion to apply — and none was applied.

### 3.5 Reset, determinism, finiteness

`reset` clears the history buffer and the reference derivatives, so a second clip
cannot inherit the first one's window; the first `act` repeats the first
observation 10 times, matching `evaluate.py`; the same input state produces the
same action bit for bit; every action is checked finite and `(29,)` before it is
returned (and again by the evaluator).

---

## 4. Results — 40 frozen clips

Full record: `reports/supermimic_canonical_eval.json` (schema produced by
`runner.build_record`, audited by `benchmark.py`).
Per-clip table: `reports/supermimic_per_clip.csv`.
Failure evidence: `reports/supermimic_diagnostics.json`.
Cost: `reports/supermimic_timing.json`.

| Metric | Value |
| --- | --- |
| Mean normalized return | **0.2221** |
| Median / best | 0.1436 / 0.7893 (`walk3_subject5`) |
| Failed clips | **32 / 40** |
| Gate | 0.82 and 0/40 — **not met** |
| Validator | `INVALID`, sole error `GPU model is not Colab G4 RTX PRO 6000 Blackwell` |

Per-clip returns (frozen metric, failed episodes masked to zero after termination):

| clip | return | failed | alive / steps | termination |
| --- | ---: | :---: | ---: | --- |
| `dance1_subject1` | 0.3044 | yes | 3617 / 6573 | ee_body_pos |
| `dance1_subject2` | 0.1348 | yes | 1708 / 6573 | ee_body_pos |
| `dance1_subject3` | 0.1523 | yes | 3306 / 6573 | ee_body_pos |
| `dance2_subject1` | 0.1170 | yes | 1940 / 11283 | ee_body_pos |
| `dance2_subject2` | 0.1720 | yes | 5240 / 11283 | ee_body_pos |
| `dance2_subject3` | 0.4074 | yes | 6900 / 11283 | ee_body_pos |
| `dance2_subject4` | 0.0157 | yes | 2967 / 11283 | ee_body_pos |
| `dance2_subject5` | 0.1614 | yes | 2919 / 11283 | ee_body_pos |
| `fallAndGetUp1_subject1` | 0.0019 | yes | 345 / 8409 | ee_body_pos |
| `fallAndGetUp1_subject4` | 0.0000 | yes | 846 / 8409 | ee_body_pos |
| `fallAndGetUp1_subject5` | 0.0198 | yes | 457 / 8409 | ee_body_pos |
| `fallAndGetUp2_subject2` | 0.0000 | yes | 4673 / 8194 | ee_body_pos |
| `fallAndGetUp2_subject3` | 0.0606 | yes | 788 / 8194 | ee_body_pos |
| `fallAndGetUp3_subject1` | 0.0620 | yes | 1239 / 5108 | ee_body_pos |
| `fight1_subject2` | 0.0315 | yes | 949 / 12243 | ee_body_pos |
| `fight1_subject3` | 0.0000 | yes | 2285 / 12243 | ee_body_pos |
| `fight1_subject5` | 0.0631 | yes | 1188 / 12243 | ee_body_pos |
| `fightAndSports1_subject1` | 0.0198 | yes | 257 / 12256 | ee_body_pos |
| `fightAndSports1_subject4` | 0.0000 | yes | 1478 / 12256 | ee_body_pos |
| `jumps1_subject1` | 0.2577 | yes | 5103 / 12221 | ee_body_pos |
| `jumps1_subject2` | 0.0434 | yes | 580 / 12221 | ee_body_pos |
| `jumps1_subject5` | 0.0108 | yes | 212 / 12221 | ee_body_pos |
| `run1_subject2` | 0.2101 | yes | 3244 / 11889 | ee_body_pos |
| `run1_subject5` | 0.1916 | yes | 5511 / 11889 | ee_body_pos |
| `run2_subject1` | 0.5121 | no | 12239 / 12239 | — |
| `run2_subject4` | 0.2369 | yes | 9933 / 12239 | ee_body_pos |
| `sprint1_subject2` | 0.5079 | yes | 10068 / 13654 | ee_body_pos |
| `sprint1_subject4` | 0.1070 | yes | 3458 / 13654 | ee_body_pos |
| `walk1_subject1` | 0.4837 | no | 13064 / 13064 | — |
| `walk1_subject2` | 0.5945 | no | 13064 / 13064 | — |
| `walk1_subject5` | 0.5580 | no | 13064 / 13064 | — |
| `walk2_subject1` | 0.0000 | yes | 9621 / 11908 | anchor_pos |
| `walk2_subject3` | 0.3034 | yes | 9438 / 11908 | ee_body_pos |
| `walk2_subject4` | 0.6815 | yes | 11178 / 11908 | ee_body_pos |
| `walk3_subject1` | 0.4645 | yes | 8208 / 12329 | ee_body_pos |
| `walk3_subject2` | 0.6376 | no | 12329 / 12329 | — |
| `walk3_subject3` | 0.0000 | yes | 8675 / 12329 | ee_body_pos |
| `walk3_subject4` | 0.5701 | no | 12329 / 12329 | — |
| `walk3_subject5` | 0.7893 | no | 12329 / 12329 | — |
| `walk4_subject1` | 0.0000 | no | 8194 / 8194 | — |

Eight clips ran the full length without terminating. Seven scored exactly 0.0
because their reward sum went slightly negative and the metric clips at zero.

### Why the clips ended

| termination term | clips |
| --- | --- |
| `ee_body_pos` | 31 |
| `anchor_pos` | 1 (`walk2_subject1`) |
| `anchor_ori`, `nonfinite` | 0 |

`ee_body_pos` fires on a per-body **z** error above 0.25 m for the four
end-effectors. Which body tripped it:

| body | clips |
| --- | --- |
| `right_ankle_roll_link` | 11 |
| `left_ankle_roll_link` | 8 |
| `right_wrist_yaw_link` | 6 |
| `left_wrist_yaw_link` | 3 |

(5 of the 31 `ee_body_pos` clips name no body: the diagnostic samples the per-body
error one physics step later than the termination check does, so the body that
tripped it can already be back inside the threshold.)

So this is not a robot lying on the floor: it is ankles and wrists drifting more
than a quarter of a metre in height, on clips that were being tracked.

### Was it tracking before it tripped?

For each clip, the mean per-step sum of the **positive** reward terms over the
window it survived (ceiling `MAX_STEP_REWARD = 0.1`):

* mean **0.0841** (84.1 % of the ceiling), range 0.0730 – 0.0949;
* mean fraction of the clip survived: 49 %.

The policy is not doing nothing. It tracks the reference reasonably well and then
crosses a gate that its training environment did not have.

---

## 5. Why this is not a statement about SuperMimic's algorithm

The score is a *cross-simulator and cross-contract* measurement. The differences,
all verified in the two code bases, are:

1. **A termination the training task does not have.** mjlab's `tracking_env_cfg`
   adds `ee_body_pos` — four bodies, z error > 0.25 m. SuperMimic's
   `MDP.termination` tests only root height (`|Δz| > 0.25`) and root tilt
   (`|ΔR[2,2]| > 0.8`). Every one of the 31 `ee_body_pos` failures is a gate that
   did not exist during training, and the training-side 0.84165 was measured
   under the looser one.
2. **A reward term the training task cannot produce.** mjlab penalises
   `self_collisions` (`weight -10`, contact force > 10 N) and builds the model
   with **full self-collision**. `supermimic/robot.py` spawns the robot with
   `enabled_self_collisions=False` and its `RewardCfg` has no such term. On
   `walk4_subject1` the penalty costs **-0.104 per step** against a positive
   reward of **+0.0915** per step — one of the highest tracking rates of the 40
   clips — so the clip finishes alive and still scores 0.0. Adding that single
   term's cost back (arithmetic on the same rollout, not a re-run with a changed
   reward) gives `0.9107` on that clip, against SuperMimic's own reported
   `0.9301`.
3. **Friction.** SuperMimic's evaluation sets robot and ground friction to 1.0;
   the canonical model gives the feet 0.6 (`FOOT_FRICTION`, `priority=1`). 0.6 is
   inside the training randomisation range `U(0.3, 1.2)`, so this is not
   obviously fatal, but it is a different contact regime.
4. **Contact discretisation.** Full-collision geoms other than the feet get
   `condim=1` here; the training stack runs the MuJoCo-Warp solver with
   `njmax=500, nconmax=128`.
5. **Reward definition.** Beyond the terms above, mjlab re-anchors the reference
   to the robot's xy and yaw for the body-pose terms while scoring root pose
   globally; SuperMimic's `MDP.reward` anchors with `yaw_mat(R @ rR^T)` and has
   `w_joint_limit = -2.0` where the canonical weight is `-10.0`. Scores computed
   under the two definitions are not interchangeable even with identical physics.
6. **Clip length.** 20 of 40 clips have `T_canonical = T_upstream - 1` (the two
   50 Hz resamplers differ at `dur/dt` integer boundaries). This changes the
   final goal frame and the metric denominator by one step out of thousands.

Initial conditions are *not* among the differences: for all 40 clips the frame-0
joint vector is inside both the soft and the hard joint limits, so the
evaluator's soft-limit clamp at reset and SuperMimic's hard-limit clamp produce
the same state.

**Conclusion:** these are real closed-loop numbers for a real SuperMimic policy
in the canonical environment, and they should be read as a transfer result. They
do not show that the algorithm fails; they show that a policy trained without
self-collision awareness and without an end-effector termination gate does not
survive the stricter contract, and that its tracking is otherwise largely intact.
The 0.84165 training-side figure is not reproducible here and is not registered
as a canonical score.

---

## 6. Inference cost

Measured on the same run as the results (`reports/supermimic_timing.json`),
macOS arm64 CPU, NumPy float32 GEMM, no GPU.

| Metric | Value |
| --- | --- |
| Control steps actually run | 224,943 (less than 441,080 because episodes ended early) |
| Policy total | **367.2 s** |
| Per step | mean **1.63 ms**, P50 **1.54 ms**, P95 **2.03 ms**, P99 **2.74 ms**, max 63.3 ms |
| Per-clip `reset` (reference derivatives + yaw frames) | P50 246 ms |
| Whole rollout wall clock | 601.9 s — policy 61 %, MuJoCo 39 % |
| Device | CPU (`arm64`), no GPU used |
| Actor | 9,118,135 numbers; `730->2048->1024->512->256->32` / `1022->2048->1024->512->256->29`, ELU |

The full protocol without early termination is 441,080 steps, i.e. ≈12 minutes of
policy time and ≈7 minutes of MuJoCo. No budget framework was added; this is a
measurement.

---

## 7. Reproducing

```bash
# evaluator environment (no torch)
python -m venv .venv && source .venv/bin/activate
pip install mujoco==3.11.0 mujoco-warp==3.11.0 warp-lang==1.16.0
python scripts/fetch_lafan1.py && python scripts/freeze_manifest.py --check

# 40 clips + cost
python scripts/profile_supermimic.py \
    --checkpoint ~/Downloads/gptfiles/research/supermimic/colab_baseline/ckpt_015000.pt \
    --training-wall-s 5223.3 \
    --out reports/supermimic_canonical_eval.json \
    --timing-out reports/supermimic_timing.json

# or the standard entry point
python scripts/run_canonical_eval.py \
    --policy-file benchmark/policies/supermimic.py:make_policy \
    --checkpoint .../ckpt_015000.pt --algorithm SuperMimic \
    --training-commit 8bfc1158bd968f729fcc65d5e897a05b05ea3d36 \
    --gpu-name "cpu-only (arm64)" --training-wall-s 5223.3 \
    --out reports/supermimic_canonical_eval_cli.json

# failure decomposition
python scripts/diagnose_supermimic.py --checkpoint .../ckpt_015000.pt

# fidelity against the upstream code (needs torch in a separate environment)
python scripts/check_supermimic_parity.py --upstream /tmp/supermimic_vibed \
    --checkpoint .../ckpt_015000.pt --clip walk4_subject1 --steps 200

# tests (76 pre-existing + 13 new)
python -m unittest discover -s tests -v
```

---

## 8. Open items

* The bf16 autocast path of the original evaluation cannot be replicated on a CPU
  host; the bf16-weight emulation bounds the deviation at 4e-2 max / 5e-3 mean on
  the action, without changing the outcome of the two clips tested. A GPU host
  would settle it exactly.
* The record is `INVALID` only because of the hardware gate. It is deliberately
  **not** registered as a leaderboard entry: the evaluator's numbers have not been
  cross-checked against mjlab's own `evaluate.py` on the same checkpoint, which
  remains the prerequisite for an official record.
* `walk2_subject1` is the one clip that fails a termination the training task
  *does* have (`anchor_pos`): a genuine loss of tracking, not a contract gap.
