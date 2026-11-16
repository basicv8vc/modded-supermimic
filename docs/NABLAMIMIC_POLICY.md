# NablaMimic native inference on the canonical evaluator

The 8-minute NablaMimic recipe (`configs/ours.args`, ∇Mimic engine) trained to
the paper's Appendix-H metric. This document records how its actor is executed
against the frozen `canonical-mjlab-g1-v1` evaluator, how that execution was
verified against the original, and where the two protocols are *not*
interchangeable.

**The same policy, not the same protocol.** The Appendix-H number (0.8345 mean /
0-of-40) and the canonical number produced here are reported side by side and
never combined.

---

## 1. Provenance

| Item | Value |
| --- | --- |
| Checkpoint | `it_001250.pkl`, 216,701,243 bytes |
| SHA256 | `7a8084c2…` (verified byte-identical against the training VM) |
| Training code | `sailab-ai/NablaMimic` @ `bfbb2a6773b4776c17402a34a93ec7589d9c7c18`, engine `c61a6fe` |
| Recipe | `configs/ours.args` (not `ours_mjw.args`; the 60-minute MJWarp recipe is a different budget) |
| Training run | `ea_s0_20261111T065918Z`, seed 0, 1261 updates, 480.32 s |
| Actor | `actor_ema`, 8,903,485 parameters |
| Policy implementation | `benchmark/policies/nablamimic.py` (NumPy; the weights are not re-trained, fine-tuned or converted) |

## 2. What the original does, and where it lives here

Read from the sources at the training commit, not inferred from tensor shapes.

| Upstream | Native behaviour | Here |
| --- | --- | --- |
| `env.proprio` | 93 = `q - q0` (29) + `qd * 0.2` (29) + `w * 0.5` (3) + gravity in the **pelvis** frame (3) + previous raw action (29) | `NablaMimicPolicy.proprioception` |
| `env._goal_features` | 68 per frame = `q - q0` (29) + `qd * 0.2` (29) + gravity (3) + `w * 0.5` (3) + heading-frame linear velocity (3) + root height (1) | `NablaMimicPolicy._goal_features` |
| `env.goal` | 10 frames from `frame` on, clamped to the clip end, plus the anchor's **orientation** (6) only — `GOAL_ANCHOR = slice(3, 9)` | `NablaMimicPolicy.goal` |
| `nets.actor_apply` | `z = tanh(enc(goal[:-6]))`, `a = dec(cat(z, hist.reshape(-1), goal[-6:]))`, flat history | `NablaMimicPolicy.forward` |
| `env` history | first observation tiled `HIST` times, then a rolling window, newest last | `NablaMimicPolicy.act` |
| residual action | target joint position is `ref[min(frame + 1, last)] + scale * a` | rewritten as `q0 + scale * a'` with `a' = a + (ref[k] - q0) / scale` |

The residual rewrite is exact by construction: the benchmark scores the
position target, so the adapter only changes *which* action vector produces the
same target. `clip`, `scale` and any smoothing are untouched.

## 3. The two places this silently goes wrong

Both were found by measurement, and both are now regression-tested.

### 3.1 The 6-D orientation has two different layouts

`motion_anchor_ori_b` takes the first two **columns** of the relative rotation
and then flattens **row-major**:

```
evaluator:  [m00, m01, m10, m11, m20, m21]     (interleaved)
NablaMimic: [m00, m10, m20, m01, m11, m21]     (columns concatenated)
```

The first version of this adapter passed the evaluator's payload straight into
the network. On `walk3_subject4` that terminated after **36 control steps**
(`ee_body_pos`) — a walking clip the policy tracks cleanly under its own
protocol. After applying the permutation, all four probed clips survive
160/160 steps.

Which layout the evaluator emits was decided by determinant, not by reading
comprehension: reading the six numbers as interleaved gives `|det - 1| <= 3e-7`
at every step, the stacked reading gives `|det - 1| ~ 0.75` — not a rotation.

### 3.2 The reference root velocities are not in `ReferenceClip`

`env._goal_features` needs the reference pelvis linear and angular velocity.
The original gets them from `mujoco.mj_differentiatePos` over the 50 Hz frames.
`ReferenceClip` exposes poses only, so the adapter rebuilds them: central
differences on positions, and the SO(3) derivative of the quaternions **in the
body frame**. Measured against the real `mj_differentiatePos` on all 12,330
frames of a clip:

| Quantity | max abs error |
| --- | --- |
| linear (world) | 0.0 |
| angular (body frame) | 1.6e-14 |
| angular (world frame, wrong reading) | 3.99 |
| joint velocity | 1.1e-6 (float32 storage) |

`ROOT_ANG_VEL_FRAME` exists only so that harness can demonstrate the body-frame
reading is the one that reproduces the original arrays.

### 3.3 Gravity has to be rebuilt through the waist chain

`StepInput` exposes the anchor (torso) orientation, not the pelvis attitude the
original uses for `_gravity_b`. The adapter composes
`A(q) · (R_torso^T R_ref_torso) · A(q_ref)^T · g_ref` with `A` the three waist
joints in `Rz(yaw) Rx(roll) Ry(pitch)` order.

The `M` in that product **must be taken at the same reference frame as
`q_ref` and `g_ref`** — frame `f`, the frame the robot's state belongs to.
Taking it one frame later does not fail loudly; it injects the reference
torso's one-frame rotation straight into `proprio[61:64]`. That was a real bug
in this adapter (0.02–0.25 of error at walking-to-running torso rates); it is
fixed, and the derivation plus measurements are in
`NABLAMIMIC_EVAL_AUDIT.md` §3.

Measured with `M` at the correct frame, the composition is exact, not merely
close: algebraically `A(q)·M·A(q_ref)^T·g_ref` reduces to `R_pelvis^T g`. On the
real clips it agrees with the original's `_gravity_b` to **5.9e-07** across 40
clips, 5 frame classes and 400 samples (`scripts/audit/parity_original_env.py`).
An earlier revision of this document reported `1.1e-3` here; by the algebra in
the audit that residual *implies a torso rate of 0.055 rad/s*, i.e. it was
measured on a near-static segment and was itself evidence that the misalignment
had not been exercised.


## 4. Verification

`tests/test_nablamimic_policy.py` — 8 tests (simulator-backed where the dataset
is present; they skip otherwise):

| Check | Result |
| --- | --- |
| `six_to_mat` inverts the evaluator's layout | 1e-12 |
| anchor permutation equals `concatenate([m[:,0], m[:,1]])` | 1e-12 |
| permutation direction (it is a 3-cycle, not an involution) | pinned |
| reference velocities vs `mj_differentiatePos` | see 3.2 |
| actor shapes vs the recipe's observation layout | 680 -> 32 -> 968 -> 29 |
| proprioception vs `MjData` (joint, velocity, angular velocity, gravity) | see 3.2 / 3.3 |

`tests/test_nablamimic_temporal_alignment.py` — 16 tests, **pure numpy, always
run, no simulator or checkpoint required**:

| Check | Result |
| --- | --- |
| adapter anchors at `f`, not `reference_index` | exact |
| swapping the two anchor fields breaks gravity (power check) | error > 0.1 |
| missing `anchor_ori_b_prev` fails closed | raises |
| gravity exact at `M(f)` across 0.5–12 rad/s torso rates | 1e-10 |
| mixed-frame error is bounded by, and tracks, the one-frame turn | monotone, ≤ `2 sin(θ/2)` |
| all 93 proprio dims against explicit formulas | 1e-12 |
| all 680 goal features against the original construction | 1e-12 |
| goal window clamps at the clip end | exact |
| history tiled then rolled, newest last | exact |
| residual rewrite commands the same setpoint | 1e-12 |
| action-rate term equivalent to the original `setpoint` rule | 1e-12 |
| raw-vs-benchmark action distinction preserved in the history | exact |
| waist rotation matches quaternion composition in the documented order | 1e-12 |

All 16 pass. Reverting the adapter to `anchor_ori_b` fails 6 of them, so the
suite is not decorative.

`tests/test_nablamimic_real_parity.py` — 4 tests that run where an upstream
`NablaMimic` checkout, `mujoco`, `jax` and one LAFAN1 CSV are present, and skip
cleanly otherwise. The ground truth is the original `TrackingEnv` on real MuJoCo
forward kinematics over a real clip, not algebra:

| Check | Result |
| --- | --- |
| 93-dim proprioception vs the original's `proprio` | ≤ 2e-4 (measured 1.5e-07 on physical channels) |
| 686-dim goal vs the original's `goal` | ≤ 2e-4 (measured 1.7e-05) |
| gravity channel vs the original's `_gravity_b` | ≤ 1e-5 (measured 5.9e-07) |
| the pre-fix adapter fails the same tolerances at the clip's fastest frame | > 1e-2, and < 5e-3 at frame 1 |

The last row is the point: it pins *why* a short test starting from rest could
not see the bug, rather than merely asserting that the new code is right.

Across all 40 clips, 5 frame classes and 2 robot states
(`scripts/audit/parity_original_env.py`), the largest deviations are 5.9e-07 on
gravity, 1.7e-05 on the 686-dim goal (all of it the 680 reference features, see
audit §3.6), and 3.6e-08 on the previous-action block. The pre-fix adapter
deviates by 1.27e-01 and 2.63e-01 respectively on the same samples.

The network itself was checked against the original JAX `actor_apply` on
identical `(history, goal)` pairs from a live rollout: max abs action
difference **1.5e-6**, mean **2.8e-7** (float32 matmuls on both sides). That
result remains valid — it tests the network given *identical inputs*, which is
precisely why it could not detect the observation bug in 3.3.

## 5. Protocol boundary

The evaluator freezes its own semantics, including the simulator device
(`TrackingEnv` builds `ManagerBasedRlEnv(cfg, device="cpu")`). The protocol's
hardware field is metadata validated by `core.py`, not a switch for the physics
backend; running the same evaluator on a GPU host does not change how it steps.

Two numbers therefore exist for this checkpoint and they are not comparable:

| Protocol | Mean normalized return | Failed clips |
| --- | --- | --- |
| Appendix H (training-time evaluator) | 0.8345 | 0 / 40 |
| `canonical-mjlab-g1-v1` (this evaluator) | 0.4819 | 16 / 40 |
| `canonical-mjlab-g1-v1`, corrected adapter | **not yet measured** | — |

The 0.4819 row was produced by an adapter that fed the actor a mixed-frame
anchor (audit §3). It is an accurate record of that run and **not** an estimate
of this checkpoint's canonical-rubric score. Do not cite it as the latter; a
re-run on the GPU host is required first (`NABLAMIMIC_EVAL_AUDIT.md` §10).

The two protocols also differ by a constant one-frame offset in the scored
reference index. They do **not** differ by a resampling frame count: measured on
all 40 CSVs against mjlab 1.6.0's own `MotionLoader`, every clip has the same
number of frames (audit §4.3). That was previously asserted as "20 of 40 clips
land one frame apart"; 20/40 is the count satisfying the divergence
precondition, and none of them actually diverges. The residual difference is the
float32 sample grid, ≤ 6.2e-04 rad of joint angle, which is not a performance
explanation.

They are recorded separately and never averaged.
