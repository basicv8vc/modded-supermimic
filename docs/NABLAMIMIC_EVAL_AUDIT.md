# NablaMimic evaluation-correctness audit

Audit of `benchmark/policies/nablamimic.py` against the original
`sailab-ai/NablaMimic` @ `bfbb2a6773b4776c17402a34a93ec7589d9c7c18`, and of the
canonical `reports/nablamimic_canonical_eval_mjlab.json` result.

Branch `feat/nablamimic-canonical-policy`, base HEAD `262768a`.

> **Scope discipline.** Nothing frozen was modified: no reward weight, no
> termination threshold, no scoring formula, no protocol file, no G1 model, no
> reference converter, no checkpoint. The one behavioural change is to the
> *policy adapter*, which is the algorithm's own code, and it is the subject of
> this document.

---

## 0. What this audit could and could not execute

Honesty first, because it bounds every claim below.

| Capability | First pass | Second pass |
| --- | --- | --- |
| Python + numpy | yes | yes |
| `mujoco` 3.11.0 + `mujoco-mjx` (CPU) | no | **yes** — the protocol's own version |
| upstream `nablamimic` package, importable and runnable | read-only | **yes** — `TrackingEnv` instantiated |
| `torch` 2.15.0, `mjlab` 1.6.0, `warp` 1.19.0 | no | **yes** (CPU) |
| LAFAN1 CSVs | no | **yes** — 40/40 matching `scripts/lafan1_g1.sha256` |
| GPU / CUDA (MuJoCo Warp needs it) | no | **no** — Apple M1, arm64 |
| NablaMimic checkpoint `it_001250.pkl` | no | **no** |
| Original upstream source | yes | yes |

What that changes:

* **Observation parity is now measured, not argued** (§3.6): the upstream
  environment itself, real FK, real clips, 400 samples. This is what the brief
  asked for and it is the strongest evidence in this document.
* **The reference-conversion axis is now measured on all 40 clips** (§4.3), and
  two previously-open items are settled — one falsified, one refuted (§4.4).
* **Still not executed: any rollout.** Phases C and D remain impossible here.
  mjlab's canonical evaluator needs MuJoCo Warp on CUDA, and no rollout of any
  kind is possible without `it_001250.pkl`. **No new evaluation JSON was
  produced**, and none is estimated. §10 gives the command for a host that has
  both.
* The reward/termination decomposition in §6 remains fully computed: it is a
  read-only re-weighting of the per-term sums the existing record already
  stores, and it needs no simulator (§6.1 explains why that is exact).

---

## 1. Verdict

| # | Finding | Status | Impact |
| --- | --- | --- | --- |
| 3 | Adapter read the anchor orientation at `reference_index` while anchoring its features at `reference_index - 1` — a mixed-frame observation | **CONFIRMED bug, fixed, re-verified on 40 clips** | Corrupted `proprio[61:64]` and `goal[680:686]` by the reference torso's one-frame turn: 5.0e-04 at rest rising to 1.27e-01. After the fix: 5.9e-07 and 1.7e-05 (§3.6) |
| 4 | 30→50 Hz conversion can differ by one frame | **Mechanism CONFIRMED, the observed claim FALSIFIED** | 0/40 clips differ, though 20/40 satisfy the divergence precondition. Residual difference is the float32 time grid: ≤6.2e-04 rad (§4.3) |
| 4b | Original floor-aligns z, canonical converter does not | **REFUTED on this dataset** | Offset ≤7.2e-08 m on all 40 clips (§4.4) |
| 5 | Action / residual / action-rate semantics | **VERIFIED equivalent** | None (one documented protocol difference: setpoint clipping) |
| 6 | 16/40 "failed" clips are all `ee_body_pos`; the original's fall criteria never fire | **CONFIRMED from record** | "Failed" does not mean "fell" |
| 6b | Self-collision penalty (−10) alone costs **0.190** of normalized return | **CONFIRMED from record** | 0.4819 → 0.6720 counterfactual |
| 7 | Physical-model differences | **PARTIALLY VERIFIED** (source-level only) | Models now compilable here, but the comparison table is not built |
| 8 | Report provenance gaps (`source_git_sha: unknown`, no CPU/thread record) | **CONFIRMED** | Reproducibility, not correctness |

**Is 0.4819 trustworthy as a canonical benchmark result? No — but the reason has
changed.** After the second pass the adapter's observation construction is
verified against the original environment to 1e-07 on the physical channels and
1.7e-05 on the goal, on real data across 40 clips. So 0.4819 is a faithful
record of a *mis-specified policy input*, not of this checkpoint. The score must
be re-produced with the fixed adapter before it is cited as the canonical
number — the adapter itself is no longer a reason to doubt the harness.

---

## 2. Original semantics, established from source

All references are to `/Users/a1-6/Downloads/NablaMimic/src/nablamimic/env.py`
unless stated.

| Function | Line | Semantics |
| --- | --- | --- |
| `State.frame` | 87–88 | "reference frame the robot is at" |
| `reset` | 293–313 | `qpos = ref_qpos[frame]`, returns `State(..., frame, ...)` — robot **is** at `frame` |
| `proprio` | 239–244 | `q - q0`, `qd*0.2`, `qvel[3:6]*0.5`, `_gravity_b(dx.qpos[3:7])`, `prev_a` |
| `_anchor_rel` | 252–254 | `_anchor_rel_pq(pose, frame=state.frame)` |
| `_anchor_rel_pq` | 256–260 | `pos` and `rot = quat_to_mat(quat_inv(ra_q) * ma_q)` at `frame`; returns `[pos, rot[:,0], rot[:,1]]` |
| `goal_ref` | 262–265 | `goal_feat[min(frame + arange(10), ref_last[frame])]` — **starts at `frame`** |
| `goal` | 267–274 | `concat(goal_ref(frame), _anchor_rel(dx, frame)[3:9])` |
| `action_base` | 320–325 | residual: `ref_qpos[min(frame + 1, ref_last[frame]), 7:]` |
| `ctrl_of` | 327–330 | `clip(action_base(frame) + action_scale * a, ctrl_lo, ctrl_hi)` |
| `action_change` | 332–353 | residual + `action_rate_space="setpoint"`: `(a_t - a_{t-1}) + (base_t - ref_qpos[f_t]) / scale`, reset case `a + (base - q0)/scale` |
| `_finish` | 372–380 | advances `frame → min(frame+1, ref_last)`, appends `proprio(dx_new, action)` |
| `reward` | 404–423 | evaluated at the **new** `state.frame` |
| `terminated` | 428–433 | `|z - z_ref| > 0.25` **or** `|g_z - g_z,ref| > 0.8`, at `state.frame` |

So the original's control step at state frame `f` is:

```
observe  proprio(robot at f)                     # gravity in the PELVIS frame
observe  goal  = features[f .. f+9] + anchor(f)   # both at f
command  ref[min(f+1, last)] + scale * a
advance  robot -> f+1
score    at f+1
```

Everything the policy sees refers to **one** frame, `f`.

---

## 3. P0 — the anchor orientation was one frame ahead  **CONFIRMED**

### 3.1 The mismatch

`benchmark/policies/nablamimic.py` before this audit (lines as of `262768a`):

```python
def act(self, step):
    k = int(step.reference_index)
    frame = k - 1                      # NablaMimic's frame
    obs = self.proprioception(step, frame)
    action_raw = self.forward(self._hist.reshape(-1), self.goal(frame, step))
```

with

```python
torso_rel = six_to_mat(step.anchor_ori_b)          # proprioception()
... anchor_ori_stacked(step.anchor_ori_b)          # goal()
```

`frame` is used for the goal features, the reference joint angles `q_ref` and the
reference gravity `g_ref`. But `step.anchor_ori_b` was **not** the anchor at
`frame`.

### 3.2 Why — mjlab's own indexing

Read from `mujocolab/mjlab` @ `0fb8a681136be94ffc636a3dd423cabb97d91f10`
(the commit the protocol freezes):

`src/mjlab/tasks/tracking/mdp/observations.py`

```python
def motion_anchor_ori_b(env, command_name):
    command = env.command_manager.get_term(command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w,
        command.anchor_pos_w, command.anchor_quat_w)
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)
```

`src/mjlab/tasks/tracking/mdp/commands.py`

```python
@property
def body_quat_w(self):     return self.motion.body_quat_w[self.time_steps]
@property
def anchor_quat_w(self):   return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

def _resample_command(self, env_ids):
    if self.cfg.sampling_mode == "start":
        self.time_steps[env_ids] = 0          # robot written at reference frame 0
    ...
def _update_command(self, env_ids=None):
    self.time_steps += 1                      # -> 1 before the first act()
```

Therefore:

* the robot stands on reference frame `k-1` when the first `StepInput` is built,
  and `reference_index == k`;
* `motion_anchor_ori_b` returns `R_robot_torso^T R_ref_torso(k)` — frame `k`, not
  `k-1`.

That is **correct for mjlab**, whose convention is "observe the frame you are
about to be scored at". It is **wrong for NablaMimic**, whose convention is
"observe the frame I am at". The adapter took the goal features from `f` and the
anchor from `f+1`.

### 3.3 What it corrupts, exactly

With `R_torso = R_pelvis · A(q)` (`A` = the three waist joints, offsets
rotation-free) and `M = R_robot_torso^T R_ref_torso`:

```
gravity_b = A(q) · M · A(q_ref)^T · g_ref
```

`A(q_ref)^T g_ref = R_ref_torso(f)^T g_world`, so with `M` at frame `f` this is
exactly `A(q) R_robot_torso^T g_world = R_robot_pelvis^T g_world` — the quantity
`env.proprio` reads straight off the root quaternion. With `M` at `f+1` it is

```
A(q) R_robot_torso^T · [ R_ref_torso(f+1) R_ref_torso(f)^T − I ] · g_world
```

i.e. the *reference torso's one-frame rotation*, injected into the gravity
channel. The same wrong `M` is also the last six goal entries, which bypass the
encoder and enter the decoder directly (`--anchor-to-decoder`).

Measured on this host with a synthetic clip whose torso turns at a controlled
rate (`tests/test_nablamimic_temporal_alignment.py`):

| torso rate (rad/s) | one-frame turn (rad) | gravity error, max abs | goal-anchor error, max abs |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.026 | 0.018 | 0.020 |
| 1.0 | 0.034 | 0.027 | 0.024 |
| 2.0 | 0.052 | 0.047 | 0.042 |
| 4.0 | 0.090 | 0.086 | 0.083 |
| 6.0 | 0.130 | 0.117 | 0.123 |
| 10.0 | 0.209 | 0.195 | 0.202 |
| 12.0 | 0.249 | 0.245 | 0.240 |

The error is bounded by `2 sin(θ/2)` and in practice ≈ θ; the test asserts both
the bound and monotonicity.

### 3.4 Why the existing test did not catch it

`tests/test_nablamimic_policy.py::test_proprioception_matches_mjdata` compared
`obs[61:64]` against the simulator's true pelvis gravity and allowed `5e-3`
(reported 1.1e-3 in `NABLAMIMIC_POLICY.md` §3.3). By the algebra above,
`error ≈ θ = ω_torso · 0.02 s`, so a measured 1.1e-3 **implies a torso rate of
0.055 rad/s**. No walking or dancing humanoid torso turns that slowly for long.

The test therefore did not refute the misalignment; it sampled the only regime
where the misalignment is invisible: the first 12 control steps (0.24 s) of a
single clip, starting from the reference's first frame, where the clip is
essentially static. The `5e-3` tolerance was not a parity claim — it was large
enough to hide a 30× error at normal walking speed.

Two further gaps: the test only ever exercised one clip, and it never compared
the *goal* vector or the actor output at all, so the six anchor entries entering
the decoder were untested.

### 3.5 The fix

The information the adapter needs does not exist in the old `StepInput`, so the
interface was extended rather than guessed around:

* `benchmark/evaluator/policy_api.py` — `StepInput` gains
  `anchor_pos_b_prev` (3,) and `anchor_ori_b_prev` (6,), documented as the same
  quantities at `reference_index - 1`. **The existing `anchor_*_b` fields keep
  their mjlab meaning**; `supermimic.py` and `evaluator/policy.py` are
  mjlab-convention policies and are untouched — and they were already
  self-consistent (`supermimic.py:472` indexes `q_ref` at `k` *and* anchors at
  `k`).
* `benchmark/evaluator/env.py` — `_anchor_relative(frame)` (new, l.227)
  reproduces mjlab's transform with mjlab's own `subtract_frame_transforms` and
  `matrix_from_quat`, using the command's own loader, with the frame passed in
  explicitly; `step_input()` populates the two new fields.
* `benchmark/policies/nablamimic.py` — `_anchor_ori_at_state_frame(step)`
  (l.164) is now the single place the adapter reads the anchor, used by both
  `proprioception` (l.496) and `goal` (l.517). It **fails closed**: if the field
  is absent it raises `NablaMimicContractError` rather than silently falling back
  to the mismatched field.

Residual action, `previous_action`, history and network code are unchanged.

### 3.6 Re-verified on the real data  **CONFIRMED**

The fix was then re-checked against the original environment itself, not against
algebra on synthetic clips. `scripts/audit/parity_original_env.py` instantiates
the upstream `TrackingEnv` (unmodified), drives it with real MuJoCo 3.11.0
forward kinematics over the real LAFAN1 G1 references, and compares its own
`proprio` / `goal` / `_anchor_rel` against the adapter's, channel by channel.

Setup: 40 clips × 5 frames (first, 10 %, middle, the frame with the highest torso
angular rate, second-to-last) × 2 robot states (reference state, and a drifted
state: 5 cm root offset, ~0.06 rad root rotation, 0.03 rad joint offsets,
0.04 rad/s velocity noise). 400 samples. All 40 CSVs verified against
`scripts/lafan1_g1.sha256` at revision `07ff8845`; mjlab 1.6.0, MuJoCo 3.11.0,
jax 0.12.0.1 on CPU.

| channel | fixed adapter | pre-fix adapter |
| --- | ---: | ---: |
| `proprio[0:29]` joint position | 1.5e-07 | 1.5e-07 |
| `proprio[29:58]` joint velocity | 1.8e-07 | 1.8e-07 |
| `proprio[58:61]` base angular velocity | 1.8e-07 | 1.8e-07 |
| **`proprio[61:64]` gravity** | **5.9e-07** | **1.27e-01** |
| `proprio[64:93]` previous action | 3.6e-08 | 3.6e-08 |
| `goal[0:680]` reference features | 1.7e-05 | 1.7e-05 |
| **`goal[680:686]` anchor orientation** | (contained above) | — |
| **`goal[0:686]` total** | **1.7e-05** | **2.63e-01** |

Reading of these numbers:

* The gravity channel is now at float noise, **and that is not a tolerance
  choice**: with the frame right, `A(q)·M·A(q_ref)ᵀ·g_ref` composes to exactly
  `R_pelvisᵀ g`, so the reconstruction is an identity, not an approximation. The
  1.1e-3 that `NABLAMIMIC_POLICY.md` §3.3 recorded was the misalignment itself,
  not the cost of reconstructing gravity through the waist.
* The remaining 1.7e-05 on the goal is the 680-feature block, not the anchor.
  Broken down per block, it is dominated by `lin_h` (root linear velocity in the
  heading frame): 1.3e-05 there, 1.9e-06 on the angular-velocity block, ≤3e-07 on
  joint position, joint velocity, gravity and root height. Cause: the adapter
  differentiates float32 root positions (`_root_velocities`, ×1/dt = ×50
  amplification of the 1e-07 quantisation), while the original reads MuJoCo's
  float64 `qvel`. It is a reconstruction cost, not a semantic difference.
* The pre-fix error tracks the *frame-local* torso rate, which is the mechanism
  of §3.4: at frame 1 (starting from rest) it is 5.0e-04; at the fastest frame of
  `walk3_subject4` it is 3.3e-02, of `fallAndGetUp2_subject3` 6.7e-02; the
  largest in the set is 1.27e-01.
* `base_ang_vel` needs no fix: the evaluator's `imu_ang_vel` gyro sensor and the
  original's `qvel[3:6]` agree to **2.2e-16** on the same state, i.e. both are
  the body-frame angular velocity. This had been an open worry; it is closed.

Not re-verified here: actor output on identical inputs. That needs the
checkpoint, which is not on this host; the 1.5e-06 measurement already recorded
in `NABLAMIMIC_POLICY.md` §4 stands, and note that the network is unchanged by
this fix, so only the observation *values* it receives have changed.

---


## 4. P0 — reference-motion conversion

### 4.1 The two grids, and a proof about when they can differ

`sailab-ai/NablaMimic` `src/nablamimic/motion.py::_resample` (l.77–87):

```python
t_out = np.arange(0.0, (n - 1) / fps_in, 1.0 / fps_out)      # float64
```

`mjlab` `scripts/csv_to_npz.py::MotionLoader._interpolate_motion`:

```python
times = torch.arange(0, self.duration, self.output_dt, dtype=torch.float32)
self.output_frames = times.shape[0]
```

with `duration = (N-1)/30` and `output_dt = 1/50`. The exact ratio is

```
duration / step = (N-1)/30 · 50 = (N-1)·5/3
```

`arange` emits `ceil(ratio)` samples when the ratio is not an integer, so a
one-ulp float difference can only change the count when the ratio lies **on** the
integer boundary — that is, **iff `(N-1) % 3 == 0`**. For every other `N` the
ratio is at least `1/3` from an integer and no float32/float64 difference can
move the ceiling.

Checked exhaustively for `N ∈ [100, 4000]`
(`scripts/audit/reference_conversion.py --mechanism`): 1301 of 3901 lengths are
ambiguous (exactly the `(N-1) % 3 == 0` class, 33.4 %), and the NumPy grid
matches the exact rational `ceil` on **all** 3901 — so any divergence comes from
the float32 side, not from NumPy.

### 4.2 What the divergence can and cannot affect

Both grids are `0, 0.02, 0.04, …`; only the final sample (`t == duration`) may be
present or absent. Therefore:

* **interior reference frames are near-identical** — the float64 sample times
  match exactly, the float32 grid is within 2.6e-05 s (§4.3, measured);
* the extra/missing frame changes `T`, hence the denominator `(T-1)` by one part
  in thousands, and shifts the last ≤10 goal windows at the clip end;
* it does **not** introduce a one-frame offset in the *scored* index, and it
  cannot explain a large score difference. `NABLAMIMIC_POLICY.md` §5 previously
  attributed part of the protocol gap to it; that attribution should not be
  read as a performance cause.

### 4.3 Per-clip table — **MEASURED: the one-frame claim does not reproduce**

`scripts/audit/reference_conversion.py --clips data/lafan1_g1` was run on all 40
real clips against mjlab 1.6.0's own `MotionLoader` (`reports/audit/reference_conversion_40clips.json`):

| quantity | result |
| --- | --- |
| clips with a non-zero frame delta | **0 / 40** |
| clips whose length *can* diverge, `(N-1) % 3 == 0` | **20 / 40** |
| max joint-position difference | 6.2e-04 rad |
| max root-position difference | 1.1e-04 m |
| max root-orientation difference | 3.4e-03 rad |

**The documented "20 of the 40 clips differ by one frame" is wrong, and the way
it is wrong is informative.** 20/40 is exactly the count of clips that satisfy
the precondition derived in §4.1 — so the earlier number appears to have been
*inferred from the residue class and reported as an observation*. Measured, no
clip diverges: the float32 ceiling happens to land on the same count for every
one of the 20.

The residual differences are not a frame shift. They are the float32 time grid:
`torch.arange(..., dtype=float32)` accumulates up to 2.6e-05 s of phase error over
a 13 655-frame clip, which at joint speeds of ~25 rad/s gives ~6e-4 rad. Interior
sample times are therefore *near*-identical, not identical as §4.2 asserted on
the float64 side.

Consequence for the audit: this axis can be removed from the list of candidate
explanations for the 0.4819 score entirely. It contributes at most a 6e-4 rad
reference perturbation — 50× smaller than the observation bug at walking speed.

> **A cross-document conflict to resolve, not silently overwritten.**
> `SUPERMIMIC_POLICY.md` §"Clip length" states that 20 of 40 clips have
> `T_canonical = T_upstream - 1`, and cites `walk4_subject1` concretely as
> canonical 8195 vs upstream 8196. This measurement does not reproduce that:
> `walk4_subject1` is 8195 on both sides, and it *is* a `(N-1) % 3 == 0` clip
> (4917 input frames), so it is a genuine boundary case that lands equal.
> Either the benchmark NPZs were produced by a different converter revision than
> mjlab 1.6.0's `MotionLoader`, or that document's "upstream" side is a
> different implementation. Resolving it needs the actual `.npz` files, which are
> not in this checkout. Until then the SuperMimic document's explanation of its
> `walk4` goal difference should be treated as unconfirmed — it is an
> explanation of a *difference*, and the difference itself is not disputed here,
> only its attribution to frame count.

### 4.4 Floor alignment — **REFUTED on this dataset**

`motion.py::process_clip` l.112–117 shifts every clip's z by the 5th percentile
of the lowest foot point:

```python
qpos[:, 2] -= np.percentile(low, 5)
```

`mjlab`'s `csv_to_npz.py` has a *different* line commented out
(`# motion[:, 2] -= 0.05`), and `benchmark/evaluator/motion.py` applies neither.

Measured on all 40 clips (`reports/audit/floor_alignment_40clips.json`): the
original-minus-canonical root height is **≤ 7.2e-08 m** (median 1.4e-08 m), and
the within-clip spread of that offset is ≤ 2.6e-06 m. The LAFAN1 G1 retargets are
already floor-aligned, so the 5th-percentile shift is zero and the two pipelines
agree. There is no per-clip constant entering goal dim 67 or the root-position
reward.

Caveat worth keeping: the two rules are *not* equivalent in general — the
original subtracts the 5th percentile of the lowest foot point, mjlab's
commented line subtracts a hard 5 cm. They coincide here only because both are
≈ 0 on this dataset. A different retarget would expose the difference.

### 4.5 Velocities

The original derives `qvel` with `mujoco.mj_differentiatePos` (central
differences, one-sided at the ends). The canonical converter uses
`torch.gradient` for linear/joint velocity and an SO(3) derivative for angular
velocity. The adapter does **not** consume the stored root velocities; it
recomputes them (`_root_velocities`) in the body frame to match
`mj_differentiatePos`, which is what the original's `_goal_features` reads
(`qvel[3:6]` of a free joint is body-frame). That choice is verified by
`test_matches_mujoco_central_difference`, which also asserts that the
world-frame reading disagrees by > 0.1 — so the test has power.

---

## 5. P0 — action and control parity  **VERIFIED**

### 5.1 Residual action

Original: `target = ref[min(f+1, last)] + scale · a` (l.320–330).
Adapter: `a' = a + (ref[k] − q0)/scale` with `k = min(reference_index, T−1)`.
Composing with the benchmark contract `target = q0 + scale · a'`:

```
q0 + scale·a' = q0 + scale·a + ref[k] − q0 = ref[k] + scale·a        ✓
```

Test `test_residual_rewrite_commands_the_same_setpoint` asserts this to `1e-12`
at frames 0, 1, 5 and 30.

### 5.2 Action-rate reward — equivalent, and this is not obvious

The canonical `action_rate_l2` sees the adapter's action `a'`, not the original's
`a`. Those differ by the reference base, so one might expect the penalty to
change. It does not, and the reason is worth recording:

```
a'_t − a'_{t−1} = (a_t − a_{t−1}) + (ref[k_t] − ref[k_{t−1}]) / scale
```

with `k_t = f_t + 1` and `k_{t−1} = f_{t−1} + 1 = f_t`, this is exactly the
original's `action_change` in `setpoint` mode, `(a_t − a_{t−1}) + (ref[f+1] −
ref[f])/scale` (l.352–353). At reset the evaluator supplies
`previous_action = 0`, which reproduces the original's `fresh` branch
`a + (base − q0)/scale`.

Asserted to `1e-12` by `test_action_rate_matches_the_original_setpoint_rule`,
including the reset case and a near-end frame.

### 5.3 The rest of the contract

| Property | How it is enforced |
| --- | --- |
| 29-DoF joint order | `reset()` raises if `context.joint_order != constants.JOINT_ORDER` |
| default pose | `reset()` raises if it differs from `env.Q0` by > 1e-12 |
| action scale | `reset()` raises if it differs from `0.25·effort/stiffness` by > 1e-8 |
| previous action | `act()` raises if the evaluator's `previous_action` drifts from the adapter's own by > 1e-9 (`test_history_desync_is_detected`) |
| previous action is the **raw** output | `proprio[64:93]` holds `a_raw`, not `a'` (`test_previous_action_is_the_raw_output_not_the_benchmark_action`) |

### 5.4 One genuine, documented difference: setpoint clipping

The original clips the setpoint to a range widened by `τ_max/kp`
(`robot.py::widen_ctrlrange`, `env.ctrl_of`). The canonical model sets
`ctrllimited=False` (`spec.py` l.125) and relies on `forcerange` instead, so
mjlab does **not** clip the position setpoint.

Since `scale = 0.25·τ_max/kp`, the widened bound corresponds to `|a| ≤ 4`; a
policy output beyond ±4 would be clipped by the original and passed through by
the canonical stack. This is a property of the frozen protocol, not of the
adapter, and it was **not** modified. It is worth a note in any comparison
between the two rubrics, and can be quantified in a rollout by counting
`|action| > 4`.

---

## 6. P1 — reward and termination differences

### 6.1 Method, and why most counterfactuals here are exact

The rollout loop is

```
action = policy.act(state)   # depends on the state only
result = env.step(action)    # physics; independent of reward
total += result.reward       # after the step
if result.terminated: stop
```

The reward never feeds back into the trajectory. A counterfactual that changes
only **reward weights** therefore leaves every state, action and termination
instant bit-identical, and its effect is an exact linear re-weighting of the
per-term sums the record already stores. Changing a **termination condition**
changes when the rollout stops, which needs a replay — those are marked as
projections, never scores.

`scripts/audit/reward_counterfactuals.py` reads
`evaluation.per_clip_diagnostics[*].mean_term_score` and reconstructs
`raw = Σ mean_term·alive_steps / ((T−1)·0.1)`, asserting the identity on all 40
clips (max residual 3.3e-8, from float32 accumulation on the torch side).
Output: `reports/audit/reward_counterfactuals.json`.

### 6.2 Headline numbers

| Variant | mean (clamped) | mean (raw) | Δ vs official |
| --- | ---: | ---: | ---: |
| **official record** | **0.481922** | 0.479826 | — |
| self-collision penalty disabled | 0.672040 | 0.672040 | **+0.190118** |
| joint-limit weight −10 → −2 | 0.484432 | 0.482379 | +0.002510 |
| both penalties at original weights | 0.674593 | 0.674593 | +0.192671 |

The self-collision penalty (`self_collisions`, weight −10, **absent from the
original reward**) is the single largest *reward-side* driver: it alone accounts
for 0.19 of normalized return. The joint-limit weight difference (−10 vs −2) is
negligible, 0.0025.

### 6.3 Terminations: "failed" does not mean "fell"

All 16 failed clips terminate on **`ee_body_pos` and nothing else** (histogram:
`{'<none>': 24, 'ee_body_pos': 16}`). The original's own criteria —
`|z − z_ref| > 0.25` and `|Δg_z| > 0.8` — **never fire on any clip**. The
canonical `ee_body_pos` term (per-end-effector z deviation > 0.25 m) is the sole
cause of every failure.

Mean alive fraction of the 16 failed clips: **0.50**. Two clips end almost
immediately: `fightAndSports1_subject1` at 233/12241 steps (1.9 %),
`jumps1_subject2` at 601/12241 (4.9 %) — while their per-alive-step positive
tracking reward is 0.97 and 0.96 of the maximum respectively. Terminating at 2 %
of a clip while tracking almost perfectly is a strong hint that the criterion is
sensitive to something other than falling (e.g. a wrist/ankle z deviation
inherent in the retarget that the policy lags), but this cannot be confirmed
without a per-body trace.

### 6.4 The `walk4_subject1` case

| quantity | value |
| --- | --- |
| alive steps | 8194 / 8194 (never terminated) |
| positive tracking reward per step | 0.0954 (95.4 % of the 0.1 ceiling) |
| self-collision per step | **−0.0996** |
| net raw | **−0.0486** → clamped to 0.0 |

The clip is tracked at 95 % of the theoretical ceiling and scores **zero**,
because the self-collision penalty alone is as large as the entire positive
reward budget. `walk3_subject3` is the same story (positive 0.0868, self-collision
−0.0908, raw −0.0353). `fallAndGetUp1_subject4` and `fallAndGetUp2_subject2`
lose 0.076 and 0.080 per step to it.

This is a legitimate property of the canonical rubric — mjlab's `FULL_COLLISION`
config does penalize self-collision, and the brief forbids weakening it. It is
reported here because it dominates the score's *composition*, and because a
retargeted clip whose reference pose is self-intersecting would be penalized
indefinitely for something the policy cannot fix. That hypothesis is
**UNVERIFIED**; it is checkable by evaluating the reference motion itself (replay
the reference qpos through the canonical model and measure the self-collision
term with no policy involved).

### 6.5 Sums, for attribution

| component | per-alive-step contribution |
| --- | --- |
| positive tracking (6 terms) | 0.086–0.095 of the 0.1 ceiling for well-tracked clips |
| `action_rate_l2` | −0.0006 … −0.0021 |
| `joint_limit` | −5e-5 … −1e-4 |
| `self_collisions` | 0 … **−0.0996** |

Reward side is therefore dominated by self-collision; joint limit and action rate
are noise. Termination side is entirely `ee_body_pos`.

---

## 7. P1 — physical model  **PARTIALLY VERIFIED**

Both models are *derived from* MuJoCo Menagerie's `unitree_g1`, but neither is
that model:

* original: `third_party/mujoco_menagerie/unitree_g1/scene_mjx.xml`, then
  rewritten by `src/nablamimic/robot.py` (mjlab's feet, explicit contact pairs,
  `solref`/`solimp`, widened ctrlrange, `frictionloss = 0`);
* canonical: mjlab v1.6.0's `unitree_g1/xmls/g1.xml` + `FULL_COLLISION`, i.e. the
  stock mjlab asset.

`src/nablamimic/robot.py` l.8–16 already documents the intended equivalence and
the intended differences:

| Aspect | Original (mjlab-matched Menagerie) | Canonical (mjlab) |
| --- | --- | --- |
| bodies / masses / inertias / joints | same | same |
| armature, PD gains, torque limits | same | same |
| feet | 7 thin capsules per sole | same construction (mjlab asset) |
| contacts | explicit pairs, parent-child excluded, 4 extra body pairs excluded | `contype/conaffinity = 1` on every `*_collision` geom, MuJoCo generates pairs |
| `solref` / `solimp` | `(0.02, 1.0)` / `(0.9, 0.95, 0.001, 0.5, 2.0)` | mjlab defaults |
| `dof_frictionloss` | forced to 0 | as in the asset |
| `ctrlrange` | joint range widened by `τ_max/kp` | `ctrllimited=False`, `forcerange` only |
| feet friction | 0.6, floor 1.0 | 0.6, floor 1.0 (`spec.py` l.91) |

**Not verified here:** the two XMLs were never compiled or diffed numerically —
the Menagerie checkout is absent from this host and neither `mujoco` nor `mjlab`
is installed. So "body masses and inertias are identical" rests on the upstream
docstring, not on a measurement. A compiled comparison (mass/inertia/jnt_range/
armature/ctrlrange arrays, plus `npair` and contact parameters) is the missing
step; `spec.py::fingerprint` already provides half of the machinery.

One checkable inconsistency: `robot.py` l.42–43 lists four excluded body pairs
(`pelvis↔hip_roll`, `elbow↔wrist_pitch`) as part of mjlab's contact set, while
`benchmark/evaluator/spec.py` applies **no** exclusions when building its
conformance model. If mjlab's real G1 `FULL_COLLISION` config excludes those
pairs, `spec.py`'s fingerprint describes a model with extra contacts. This does
not affect the evaluation (the evaluator runs mjlab's own `env_cfg`, not
`spec.py`'s model) but it does weaken `spec.py` as a conformance oracle.

---

## 8. P1 — reproducibility of the existing record

| Field | NablaMimic record | Required |
| --- | --- | --- |
| `evaluator.source_git_sha` | **`"unknown"`** | the SHA of the tree that ran |
| CPU model / architecture | **absent** | required (Warp CPU numerics are thread-sensitive) |
| thread count / affinity | **absent** | required |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `WARP_*` | **absent** | required |
| OS | absent | required |
| dirty-tree status / diff hash | **absent** | required |
| MuJoCo / MuJoCo Warp / warp-lang | `3.11.0 / 3.11.0 / 1.16.0` | present |
| mjlab commit | `0f8a681…` | present |
| robot model SHA | `ae6066…` | present |
| dataset manifest SHA | `595fd9…` | present |
| protocol SHA | `ddec7124…` | present |
| checkpoint SHA | `7a8084c2…` | present, and confirmed byte-identical by the training run |

`runner.source_git_sha()` returns `"unknown"` when `git rev-parse` fails, and it
ran in an environment without `.git` — so the record cannot say which revision
of the evaluator produced it. Combined with the documented 48-thread vs 6-thread
difference on the same clip, the existing record is **not** a controlled
comparison against `supermimic_canonical_eval_mjlab_reuse.json`, and the two must
not be differenced as if it were.

This audit host: Apple M1, arm64, 8 cores, macOS 26.3 — **not** the host that
produced either record, and not capable of producing one (no CUDA, no Warp).

---

## 9. What remains unverified

1. **The corrected adapter has never been executed in closed loop.** The fix is
   covered by 16 unit tests and by the 400-sample real-data parity of §3.6, but
   no mjlab rollout ran. Phases C and D are outstanding.
2. **The magnitude of the score change from the anchor fix is unknown.** §3.6
   establishes the input error the fix removes (1.27e-01 on the gravity channel
   and 2.63e-01 on the goal at the worst sampled frame); it does not establish
   the return delta.
3. **The physical-model table** (§7) needs a compiled comparison. Both models are
   now loadable here, so this is tractable but not done.
4. **The `ee_body_pos` mechanism** (§6.3) needs a per-body z trace to explain why
   two clips terminate at ~2 % while tracking at 96–97 %.
5. **The self-collision plausibility question** (§6.4) needs a reference-only
   replay.
6. **Thread-sensitivity of the canonical evaluator** is documented but not
   quantified by this audit.
7. **Actor output parity** was not re-measured (no checkpoint here); the recorded
   1.5e-06 stands, and the network is unchanged by this fix.

Settled in the second pass, previously open: per-clip reference conversion
(§4.3, measured — claim falsified), floor alignment (§4.4, refuted), and
`base_ang_vel` frame convention (§3.6, closed at 2.2e-16).

---

## 10. Next steps (in order)

1. On the GPU host, with the dataset and checkpoint present:

   ```bash
   # parity gate: unit tests (no simulator) + real-data parity (needs upstream + data)
   python -m unittest tests.test_nablamimic_temporal_alignment
   NABLAMIMIC_ROOT=../NablaMimic LAFAN1_G1_DIR=data/lafan1_g1 \
       python -m scripts.audit.parity_original_env \
       --out reports/audit/parity_original_env.json
   ```

2. Phase C — four clips (`walk3_subject4`, `dance1_subject1`, `run1_subject2`,
   `fallAndGetUp2_subject3`) with the fixed adapter, checking termination reason
   and per-term reward. Pin the *same* host and thread configuration for both
   algorithms. The observation-parity part of this gate now already passes
   (§3.6), so what remains here is the closed-loop half.

3. Phase D — the full 40 clips with the **same checkpoint**
   (`7a8084c2…`) and the frozen protocol, written to a **new** filename
   (e.g. `nablamimic_canonical_eval_mjlab_anchorfix.json`). Do not overwrite
   `nablamimic_canonical_eval_mjlab.json`.

4. Record `source_git_sha`, CPU model, thread count, affinity and the
   `OMP`/`MKL`/`WARP` environment in the new record (`runner.build_record` has a
   free-form `extra` slot; the identity fields belong in `evaluator`).

5. Re-evaluate SuperMimic with the same environment if any shared evaluator code
   changed. **No shared evaluator code changed in this audit** — the change is
   additive to `StepInput` plus one new method, and SuperMimic reads neither —
   so a SuperMimic re-run is not required by this fix, but is required before any
   *cross-algorithm* claim made with a new execution configuration.

6. Then, and only then, decide whether 0.4819 is replaced.

---

## Appendix — files and tests

Changed:

| File | Change |
| --- | --- |
| `benchmark/evaluator/policy_api.py` | `StepInput` gains `anchor_pos_b_prev`, `anchor_ori_b_prev`; docstring states the two-frame rule |
| `benchmark/evaluator/env.py` | new `_anchor_relative(frame)`; `step_input()` populates the new fields |
| `benchmark/policies/nablamimic.py` | single anchor accessor at frame `f`, fail-closed; docstrings corrected |
| `docs/POLICY_INTERFACE.md` | documents both anchor conventions |
| `docs/NABLAMIMIC_POLICY.md` | §3.3/§5 corrected (see that file) |
| `docs/NABLAMIMIC_CANONICAL_EVAL.md` | status of 0.4819 qualified |

Added:

| File | Purpose |
| --- | --- |
| `tests/test_nablamimic_temporal_alignment.py` | 16 tests: frame selection, gravity across angular velocities, all 93 proprio dims, all 680 goal dims, history, residual and action-rate parity, waist rotation |
| `tests/test_nablamimic_real_parity.py` | runs the upstream env against the adapter on one real clip; skips cleanly where the upstream checkout, the dataset or MuJoCo are absent |
| `scripts/audit/reward_counterfactuals.py` | per-clip term decomposition + exact counterfactuals |
| `scripts/audit/reference_conversion.py` | resampling mechanism proof + per-clip comparison tool + floor-alignment check |
| `scripts/audit/parity_original_env.py` | **the real-data parity harness**: upstream `TrackingEnv` vs adapter, 93 + 686 dims, 5 frame classes × 2 state modes × N clips |
| `reports/audit/reward_counterfactuals.json` | machine-readable decomposition, 40 clips |
| `reports/audit/reference_conversion_mechanism.json` | data-free resampling analysis |
| `reports/audit/reference_conversion_40clips.json` | per-clip original-vs-canonical resampling table, real data |
| `reports/audit/floor_alignment_40clips.json` | per-clip root-height offset |
| `reports/audit/parity_original_env_40clips.json` | 400-sample observation parity, per clip/frame/channel, both adapters |
| `reports/audit/parity_smoke_4clips.json` | the four representative clips, same harness |

Deliberately **not** changed: `benchmark/protocol.json`, `constants.py` (rewards,
termination thresholds, `MAX_STEP_REWARD`), `runner.py` scoring,
`motion.py` converter, `assets/g1/g1.xml`, `spec.py`, the checkpoint, both
existing evaluation records, and `supermimic.py`.
