# Policy interface — what the benchmark fixes and what it doesn't

`modded-supermimic` follows [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt):
**fix the task, the hardware and the score; leave the algorithm open.**

That means the evaluator must not assume anything about how a submission turns
robot state into motor commands. It calls one entry point, receives 29 numbers,
and scores the resulting trajectory.

## 1. The boundary

| Fixed by the benchmark | Owned by the algorithm |
| --- | --- |
| G1 29-DoF model, official mjlab physics, full self-collision | observation layout and encoding |
| the 40 LAFAN1 motions | history / look-ahead window |
| reward, termination, `normalized_return`, the 0.82 gate | network architecture |
| control interface: 29 dims, joint order, 50 Hz, action scale, PD gains | checkpoint format and loader |
| reference motion exposed to every submission | training algorithm and training simulator |

Nothing about mjlab is required. The bundled mjlab-native MLP is **one**
implementation of the interface (§5), kept because it is a natural baseline —
not a shape every submission must match.

## 2. The interface: two methods

```python
class MyPolicy:
    def reset(self, context): ...   # once per clip, before the first step
    def act(self, step):            # once per control step (50 Hz)
        return action               # array-like, shape (29,)
```

That is the whole contract. No registry, no weight converter, no base class to
inherit. The evaluator verifies the object has both methods and fails with a
readable error if it doesn't (`PolicyInterfaceError`).

### `reset(context: ClipContext)`

| field | meaning |
| --- | --- |
| `clip` | clip id, e.g. `dance1_subject1` |
| `joint_order` | 29 joint names, the index order of every action |
| `default_joint_pos` | (29,) radians — the pose a zero action commands |
| `action_scale` | (29,) radians per unit action |
| `control_dt` | 0.02 s (50 Hz) |
| `reference` | read-only `ReferenceClip` (see below) |

Per-clip state must be cleared here: one policy object is reused across all 40
clips, so a hidden buffer from the previous clip is a bug the algorithm owns.

### `act(step: StepInput) -> array-like (29,)`

| field | shape | meaning |
| --- | --- | --- |
| `step` | scalar | 0-based control step within the clip |
| `reference_index` | scalar | frame index into `reference` arrays |
| `time_s` | scalar | `step * control_dt` |
| `joint_pos` | (29,) | absolute joint positions, radians |
| `joint_vel` | (29,) | joint velocities, rad/s |
| `base_lin_vel` | (3,) | pelvis IMU linear velocity, body frame |
| `base_ang_vel` | (3,) | pelvis IMU angular velocity, body frame |
| `anchor_pos_b` | (3,) | reference `torso_link` position in the robot `torso_link` frame, at `reference_index` |
| `anchor_ori_b` | (6,) | first two columns of that relative rotation matrix, at `reference_index` |
| `previous_action` | (29,) | raw action issued on the previous step (zeros at step 0) |
| `reference` | — | same `ReferenceClip` as in the context |
| `anchor_pos_b_prev` | (3,) | the same quantity at `reference_index - 1` |
| `anchor_ori_b_prev` | (6,) | the same quantity at `reference_index - 1` |

All arrays are copies, so a policy may keep them. These are physical quantities a
real robot would have plus the task specification; there is no privileged
simulator state.

#### Two frames are live at every step, and they differ

`reference_index` is the frame the **upcoming** reward and termination compare
against: mjlab advances `MotionCommand.time_steps` at reset, so the first step
has `reference_index == 1` while the robot still stands on frame 0. That makes
`reference_index - 1` the frame the robot's **own state** belongs to.

The anchor fields exist twice because both conventions are legitimate:

* `anchor_*_b` follow mjlab's observation terms, which always read
  `time_steps` — the upcoming frame. mjlab-trained policies (`supermimic.py`,
  `evaluator/policy.py`) observe this one and must keep using it.
* `anchor_*_prev` carry the same two quantities one frame earlier, at
  `reference_index - 1`. An algorithm whose observation describes its current
  state — as `nablamimic.py` does — must anchor there.

Combining frame-`f` reference features with the frame-`f+1` anchor produces a
mixed-frame observation. That is not a shape error and no dimension check can
catch it; it is a silent numerical corruption of the gravity channel. See
`NABLAMIMIC_EVAL_AUDIT.md` §3.

### `ReferenceClip` (read-only)

| field | shape | meaning |
| --- | --- | --- |
| `num_frames` | scalar | `T`; the episode has `T - 1` control steps |
| `control_dt` | scalar | 0.02 s |
| `joint_pos` | (T, 29) | reference joint angles, radians |
| `joint_vel` | (T, 29) | reference joint velocities, rad/s |
| `root_pos_w` | (T, 3) | reference pelvis position, world |
| `root_quat_w` | (T, 4) | reference pelvis orientation, wxyz |

The **full** clip is exposed, not just the current frame, so that algorithms
using look-ahead are not penalised. Every submission sees the same data, so this
is part of the task description rather than an advantage. The arrays are marked
non-writeable; mutating them raises.

## 3. The control interface is not negotiable

```python
target[i] = action[i] * action_scale[i] + default_joint_pos[i]   # radians
```

* index `i` is `joint_order[i]` — always.
* 50 Hz: one action, then four physics substeps of 5 ms; the target is held.
* the actuator PD gains, force limits and the physics are frozen. A policy cannot
  change them; it never receives the model.
* **every action is validated** before stepping: exact shape `(29,)`, numeric
  dtype, all finite. A wrong width, a `(1, 29)` array, a `None` or a `NaN` aborts
  the run with `ActionFormatError` naming the clip and step. The evaluator does
  not clip, reshape or repair: silently fixing a bad action would hide a bug and
  quietly change what was measured.

## 4. Plugging in

```
python scripts/run_canonical_eval.py \
    --policy-file ../my_algo/infer.py:make_policy \
    --checkpoint ../my_algo/ckpt_30000.pt \
    --algorithm MyAlgo --training-commit <40-hex> --seed 0 \
    --gpu-name "..." --training-wall-s 5220 \
    --out reports/my_algo.json
```

* `--policy-file PATH:FACTORY` — `PATH` is a `.py` file or an importable module,
  `FACTORY` is called as `factory(checkpoint_path)` when `--checkpoint` is given,
  else `factory()`. It must return an object with `reset`/`act`.
* `--checkpoint` is **hashed for the record and handed to your factory**. The
  evaluator never opens, parses or converts it.
* A minimal working example lives in
  [`examples/scripted_replayer.py`](../examples/scripted_replayer.py) (~40 lines).

Without `--policy-file`, `--checkpoint` is loaded by the bundled mjlab-native
loader (§5) for backwards compatibility.

## 5. The bundled mjlab-native implementation

`benchmark/evaluator/policy.py` implements the interface for rsl_rl actors
trained by mjlab:

* `reset` records `default_joint_pos`; `act` builds mjlab's 160-dim vector via
  `mjlab_observation(step, default_joint_pos)` and runs the MLP
  (optional `EmpiricalNormalization`, `eps=1e-2`, deterministic mean output).
* it accepts `.npz` exports (`scripts/export_actor_npz.py`) and rsl_rl `.pt`
  files, and it **rejects** a checkpoint whose `obs_dim`, joint order or action
  scale does not match the build. That strictness is a property of this loader,
  not of the benchmark: another algorithm may use any format it likes, including
  one that contains no `actor`/`critic` keys at all.
* `forward(obs)` exposes the raw network call for tests.

## 6. What the evaluator guarantees, and what it can't

Guaranteed: identical robot, physics, motions, reward, terminations, metric,
control interface and simulator build for every submission; deterministic reruns;
a record that `benchmark/core.py` can audit; and 40 clips evaluated with one
checkpoint hash.

Not guaranteed, and worth stating explicitly:

* **fairness across training simulators.** A policy trained in IsaacLab, MJX or
  MuJoCo Warp is scored in *this* MuJoCo model. Sim-to-sim gaps are part of the
  measured difference, not an artefact of the evaluator — but it does mean scores
  from different training stacks are not strictly like-for-like in physics terms.
* **protection against a policy that peeks at the reference.** The full reference
  is exposed by design; using it is legitimate, and cheating by memorising the 40
  clips is a different problem (the protocol's answer is that the clips are public
  and the metric is per-clip tracking, so memorisation is the task).
* **that a submitted checkpoint is the one that was trained.** The evaluator
  records a SHA of whatever file it was given.

## 7. Status

* No trained checkpoint exists in this repository, so no formal score has been
  produced with this interface.
* Rollouts run on the `mujoco` CPU backend (~1 ms/step). `mujoco-warp` is pinned
  but not wired in, so the protocol's GPU field cannot be satisfied here.
* A side-by-side parity run against mjlab's own `evaluate.py` (same checkpoint,
  same clips) is still pending; it needs a real mjlab checkpoint.
