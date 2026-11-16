# Canonical Evaluator v1 — `canonical-mjlab-g1-v1`

A runnable, deterministic physics evaluator for the
`modded-supermimic-g1-lafan1-v1` protocol (G1 29-DoF motion tracking on the 40
retargeted LAFAN1 clips).

It reproduces the **evaluation-relevant semantics** of mjlab v1.6.0
`Mjlab-Tracking-Flat-Unitree-G1` — robot model, actuators, contacts, control
rate, observation layout, reward terms, termination conditions — and drops
everything that only exists for RL training (randomization, curriculum,
auto-reset, observation corruption).

`benchmark/core.py`, `benchmark.py` and `benchmark/protocol.json` are unchanged:
the evaluator only *produces* records for them.

---

## 1. Frozen provenance

| item | value |
| --- | --- |
| mjlab repo / tag | `mujocolab/mjlab` @ `v1.6.0` |
| mjlab commit | `0fb8a681136be94ffc636a3dd423cabb97d91f10` |
| task / config | `Mjlab-Tracking-Flat-Unitree-G1` (`tasks/tracking/config/g1/env_cfgs.py`) |
| robot model | `asset_zoo/robots/unitree_g1/xmls/g1.xml` + 35 STL (vendored under `assets/g1/`) |
| robot model SHA256 | see `benchmark/evaluator/manifest.json` |
| dataset | `lvhaidong/LAFAN1_Retargeting_Dataset` `g1/` @ revision `07ff8845a94d59d4195a8ac04fe918d4e6338239` |
| dataset manifest SHA256 | see `benchmark/evaluator/manifest.json` |
| physics stack | `mujoco==3.11.0`, `mujoco-warp==3.11.0`, `warp-lang==1.16.0` |

`python scripts/freeze_manifest.py --check` re-verifies both hashes.

The dataset hashes were additionally cross-checked against the checksums that
ship with the NablaMimic baseline (`scripts/lafan1_g1.sha256`): all 40 match.

---

## 2. Physics and control

| knob | value | source |
| --- | --- | --- |
| DoF | 29 (1 free joint + 29 hinges) | `g1.xml` |
| self-collision | full (`contype=conaffinity=1` on all 33 `*_collision` geoms) | `FULL_COLLISION` |
| non-collision geoms | `contype=conaffinity=0` | `CollisionCfg(disable_other_geoms=True)` |
| feet | `condim=3`, `priority=1`, `friction[0]=0.6` (14 geoms) | `FULL_COLLISION` |
| ground | plane, MuJoCo default contact params | `TerrainEntityCfg(terrain_type="plane")` |
| actuators | 29 MuJoCo `<position>`, `ctrllimited=false`, `forcerange=±effort_limit` | `BuiltinPositionActuatorCfg` |
| gains | `kp = armature·ω²`, `kv = 2·ζ·armature·ω`, `ω = 10·2π`, `ζ = 2` | `g1_constants.py` |
| armature groups | 5020 / 7520-14 / 7520-22 / 4010 / waist(2×5020) / ankle(2×5020) | `g1_constants.py` |
| timestep | 0.005 s, `implicitfast`, `pyramidal`, Newton, 10 / 20 iterations | `MujocoCfg` |
| decimation | 4 → **50 Hz** control | `tracking_env_cfg` |
| joint soft limits | factor 0.9 | `soft_joint_pos_limit_factor` |
| gravity | (0, 0, −9.81) | `MujocoCfg` |

Compiled-model fingerprint (recorded in the manifest):
`nbody=31, njnt=30, ngeom=69 (68 robot + plane), nq=36, nv=35, nu=29`.

---

## 3. What a policy is given

**The benchmark does not define an observation.** It defines the information a
policy may use and the one thing it reads back: a 29-dim action. The interface is
two methods (see [`POLICY_INTERFACE.md`](POLICY_INTERFACE.md)):

```
reset(context)   once per clip: joint order, default pose, action scale,
                 control rate, read-only reference motion
act(step)        -> 29 joint targets
```

`step` carries proprioception plus the reference index: absolute joint
positions and velocities, base linear/angular velocity, the reference anchor pose
expressed in the robot anchor frame, and the previous raw action. Nothing else --
no `MjModel`, no `MjData`, no reward or termination object.

mjlab's 160-dim actor vector still exists, because the reference MLP was trained
on it, but it is now **one encoding built by one policy implementation**
(`policy.mjlab_observation`), not a requirement on submissions. An algorithm with
a 7-dim observation, a 3-frame history or a transformer is equally legal.

For reference, that encoding is `float32`-equivalent, concatenated in this exact
order (`tracking_env_cfg.observations["actor"]`, `play=True` ⇒ no corruption, no
noise):

| offset | dims | term | content |
| --- | --- | --- | --- |
| 0 | 58 | `command` | reference `joint_pos` (29) ++ `joint_vel` (29) at the current reference index |
| 58 | 3 | `motion_anchor_pos_b` | `torso_link` reference position in the robot `torso_link` frame |
| 61 | 6 | `motion_anchor_ori_b` | first two columns of the relative rotation matrix, row-major |
| 67 | 3 | `base_lin_vel` | `robot/imu_lin_vel` sensor (pelvis IMU site) |
| 70 | 3 | `base_ang_vel` | `robot/imu_ang_vel` sensor |
| 73 | 29 | `joint_pos` | `qpos − default_joint_pos` (encoder bias frozen to 0) |
| 102 | 29 | `joint_vel` | `qvel` |
| 131 | 29 | `actions` | previous *raw* action (zeros at reset) |

`default_joint_pos` is the mjlab `KNEES_BENT_KEYFRAME` resolved onto the 29 joints
(unlisted joints ⇒ 0.0).

## 4. Action contract

This part *is* fixed: it is the robot's interface, not the algorithm's choice.

* 29-dim, index `i` ⇔ `constants.JOINT_ORDER[i]` (mjlab "natural" joint order,
  which for this MJCF equals the LAFAN1 CSV column order — asserted at build).
* position target: `action[i] * G1_ACTION_SCALE[i] + default_joint_pos[i]`,
  with `G1_ACTION_SCALE[i] = 0.25 · effort_limit / stiffness` per actuator group
  (0.4386 shoulder/elbow, 0.5475 hip-pitch/yaw, 0.3507 hip-roll/knee,
  0.0745 wrist pitch/yaw).
* the target is held for all four physics substeps; no clipping, no smoothing.
* the evaluator validates every action before stepping the simulator: exact shape
  `(29,)`, numeric dtype, all finite. Anything else aborts the run with
  `ActionFormatError` naming the clip and step — silently repairing a bad action
  would hide a bug in the algorithm.
* how the action is produced is open. The mjlab-native implementation is
  deterministic (obs normalizer with `eps=1e-2`, then a `512-256-128` ELU MLP,
  mean output, no sampling); another algorithm may sample, smooth or use a
  completely different network, as long as it emits 29 numbers per 50 Hz step.

## 5. Reward (per control step, `dt = 0.02`)

`r_t = Σ_i w_i · dt · v_i` (`scale_rewards_by_dt=True`):

| term | weight | value |
| --- | --- | --- |
| `motion_global_root_pos` | +0.5 | `exp(−‖p_anchor − p_robot_anchor‖² / 0.3²)` |
| `motion_global_root_ori` | +0.5 | `exp(−quat_err² / 0.4²)` |
| `motion_body_pos` | +1.0 | `exp(−mean_i‖·‖² / 0.3²)` on the 14 tracked bodies, yaw-aligned relative frame |
| `motion_body_ori` | +1.0 | `exp(−mean_i quat_err² / 0.4²)` |
| `motion_body_lin_vel` | +1.0 | `exp(−mean_i‖·‖² / 1.0²)` |
| `motion_body_ang_vel` | +1.0 | `exp(−mean_i‖·‖² / 3.14²)` |
| `action_rate_l2` | −0.1 | `Σ (a_t − a_{t−1})²` on raw actions |
| `joint_limit` | −10.0 | `Σ` soft-limit violation |
| `self_collisions` | −10.0 | number of the 4 substeps with a robot–robot contact force > 10 N |

`MAX_STEP_REWARD = (0.5+0.5+1+1+1+1) · 0.02 = 0.1`.

## 6. Terminations

| term | condition |
| --- | --- |
| `time_out` | never fires during a canonical clip (episode length is the clip) |
| `anchor_pos` | `|z_ref(torso) − z_robot(torso)| > 0.25` |
| `anchor_ori` | `|gz_ref − gz_robot| > 0.8` (z of the projected gravity in each anchor frame) |
| `ee_body_pos` | any of `left/right ankle_roll`, `left/right wrist_yaw` with `|z_rel − z_robot| > 0.25` |
| `nonfinite` | NaN/Inf in `qpos`/`qvel` (robustness guard, never observed) |

`failed = any of the above fired before the clip ended`.

## 7. Return definition (no new metric)

```
n_steps = T − 1                      # T = number of 50 Hz reference frames
normalized_return = Σ_{t} r_t / (n_steps · 0.1)   # r_t = 0 once failed
```

The terminating step's reward still counts; later steps are masked to 0 and the
rollout stops (identical value, less compute). This is the published definition
(`NablaMimic/src/nablamimic/evaluate.py`: `full_len = lengths − 1`,
`tsum += where(alive, reward, 0)`, `tsum.sum(1) / (full_len · max_reward)`), and
`0.1` reproduces its `max_reward`, so the `0.82` gate in `protocol.json` is
comparable. The record's mean is the unweighted mean over the 40 clips.

## 8. Determinism

Guaranteed: identical binary + identical model + identical dataset ⇒ bit-identical
per-clip returns. Concretely:

* reset always writes reference frame 0 (root pose, root world velocity, joint
  state clipped to the soft limits) — `sampling_mode="start"`, no RSI;
* **all mjlab startup randomizations are disabled** (see §9);
* no observation noise / corruption, no `push_robot` interval event;
* the only stochastic element in a real policy (`GaussianDistribution` sampling)
  is replaced by the deterministic mean action.

## 9. Ambiguities that had to be frozen

Each of these is a decision the evaluator makes explicitly. Report them rather
than silently changing a quality gate.

1. **Startup domain randomization.** mjlab's `play=True` override removes
   `push_robot`, the pose/velocity ranges and obs corruption, but *keeps* the
   `startup` events `encoder_bias` (±0.01 rad), `base_com` (±2.5/5/5 cm) and
   `foot_friction` (0.3–1.2). Those make mjlab's own `evaluate.py` stochastic.
   → **Frozen: all three are disabled** (identity values). This is the only way
   to satisfy "deterministic rollout"; it also matches the published baselines,
   which report no randomization.
2. **Episode length.** mjlab's play config uses an effectively infinite episode,
   so `time_out` never fires and the loop length is a wrapper decision.
   → **Frozen: `n_steps = T − 1`**, matching the published baseline denominator.
   Using `T` instead would shift every score by ~1/500 of a step.
3. **Stale kinematics in reward/termination.** `ManagerBasedRlEnv.step` calls
   `sim.forward()` only *after* the termination/reward managers, so both see
   quantities that lag `qpos` by one physics substep. → **Frozen: replicated
   exactly**, because a policy trained in that MDP must be graded in the same
   one-substep-lagged MDP. Removing the lag would be "more correct physics" but
   a different (unpublished) metric.
4. **Relative-body-pose lag.** `body_pos_relative_w` is refreshed together with
   the reference-index increment, i.e. the `motion_body_*` terms at reference
   index `k` are anchored on the robot pose reached at `k−1`. → **Frozen:
   replicated.**
5. **Reward at the terminating step.** Counted (before masking), not dropped.
6. **mjlab import.** `mjlab==1.6.0` cannot be installed in this environment, so
   the evaluator re-implements the env semantics with `mujoco` directly, using
   the official `g1.xml`, the official `g1_constants.py` numbers and the official
   `tracking_env_cfg.py` terms. It does not call mjlab code. The conformance
   tests (`tests/test_evaluator.py`) pin the resulting model structure, gains,
   action scale, observation layout and reward magnitudes, so a future mjlab
   upgrade cannot silently change the numbers.
7. **Backend.** `mujoco` (CPU, double precision, deterministic) is the reference
   backend. `mujoco-warp` is installed and version-recorded but not yet used for
   rollout (see §11).
8. **Update (official mjlab reuse refactor).** The premise of item 6 is no
   longer true: `mjlab==1.6.0` (commit `0fb8a681136be94ffc636a3dd423cabb97d91f10`)
   now installs in the pinned benchmark environment and the evaluator drives the
   official `ManagerBasedRlEnv` through thin glue
   (`benchmark/evaluator/{motion,env,runner}.py`) instead of re-implementing
   physics, rewards, terminations and motion tracking. The frozen score rules of
   items 1-5 are unchanged, with two consequences reported in
   `reports/alignment_v1_vs_mjlab.md`: (a) the frame-index pairing is mjlab's
   own (the first scored frame after reset is 1, a constant +1 offset against
   the pre-refactor evaluator); (b) rollout physics runs the official warp
   float32 backend of item 7, not double-precision `mujoco`. The conformance
   role of `tests/test_evaluator.py` and the manifest fingerprint is kept.

## 10. Running it

Formal run with the bundled mjlab-native loader (one real checkpoint, all 40
clips, one GPU):

```bash
python scripts/export_actor_npz.py \
    --checkpoint logs/rsl_rl/g1_tracking/<run>/model_30000.pt \
    --out checkpoints/g1_tracking_model_30000.npz

python scripts/run_canonical_eval.py \
    --checkpoint checkpoints/g1_tracking_model_30000.npz \
    --algorithm SuperMimic \
    --training-commit <40-hex git SHA of the training code> \
    --seed 0 \
    --gpu-name "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
    --training-wall-s 5220 \
    --out reports/canonical_eval.json

python benchmark.py validate reports/canonical_eval.json
```

Formal run with your own algorithm (any observation, network or checkpoint
format -- see [`POLICY_INTERFACE.md`](POLICY_INTERFACE.md)):

```bash
python scripts/run_canonical_eval.py \
    --policy-file ../my_algo/infer.py:make_policy \
    --checkpoint ../my_algo/ckpt_30000.pt \
    --algorithm MyAlgo \
    --training-commit <40-hex> --seed 0 \
    --gpu-name "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
    --training-wall-s 5220 \
    --out reports/my_algo.json
```

`--checkpoint` is hashed for the record and passed to your factory; the evaluator
never parses it. Without `--policy-file` it falls back to the bundled loader.

Smoke test (no checkpoint; scripted policy; **not submittable**):

```bash
python scripts/run_canonical_eval.py --smoke-policy replay \
    --clips walk1_subject1 --out reports/smoke.json
```

`--smoke-policy hold` holds the default pose, `--smoke-policy replay` PD-tracks the
reference joint angles. Both exist only to prove that the model loads, that
physics runs in closed loop, and that rewards/terminations fire; the resulting
record carries `evaluation.mode = "smoke"` and
`evaluation.policy_kind = "smoke_scripted"`, which the submission gate rejects.

## 11. Known gaps before a formal 40-clip run

1. **No trained checkpoint in this repo.** mjlab distributes G1 tracking weights
   as a W&B artifact (`evaluate.py` downloads `wandb:...`), which needs
   credentials. The checkpoint path is implemented and unit-tested with a
   synthetic `.npz`, but a *real* 40-clip score cannot be produced here.
   An audit of every prior SuperMimic / NablaMimic artifact (see
   `docs/CHECKPOINT_AUDIT.md`) found exactly one real checkpoint on disk, and it
   was trained for a **different task interface**: a goal-conditioned world-model
   policy (`pi.enc` 730→32, `pi.dec` 1022→29, 10-frame history) trained in
   IsaacLab, not a 29-DoF tracking policy on these 40 clips. `.pt` checkpoints are
   now read without torch (`torchzip.py`), so that rejection is structural rather
   than environmental.
   Note what changed with the policy interface (`POLICY_INTERFACE.md`): the
   observation encoding is no longer a benchmark-level constraint, because an
   algorithm may bring its own encoder and loader. What is still missing is
   narrower and unavoidable -- a checkpoint *trained for this task*, i.e. G1,
   these 40 motions, this reward. No amount of interface work produces one.
2. **CPU backend only.** The rollout uses `mujoco` on CPU (measured 0.99 ms per
   50 Hz step, so the 441,080 steps of all 40 clips need ≈7 min of rollout plus
   ≈3 min of reference building). `mujoco-warp` is not wired up yet; the
   protocol's GPU field therefore cannot be satisfied by this machine.
3. **Official-environment parity run.** Once mjlab is installable, running
   `evaluate.py` and this evaluator side by side on the same checkpoint is the
   remaining end-to-end verification.

## 12. What the smoke runs prove — and what they do not

`--smoke-policy replay` feeds `a_t = (q_ref(t) − q_default) / action_scale`, i.e.
the reference joint angles as position targets, with no feedback.

**Observed** (`reports/smoke_replay_40clips.json`): all 40 clips load, reset to
their own first frame and run; every episode ends between step 15 and 42 with
`ee_body_pos`, mean normalized return `0.00199`, 40 failed.

**Diagnosed** (`scripts/diagnose_clip.py`, `dance1_subject1`):

| step | pelvis Δz | Δ projected gravity | ankle z err | wrist z err |
| --- | --- | --- | --- | --- |
| 10 | 0.018 | 0.0004 | 0.014 | 0.177 |
| 20 | 0.035 | 0.0069 | 0.014 | 0.214 |
| 39 | 0.159 | 0.0038 | 0.010 | 0.261 → `ee_body_pos` |

The robot does **not** fall: `anchor_pos` (0.25) and `anchor_ori` (0.8) are never
close, the ankles track to 1 cm, and the reference wrist barely moves over the
same span (1.059 m → 1.046 m). The wrists sag ~26 cm while the torso stands.

This is the *soft* mjlab servo doing exactly what it is configured to do. Arm
stiffness is `kp = 0.25 · effort_limit / action_scale` — for the 5020 arm group
`kp = 14.25 Nm/rad`, so holding a ~0.35 m arm out needs a steady-state offset of
`τ/kp ≈ 0.5 rad`; a PD position servo has no gravity feed-forward. Re-checked
against the frozen constants (`_reflected_inertia` two-stage planetary:
`0.139e-4 · (3.5556 · 4.5)² + … = 3.610e-3` ⇒ `kp = 3.610e-3 · 62.83² = 14.25`),
and the leg groups agree too (`7520_14 → 40.18`, `7520_22 → 99.10`), so this is
the official model, not a transcription error. A trained policy outputs the
compensating offset inside its action; an open-loop replay cannot.

Consequences, stated plainly:

* the smoke record **validates mechanism** — 40 deterministic per-clip rollouts,
  faithful resets, faithful terminations, a record the validator accepts as
  structurally complete;
* it says **nothing** about the evaluator's discriminative power, and `0.00199`
  must never be read as a baseline or compared to the `0.82` gate;
* the harness itself is not step-limited: `evaluate_clip` runs until the clip
  ends and only stops early on a real termination (post-termination steps are
  masked to 0 anyway, so the return is identical). A policy that survives runs
  all 13,654 steps of `dance1_subject1`.

