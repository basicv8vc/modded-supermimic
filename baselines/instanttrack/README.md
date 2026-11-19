# InstantTrack v1.4 / BlinkMimic — Colab G4 + V2 Workflow

Integrates [`Huy-Quang-Dao/InstantTrack`](https://github.com/Huy-Quang-Dao/InstantTrack) (`v1.4.0`, which incorporates [`fuyichen0725/blinkmimic`](https://github.com/fuyichen0725/blinkmimic) `v1.0.0`) into the `modded-supermimic` V2 speedrun benchmark without modifying any frozen referee file.

## Why V2 Alignment Is Required Before Training

Upstream `InstantTrack` and `BlinkMimic` inherit `supermimic_vibed`'s G1 URDF setup with **self-collisions disabled** (`contype=0, conaffinity=1`) and download LAFAN1 from HuggingFace `main`. Under V2's frozen `FULL_COLLISION` referee, policies trained without self-collision suffer severe degradation (see [V2 Diagnostic Report](../../docs/V2_G4_BASELINE_AND_DIAGNOSTIC_REPORT.md)).

During setup, [`vm/patch_full_collision.py`](vm/patch_full_collision.py) aligns the training environment to V2:
- **G1 model**: frozen [`assets/g1/g1.xml`](../../assets/g1/g1.xml) with `FULL_COLLISION` matching [`benchmark/evaluator/spec.py`](../../benchmark/evaluator/spec.py) (`*_collision` `contype=1, conaffinity=1`; feet `condim=3, priority=1, friction=0.6`; visual geoms disabled).
- **LAFAN1 motions**: pinned 40-clip dataset from [`scripts/fetch_lafan1.py`](../../scripts/fetch_lafan1.py).
- **Algorithm hyperparameters**: unmodified upstream `PRESET[instanttrack]` (`v1.4.0`) or `PRESET[main]` (`BlinkMimic`) with `--seed 0`.

## Policy Interface Adapter

[`policy.py`](policy.py) implements the pure-NumPy `reset(context)` / `act(step)` contract for `scripts/v2/fast_eval.py`:
- Reconstructs the 99-dim proprioceptive observation from `StepInput` (`anchor_ori_b_prev` + waist kinematics).
- Matches `MotionLib.goal` end-of-clip clamping and supports both 3- and 4-hidden-layer MLP depths.
- Applies the `--pi_ref_skip` residual `(q_ref(f+1) - ref_q0) / ref_alpha` when `pi.ref_q0` / `pi.ref_alpha` buffers are present.

## Quick Start (Colab G4)

```bash
cd baselines/instanttrack

# Default: InstantTrack v1.4.0 (commit b9512a96...), seed 0, 15 min budget
./run.sh all

# Or run BlinkMimic v1.0.0 (commit d264658d...) explicitly
UPSTREAM=blinkmimic ./run.sh all

# Monitor / bisect / collect / stop
./run.sh status
./run.sh step bisect
./run.sh collect
./run.sh down
```
