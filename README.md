<h1 align="center">modded-supermimic</h1>

<p align="center">
  <b>Unitree G1 Motion-Tracking Speedrun Benchmark.</b><br>
  Train a deployable 29-DoF policy from scratch to ≥ 0.82 normalized return and 0/40 failures across 40 LAFAN1 clips as fast as possible on 1× Colab G4 (RTX PRO 6000).
</p>

<p align="center">
  English · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/basicv8vc/modded-supermimic/actions/workflows/ci.yml"><img src="https://github.com/basicv8vc/modded-supermimic/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status"></a>
  <a href="benchmark/v2/protocol_v2.json"><img src="https://img.shields.io/badge/protocol-FULL__COLLISION-2f6feb" alt="Protocol FULL_COLLISION"></a>
  <img src="https://img.shields.io/badge/hardware-1%C3%97%20Colab%20G4%20(sm__120)-lightgrey" alt="1x Colab G4">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT%20%2B%20Apache--2.0%20assets-green" alt="MIT and Apache-2.0 assets"></a>
</p>

<p align="center">
  <img src="assets/g1-flash.png" alt="G1 Flash — Faster Unitree G1 Training Speedrun" width="160"><br>
  <sub><i>Faster training for Unitree G1 — from scratch to full 40-clip mastery at speedrun pace.</i></sub>
</p>

Inspired by [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt): **fix the task, hardware, and independent referee; leave the training algorithm, framework, and simulator open.**

---

## World Record Progression

Evaluated independently by the frozen referee ([protocol](benchmark/v2/protocol_v2.json)) on **Google Colab RTX PRO 6000**:

- **Physics**: FULL_COLLISION enabled
- **Runtime**: mjlab 1.6.0 · mujoco 3.11.0 · mujoco-warp 3.11.0 · warp-lang 1.16.0
- **Scoring**: Unclipped returns (no [0, 1] clamping)
- **Reward**: Zero self-collision reward penalty

| #     | Record                  | Training Time (0/40 & ≥0.82) | Trainer / Simulator                                                          | Return      | Fails      | Paper / Repo                                                                                           | Artifacts                                                                                                                                                            |
|:-----:|:----------------------- |:----------------------------:|:---------------------------------------------------------------------------- |:-----------:|:----------:|:------------------------------------------------------------------------------------------------------ |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **InstantTrack v1.4**   | **6m 45s**                   | PyTorch + mjlab / MuJoCo Warp<br><sub>InstantTrack v1.4.0 + BlinkMimic</sub> | **0.82169** | **0 / 40** | [BlinkMimic](https://arxiv.org/abs/2611.16422) · [Repo](https://github.com/Huy-Quang-Dao/InstantTrack) | [Record Card](records/v2/003_instanttrack_v14/README.md) · [Eval JSON](reports/v2/instanttrack_v14_s0/ckpt_002300_v2.json)                                           |
| **2** | **NablaMimic**          | **9m 32s**                   | JAX + NablaEngine 0.1.0                                                      | **0.84108** | **0 / 40** | [∇Mimic](https://arxiv.org/abs/2610.41654) · [Repo](https://github.com/sailab-ai/NablaMimic)           | [Record Card](records/v2/002_nablamimic_nablaengine/README.md) · [Eval JSON](reports/v2/nablamimic_25m_bisect/v2_it_001500.json)                                     |
| **3** | **SuperMimic Official** | **36m 51s**                  | PyTorch + mjlab<br><sub>HybridRobotics/supermimic</sub>                      | **0.82641** | **0 / 40** | [SuperMimic](https://arxiv.org/abs/2610.28480) · [Repo](https://github.com/HybridRobotics/supermimic)  | [Record Card](records/v2/001_supermimic_official_mjlab/README.md) · [Bisect Report](baselines/supermimic_mjlab/out/official_35k_20261115T120918Z/bisect_report.json) |

- **#1 InstantTrack v1.4** (Fastest Overall & mjlab): **6m 45s** (405.23 s raw, iter 2,300; net train ≈ 3m 24s / 204.16 s excluding 201.07 s initial compile; 15m end return: 0.84308) — incorporating [BlinkMimic](https://arxiv.org/abs/2611.16422) aligned to FULL_COLLISION (≈ **5.5× faster** raw / **10.8× faster** net than official HybridRobotics/supermimic, and ≈ **1.4× faster** raw than NablaMimic).
- **#2 NablaMimic** (Fastest on NablaEngine / JAX): **9m 32s** (572.18 s, iter 1,500; 25m end return: 0.86896) — [∇Mimic](https://arxiv.org/abs/2610.41654).
- **#3 SuperMimic Official** (Baseline): **36m 51s** (2,210.83 s, iter 15,000; stable 0/40 onset at 40m 53s / 2,453.09 s, return 0.83635; peak: 0.85751) — HybridRobotics/supermimic.
- **Machine-readable registry**: [reports/v2/registry.json](reports/v2/registry.json) (audited by CI).

---

## Rules

1. **Frozen Hardware**: 1× Google Colab G4 (NVIDIA RTX PRO 6000, 96 GB).
2. **Target Contract (Single Checkpoint)**: A single checkpoint evaluated across all 40 canonical LAFAN1 clips must satisfy:
   - **Aggregate normalized return ≥ 0.82** (unweighted mean of full-clip normalized returns, without clamping).
   - **Failed clips = 0 / 40** (no root_height < 0.25 m, root_tilt > 0.8 rad, or non-finite termination before step T - 1).
3. **Training-Only Wall Time**: Ranked by effective training wall-clock time from step 0 until the qualifying checkpoint is written. Excluding dataset fetch, environment setup, one-time JIT/CUDA-graph compilation, and post-hoc evaluation.
4. **From Scratch, Deployable Control**:
   - Weights must be trained from scratch on the 40 canonical LAFAN1 clips (no external pre-trained weights).
   - At test time, the frozen referee exposes only [StepInput](docs/POLICY_INTERFACE.md) (joint_pos, joint_vel, base_lin_vel, base_ang_vel, anchor_pos_b / anchor_pos_b_prev, anchor_ori_b / anchor_ori_b_prev, previous_action, and read-only ReferenceClip) and accepts (29,) joint actions at 50 Hz (target = action * action_scale + default_joint_pos).
5. **Open Training Stack, Frozen Referee**:
   - Any training framework, observation encoding, history/lookahead design, RL/imitation algorithm, or GPU training simulator is allowed (labelled explicitly under Trainer / Simulator on the leaderboard).
6. **Reproducibility & Proof**:
   - Single-seed runs (seed=0) are accepted for leaderboard exploration; high-scrutiny claims may request multi-seed verification.
   - Every submission PR must include a reproducible launch script (or pinned commit + patch), training log, checkpoint_sha256, and the complete 40-clip evaluation JSON generated by scripts/v2/fast_eval.py.

---

## Quick Start

```bash
git clone https://github.com/basicv8vc/modded-supermimic.git
cd modded-supermimic

# 1. Install pinned runtime on Linux CUDA (Colab G4)
pip install "mujoco==3.11.0" "mujoco-warp==3.11.0" "warp-lang==1.16.0" pytest
pip install "mjlab @ git+https://github.com/mujocolab/mjlab.git@0fb8a681136be94ffc636a3dd423cabb97d91f10"
pip install --no-deps --force-reinstall "mujoco==3.11.0" "mujoco-warp==3.11.0" "warp-lang==1.16.0"

# 2. Fetch pinned 40 LAFAN1 clips & verify hardware/manifest
python3 scripts/fetch_lafan1.py
python3 scripts/freeze_manifest.py --check
python3 scripts/v2/gpu_runtime_check.py --scope acceptance

# 3. Evaluate any policy implementing reset(context) / act(step) (~150-170 s on G4)
python3 scripts/v2/fast_eval.py \
    --backend batched --batch-envs 40 --cache-references \
    --policy-file ../my_algo/policy.py:make_policy \
    --checkpoint ../my_algo/ckpt.pt \
    --algorithm MyAlgo \
    --training-commit <40-hex> \
    --training-wall-s <seconds> \
    --seed 0 \
    --out reports/v2/my_algo_seed0.json

# 4. Audit format & registry consistency (CPU, no GPU required)
python3 scripts/v2/audit_registry.py
```

See [docs/POLICY_INTERFACE.md](docs/POLICY_INTERFACE.md) for the exact reset(context) / act(step) contract, and [baselines/](baselines/README.md) to reproduce any baseline end-to-end on Colab G4.

---

## Repository Layout

| Path                                                                           | Role                                                                                                                                                                                                                                     |
|:------------------------------------------------------------------------------ |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [benchmark/](benchmark/), [assets/g1/](assets/g1/), [scripts/v2/](scripts/v2/) | **Frozen referee** (mjlab 1.6.0 + mujoco-warp FULL_COLLISION), policy interface, and CLI/parity tools (benchmark.py).                                                                                                                 |
| [baselines/](baselines/README.md)                                              | One-command Colab G4 + Google Drive training & evaluation workflows for [instanttrack](baselines/instanttrack/README.md), [nablamimic](baselines/nablamimic/README.md), and [supermimic_mjlab](baselines/supermimic_mjlab/README.md). |
| [records/v2/](records/README.md)                                               | Human-readable speedrun record cards (001_..., 002_..., 003_...) paired with audited 40-clip JSONs under [reports/v2/](reports/v2/registry.json).                                                                                        |
| [docs/](docs/)                                                                 | Protocol design, policy interface spec, and G4 baseline & diagnostic report.                                                                                                                                                          |
| [tests/](tests/)                                                               | CPU contract/parity tests (test_benchmark_v2.py) and G4 CUDA runtime acceptance tests (test_v2_gpu_runtime.py).                                                                                                                          |

---

## Submitting a Record

1. Train from scratch on 1× Colab G4, run the 40-clip evaluation (scripts/v2/fast_eval.py --backend batched --batch-envs 40), and run a 2-clip serial-vs-batched parity check (scripts/v2/compare_evals.py).
2. Add a folder records/v2/\<NNN\>\_\<name\>/ containing a concise README.md, training config/patch/launch script, train.log, and the complete 40-clip evaluation JSON (under reports/v2/ or inside the record folder, registered in [reports/v2/registry.json](reports/v2/registry.json)).
3. Run python3 scripts/v2/audit_registry.py and pytest locally, then open a PR **without modifying any file under benchmark/ or assets/g1/**.

---

## License

Source code and documentation authored in this repository are released under the **MIT License** — see [LICENSE](LICENSE).

Third-party exception: assets/g1/ (Unitree G1 29-DoF MJCF + 35 STL meshes copied verbatim from [mujocolab/mjlab](https://github.com/mujocolab/mjlab)) remains under **Apache-2.0** ([licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)). LAFAN1 motion data is not redistributed; scripts/fetch_lafan1.py downloads it at a pinned revision. Full attribution in [NOTICE](NOTICE).
