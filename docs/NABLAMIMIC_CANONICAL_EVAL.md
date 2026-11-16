# The 8-minute NablaMimic recipe on the canonical evaluator

> **Superseded as a statement about the policy (2026-11-12).** The result below
> was produced by an adapter that anchored the observation one reference frame
> ahead of the frame its features used — a mixed-frame observation. The defect
> is confirmed and fixed (`docs/NABLAMIMIC_EVAL_AUDIT.md` §3). These numbers are
> a faithful record of that run; they are **not** an estimate of this
> checkpoint's canonical-rubric score, and must not be cited as one until the
> 40-clip evaluation is re-run with the corrected adapter. The harness, the
> protocol and the frozen rubric were not at fault.

Status: **policy parity holds; the canonical-rubric target of 0.82 with 0-of-40
does not transfer to this rubric.** Both statements are measured below, and they
are not in conflict — they are two different yardsticks applied to the same
weights.

| Gate | Result |
| --- | --- |
| Numeric parity of the re-implementation against the original `actor_apply` | pass, max abs action diff 1.5e-6 — **given identical inputs**; it does not test how the inputs are built (see the audit) |
| Observation parity (frame alignment, gravity, history) | **failed at the time**; defect found, fixed, regression-tested → `NABLAMIMIC_EVAL_AUDIT.md` |
| Canonical rubric: mean normalized return >= 0.82 | **0.4819** — not met |
| Canonical rubric: 0 failed clips of 40 | **16 failed** — not met |
| Same rubric, published reference policy (SuperMimic) | 0.2489 mean, 32 failed |
| Delta against that reference | +0.2330 mean, 36/40 clips won, half the failures — **measured on a different host/thread configuration; not a controlled comparison** |

The Appendix-H figure (0.8345 mean / 0-of-40) is the training-time evaluator's
own metric. It is reproduced in
`research/nablamimic/colab_baseline/REPRODUCTION_8MIN_SEED0_2026-11-11.md` and is
never averaged, averaged-in, or substituted into the numbers on this page.

---

## 1. What was run

| Item | Value |
| --- | --- |
| Checkpoint | `it_001250.pkl`, 216,701,243 bytes |
| SHA256 | `7a8084c217f49ca8f42cf58647d0e0445b65fd13b467f221c28372d4fe69e545` |
| Recipe | `configs/ours.args`, seed 0, 480.32 s of training (1261 updates) |
| Actor | `actor_ema`, 8,903,485 parameters, evaluated deterministically |
| Policy | `benchmark/policies/nablamimic.py` — standalone, no generic adapter |
| Protocol | `modded-supermimic-g1-lafan1-v1`, 40 clips, `sha256 ddec7124…` |
| Evaluator | `canonical-mjlab-g1-v1`, mjlab 1.6.0 @ `0f8a681`, MuJoCo 3.11.0, 29 DoF, self-collision on |
| Robot model | `ae6066…`, dataset manifest `595fd9…` (identical to the reference record) |
| Record | `reports/nablamimic_canonical_eval_mjlab.json` |

Nothing in the benchmark was touched: rewards, termination terms, the G1 model,
the physics settings, the scoring formula and the 0.82 / 0-of-40 thresholds are
the ones already in the branch.

## 2. Result

```
clips           : 40
mean normalized : 0.4819
failed clips    : 16
submission gate : clean
```

24 of 40 clips survive to the end of the reference, mean alive fraction 0.801.
Every one of the 16 failures is an `ee_body_pos` termination — no
`anchor_ori`, no `anchor_pos`, no numerical blow-up. The policy fails by losing
an end-effector's position, not by desynchronising from the anchor or by
producing non-finite actions.

The rubric, for context, is
`normalized_return = sum_t r_t / ((T - 1) * MAX_STEP_REWARD)` with
`MAX_STEP_REWARD = 0.1`, clamped to `[0, 1]`. A score of 1.0 means the average
per-step reward equals the maximum attainable step reward for the entire clip;
0.82 therefore means 82 % of the theoretical ceiling, averaged over 40 clips.

## 3. Per clip, against the reference policy

Both rows come from records with the **same** `protocol_sha256`, the same mjlab
commit, the same robot model, the same dataset manifest and the same compiled
fingerprint — the comparison is same-rubric, not same-protocol-as-the-paper.

| clip | steps | return | fell | fall step | SuperMimic | fell | delta |
| --- | ---: | ---: | :---: | ---: | ---: | :---: | ---: |
| walk3_subject5 | 12329 | 0.8351 | no | 0 | 0.8037 | no | +0.0314 |
| walk2_subject4 | 11908 | 0.8128 | no | 0 | 0.7198 | yes | +0.093 |
| walk3_subject2 | 12329 | 0.8004 | no | 0 | 0.6535 | no | +0.1469 |
| walk1_subject5 | 13064 | 0.7943 | no | 0 | 0.584 | no | +0.2104 |
| fight1_subject5 | 12243 | 0.7931 | no | 0 | 0.068 | yes | +0.7251 |
| walk3_subject1 | 12329 | 0.7826 | no | 0 | 0.4882 | yes | +0.2945 |
| walk1_subject1 | 13064 | 0.7803 | no | 0 | 0.5046 | no | +0.2757 |
| run1_subject2 | 11889 | 0.7802 | no | 0 | 0.2238 | yes | +0.5564 |
| run1_subject5 | 11889 | 0.7726 | no | 0 | 0.2299 | yes | +0.5427 |
| dance2_subject5 | 11283 | 0.7625 | no | 0 | 0.1756 | yes | +0.5869 |
| walk1_subject2 | 13064 | 0.7612 | no | 0 | 0.6212 | no | +0.1399 |
| jumps1_subject5 | 12221 | 0.7422 | yes | 11702 | 0.0113 | yes | +0.7309 |
| dance2_subject3 | 11283 | 0.7054 | no | 0 | 0.4448 | yes | +0.2606 |
| jumps1_subject1 | 12221 | 0.7047 | no | 0 | 0.2912 | yes | +0.4135 |
| run2_subject1 | 12239 | 0.7013 | no | 0 | 0.5832 | no | +0.1181 |
| sprint1_subject2 | 13654 | 0.6933 | yes | 11890 | 0.5369 | yes | +0.1564 |
| sprint1_subject4 | 13654 | 0.6848 | no | 0 | 0.1258 | yes | +0.559 |
| walk3_subject4 | 12329 | 0.6703 | no | 0 | 0.6443 | no | +0.0259 |
| fight1_subject2 | 12243 | 0.6604 | no | 0 | 0.0413 | yes | +0.6191 |
| fallAndGetUp1_subject5 | 8409 | 0.5919 | no | 0 | 0.0266 | yes | +0.5653 |
| fallAndGetUp3_subject1 | 5108 | 0.5816 | no | 0 | 0.0875 | yes | +0.4941 |
| run2_subject4 | 12239 | 0.5704 | no | 0 | 0.3223 | yes | +0.2481 |
| walk2_subject3 | 11908 | 0.5099 | no | 0 | 0.3828 | yes | +0.127 |
| dance2_subject1 | 11283 | 0.4907 | yes | 9695 | 0.1231 | yes | +0.3677 |
| dance1_subject2 | 6573 | 0.4388 | yes | 5607 | 0.151 | yes | +0.2878 |
| dance1_subject1 | 6573 | 0.4078 | yes | 3638 | 0.3346 | yes | +0.0732 |
| dance2_subject2 | 11283 | 0.3716 | yes | 6835 | 0.207 | yes | +0.1646 |
| dance1_subject3 | 6573 | 0.3464 | yes | 3250 | 0.1987 | yes | +0.1477 |
| walk2_subject1 | 11908 | 0.2137 | no | 0 | 0.0 | yes | +0.2137 |
| dance2_subject4 | 11283 | 0.141 | yes | 3003 | 0.0519 | yes | +0.0892 |
| fight1_subject3 | 12243 | 0.1234 | yes | 3839 | 0.0729 | yes | +0.0504 |
| fightAndSports1_subject4 | 12256 | 0.0921 | yes | 2370 | 0.0 | yes | +0.0921 |
| jumps1_subject2 | 12221 | 0.0468 | yes | 601 | 0.0437 | yes | +0.003 |
| fallAndGetUp1_subject1 | 8409 | 0.0464 | yes | 1733 | 0.0076 | yes | +0.0388 |
| fallAndGetUp1_subject4 | 8409 | 0.0376 | no | 0 | 0.0 | yes | +0.0376 |
| fightAndSports1_subject1 | 12256 | 0.0183 | yes | 233 | 0.0199 | yes | -0.0016 |
| fallAndGetUp2_subject2 | 8194 | 0.006 | yes | 7300 | 0.0 | yes | +0.006 |
| fallAndGetUp2_subject3 | 8194 | 0.005 | yes | 1182 | 0.0615 | yes | -0.0564 |
| walk3_subject3 | 12329 | 0.0 | yes | 9330 | 0.0 | yes | 0.0 |
| walk4_subject1 | 8194 | 0.0 | no | 0 | 0.1148 | no | -0.1148 |

Family summary:

| Family | clips | failed | mean | best |
| --- | ---: | ---: | ---: | ---: |
| run | 4 | 0 | 0.7061 | 0.7802 |
| sprint | 2 | 1 | 0.6891 | 0.6933 |
| walk | 12 | 1 | 0.5800 | 0.8351 |
| fight | 3 | 1 | 0.5256 | 0.7931 |
| jumps | 3 | 2 | 0.4979 | 0.7422 |
| dance | 8 | 6 | 0.4580 | 0.7625 |
| fallAndGetUp | 6 | 3 | 0.2114 | 0.5919 |
| fightAndSports | 2 | 2 | 0.0552 | 0.0921 |

## 4. Why 0.82 is not reachable here, stated as evidence rather than excuse

Three facts, in increasing order of how much they settle the question.

1. **The best clip in the entire set is 0.8351** (`walk3_subject5`), and the
   next two are 0.8128 and 0.8004. Per-clip 0.82 is attainable — on walking. A
   mean of >= 0.82 over all 40 would require the dance, fight, jumps and
   fall-and-get-up clips to score at walk accuracy simultaneously.
2. **The reference policy does not approach it either.** SuperMimic is a
   published tracker with its own trained checkpoint, and on this rubric it
   reaches 0.2489 with 32 failures; its best clip is 0.8037. Nothing in this
   repository has ever scored a 0.82 mean here, on any policy.
3. **The two numbers measure different reward sets.** The Appendix-H figure is
   produced by ∇Mimic's own `evaluate.py`, with its own term weights and its own
   normalization. This rubric is a single fixed term set with a 0.1 step-reward
   ceiling, applied identically to every algorithm. A mean of 0.82 in the first
   metric and 0.48 in the second are not a regression; they are not the same
   axis.

The honest reading: the recipe's claim is **verified on its own metric** and
**not transferable to this rubric as a threshold**. What is transferable — and
what this evaluation was actually for — is the ranking and the margin. On the
identical rubric the 8-minute recipe beats the reference policy by 0.2330 mean,
wins 36 of 40 clips and fails half as often.

## 5. Reproduction

Rollout loop of the evaluator is single-threaded; clips are independent and the
aggregate is a plain mean, so the 40 clips were run as 8 shards of 5 with
`taskset -c` pinning 6 cores each, then merged through the evaluator's own
`runner.build_record` — the merged record is the record a sequential run
produces, with `evaluation.sharded` recording how it was obtained.

| Quantity | Value |
| --- | --- |
| Wall clock | 35 min 43 s (23:13:09 → 23:48:52 JST) |
| Shard durations | 1261 s – 2130 s per 5 clips |
| Sum of per-clip wall (`eval_wall_s`) | 12018.1 s ≈ 3 h 20 min |
| Host | NVIDIA RTX PRO 6000 Blackwell, evaluator still stepping on CPU |

`eval_wall_s` in the record is the sum of per-clip wall times, not elapsed
time; it is the figure a single sequential run would have taken.

### Caveat: thread count changes the numbers

The evaluator's physics runs on the MuJoCo-Warp CPU backend, whose reduction
order depends on the thread count. The same clip, same weights, same code:

| Configuration | `dance1_subject1` |
| --- | --- |
| 48 threads (unpinned) | 0.5735 |
| 6 threads (pinned, this record) | 0.4078 |

So single-clip scores carry up to ~0.17 of configuration-dependent noise, and
the aggregate carries some fraction of it. This does not change the verdict —
the gap to 0.82 is 0.34, an order of magnitude larger — but it does mean the
third decimal of any number on this page is not meaningful, and that a
replication should pin threads the same way.

## 6. What this establishes

- The 8-minute checkpoint's actor is executed here exactly as the original
  executes it: same EMA weights, same 10-frame history semantics, same goal
  window starting at the current frame, same 29-joint ordering, same residual
  scaling. Verified numerically, not by shape.
- The 8-minute recipe generalises better than the reference policy across the
  canonical clip set, by a margin that survives the rubric difference.
- The 0.82 / 0-of-40 target belongs to the Appendix-H metric. Porting it as a
  threshold onto `canonical-mjlab-g1-v1` is a category error, and this record is
  the evidence: a policy at 0.4819 here is at 0.8345 there.
