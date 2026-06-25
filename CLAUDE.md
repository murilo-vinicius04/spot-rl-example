# spot-rl-example — deploy context

This repo deploys Spot locomanipulation policies (ONNX) onto the real robot via the Boston
Dynamics SDK. Policies are **trained** in the sibling `spot-locomanipulation` workspace and
**exported to ONNX** from inside the Isaac Sim container. The shipped policies live in `models/`
(see README). This file documents only the IsaacLab-container interaction needed to (re)export.

## Policies are 19-DoF whole-body, obs `[1,69]` → actions `[1,19]`

All three `models/*` policies share one observation/action layout, so they are interchangeable on
the Phase-2 deploy path. The 69-vector is:
`base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) + joint_pos(12) +
joint_vel(12) + arm_joint_pos(7) + arm_joint_vel(7) + actions(19)`. Action scale `0.2`,
`use_default_offset=True`; the 19 default joint offsets are identical across phase2 / v9 / wrench
(so one `env.yaml` is valid for all — that's why `models/*/env.yaml` are copies of the deploy config).

## v9/wrench are COMPOSITE actors — the stock exporter is wrong

v9 and wrench are `SerialCorrectionActor`s: a **skill-drive** net (gait) + a **balance-corrector**
net (cerebellar stabilizer); the action is the sum. The stock rsl_rl exporter (`play.py`) only saves
`policy.actor` (== the skill-drive `mlp`) and **silently drops the corrector** → a walk-only, much
less stable ONNX (4-Gemm, no Add). A **correct composite ONNX has 7 Gemm + 1 Add** (4 skill + 3
corrector + residual sum) plus a Slice/Concat that rebuilds the corrector's `balance` obs (the
69-vector minus `velocity_commands[9:12]`). `models/wrench`'s inert wrench head is dropped at export
(no privileged input). `phase2` is a plain MLP — its stock export is complete.

## Exporting the composite to ONNX (Isaac Sim container)

Use the composite-aware exporter, NOT `play.py`. It traces the full `forward()` (skill + corrector),
writes `models/{v9,wrench}/policy.onnx`, and asserts the ONNX matches the real torch `forward()` to
~1e-5. The host repo is bind-mounted to `/workspace/spot-locomanipulation` (host edits seen in-container).

```bash
docker exec -e TERM=xterm -w /workspace/spot-locomanipulation spot-teleop-isaac-sim-1 \
  bash -c 'export TERM=xterm; ./IsaacLab/isaaclab.sh -p scripts/rsl_rl/export_composite_onnx.py --headless'
# then copy spot-locomanipulation/models/{v9,wrench}/policy.onnx into models/{v9_serial,wrench}/
```

- Verify after export: `onnx.load(...).graph.input` should be a single `obs [1,69]`, output
  `actions [1,19]`, with **7 Gemm + 1 Add** (skill + corrector). 4 Gemm / no Add = walk-only, wrong.
- Sim-side (`.pt`) deploy of the full composite with live lesioning is
  `spot-locomanipulation/scripts/rsl_rl/deploy_spot_walkreflex_standalone.py --serial [--wrench]`.

### `TERM=xterm` is mandatory (the gotcha)

Running `isaaclab.sh` non-interactively via `docker exec` **requires `export TERM=xterm`** first.
`isaaclab.sh` runs `set -e` then `tabs 4` (line ~16); with an unset/`dumb` TERM the `tabs` command
returns nonzero and the whole script aborts after printing only
`'ansi+tabs': unknown terminal type`. Symptom: the process is gone and the logfile has just that one
line. Always set `TERM=xterm` in the `bash -c` for both the outer exec env (`-e TERM=xterm`) and
inside the command.
