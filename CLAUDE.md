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

`models/wrench` was exported with the **wrench head pruned** (`--lesion wrench`): a lesion study found
the privileged-wrench head inert (the balance corrector braces from proprioception alone), so the
deployed graph has **no privileged input** and is shape-identical to v9.

## Exporting a policy to ONNX (Isaac Sim container)

The exporter is `spot-locomanipulation/scripts/rsl_rl/play.py` — it loads a checkpoint, optionally
lesions, writes `<run>/exported/policy.{onnx,pt}`, then starts a sim loop (kill it once the ONNX is
written). The host repo is bind-mounted to `/workspace/spot-locomanipulation`, so host edits are seen
in-container with no rebuild.

```bash
docker exec -e TERM=xterm -w /workspace/spot-locomanipulation spot-teleop-isaac-sim-1 \
  bash -c 'export TERM=xterm; ./IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
    --task Isaac-Locomanipulation-Flat-Spot-WalkSerialWrench-Play-v0 --num_envs 1 --headless \
    --lesion wrench \
    --checkpoint logs/rsl_rl/spot_walk_serial_wrench/<run>/model_wrench_v1_GOOD.pt'
```

- **`--checkpoint` must be a path** (resolved from CWD via `retrieve_file_path`), **not a bare
  filename** — a bare name is looked up relative to CWD, not inside the run dir, and fails with
  `Unable to find the file`.
- **`--lesion wrench`** drops the inert wrench head → clean 69→19 (no second input). Omit it for
  non-wrench actors (e.g. v9), which already export to 69→19.
- Verify after export: `onnx.load(...).graph.input` should be a single `obs [1,69]`, output
  `actions [1,19]`.

### `TERM=xterm` is mandatory (the gotcha)

Running `isaaclab.sh` non-interactively via `docker exec` **requires `export TERM=xterm`** first.
`isaaclab.sh` runs `set -e` then `tabs 4` (line ~16); with an unset/`dumb` TERM the `tabs` command
returns nonzero and the whole script aborts after printing only
`'ansi+tabs': unknown terminal type`. Symptom: the process is gone and the logfile has just that one
line. Always set `TERM=xterm` in the `bash -c` for both the outer exec env (`-e TERM=xterm`) and
inside the command.
