# regional_trunk_soft — model card

Added 2026-07-31.

## Provenance

| | |
|---|---|
| checkpoint | `spot-locomanipulation/logs/rsl_rl/spot_regional_trunk_soft/2026-07-31_12-38-11/model_3897.pt` |
| training task | `Isaac-Locomanipulation-Flat-Spot-RegionalTrunkSoftMC-v0` |
| env | `SpotLocomanipulationEnvCfg_WalkSerialContinuousStandRew` |
| runner cfg | `SpotRegionalTrunkSoftMCPPORunnerCfg` |
| exported via | `scripts/rsl_rl/play.py --task ...-RegionalTrunkSoftMC-Play-v0 --num_envs 1 --headless` |
| `policy.onnx` sha256[:16] | `f18264ef19473579` |

## Architecture — one trunk, two regions

A **single MLP** (`RegionalMLPModel`), not a composite. Its first hidden layer is partitioned into two
named regions by *what each may read*:

| region | units | blocked obs |
|---|---|---|
| `drive` | 0–256 | none — reads all 69 |
| `balance` | 256–512 | **9, 10, 11** (the velocity command) |

This reproduces v9's drive/balance seam inside one set of weights instead of two networks. "Soft" =
phase 2 with routed two-critic training at `route_alpha=0.5`.

**It can stand.** Trained on the full stand + walk + transition continuum (`cmd=0` included), unlike
the phase-1 `spot_walk_only_trunk`, which excludes `cmd=0` entirely and **must never be deployed**.

## Export verification — done, not assumed

`RegionalMLPModel` applies its mask *inside* forward (`F.linear(x, weight * in_mask, bias) *
unit_gate`), so a trace has to fold it into constants. This repo has already shipped a silently
crippled export once (the stock exporter dropped v9's entire balance corrector), so the fold was
checked rather than assumed:

```
$ python3 scripts/rsl_rl/verify_regional_onnx.py .../policy.onnx --regions '[...]'
input  obs: [1, 69]        output actions: [1, 19]
first-layer weight: (512, 69)
  balance units[256:512] x obs[9,10,11]  max|w| = 0.000e+00   OK
  balance unblocked columns              max|w| = 2.420e+00   (not vacuous)
RESULT: PASS - mask is baked into the exported weights
```

Graph: `{Gemm: 4, Elu: 3, Constant: 1, Mul: 1}`. The `Mul` is the constant `unit_gate`, correctly
baked in. Note the "**7 Gemm + 1 Add**" rule in the repo `CLAUDE.md` applies to *composite*
`SerialCorrectionActor` exports only — for a single-trunk model 4 Gemm is complete, not truncated.

Region **dropout** is not a concern here: it exists only in the phase-3 `RegionalTrunkDrop` variant.
(If that variant is ever shipped, dropout must be forced off first — the v4 subsumption stack shipped
with `p=0.2` still live once.)

## ★ What is NOT known

**This policy has no quantitative evaluation yet.** It was selected by visual judgement in the
standalone Isaac deploy (`scripts/rsl_rl/deploy_regional_standalone.sh`), which in this project is a
legitimate and repeatedly-vindicated signal — deterministic probes have passed and visuals failed
more than once. But it means there is no fall rate, no `error_vel`, no foot-crossing number for it.

For context, the policies that *have* been measured (seed 0, 128 envs × 1500 steps, on the
walk-only distribution — **not** this policy's own distribution, so indicative only):

| policy | error_vel_xy | falls | crossing |
|---|---|---|---|
| walk-only trunk | 0.2449 | 0.78% | 1.01% |
| v9 serial | 0.2960 | 2.99% | 3.55% |
| wrench-good | 0.3154 | **0.00%** | — |

**Before a first hardware run, prefer the policy with the lowest fall rate over the one with the best
tracking.** A 20% worse velocity error costs nothing; one fall costs the sensor rig. On present
evidence that is `wrench` (0.00% falls over 128 episodes), and this model has no fall number at all.

The deploy-distribution ranking (`scripts/rsl_rl/rank_policies_deploy.sh`, which includes this
checkpoint) was queued but had not completed when this was packaged. **Fill this section in from it
before trusting the model on hardware.**

## Running it

```bash
uv run rl_deploy/spot_rl_demo.py 10.0.0.3 --mock \
  -policy_file_path models/regional_trunk_soft \
  --run_label mock_check --payload "lidar+jetson"
```

Drop `--mock` for the real robot. `--run_label` / `--payload` / `--run_notes` are recorded into the
HDF5 attrs along with this ONNX's sha256, so a log can always be traced back to these exact weights.
