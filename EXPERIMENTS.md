# Deploy-day experiment protocol

Written 2026-07-31 for the **first hardware session**. Companion to
`spot-locomanipulation/SIM2REAL_GAP.md` — the blocks below are numbered against its §6 items.

**Governing principle: robot time is the scarce resource, and unlabelled data is wasted data.**
Everything here is ordered by (value × irreplaceability), safety-first. Blocks A–B need no walking.

---

## 0. Pre-flight (do before the robot is powered)

- [ ] `git status` clean, on `add-loco-policies`, pushed to `fork`.
- [ ] **Decide and write down which policy you are running.** `models/` ships three:
      `phase2` (plain MLP, complete stock export), `v9_serial` and `wrench` (composite actors —
      7 Gemm + 1 Add; 4 Gemm / no Add means the corrector was silently dropped, see `CLAUDE.md`).
      Verify the Gemm/Add count before trusting a composite policy on hardware.
- [ ] Dry-run everything in `--mock` first: `uv run rl_deploy/spot_rl_demo.py 10.0.0.3 --mock -policy_file_path models/wrench`
- [ ] Confirm the new metadata prints: the run should log a `[run metadata]` line at startup.
      If `policy_sha256_16` says `unavailable`, the path is wrong — fix before collecting.

### ★ Every run must carry its label
HDF5 logging is **on by default** (a timestamped file per run), but until now it wrote **zero
attributes** — a day of runs was indistinguishable afterwards. Three new flags fix that:

```bash
uv run rl_deploy/spot_rl_demo.py 10.0.0.3 \
  -policy_file_path models/wrench \
  --run_label  stand_still_60s \
  --payload    "lidar+jetson" \
  --run_notes  "concrete floor, no ZED yet"
```

`--payload` is not optional bookkeeping — it **is** the S5/S5b experimental variable. Log it on
every single run, including the ones where it is `none`.

---

## 1. Physical measurements — robot powered OFF (§6.2, §6b)

No control code, ~20 minutes, and it unblocks the sim work that is currently guessing.

- [ ] **Mass of each bolted-on item**, separately: lidar, mast/mount, Jetson + enclosure, ZED +
      bracket (if fitted), cabling/plate. Kitchen scale is fine.
- [ ] **Mount offset of each** from the base origin (x fwd, y left, z up), tape measure, ±1 cm.
- [ ] **Total robot mass** with the rig fitted — compare against the URDF total
      (`uv run print_urdf_masses.py`).
- [ ] Photograph the rig from the side and back, with a ruler in frame.

Feeds: the URDF rig bodies, and the `_RIG_MASS_RANGE` / `_RIG_COM_RANGE` constants that are
currently **provisional estimates** in `spot_locomanipulation_env_cfg.py`.

---

## 2. Block A — standing still, policy running (§6.3 state estimator)

**60 s, robot standing, zero command, do not touch the gamepad.**

```bash
--run_label stand_still_60s --payload "<rig>"
```

Answers: the state-estimator **bias and noise floor**. `raw_base_linear_velocity` should be ~0;
whatever it actually is, is the bias our sim models as zero-mean ±0.1 uniform noise. This is the
single cheapest check on the observation-distribution question, and it needs no floor space.

Run it **twice**: once with the rig fitted, once bare if the rig is removable.

---

## 3. Block B — straight-line walk over a measured distance (§6.3, §6.5)

Tape-measure a **5 m** straight line on the floor. Walk it at a steady moderate command, stop.

```bash
--run_label walk_straight_5m_slow  --run_notes "5.0 m tape, concrete"
```

Answers: **state-estimator scale and drift** — integrate `raw_base_linear_velocity` over the run
and compare to 5.0 m. Also gives the first real `error_vel_xy` analogue.

Repeat at a faster command (`walk_straight_5m_fast`) if the space allows.

---

## 4. Block C — ★ the payload A/B (S5b / S5) — highest value of the day

**Run Blocks A and B again with the payload configuration changed** (rig fitted vs removed, or
lidar-only vs lidar+Jetson — whatever swap is practical).

```bash
--run_label stand_still_60s      --payload none
--run_label walk_straight_5m_slow --payload none
```

Why this is the most valuable hour: the payload is the **one place our robot provably differs from
the configuration that was already validated on hardware** (upstream's Spot policy shipped on a
*bare* robot — `SIM2REAL_GAP.md` §3b). Every other sim2real concern was retired by that existence
proof. This A/B directly measures the thing we are currently guessing at, and it is free — same
manoeuvres, one hardware change between them.

If you only have time for one block today, do this one.

---

## 5. Block D — command sweep for sim↔real replay (§6.4)

A varied but *unremarkable* command sequence: forward, stop, backward, left, right, yaw both ways,
a few stop-starts. 60–90 s. Nothing aggressive.

```bash
--run_label cmd_sweep_90s
```

This is the input to the existing replay tooling, which already does the distributional comparison:

```bash
uv run rl_deploy/scripts/replay_and_compare_sim_real.py --hdf5_file <run>.hdf5
uv run rl_deploy/scripts/compare_actuator_loads.py     --hdf5_file <run>.hdf5
```

---

## 6. What you get for free from any run (no extra experiment)

These are recoverable post-hoc from data already logged, so **do not spend robot time on them**:

- **Command→motion latency bound (§6.1).** Cross-correlate `commanded_action` against
  `raw_joint_positions`. At ~48 Hz control the resolution is ~21 ms — coarse, but enough to
  distinguish our modelled 0–8 ms from a real 20–50 ms. `dt_*` fields already cover the internal
  compute path; this covers the actuator path they miss.
- **Effective actuator parameters (§6.0, partial).** `raw_joint_loads` + `raw_joint_positions` +
  `raw_joint_velocities` from an ordinary walk supports fitting an effective stiffness/damping/
  friction. This is a poor substitute for a proper chirp/step excitation sweep, but it is free and
  safe today, and it is the first real data our Kp=60/Kd=1.5/armature=0 numbers have ever seen.
- **Torque headroom.** Compare `raw_joint_loads` peaks against the 45 N·m hip limit — and note
  whether the robot ever sustains near-limit torque, which is what the missing continuous-vs-peak
  distinction would bite on.

---

## 7. Explicitly NOT today

- **A joint-level excitation sweep (chirp/step per joint, §6.0 proper).** This is the highest-value
  dataset for PACE/P1 identification, but it needs a new low-level joint-command script that has
  never been run on hardware. Writing and debugging that on a first deploy day is how robots get
  damaged. Schedule it as its own session, on a stand, once the basic loop is trusted.
- **Anything with the arm under load.** The arm plant is the one subsystem with no hardware
  validation at all (§3b: upstream shipped the armless `SPOT_CFG`), so it deserves a deliberate
  session, not an improvised one.
- **Chasing performance numbers.** Today is about *provenance and coverage*, not about whether
  `error_vel_xy` looks good.

---

## 8. After the session

- [ ] Copy every `.hdf5` somewhere backed up **before** analysing anything.
- [ ] `python3 -c "import h5py;print(dict(h5py.File('<f>.hdf5').attrs))"` on one file — confirm the
      attrs are populated. If they are empty, the run predates the metadata change and you should
      write the label into a sidecar `.txt` immediately, while you still remember.
- [ ] Record measured masses/offsets into `SIM2REAL_GAP.md` §6b and replace the provisional
      `_RIG_MASS_RANGE` / `_RIG_COM_RANGE` constants.
- [ ] Note anything the robot did that the sim does not — behaviour, sound, hesitation. This
      project's record is that visual/interactive observation catches what deterministic probes
      miss, repeatedly.
