#!/usr/bin/env python3
"""Quantify the sim-vs-real gap from a replay dump, per joint, distributionally.

WHY DISTRIBUTIONAL AND NOT POINTWISE. The replay feeds the same COMMAND sequence to sim and to the
real robot, but the dynamics run closed-loop, so the two trajectories diverge within a second or two
of the first command. Comparing sample i to sample i after that is comparing two different
experiments. What survives divergence is the DISTRIBUTION of each signal under a matched command
regime -- which is also exactly how P1 (arXiv:2504.17857, the Spot paper) frames sim-parameter
identification: match distributional measures, not trajectories.

Joint ordering: the real log uses ORDERED_JOINT_NAMES_SPOT (19: 12 legs then 7 arm); the sim env
uses ORDERED_JOINT_NAMES_SPOT_BASE (12 legs). The first 12 entries are in the same order, so leg
index i maps to leg index i. The arm exists ONLY on the real side -- that asymmetry is itself a
model difference and is reported, not hidden.

Usage:
    uv run rl_deploy/scripts/compare_sim_real_distributions.py [--npz logs/replay_arrays.npz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEG_NAMES = ["fl_hx", "fl_hy", "fl_kn", "fr_hx", "fr_hy", "fr_kn",
             "hl_hx", "hl_hy", "hl_kn", "hr_hx", "hr_hy", "hr_kn",
             "arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x"]


def wasserstein(a, b):
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        # quantile-based fallback: mean |Q_a - Q_b| over a fixed grid
        q = np.linspace(0.01, 0.99, 99)
        return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def summarize(name, sim, real, unit):
    w = wasserstein(sim, real)
    return dict(name=name, unit=unit,
                sim_mean=sim.mean(), real_mean=real.mean(),
                sim_std=sim.std(), real_std=real.std(),
                d_mean=sim.mean() - real.mean(),
                std_ratio=(sim.std() / real.std()) if real.std() > 1e-9 else np.nan,
                wass=w)


def print_table(rows, title):
    print(f"\n=== {title}")
    print(f"{'signal':>10} {'sim mean':>10} {'real mean':>10} {'Δmean':>9} "
          f"{'sim std':>9} {'real std':>9} {'std ratio':>10} {'W1':>9}")
    for r in rows:
        print(f"{r['name']:>10} {r['sim_mean']:10.3f} {r['real_mean']:10.3f} {r['d_mean']:+9.3f} "
              f"{r['sim_std']:9.3f} {r['real_std']:9.3f} {r['std_ratio']:10.2f} {r['wass']:9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=Path("logs/replay_arrays.npz"))
    ap.add_argument("--out", type=Path, default=Path("logs"))
    ap.add_argument("--moving_only", action="store_true",
                    help="Restrict to steps where the linear command exceeds 0.05 m/s. Standing and "
                         "walking are different regimes and pooling them hides both.")
    a = ap.parse_args()

    d = np.load(a.npz)
    sim_pos, real_pos = d["sim_positions"], d["real_positions"]
    sim_load, real_load = d["sim_loads"], d["real_loads"]
    # real_* come from the HDF5 as (N, 1, 3); sim_* are (N, 3). Squeeze so both index alike.
    sq = lambda x: np.asarray(x).reshape(len(x), -1)
    sim_lv, real_lv = sq(d["sim_lin_vel"]), sq(d["real_lin_vel"])
    sim_av, real_av = sq(d["sim_ang_vel"]), sq(d["real_ang_vel"])
    cmds = d["vel_cmds"]

    n = min(len(sim_pos), len(real_pos))
    mask = np.ones(n, bool)
    if a.moving_only:
        mask = np.linalg.norm(cmds[:n, :2], axis=1) > 0.05
    print(f"samples: {n}  (using {mask.sum()}{' moving-only' if a.moving_only else ''})")
    print(f"sim joints: {sim_pos.shape[1]}   real joints: {real_pos.shape[1]}")
    if real_pos.shape[1] > sim_pos.shape[1]:
        print(f"NOTE: the real robot has {real_pos.shape[1] - sim_pos.shape[1]} joints the sim model "
              f"does not (the arm). Those are excluded below and are themselves a model difference.")

    nleg = min(len(LEG_NAMES), sim_pos.shape[1], real_pos.shape[1])

    pos_rows = [summarize(LEG_NAMES[j], sim_pos[:n][mask, j], real_pos[:n][mask, j], "rad")
                for j in range(nleg)]
    load_rows = [summarize(LEG_NAMES[j], sim_load[:n][mask, j], real_load[:n][mask, j], "Nm")
                 for j in range(nleg)]
    base_rows = ([summarize(f"lin_{c}", sim_lv[:n][mask, i], real_lv[:n][mask, i], "m/s")
                  for i, c in enumerate("xyz")]
                 + [summarize(f"ang_{c}", sim_av[:n][mask, i], real_av[:n][mask, i], "rad/s")
                    for i, c in enumerate("xyz")])

    print_table(base_rows, "BASE velocity  (real side is the STATE ESTIMATOR, sim side is ground truth)")
    print_table(pos_rows, "LEG joint positions [rad]")
    print_table(load_rows, "LEG joint loads [Nm]")

    worst = sorted(pos_rows, key=lambda r: -r["wass"])[:3]
    print("\nlargest positional gaps:", ", ".join(f"{r['name']} (W1={r['wass']:.3f})" for r in worst))

    a.out.mkdir(parents=True, exist_ok=True)
    for tag, sim_a, real_a, rows, unit in [
        ("joint_positions", sim_pos, real_pos, pos_rows, "rad"),
        ("joint_loads", sim_load, real_load, load_rows, "Nm"),
    ]:
        ncol = 4; nrow = int(np.ceil(nleg / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 2.8*nrow))
        for j, ax in enumerate(axes.flat):
            if j >= nleg:
                ax.axis("off"); continue
            s, r = sim_a[:n][mask, j], real_a[:n][mask, j]
            lo = min(s.min(), r.min()); hi = max(s.max(), r.max())
            bins = np.linspace(lo, hi, 60)
            ax.hist(r, bins=bins, alpha=0.55, label="real", color="tab:blue", density=True)
            ax.hist(s, bins=bins, alpha=0.55, label="sim", color="tab:red", density=True)
            ax.set_title(f"{LEG_NAMES[j]}  W1={rows[j]['wass']:.3f}", fontsize=9)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.legend(fontsize=8)
        fig.suptitle(f"sim vs real — {tag.replace('_',' ')} [{unit}]"
                     f"{' (moving only)' if a.moving_only else ''}")
        fig.tight_layout()
        p = a.out / f"dist_{tag}.png"
        fig.savefig(p, dpi=110); plt.close(fig)
        print(f"saved {p}")

    # time series of the two things the operator actually reported: forward speed and pitch
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(real_lv[:n, 0], label="real vx (estimator)", lw=0.8)
    axes[0].plot(sim_lv[:n, 0], label="sim vx (truth)", lw=0.8)
    axes[0].plot(cmds[:n, 0], "k--", label="cmd vx", lw=0.8)
    axes[0].set_ylabel("m/s"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    axes[1].plot(real_av[:n, 1], label="real pitch rate", lw=0.8)
    axes[1].plot(sim_av[:n, 1], label="sim pitch rate", lw=0.8)
    axes[1].set_ylabel("rad/s"); axes[1].set_xlabel("step"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.suptitle("forward speed and pitch rate — trajectories DIVERGE, read distributions not overlap")
    fig.tight_layout(); fig.savefig(a.out / "timeseries_vx_pitch.png", dpi=110); plt.close(fig)
    print(f"saved {a.out/'timeseries_vx_pitch.png'}")


if __name__ == "__main__":
    main()
