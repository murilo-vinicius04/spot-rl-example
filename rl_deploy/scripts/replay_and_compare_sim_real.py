import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(
        description="Replay velocity commands from HDF5 and compare with real Spot."
    )
    parser.add_argument(
        "--hdf5_file",
        type=Path,
        default=Path("spot_isaac_real.hdf5"),
        help="Path to HDF5 log.",
    )
    parser.add_argument(
        "--limit", type=int, default=-1, help="Max steps to replay. -1 for all."
    )
    parser.add_argument(
        "-policy_file_path",
        type=str,
        default="rl_deploy/configs",
        help="Directory holding policy.onnx + env.yaml to replay. Defaults to the historical "
        "hardcoded rl_deploy/configs so existing behaviour is unchanged.",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Use Phase2OnnxCommandGenerator (single 69-dim concatenated observation) instead of "
        "the legacy OnnxCommandGenerator (7 named tensors). REQUIRED for v9/wrench/trunk-family "
        "policies -- they are 69->19 and running them through the legacy path silently feeds the "
        "network a differently-shaped observation rather than raising.",
    )
    AppLauncher.add_app_launcher_args(parser)

    args_cli = parser.parse_args()
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedEnv

    from rl_deploy.orbit import orbit_configuration

    if args_cli.phase2:
        # 69-dim concatenated observation path -- the one v9/wrench/trunk policies actually run on
        # the robot via spot_rl_demo.py. Same constructor shape, and this module ships its own
        # matching OnnxControllerContext/StateHandler.
        from rl_deploy.orbit.phase2_onnx_command_generator import (
            OnnxControllerContext,
            Phase2OnnxCommandGenerator as _CommandGenerator,
            StateHandler,
        )
    else:
        from rl_deploy.orbit.onnx_command_generator import (
            OnnxCommandGenerator as _CommandGenerator,
            OnnxControllerContext,
            StateHandler,
        )
    from rl_deploy.isaaclab_spot.isaac_spot import IsaacMockSpot
    from rl_deploy.isaaclab_spot.spot_env import SpotFlatEnvCfg
    from rl_deploy.spot.constants import ORDERED_JOINT_NAMES_SPOT

    hdf5_path = args_cli.hdf5_file
    if not hdf5_path.exists():
        print(f"File {hdf5_path} does not exist.")
        simulation_app.close()
        return

    print(f"Reading {hdf5_path}...")
    with h5py.File(hdf5_path, "r") as f:
        vel_cmds = f["preprocessed_velocity_cmd"][:]
        real_positions = f["spot_current_positions"][:]
        real_loads = f["raw_joint_loads"][:]
        real_commanded = f["commanded_action"][:]
        real_lin_vel = f["preprocessed_base_linear_velocity"][:]
        real_ang_vel = f["preprocessed_base_angular_velocity"][:]

    num_steps = vel_cmds.shape[0]
    if args_cli.limit > 0:
        num_steps = min(num_steps, args_cli.limit)

    export_model_dir = args_cli.policy_file_path
    env_config = orbit_configuration.detect_config_file(export_model_dir)
    policy_file = orbit_configuration.detect_policy_file(export_model_dir)
    config = orbit_configuration.load_configuration(env_config)

    env_cfg = SpotFlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedEnv(env_cfg)

    context = OnnxControllerContext()
    state_handler = StateHandler(context)
    command_generator = _CommandGenerator(
        context, config, policy_file, False, logger=None
    )

    spot = IsaacMockSpot()
    spot.start_state_stream(state_handler)

    obs_dict, _ = env.reset()
    obs = obs_dict["debug"]
    spot.set_state(obs)
    spot.start_command_stream(command_generator)

    # Storage for comparison
    sim_positions = []
    sim_lin_vel = []
    sim_ang_vel = []
    sim_loads = []
    sim_commanded = []

    print(f"Starting replay for {num_steps} steps...")
    for i in range(num_steps):
        # Override velocity command directly to the context
        context.velocity_cmd = vel_cmds[i].tolist()

        with torch.inference_mode():
            actions = spot.command_update().to(env_cfg.sim.device)
            obs_dict, _ = env.step(actions)
            obs = obs_dict["spot"]
            spot.set_state(obs)

            # Record simulated states corresponding to the observation
            sim_pos = obs["joint_pos_rel"][0].cpu().numpy()
            sim_lv = obs["base_lin_vel"][0].cpu().numpy()
            sim_av = obs["base_ang_vel"][0].cpu().numpy()
            sim_eff = obs["joint_effort"][0].cpu().numpy()

            sim_positions.append(sim_pos)
            sim_lin_vel.append(sim_lv)
            sim_ang_vel.append(sim_av)
            sim_loads.append(sim_eff)
            sim_commanded.append(actions[0].cpu().numpy())

    env.close()

    sim_positions = np.array(sim_positions)
    sim_lin_vel = np.array(sim_lin_vel)
    sim_ang_vel = np.array(sim_ang_vel)
    sim_loads = np.array(sim_loads)
    sim_commanded = np.array(sim_commanded)

    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Plot Base Linear Velocity comparisons
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = ["X", "Y", "Z"]
    for j in range(3):
        axes[j].plot(real_lin_vel[:num_steps, j], label="Real", color="blue", alpha=0.7)
        axes[j].plot(
            sim_lin_vel[:, j], label="Sim", color="red", linestyle="--", alpha=0.7
        )
        axes[j].set_title(f"Base Linear Velocity {labels[j]}")
        axes[j].set_ylabel("Velocity (m/s)")
        axes[j].legend()
        axes[j].grid(True)
    axes[2].set_xlabel("Timesteps")
    plt.tight_layout()
    plt.savefig(out_dir / "compare_base_lin_vel.png")
    plt.close(fig)
    print("Saved compare_base_lin_vel.png")

    # Plot Base Angular Velocity comparisons
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for j in range(3):
        axes[j].plot(real_ang_vel[:num_steps, j], label="Real", color="blue", alpha=0.7)
        axes[j].plot(
            sim_ang_vel[:, j], label="Sim", color="red", linestyle="--", alpha=0.7
        )
        axes[j].set_title(f"Base Angular Velocity {labels[j]}")
        axes[j].set_ylabel("Velocity (rad/s)")
        axes[j].legend()
        axes[j].grid(True)
    axes[2].set_xlabel("Timesteps")
    plt.tight_layout()
    plt.savefig(out_dir / "compare_base_ang_vel.png")
    plt.close(fig)
    print("Saved compare_base_ang_vel.png")

    from rl_deploy.spot.constants import ORDERED_JOINT_NAMES_SPOT_BASE

    knee_names = [name for name in ORDERED_JOINT_NAMES_SPOT if name.endswith("_kn")]

    # Plot Knee Positions
    fig, axes = plt.subplots(
        len(knee_names), 1, figsize=(10, 3 * len(knee_names)), sharex=True
    )
    for i, name in enumerate(knee_names):
        real_idx = ORDERED_JOINT_NAMES_SPOT.index(name)
        sim_idx = ORDERED_JOINT_NAMES_SPOT_BASE.index(name)

        axes[i].plot(
            real_positions[:num_steps, real_idx],
            label="Real Position",
            color="blue",
            alpha=0.7,
        )
        axes[i].plot(
            sim_positions[:, sim_idx],
            label="Sim Position",
            color="red",
            linestyle="--",
            alpha=0.7,
        )
        axes[i].plot(
            real_commanded[:num_steps, real_idx],
            label="Real Commanded",
            color="green",
            linestyle=":",
            alpha=0.7,
        )
        axes[i].plot(
            sim_commanded[:, sim_idx],
            label="Sim Commanded",
            color="orange",
            linestyle="-.",
            alpha=0.7,
        )

        axes[i].set_title(f"Joint Position/Command {name}")
        axes[i].set_ylabel("Angle (rad)")
        axes[i].legend()
        axes[i].grid(True)
    axes[-1].set_xlabel("Timesteps")
    plt.tight_layout()
    plt.savefig(out_dir / "compare_joint_pos_knees.png")
    plt.close(fig)
    print("Saved compare_joint_pos_knees.png")

    # Plot Knee Loads
    fig, axes = plt.subplots(
        len(knee_names), 1, figsize=(10, 3 * len(knee_names)), sharex=True
    )
    for i, name in enumerate(knee_names):
        real_idx = ORDERED_JOINT_NAMES_SPOT.index(name)
        sim_idx = ORDERED_JOINT_NAMES_SPOT_BASE.index(name)

        axes[i].plot(
            real_loads[:num_steps, real_idx], label="Real Load", color="blue", alpha=0.7
        )
        # Note: sim_eff represents predicted effort applied
        axes[i].plot(
            sim_loads[:, sim_idx],
            label="Sim Load",
            color="red",
            linestyle="--",
            alpha=0.7,
        )

        axes[i].set_title(f"Joint Effort {name}")
        axes[i].set_ylabel("Effort (N*m)")
        axes[i].legend()
        axes[i].grid(True)
    axes[-1].set_xlabel("Timesteps")
    plt.tight_layout()
    plt.savefig(out_dir / "compare_joint_efforts_knees.png")
    plt.close(fig)
    print("Saved compare_joint_efforts_knees.png")

    simulation_app.close()


if __name__ == "__main__":
    main()
