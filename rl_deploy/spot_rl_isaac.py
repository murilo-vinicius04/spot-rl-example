import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Setting up a Spot Gripper environment.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

parser.add_argument(
    "-policy_file_path", type=str, default="rl_deploy/configs",
    help="Directory with policy.onnx + env.yaml. Default keeps the historical hardcoded path.")
parser.add_argument(
    "--phase2", action="store_true",
    help="Use Phase2OnnxCommandGenerator -- the SAME generator spot_rl_demo.py runs on the real "
         "robot. Required for v9/wrench/trunk (69->19); the legacy generator builds a different obs.")
parser.add_argument(
    "--max_steps", type=int, default=20000, help="Steps then exit (headless scripted runs).")
parser.add_argument(
    "--csv", type=str, default=None, help="Per-step CSV: cmd, base vel, projected gravity, q, tau.")
parser.add_argument(
    "--routine", action="store_true",
    help="Drive a scripted velocity routine instead of the keyboard, so a headless run produces "
         "command onsets comparable to a real log.")
parser.add_argument(
    "--hdf5_log",
    type=str,
    default="spot_isaac_sim.hdf5",
    help="Path to save HDF5 log of observations.",
)

# parse the arguments
args_cli = parser.parse_args()
# args_cli.experience = "isaacsim.exp.full.kit"  # Set the experience here

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os
import sys

# This allows for absolute imports from 'spot_mgrasping'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from isaaclab.envs import ManagerBasedEnv
from utils.hdf5_logger import HDF5Logger

from rl_deploy.hid.terminal_keyboard import TerminalKeyboard
from rl_deploy.orbit import orbit_configuration
if args_cli.phase2:
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


def main():
    """Main function."""
    export_model_dir = args_cli.policy_file_path
    env_config = orbit_configuration.detect_config_file(export_model_dir)
    policy_file = orbit_configuration.detect_policy_file(export_model_dir)
    config = orbit_configuration.load_configuration(env_config)

    env_cfg = SpotFlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    # wrap for video recording
    env = ManagerBasedEnv(env_cfg)

    obs, _ = env.reset()
    logger = HDF5Logger(args_cli.hdf5_log)
    context = OnnxControllerContext()
    state_handler = StateHandler(context)
    command_generator = _CommandGenerator(
        context, config, policy_file, False, logger=logger
    )
    # TerminalKeyboard needs a tty (termios); headless/scripted runs have none, and --routine
    # drives the command anyway.
    gamepad = None
    if not args_cli.routine:
        try:
            gamepad = TerminalKeyboard(context, x_vel=0.0, y_vel=0.0, yaw=0.0)
        except Exception as e:
            print(f"[WARN] TerminalKeyboard unavailable ({e}); commands stay zero.")

    spot = IsaacMockSpot()

    # Start streams
    spot.start_state_stream(state_handler)

    obs_dict, _ = env.reset()
    spot.set_state(obs_dict["spot"])
    spot.start_command_stream(command_generator)
    # gamepad.start_listening()

    # Matches the real 2026-07-31 run: 0.5 m/s steps with stops between, not 1.5.
    ROUTINE = [(0.0, 0.0, 0.0, 4.0), (0.5, 0.0, 0.0, 5.0), (0.0, 0.0, 0.0, 4.0),
               (-0.5, 0.0, 0.0, 5.0), (0.0, 0.0, 0.0, 4.0), (0.0, 0.5, 0.0, 5.0),
               (0.0, 0.0, 0.0, 4.0)]
    HZ = 50.0
    csv_f = None
    if args_cli.csv:
        csv_f = open(args_cli.csv, "w", buffering=1)
        csv_f.write("t,cmd_x,cmd_y,cmd_wz,vw_x,vw_y,vw_z,g_x,g_y,g_z,"
                    + ",".join(f"q{j}" for j in range(19)) + ","
                    + ",".join(f"tau{j}" for j in range(19)) + ","
                    + ",".join(f"act{j}" for j in range(19)) + "\n")

    for i in range(args_cli.max_steps):
        if args_cli.routine:
            tt = i / HZ
            acc = 0.0
            for vx, vy, wz, dur in ROUTINE:
                if tt < acc + dur:
                    context.velocity_cmd = [vx, vy, wz]
                    break
                acc += dur
            else:
                context.velocity_cmd = [0.0, 0.0, 0.0]
        # run everything in inference mode
        with torch.inference_mode():
            actions = spot.command_update().to(env_cfg.sim.device)
            obs_dict, _ = env.step(actions)
            spot.set_state(obs_dict["spot"])
            if csv_f is not None:
                so, dbg = obs_dict["spot"], obs_dict["debug"]
                row = ([i / HZ] + list(context.velocity_cmd)
                       + so["root_lin_vel_w"][0].cpu().tolist()
                       + dbg["projected_gravity"][0].cpu().tolist()
                       + so["joint_pos"][0].cpu().tolist()
                       + so["joint_effort"][0].cpu().tolist()
                       + actions[0].cpu().tolist())
                csv_f.write(",".join(f"{v:.5f}" for v in row) + "\n")
            # The logger object might not have logger.log so let's log safe
            if logger and hasattr(logger, "log"):
                logger.log(obs_dict)
        if gamepad is not None:
            gamepad.listen_loop()

    if csv_f is not None:
        csv_f.close()
        print(f"[LOG] wrote {args_cli.csv}")

    # close the simulator
    env.close()
    logger.save()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
