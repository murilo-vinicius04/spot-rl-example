# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
"""Run a Phase-2 / v9 composite ONNX policy in Isaac Sim through the SAME deploy
code used on the real robot (``Phase2OnnxCommandGenerator``), driven by
``IsaacMockSpot``. Use this to confirm the policy stands / walks in sim BEFORE
touching hardware — if it falls here, the obs/action wiring is still off.

Unlike ``spot_rl_isaac.py`` (which wires the phase-1 ``OnnxCommandGenerator`` +
``rl_deploy/configs``), this points the Phase-2 generator at a composite model dir
(``models/v9_serial`` by default; also ``models/wrench`` / ``models/phase2``).

Run inside the Isaac Sim container (TERM=xterm):
    ./IsaacLab/isaaclab.sh -p rl_deploy/spot_rl_isaac_v9.py --headless
    # viewer instead of headless: drop --headless ; WebRTC: add --livestream 2
    # other policy:               --model_dir models/wrench
Drive velocity from the terminal (TerminalKeyboard); a zero command makes v9 try
to stand (it has a known weak stand-still), so nudge it forward to see the gait.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="v9/Phase-2 composite policy in Isaac Sim via the real-robot deploy path."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--model_dir",
    type=str,
    default="models/v9_serial",
    help="Dir holding env.yaml + policy.onnx (models/v9_serial | models/wrench | models/phase2).",
)
parser.add_argument("--hdf5_log", type=str, default="spot_isaac_v9.hdf5", help="HDF5 log path.")
parser.add_argument("--steps", type=int, default=20000, help="Number of sim steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app first (IsaacLab imports below need it up)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""
import os
import sys

# allow `rl_deploy.*` absolute imports when run as a script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from isaaclab.envs import ManagerBasedEnv

from rl_deploy.utils.hdf5_logger import HDF5Logger
from rl_deploy.hid.terminal_keyboard import TerminalKeyboard
from rl_deploy.orbit import orbit_configuration
from rl_deploy.orbit.phase2_onnx_command_generator import (
    Phase2OnnxCommandGenerator,
    OnnxControllerContext,
    StateHandler,
)
from rl_deploy.isaaclab_spot.isaac_spot import IsaacMockSpot
from rl_deploy.isaaclab_spot.spot_env import SpotFlatEnvCfg


def main():
    """Drive the Phase-2 generator against Isaac Sim via IsaacMockSpot."""
    env_config = orbit_configuration.detect_config_file(args_cli.model_dir)
    policy_file = orbit_configuration.detect_policy_file(args_cli.model_dir)
    config = orbit_configuration.load_configuration(env_config)
    print(f"[v9-isaac] model_dir={args_cli.model_dir}")
    print(f"[v9-isaac]   env_cfg={env_config}")
    print(f"[v9-isaac]   policy ={policy_file}")

    env_cfg = SpotFlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedEnv(env_cfg)

    logger = HDF5Logger(args_cli.hdf5_log)
    context = OnnxControllerContext()
    state_handler = StateHandler(context)
    # Same generator class the real robot uses — this is the whole point.
    command_generator = Phase2OnnxCommandGenerator(
        context, config, policy_file, verbose=False, logger=logger
    )
    gamepad = TerminalKeyboard(context, x_vel=0.0, y_vel=0.0, yaw=0.0)

    spot = IsaacMockSpot()
    spot.start_state_stream(state_handler)

    # Prime one state so the generator has latest_state before the first command.
    obs_dict, _ = env.reset()
    spot.set_state(obs_dict["spot"])
    spot.start_command_stream(command_generator)

    for _ in range(args_cli.steps):
        with torch.inference_mode():
            actions = spot.command_update().to(env_cfg.sim.device)  # (1, 19) abs targets, SPOT order
            obs_dict, _ = env.step(actions)
            spot.set_state(obs_dict["spot"])
        gamepad.listen_loop()

    env.close()
    logger.save()


if __name__ == "__main__":
    main()
    simulation_app.close()
