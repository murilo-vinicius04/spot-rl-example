# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import sys
from pathlib import Path

import bosdyn.client.util
import orbit.orbit_configuration
from rl_deploy.hid.terminal_keyboard import TerminalKeyboard
from rl_deploy.orbit.phase2_onnx_command_generator import (
    Phase2OnnxCommandGenerator,
    OnnxControllerContext,
    StateHandler,
)
from rl_deploy.spot.mock_spot import MockSpot
from rl_deploy.spot.spot import Spot
from rl_deploy.utils.event_divider import EventDivider
from rl_deploy.utils.hdf5_logger import HDF5Logger

from datetime import datetime


def main():
    """Command line interface. change that is ok"""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        "-policy_file_path",
        type=Path,
        default=Path(__file__).parent / "configs",
        help="Path to the policy file or directory containing the policy file.",
    )
    parser.add_argument("-m", "--mock", action="store_true")
    parser.add_argument(
        "--hdf5_log",
        type=str,
        default=f"spot_isaac_real_{datetime.now().strftime('%Y%m%d_%H%M%S')}.hdf5",
        help="Path to save HDF5 log of observations.",
    )
    options = parser.parse_args()

    env_config = orbit.orbit_configuration.detect_config_file(options.policy_file_path)
    policy_file = orbit.orbit_configuration.detect_policy_file(options.policy_file_path)

    config = orbit.orbit_configuration.load_configuration(env_config)
    print("Loaded configs: ", config)

    context = OnnxControllerContext()
    state_handler = StateHandler(context)
    print("Verbose option: ", options.verbose)

    logger = HDF5Logger(options.hdf5_log)
    command_generator = Phase2OnnxCommandGenerator(
        context, config, policy_file, options.verbose, logger=logger
    )
    gamepad = TerminalKeyboard(context)
    # 333 Hz state update / 7 => ~48 Hz control updates (closer to Phase 2's 50 Hz target)
    timeing_policy = EventDivider(context, 7)

    if options.mock:
        spot = MockSpot()
    else:
        print("Connecting Spot")
        spot = Spot(options)
        print("OK")

    with spot.lease_keep_alive():
        try:
            print("Powering on and standing")
            spot.power_on()
            spot.stand(0.0)
            print("start state stream")
            spot.start_state_stream(state_handler)

            input(
                "\n>>> Robot is STANDING under Boston Dynamics control (no policy commands yet).\n"
                ">>> Confirm the gantry is taking weight and the E-stop/hijack is in hand.\n"
                ">>> Press ENTER to hand control to the POLICY, or Ctrl-C to abort: "
            )
            print("start command stream")
            spot.start_command_stream(command_generator, timeing_policy)
            gamepad.listen()

        except KeyboardInterrupt:
            print("killed with ctrl-c")
        finally:
            print("stop command stream")
            spot.stop_command_stream()
            print("stop state stream")
            spot.stop_state_stream()
            print("stop game pad")
            logger.save()


if __name__ == "__main__":
    if not main():
        sys.exit(1)
