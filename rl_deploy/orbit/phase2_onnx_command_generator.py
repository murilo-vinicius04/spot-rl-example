# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import os
import time
from dataclasses import dataclass
from threading import Event
from typing import List

import numpy as np
import onnxruntime as ort
from bosdyn.api import robot_command_pb2
from bosdyn.api.robot_command_pb2 import JointControlStreamRequest
from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse
from bosdyn.util import seconds_to_timestamp, set_timestamp_from_now, timestamp_to_sec

import rl_deploy.orbit.observations as ob
from rl_deploy.orbit.joint_mapping import (
    reorder_spot_to_policy_leg,
    reorder_policy_to_spot_leg,
)
from rl_deploy.orbit.orbit_configuration import OrbitConfig
from rl_deploy.spot.constants import (
    DEFAULT_K_Q_P,
    DEFAULT_K_QD_P,
    JOINT_LIMITS,
    JOINT_SOFT_LIMITS,
    ORDERED_JOINT_NAMES_SPOT,
)
from rl_deploy.utils.dict_tools import dict_to_list
from rl_deploy.utils.hdf5_logger import HDF5Logger


@dataclass
class OnnxControllerContext:
    """data class to hold runtime data needed by the controller"""

    event = Event()
    latest_state = None
    velocity_cmd = [0.0, 0.0, 0.0]
    count = 0

    def __post_init__(self):
        self.timing_dict = {}


class StateHandler:
    """Class to be used as callback for state stream to put state date
    into the controllers context
    """

    def __init__(self, context: OnnxControllerContext) -> None:
        self._context = context

    def __call__(self, state: RobotStateStreamResponse):
        """make class a callable and handle incoming state stream when called

        arguments
        state -- proto msg from spot containing most recent data on the robots state"""
        self._context.latest_state = state
        if hasattr(self._context, "timing_dict"):
            self._context.timing_dict["state_arrival"] = time.perf_counter()
        self._context.event.set()


class Phase2OnnxCommandGenerator:
    """Class to deploy Phase 2 ONNX policy to real Spot robot.

    Key differences from OnnxCommandGenerator:
    - Uses 69-dim concatenated observation tensor (vs 7 named tensors)
    - Maintains 19-dim last_action buffer (vs 12-dim)
    - Applies 0.2 scaling factor to policy outputs
    - Arm remains fixed in carry pose (same as original)
    """

    def __init__(
        self,
        context: OnnxControllerContext,
        config: OrbitConfig,
        policy_file_name: os.PathLike | str,
        verbose: bool,
        logger: HDF5Logger | None = None,
        mock: bool = False,
    ):
        self._context = context
        self._config = config
        self.logger = logger
        self.mock = mock
        self._inference_session = ort.InferenceSession(policy_file_name)
        self._last_action_policy = np.array([0.0] * 19)  # In Policy order for observations
        self._last_action_spot = np.array([0.0] * 19)   # In Spot order for robot commands
        self._count = 1
        self._init_pos = None
        self._init_load = None
        self.verbose = verbose

        self.joints_offsets_ordered_spot = dict_to_list(
            self._config.default_joints, ORDERED_JOINT_NAMES_SPOT
        )

        # Arm carry pose offsets (same as original OnnxCommandGenerator)
        self.arm_offsets_ordered = [0.0, -3.1415, 3.1415, 1.5655, 0.00, -1.5655, 0.0]

        self._triggered_safety = False
        self._safety_pos = None

        self._safe_limits = self._generate_safe_limits()

        # Joint indices for separating leg and arm data
        self._leg_joint_indices = range(12)  # First 12 joints are legs
        self._arm_joint_indices = range(12, 19)  # Last 7 joints are arm

    def _generate_safe_limits(self):
        """
        Generate safe limits for each joint based on the joint limits and soft limits.

        The soft limits were generated from simulated data, using the formula:

        max_val, min_val = max and min needed during simulation
        max, min = max and min of the joint limit range

        middle = (max + min)/2
        full_range = max - min

        min_margin = (middle - min_val)/full_range * 2
        max_margin = (max_val - middle)/full_range * 2
        """
        safe_limits = {}
        for joint_name in JOINT_SOFT_LIMITS:
            lower = JOINT_LIMITS[joint_name]["lower"]
            upper = JOINT_LIMITS[joint_name]["upper"]
            middle = (lower + upper) / 2
            full_range = upper - lower

            min_margin, max_margin = JOINT_SOFT_LIMITS[joint_name]
            min_val = middle - (min_margin * full_range / 2)
            max_val = middle + (max_margin * full_range / 2)

            safe_limits[joint_name] = (min_val, max_val)

        msg = "\nSafety Limits:\n"
        msg += "\n".join(
            [
                f"  {joint_name}: [{min_val:.3f}, {max_val:.3f}]\n"
                for joint_name, (min_val, max_val) in safe_limits.items()
            ]
        )
        print(msg)

        return safe_limits

    def __call__(self):
        """Makes class a callable and computes model output for latest controller context.

        Returns:
            proto message to be used in Spot's command stream
        """
        t_start_call = time.perf_counter()
        if hasattr(self._context, "timing_dict"):
            last_call = self._context.timing_dict.get("last_call_time", t_start_call)
            dt_total_step = t_start_call - last_call
            self._context.timing_dict["last_call_time"] = t_start_call
            dt_divider_to_onnx = t_start_call - self._context.timing_dict.get(
                "divider_end", t_start_call
            )
            dt_state_arrival_to_compute = t_start_call - self._context.timing_dict.get(
                "state_arrival", t_start_call
            )
        else:
            dt_total_step, dt_divider_to_onnx, dt_state_arrival_to_compute = (
                0.0,
                0.0,
                0.0,
            )

        # Cache initial joint position when command stream starts
        if self._init_pos is None:
            self._init_pos = self._context.latest_state.joint_states.position
            self._init_load = self._context.latest_state.joint_states.load

        if self._safety_pos is not None:
            return self.create_proto(self._safety_pos)

        # Extract observation data from latest Spot state data
        inputs_dict = self.collect_inputs(self._context.latest_state, self._config)

        current_positions_map = dict(
            zip(
                ORDERED_JOINT_NAMES_SPOT,
                self._context.latest_state.joint_states.position,
            )
        )

        # Safety Check
        self._triggered_safety = False  # self._check_safety(current_positions_map)

        if self._triggered_safety:
            print("Triggered safety")
            # Create hold command from current positions
            hold_pos = [
                current_positions_map[name] for name in ORDERED_JOINT_NAMES_SPOT
            ]
            self._safety_pos = hold_pos
            return self.create_proto(hold_pos)

        if self.mock:
            # Action of zeros results in default joint values after post-processing
            mocked_action = [0.0] * 12
            output = mocked_action
            t_onx_start = t_onx_end = time.perf_counter()
        else:
            t_onx_start = time.perf_counter()
            output = self._compute_action(inputs_dict)
            t_onx_end = time.perf_counter()

        t_post_start = time.perf_counter()
        action = output
        t_post_end = time.perf_counter()

        # Generate proto message from target joint positions
        proto = self.create_proto(action + self.arm_offsets_ordered)

        if self.logger is not None:
            dt_onnx = t_onx_end - t_onx_start
            dt_post = t_post_end - t_post_start
            dt_divider_wait = (
                self._context.timing_dict.get("dt_divider_wait", 0.0)
                if hasattr(self._context, "timing_dict")
                else 0.0
            )

            raw_state = self._context.latest_state
            self.logger.log_state(
                raw_base_linear_velocity=ob.get_base_linear_velocity(raw_state),
                raw_base_angular_velocity=ob.get_base_angular_velocity(raw_state),
                raw_projected_gravity=ob.get_projected_gravity(raw_state),
                raw_joint_positions=ob.get_joint_positions(
                    raw_state, self.joints_offsets_ordered_spot
                ),
                raw_joint_velocities=ob.get_joint_velocity(raw_state),
                raw_joint_loads=ob.get_join_load(raw_state),
                response_timestamp=ob.get_response_timestamp(raw_state),
                spot_current_positions=list(raw_state.joint_states.position),
                spot_current_velocities=list(raw_state.joint_states.velocity),
                preprocessed_base_linear_velocity=inputs_dict["base_linear_velocity"],
                preprocessed_base_angular_velocity=inputs_dict["base_angular_velocity"],
                preprocessed_projected_gravity=inputs_dict["projected_gravity"],
                preprocessed_velocity_cmd=inputs_dict["velocity_commands"],
                preprocessed_joint_positions=inputs_dict["joint_positions"],
                preprocessed_joint_velocities=inputs_dict["joint_velocities"],
                preprocessed_last_action=inputs_dict["last_actions"],
                commanded_action=action,
                dt_divider_wait=dt_divider_wait,
                dt_divider_to_onnx=dt_divider_to_onnx,
                dt_onnx_compute=dt_onnx,
                dt_post_process=dt_post,
                dt_total_step=dt_total_step,
                dt_state_arrival_to_compute=dt_state_arrival_to_compute,
                raw_state_proto_bytes=raw_state.SerializeToString(),
                proto_bytes=proto.SerializeToString(),
            )

        # Update counters
        self._count += 1
        self._context.count += 1

        if self.mock:
            mocked_action = [0.0] * 12
            return self.create_proto(mocked_action)

        return proto

    def _check_safety(self, current_positions_map):
        for joint_name, (safe_min, safe_max) in self._safe_limits.items():
            current_val = current_positions_map.get(joint_name)

            if current_val is None:
                print(f"[SAFETY STOP] Joint {joint_name} value is None")
                return True

            if current_val < safe_min or current_val > safe_max:
                print(
                    f"[SAFETY STOP] Joint {joint_name} value {current_val:.4f} outside safe range [{safe_min:.4f}, {safe_max:.4f}]"
                )
                return True
        return False

    def _compute_action(self, input_dict: dict[str, np.ndarray]):
        """Execute ONNX model and process output with 0.2 scaling factor.

        The Phase 2 policy outputs 19 dimensions (all joints), but we only use
        the first 12 for leg control. The last 7 (arm) are ignored to keep the
        arm fixed in carry pose.

        CRITICAL: Policy outputs leg actions in Policy order, but we need to
        reorder them back to Spot robot order before sending to the robot.
        """
        # Execute model from ONNX file (outputs 19 dimensions)
        onnx_input = {"obs": input_dict["observations"]}
        output = self._inference_session.run(None, onnx_input)[0][0]

        # Only use first 12 dimensions for legs, apply 0.2 scaling factor
        # Last 7 dimensions (arm) are ignored
        leg_action_policy = output[:12] * 0.2

        # CRITICAL: Reorder leg actions from Policy order back to Spot robot order
        # Policy:     [fl_hx, fr_hx, hl_hx, hr_hx, fl_hy, fr_hy, hl_hy, hr_hy, fl_kn, fr_kn, hl_kn, hr_kn]
        # Spot:      [fl_hx, fl_hy, fl_kn, fr_hx, fr_hy, fr_kn, hl_hx, hl_hy, hl_kn, hr_hx, hr_hy, hr_kn]
        leg_action_spot = reorder_policy_to_spot_leg(leg_action_policy)

        # Store both versions: Policy order for next observation, Spot order for robot
        self._last_action_policy = np.concatenate([leg_action_policy, [0.0] * 7])  # Keep in Policy order
        self._last_action_spot = np.concatenate([leg_action_spot, [0.0] * 7])   # For robot

        return leg_action_spot.tolist()

    def collect_inputs(
        self,
        state: RobotStateStreamResponse,
        config: OrbitConfig,
        joint_commands: List[float] | None = None,
    ) -> dict:
        """Extract observation data from Spot's current state and format for Phase 2 ONNX.

        Phase 2 expects a single 69-dim concatenated tensor:
        [lin_vel(3), ang_vel(3), gravity(3), cmd(3),
         leg_pos(12), leg_vel(12), arm_pos(7), arm_vel(7),
         last_action(19)] = 69 dimensions

        Args:
            state: Proto msg with Spot's latest state
            config: Model configuration data from Orbit
            joint_commands: Optional joint commands (not used in Phase 2)

        Returns:
            Dict with single 69-dim observation tensor
        """
        if self.verbose:
            print("[INFO] cmd", self._context.velocity_cmd)

        # Extract base state
        base_lin_vel = ob.get_base_linear_velocity(state)  # [3]
        base_ang_vel = ob.get_base_angular_velocity(state)  # [3]
        gravity = ob.get_projected_gravity(state)  # [3]
        cmd = np.array(self._context.velocity_cmd)  # [3]

        # Extract joint states
        joint_pos = np.array(state.joint_states.position)  # [19]
        joint_vel = np.array(state.joint_states.velocity)  # [19]

        # Separate leg and arm data (in Spot robot order)
        leg_pos_spot = joint_pos[self._leg_joint_indices]  # [12]
        leg_vel_spot = joint_vel[self._leg_joint_indices]  # [12]
        arm_pos = joint_pos[self._arm_joint_indices]  # [7]
        arm_vel = joint_vel[self._arm_joint_indices]  # [7]

        # CRITICAL: Reorder leg data from Spot order to Phase 2 policy order
        # Real Spot: [fl_hx, fl_hy, fl_kn, fr_hx, fr_hy, fr_kn, hl_hx, hl_hy, hl_kn, hr_hx, hr_hy, hr_kn]
        # Policy:     [fl_hx, fr_hx, hl_hx, hr_hx, fl_hy, fr_hy, hl_hy, hr_hy, fl_kn, fr_kn, hl_kn, hr_kn]
        leg_pos_policy = reorder_spot_to_policy_leg(leg_pos_spot)  # [12]
        leg_vel_policy = reorder_spot_to_policy_leg(leg_vel_spot)  # [12]

        # Concatenate all components into 69-dim observation
        obs = np.concatenate([
            base_lin_vel,           # 3
            base_ang_vel,           # 3
            gravity,                # 3
            cmd,                    # 3
            leg_pos_policy,         # 12 (reordered to policy order)
            leg_vel_policy,         # 12 (reordered to policy order)
            arm_pos,                # 7 (arm order is same)
            arm_vel,                # 7 (arm order is same)
            self._last_action_policy # 19 (in Policy order for observations)
        ])  # Total: 69

        # Return in format compatible with both logging and ONNX
        # For ONNX: single concatenated tensor with policy-ordered leg data
        # For logging: use original Spot order for consistency
        return {
            "observations": obs.astype(np.float32).reshape(1, 69),  # For ONNX (policy order)
            "base_linear_velocity": base_lin_vel.astype(np.float32).reshape(1, 3),
            "base_angular_velocity": base_ang_vel.astype(np.float32).reshape(1, 3),
            "projected_gravity": gravity.astype(np.float32).reshape(1, 3),
            "velocity_commands": cmd.astype(np.float32).reshape(1, 3),
            "joint_positions": joint_pos.astype(np.float32).reshape(1, -1),  # Original Spot order
            "joint_velocities": joint_vel.astype(np.float32).reshape(1, -1),  # Original Spot order
            "last_actions": self._last_action_spot.astype(np.float32).reshape(1, -1),  # Spot order for logging
        }

    def create_proto(self, pos_command: List[float]):
        """Generate a proto msg for Spot with a given pos_command.

        Args:
            pos_command: List of joint positions (see spot.constants for order)

        Returns:
            Proto message to send in Spot's command stream
        """
        update_proto = robot_command_pb2.JointControlStreamRequest()
        update_proto.Clear()

        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_phase2_client"

        k_q_p = dict_to_list(self._config.kp, ORDERED_JOINT_NAMES_SPOT)
        k_qd_p = dict_to_list(self._config.kd, ORDERED_JOINT_NAMES_SPOT)

        N_DOF = len(pos_command)
        pos_cmd = [0] * N_DOF
        vel_cmd = [0] * N_DOF
        load_cmd = [0] * N_DOF

        for joint_ind in range(N_DOF):
            pos_cmd[joint_ind] = pos_command[joint_ind]
            vel_cmd[joint_ind] = 0
            load_cmd[joint_ind] = 0

        # Fill in gains the first dt
        if self._count <= 3:
            update_proto.joint_command.gains.k_q_p.extend(k_q_p)
            update_proto.joint_command.gains.k_qd_p.extend(k_qd_p)

        update_proto.joint_command.position.extend(pos_cmd)
        update_proto.joint_command.velocity.extend(vel_cmd)
        update_proto.joint_command.load.extend(load_cmd)

        observation_time = self._context.latest_state.joint_states.acquisition_timestamp
        end_time = seconds_to_timestamp(timestamp_to_sec(observation_time) + 0.1)
        update_proto.joint_command.end_time.CopyFrom(end_time)

        # Let it extrapolate the command a little
        update_proto.joint_command.extrapolation_duration.nanos = int(5 * 1e6)

        # Set user key for latency tracking
        update_proto.joint_command.user_command_key = self._count
        return update_proto

    def create_proto_hold(self):
        """Generate a proto msg that holds Spot's current pose (useful for debugging).

        Returns:
            Proto message to send in Spot's command stream
        """
        update_proto = robot_command_pb2.JointControlStreamRequest()
        update_proto.Clear()
        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_phase2_client"

        k_q_p = DEFAULT_K_Q_P[0:19]
        k_qd_p = DEFAULT_K_QD_P[0:19]

        N_DOF = 19
        pos_cmd = [0] * N_DOF
        vel_cmd = [0] * N_DOF
        load_cmd = [0] * N_DOF

        for joint_ind in range(N_DOF):
            pos_cmd[joint_ind] = self._init_pos[joint_ind]
            vel_cmd[joint_ind] = 0
            load_cmd[joint_ind] = self._init_load[joint_ind]

        # Fill in gains the first dt
        if self._count == 1:
            update_proto.joint_command.gains.k_q_p.extend(k_q_p)
            update_proto.joint_command.gains.k_qd_p.extend(k_qd_p)

        update_proto.joint_command.position.extend(pos_cmd)
        update_proto.joint_command.velocity.extend(vel_cmd)
        update_proto.joint_command.load.extend(load_cmd)

        observation_time = self._context.latest_state.joint_states.acquisition_timestamp
        end_time = seconds_to_timestamp(timestamp_to_sec(observation_time) + 0.1)
        update_proto.joint_command.end_time.CopyFrom(end_time)

        # Let it extrapolate the command a little
        update_proto.joint_command.extrapolation_duration.nanos = int(5 * 1e6)

        # Set user key for latency tracking
        update_proto.joint_command.user_command_key = self._count
        return update_proto
