# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import datetime
import os
from typing import Dict, List

import h5py
import numpy as np


class HDF5Logger:
    """Class to buffer and save robot states and observations to an HDF5 file."""

    def __init__(self, log_path: str, metadata: Dict[str, str] | None = None):
        self.log_path = log_path
        self._first_timestamp = None
        # Run provenance, written as HDF5 root attrs by save(). Without this a day of robot
        # logs is a pile of same-shaped arrays with no way to tell which policy, which payload
        # configuration, or which manoeuvre produced them. See EXPERIMENTS.md.
        self.metadata: Dict[str, str] = dict(metadata or {})
        self.metadata.setdefault(
            "wall_clock_start", datetime.datetime.now().astimezone().isoformat()
        )
        self.data: Dict[str, List] = {
            "raw_base_linear_velocity": [],
            "raw_base_angular_velocity": [],
            "raw_projected_gravity": [],
            "raw_joint_positions": [],
            "raw_joint_velocities": [],
            "raw_joint_loads": [],
            "commanded_action": [],
            "spot_current_positions": [],
            "spot_current_velocities": [],
            "preprocessed_base_linear_velocity": [],
            "preprocessed_base_angular_velocity": [],
            "preprocessed_projected_gravity": [],
            "preprocessed_velocity_cmd": [],
            "preprocessed_joint_positions": [],
            "preprocessed_joint_velocities": [],
            "preprocessed_last_action": [],
            "response_timestamp": [],
            "dt_divider_wait": [],
            "dt_divider_to_onnx": [],
            "dt_onnx_compute": [],
            "dt_post_process": [],
            "dt_total_step": [],
            "dt_state_arrival_to_compute": [],
            "raw_state_proto_bytes": [],
            "proto_bytes": [],
        }

    def log_state(
        self,
        raw_base_linear_velocity: List[float],
        raw_base_angular_velocity: List[float],
        raw_projected_gravity: List[float],
        raw_joint_positions: List[float],
        raw_joint_velocities: List[float],
        raw_joint_loads: List[float],
        spot_current_positions: List[float],
        spot_current_velocities: List[float],
        preprocessed_base_linear_velocity: List[float],
        preprocessed_base_angular_velocity: List[float],
        preprocessed_projected_gravity: List[float],
        preprocessed_velocity_cmd: List[float],
        preprocessed_joint_positions: List[float],
        preprocessed_joint_velocities: List[float],
        preprocessed_last_action: List[float],
        commanded_action: List[float],
        response_timestamp: datetime.datetime,
        dt_divider_wait: float,
        dt_divider_to_onnx: float,
        dt_onnx_compute: float,
        dt_post_process: float,
        dt_total_step: float,
        dt_state_arrival_to_compute: float,
        raw_state_proto_bytes: bytes,
        proto_bytes: bytes,
    ):
        """Append a single step of data to the buffers."""
        self.data["raw_base_linear_velocity"].append(raw_base_linear_velocity)
        self.data["raw_base_angular_velocity"].append(raw_base_angular_velocity)
        self.data["raw_projected_gravity"].append(raw_projected_gravity)
        self.data["raw_joint_positions"].append(raw_joint_positions)
        self.data["raw_joint_velocities"].append(raw_joint_velocities)
        self.data["raw_joint_loads"].append(raw_joint_loads)
        self.data["spot_current_positions"].append(spot_current_positions)
        self.data["spot_current_velocities"].append(spot_current_velocities)
        self.data["preprocessed_base_linear_velocity"].append(
            preprocessed_base_linear_velocity
        )
        self.data["preprocessed_base_angular_velocity"].append(
            preprocessed_base_angular_velocity
        )
        self.data["preprocessed_projected_gravity"].append(
            preprocessed_projected_gravity
        )
        self.data["preprocessed_velocity_cmd"].append(preprocessed_velocity_cmd)
        self.data["preprocessed_joint_positions"].append(preprocessed_joint_positions)
        self.data["preprocessed_joint_velocities"].append(preprocessed_joint_velocities)
        self.data["preprocessed_last_action"].append(preprocessed_last_action)
        self.data["commanded_action"].append(commanded_action)
        self.data["dt_divider_wait"].append(dt_divider_wait)
        self.data["dt_divider_to_onnx"].append(dt_divider_to_onnx)
        self.data["dt_onnx_compute"].append(dt_onnx_compute)
        self.data["dt_post_process"].append(dt_post_process)
        self.data["dt_total_step"].append(dt_total_step)
        self.data["dt_state_arrival_to_compute"].append(dt_state_arrival_to_compute)
        if self._first_timestamp is None:
            self._first_timestamp = response_timestamp
        delta_time = (response_timestamp - self._first_timestamp).total_seconds()
        self.data["response_timestamp"].append(delta_time)
        self.data["raw_state_proto_bytes"].append(raw_state_proto_bytes)
        self.data["proto_bytes"].append(proto_bytes)

    def save(self):
        """Write all buffered data to the HDF5 file."""
        if not self.log_path:
            return

        print(f"Saving HDF5 log to {self.log_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        # Create variable-length datatype for raw bytes arrays
        vlen_bytes_dtype = h5py.vlen_dtype(np.uint8)
        with h5py.File(self.log_path, "w") as f:
            # Provenance first, and defensively: a metadata bug must NEVER cost the run data.
            try:
                self.metadata.setdefault(
                    "wall_clock_end", datetime.datetime.now().astimezone().isoformat()
                )
                self.metadata.setdefault("n_steps", str(len(self.data["response_timestamp"])))
                for k, v in self.metadata.items():
                    f.attrs[k] = str(v)
            except Exception as exc:  # noqa: BLE001 - never lose a robot run over metadata
                print(f"[HDF5Logger] WARNING: failed to write metadata attrs: {exc}")

            for key, val in self.data.items():
                if len(val) > 0:
                    if key == "raw_state_proto_bytes" or key == "proto_bytes":
                        # Convert bytes strings into variable length numpy arrays of uint8
                        byte_arrays = [np.frombuffer(b, dtype=np.uint8) for b in val]
                        ds = f.create_dataset(key, (len(val),), dtype=vlen_bytes_dtype)
                        for i, arr in enumerate(byte_arrays):
                            ds[i] = arr
                    else:
                        f.create_dataset(key, data=np.array(val, dtype=np.float32))
        print("HDF5 log saved successfully.")
