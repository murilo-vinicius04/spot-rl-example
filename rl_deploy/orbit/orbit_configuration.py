# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from rl_deploy.orbit.orbit_constants import ORDERED_JOINT_NAMES_ARM_ISAAC
import json
import os
import re
from dataclasses import dataclass
from typing import List

import yaml

from rl_deploy.orbit.orbit_constants import ORDERED_JOINT_NAMES_ISAAC
from rl_deploy.utils.dict_tools import dict_from_lists, set_matching
from rl_deploy.spot.constants import DEFAULT_K_Q_P, DEFAULT_K_QD_P, DOF


class Ref(yaml.YAMLObject):
    yaml_tag = "tag:yaml.org,2002:python/tuple"

    def __init__(self, val):
        self.val = val

    @classmethod
    def from_yaml(cls, loader, node):
        return tuple(node.value)


class Slices(yaml.YAMLObject):
    yaml_tag = "python/object/apply:builtins.slice"

    @classmethod
    def from_yaml(cls, loader, node):
        values = node.value
        if len(values) == 1:
            return slice(values[0].value)
        elif len(values) == 2:
            return slice(values[0].value, values[1].value)
        elif len(values) == 3:
            return slice(values[0].value, values[1].value, values[2].value)


yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", Ref.from_yaml)
yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice", Slices.from_yaml
)


@dataclass
class OrbitConfig:
    """dataclass holding data extracted from orbits training configuration"""

    kp: List[float]
    kd: List[float]
    default_joints: List[float]
    standing_height: float
    action_scale: float


def detect_config_file(directory: os.PathLike | str) -> dict | None:
    """find and parse json or yaml file in policy directory

    arguments
    directory -- path where policy and training configuration can be found

    return dictionary from config file
    """
    files = [f for f in os.listdir(directory) if f.endswith("env.json")]
    if len(files) == 1:
        filepath = os.path.join(directory, files[0])
        with open(filepath) as f:
            return json.load(f)

    files = [f for f in os.listdir(directory) if f.endswith("env.yaml")]
    if len(files) == 1:
        filepath = os.path.join(directory, files[0])
        with open(filepath) as f:
            return yaml.safe_load(f)

    return None


def detect_policy_file(directory: os.PathLike | str) -> str | None:
    """find onnx file in policy directory

    arguments
    directory -- path where policy and training configuration can be found

    return filepath to onnx file
    """
    files = [f for f in os.listdir(directory) if f.endswith(".onnx")]
    if len(files) == 1:
        return os.path.join(directory, files[0])
    return None


def load_configuration(env_config: dict) -> OrbitConfig:
    """parse json file and populate an OrbitConfig dataclass

    arguments
    file -- the path to the json file containing training configuration

    return OrbitConfig containing needed training configuration
    """

    joint_kp = dict_from_lists(ORDERED_JOINT_NAMES_ISAAC, [None] * 19)
    joint_kd = dict_from_lists(ORDERED_JOINT_NAMES_ISAAC, [None] * 19)
    joint_offsets = dict_from_lists(ORDERED_JOINT_NAMES_ISAAC, [None] * 19)

    actuators = env_config["scene"]["robot"]["actuators"]

    for group in actuators.keys():
        regex = re.compile(actuators[group]["joint_names_expr"][0])

        set_matching(joint_kp, regex, actuators[group]["stiffness"])
        set_matching(joint_kd, regex, actuators[group]["damping"])

    default_joint_data = env_config["scene"]["robot"]["init_state"]["joint_pos"]
    default_joint_expressions = default_joint_data.keys()
    for expression in default_joint_expressions:
        regex = re.compile(expression)
        set_matching(joint_offsets, regex, default_joint_data[expression])

    if "joint_pos" in env_config["actions"]:
        action_scale = env_config["actions"]["joint_pos"]["scale"]
    elif "leg_joint_pos" in env_config["actions"]:
        action_scale = env_config["actions"]["leg_joint_pos"]["scale"]
    else:
        raise KeyError("Could not find 'joint_pos' or 'leg_joint_pos' in actions configuration.")
    standing_height = env_config["scene"]["robot"]["init_state"]["pos"][2]

    # Override the arm with default values for kp, kd
    for joint_name in ORDERED_JOINT_NAMES_ARM_ISAAC:
        joint_kp[joint_name] = DEFAULT_K_Q_P[DOF[joint_name.upper()]]
        joint_kd[joint_name] = DEFAULT_K_QD_P[DOF[joint_name.upper()]]
        print(
            f"Setting {joint_name} kp to {joint_kp[joint_name]} and kd to {joint_kd[joint_name]}"
        )

    return OrbitConfig(
        kp=joint_kp,
        kd=joint_kd,
        default_joints=joint_offsets,
        standing_height=standing_height,
        action_scale=action_scale,
    )
