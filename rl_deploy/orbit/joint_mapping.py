# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
"""Joint mapping between real Spot robot order and Phase 2 policy order."""

import numpy as np

# Real Spot robot joint order (from spot/constants.py ORDERED_JOINT_NAMES_SPOT)
SPOT_JOINT_ORDER = [
    "fl_hx", "fl_hy", "fl_kn",  # Front Left
    "fr_hx", "fr_hy", "fr_kn",  # Front Right
    "hl_hx", "hl_hy", "hl_kn",  # Hind Left
    "hr_hx", "hr_hy", "hr_kn",  # Hind Right
    "arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x"  # Arm
]

# Phase 2 policy expected leg joint order (from deploy_spot_phase2_standalone.py)
POLICY_LEG_ORDER = [
    "fl_hx", "fr_hx", "hl_hx", "hr_hx",  # All hip x
    "fl_hy", "fr_hy", "hl_hy", "hr_hy",  # All hip y
    "fl_kn", "fr_kn", "hl_kn", "hr_kn",  # All knees
]

# Phase 2 policy expected arm joint order (same as real robot)
POLICY_ARM_ORDER = [
    "arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x"
]

# Create inverse mapping: spot_index -> policy_index
# For each spot joint, find its index in the policy order
SPOT_TO_POLICY_LEG_MAPPING = [
    POLICY_LEG_ORDER.index(joint) for joint in SPOT_JOINT_ORDER[:12]
]

# Create inverse mapping: policy_index -> spot_index
# For each policy joint, find its index in the spot order
POLICY_TO_SPOT_LEG_MAPPING = [
    SPOT_JOINT_ORDER.index(joint) for joint in POLICY_LEG_ORDER
]

def reorder_spot_to_policy_leg(data):
    """
    Reorder leg data from real Spot robot order to Phase 2 policy order.

    Args:
        data: array-like of length 12 in real Spot robot order

    Returns:
        array of length 12 in Phase 2 policy order
    """
    data_array = np.array(data)
    # For each policy position i, find the spot joint that belongs there
    # Create inverse of SPOT_TO_POLICY_LEG_MAPPING
    policy_to_spot_inverse = [0] * 12
    for spot_idx, policy_idx in enumerate(SPOT_TO_POLICY_LEG_MAPPING):
        policy_to_spot_inverse[policy_idx] = spot_idx

    return data_array[policy_to_spot_inverse]

def reorder_policy_to_spot_leg(data):
    """
    Reorder leg data from Phase 2 policy order to real Spot robot order.

    Args:
        data: array-like of length 12 in Phase 2 policy order

    Returns:
        array of length 12 in real Spot robot order
    """
    data_array = np.array(data)
    # For each spot position i, find the policy joint that belongs there
    # Create inverse of POLICY_TO_SPOT_LEG_MAPPING
    spot_to_policy_inverse = [0] * 12
    for policy_idx, spot_idx in enumerate(POLICY_TO_SPOT_LEG_MAPPING):
        spot_to_policy_inverse[spot_idx] = policy_idx

    return data_array[spot_to_policy_inverse]

if __name__ == "__main__":
    # Test the mappings
    print("Real Spot Robot Leg Order:")
    print(SPOT_JOINT_ORDER[:12])

    print("\nPhase 2 Policy Leg Order:")
    print(POLICY_LEG_ORDER)

    print("\nSpot to Policy Mapping:")
    print(f"  {SPOT_TO_POLICY_LEG_MAPPING}")

    print("\nPolicy to Spot Mapping:")
    print(f"  {POLICY_TO_SPOT_LEG_MAPPING}")

    # Test with sample data
    # Simulate real robot data (in SPOT order)
    spot_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    print("\nOriginal Spot data:")
    print(f"  {spot_data}")

    # Reorder to policy order
    policy_data = reorder_spot_to_policy_leg(spot_data)
    print("\nReordered to Policy order:")
    print(f"  {policy_data}")

    # Reorder back to Spot order
    spot_data_back = reorder_policy_to_spot_leg(policy_data)
    print("\nReordered back to Spot order:")
    print(f"  {spot_data_back}")

    # Verify round-trip
    if np.allclose(spot_data, spot_data_back):
        print("\n✅ Round-trip test PASSED!")
    else:
        print("\n❌ Round-trip test FAILED!")
        print(f"   Difference: {np.abs(spot_data - spot_data_back)}")
