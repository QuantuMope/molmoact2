"""Piper-X TCP kinematics and base-frame EE deltas (metres and radians)."""

from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation


# Home TCP relative to each arm base, with a 0.13503 m offset along link6 +Z.
# TCP axes (+X, +Y, +Z) correspond to link6 (+Z, -X, -Y).
PIPER_M = np.array([
    [0.995687218, -0.000003846, -0.092773725, 0.232276698],
    [0.000004475, 1.000000000, 0.000006569, 0.000001714],
    [0.092773725, -0.000006956, 0.995687218, 0.229485257],
    [0.0, 0.0, 0.0, 1.0],
])
PIPER_S = np.array([
    [0., -0.000007844, -0.000007844, -0.000007844, 0.092770068, 0.995687218],
    [0., 1.000000000, 1.000000000, 1.000000000, -0.000006569, 0.000004475],
    [1., -0.000003639, -0.000003639, -0.000003639, -0.995687559, 0.092773725],
    [0., -0.123000000, -0.161605158, -0.207202299, 0.000000583, -0.000000868],
    [0., -0.000000965, -0.000002295, -0.000001675, 0.082534381, 0.206946363],
    [0., 0.000000000, -0.282403510, -0.013603454, -0.000000490, -0.000000667],
])
PIPER_JOINT_LOWER = np.array([-2.618, 0., -2.9671, -1.57, -1.57, -2.879793])
PIPER_JOINT_UPPER = np.array([2.618, 3.14, 0., 1.57, 1.57, 2.879793])

# Joint-limit-aware TCP workspace bounds with margin, in each arm's base frame.
PIPER_EE_XYZ_MIN = [-.85, -.85, -.45]
PIPER_EE_XYZ_MAX = [.85, .85, .95]


def append_piper_ee_pose(states: np.ndarray) -> np.ndarray:
    """Append each arm's [xyz, R[:, 0], R[:, 1]] to 7-D or 14-D joint states."""
    states = np.asarray(states, dtype=np.float32)
    if states.shape[-1] not in (7, 14):
        raise ValueError("Piper EE observations require 7 or 14 joint/gripper values.")
    poses = piper_fk(states)
    ee = np.concatenate((poses[..., :3, 3], poses[..., :3, 0], poses[..., :3, 1]), axis=-1)
    ee = ee.reshape(*states.shape[:-1], states.shape[-1] // 7 * 9).astype(np.float32)
    return np.concatenate((states, ee), axis=-1)


def piper_ee_limits(xyz_limit: float = .04, rpy_limit: float = .08) -> np.ndarray:
    # Defaults match ren_openpi0. Absolute grippers are not clipped here.
    return np.tile([xyz_limit] * 3 + [rpy_limit] * 3 + [np.inf], 2)


@lru_cache(maxsize=1)
def piper_solver():
    # Joint-only training does not need the compiled extension.
    import kincpp

    return kincpp.KinematicsSolver(PIPER_M, PIPER_S, PIPER_JOINT_LOWER, PIPER_JOINT_UPPER)


def piper_fk(states: np.ndarray) -> np.ndarray:
    """Return [N, arms, 4, 4] TCP poses using one batched FK call."""
    num_arms = states.shape[-1] // 7
    joints = np.asarray(states).reshape(-1, num_arms, 7)[..., :6].reshape(-1, 6)
    return np.asarray(piper_solver().batched_forward_kinematics(joints)).reshape(-1, num_arms, 4, 4)


def base_frame_ee_delta(current: np.ndarray, following: np.ndarray) -> np.ndarray:
    """Invert p_next = p + dp, R_next = Rx(dr) Ry(dp) Rz(dy) R (ren_openpi0 sim)."""
    delta_rotation = following[..., :3, :3] @ np.swapaxes(current[..., :3, :3], -1, -2)
    rpy = Rotation.from_matrix(delta_rotation.reshape(-1, 3, 3)).as_euler("XYZ")
    rpy = rpy.reshape(*delta_rotation.shape[:-2], 3)
    return np.concatenate((following[..., :3, 3] - current[..., :3, 3], rpy), axis=-1)


def adaptive_stride_offsets(poses, action_stride, xyz_limit, rpy_limit):
    """Choose the longest bounded EE transition, falling back to stride 1."""
    limits = np.array([xyz_limit] * 3 + [rpy_limit] * 3)
    offsets = [0]
    while offsets[-1] < len(poses) - 1:
        start = offsets[-1]
        stride = min(action_stride, len(poses) - 1 - start)
        while stride > 1:
            delta = base_frame_ee_delta(poses[start], poses[start + stride]).astype(np.float32)
            if np.all(np.abs(delta) <= limits):
                break
            stride -= 1
        offsets.append(start + stride)
    return np.asarray(offsets)


def piper_delta_ee(current_states: np.ndarray, next_states: np.ndarray) -> np.ndarray:
    """Return raw 14-D EE deltas; callers fill the two absolute gripper actions."""
    poses = piper_fk(np.concatenate((current_states, next_states)))
    count = len(current_states)
    actions = np.zeros((count, 2, 7), dtype=np.float32)
    actions[..., :6] = base_frame_ee_delta(poses[:count], poses[count:])
    return actions.reshape(count, 14)
