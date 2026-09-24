import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from olmo.piper_kinematics import (
    PIPER_M,
    base_frame_ee_delta,
    piper_delta_ee,
    piper_fk,
    piper_solver,
)


def test_base_frame_delta_reconstructs_noncommuting_global_rotation():
    current = np.eye(4)[None, None].repeat(2, axis=1)
    current[..., :3, :3] = Rotation.from_euler("XYZ", [.7, -.4, .5]).as_matrix()
    current[..., :3, 3] = [.2, -.3, .4]
    expected = np.array([.01, -.02, .03, .06, -.04, .07])
    following = current.copy()
    following[..., :3, 3] += expected[:3]
    following[..., :3, :3] = Rotation.from_euler("XYZ", expected[3:]).as_matrix() @ current[..., :3, :3]

    actual = base_frame_ee_delta(current, following)

    np.testing.assert_allclose(actual, np.broadcast_to(expected, (1, 2, 6)), atol=1e-12)


def test_piper_batched_fk_matches_scalar_fk_for_both_arms():
    states = np.zeros((3, 14))
    states[1, :6] = [.1, .2, -.3, .1, -.2, .3]
    states[2, 7:13] = [-.1, .3, -.2, -.1, .2, -.3]
    actual = piper_fk(states)

    np.testing.assert_allclose(actual[0, 0], PIPER_M)
    np.testing.assert_allclose(actual[0, 1], PIPER_M)
    for index in range(3):
        for arm in range(2):
            expected = piper_solver().forward_kinematics(states[index, arm * 7:arm * 7 + 6])
            np.testing.assert_allclose(actual[index, arm], expected)


def test_piper_delta_ee_reconstructs_fk_targets():
    current = np.zeros((2, 14))
    current[:, :6] = [.2, .3, -.4, .1, -.2, .3]
    current[:, 7:13] = [-.2, .4, -.3, -.1, .2, -.3]
    following = current.copy()
    following[0, :6] += [.05, .02, -.03, .04, -.01, .02]
    following[1, 7:13] += [-.02, -.03, .01, .02, .04, -.05]
    delta = piper_delta_ee(current, following).reshape(2, 2, 7)
    current_poses, next_poses = piper_fk(current), piper_fk(following)

    np.testing.assert_allclose(current_poses[..., :3, 3] + delta[..., :3], next_poses[..., :3, 3], atol=1e-7)
    rotations = Rotation.from_euler("XYZ", delta[..., 3:6].reshape(-1, 3)).as_matrix().reshape(2, 2, 3, 3)
    np.testing.assert_allclose(rotations @ current_poses[..., :3, :3], next_poses[..., :3, :3], atol=1e-7)
    np.testing.assert_array_equal(delta[..., 6], 0)


def test_gripper_state_does_not_affect_tcp_delta():
    current = np.zeros((1, 14))
    following = current.copy()
    following[:, (6, 13)] = .05
    np.testing.assert_allclose(piper_delta_ee(current, following), 0, atol=1e-7)


@pytest.mark.parametrize("norm_mode", ["min_max", "q01_q99"])
@pytest.mark.parametrize("xyz_limit,rpy_limit", [(.04, .08), (.01, .03)])
def test_ee_normalization_uses_ee_bounds_and_preserves_joint_state_stats(norm_mode, xyz_limit, rpy_limit):
    from copy import deepcopy
    from launch_scripts.train_lerobot import _apply_delta_action_stats
    from olmo.data.robot_processing import RobotProcessorConfig

    stats = {"piper_x": {
        "action": {"min": [-2.] * 14, "max": [2.] * 14, "q01": [-1.] * 14, "q99": [1.] * 14},
        "observation.state": {"min": [-2.] * 14, "max": [2.] * 14, "q01": [-1.] * 14, "q99": [1.] * 14},
    }}
    for key in ("min", "q01"):
        for index in (6, 13):
            stats["piper_x"]["action"][key][index] = 0.
    for key in ("max", "q99"):
        for index in (6, 13):
            stats["piper_x"]["action"][key][index] = .05
    original_state_stats = deepcopy(stats["piper_x"]["observation.state"])
    metadata = {"piper_x": {
        "action_dim": 14, "action_key": "action", "state_keys": ["observation.state"],
    }}
    _apply_delta_action_stats(
        stats, metadata, -1, action_stride=2, delta_action_mode="ee",
        delta_ee_xyz_limit=xyz_limit, delta_ee_rpy_limit=rpy_limit,
    )
    assert metadata["piper_x"]["delta_ee_xyz_limit"] == xyz_limit
    assert metadata["piper_x"]["delta_ee_rpy_limit"] == rpy_limit
    assert metadata["piper_x"]["delta_ee_limits"] == [xyz_limit] * 3 + [rpy_limit] * 3
    assert metadata["piper_x"]["delta_action_mode"] == "ee"
    assert metadata["piper_x"]["delta_ee_frame"] == "base"
    assert metadata["piper_x"]["delta_ee_euler_convention"] == "XYZ"
    assert stats["piper_x"]["observation.state"] == original_state_stats
    processor = RobotProcessorConfig.from_stats(
        stats, tag_metadata=metadata, repo_to_tag={"lerobot_test": "piper_x"}, norm_mode=norm_mode,
    ).build_processor()
    action = np.tile([xyz_limit / 2, -xyz_limit / 2, xyz_limit / 2,
                      rpy_limit / 2, -rpy_limit / 2, rpy_limit / 2, .025], 2)[None].astype(np.float32)
    normalized = processor.normalize_action(action, "lerobot_test")
    np.testing.assert_allclose(normalized, np.tile([.5, -.5, .5, .5, -.5, .5, 0], 2)[None], atol=1e-6)
    np.testing.assert_allclose(processor.unnormalize_action(normalized, "lerobot_test"), action, atol=1e-7)
