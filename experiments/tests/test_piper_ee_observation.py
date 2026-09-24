from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

from launch_scripts.train_lerobot import _append_ee_pose_normalization_stats
from olmo.data.robot_processing import RobotProcessorConfig
from olmo.hf_model.modeling_molmoact2 import _RobotStats
from olmo.piper_kinematics import (
    PIPER_EE_XYZ_MIN, PIPER_EE_XYZ_MAX, append_piper_ee_pose, piper_fk,
)


def make_processor_config(norm_mode="min_max"):
    state_stats = {
        "min": [-2.] * 14, "max": [2.] * 14,
        "q01": [-2.] * 14, "q99": [2.] * 14,
        "q10": [-2.] * 14, "q90": [2.] * 14,
        "mean": [0.] * 14, "std": [2.] * 14,
        "mask": [True] * 14, "names": [f"joint_{i}" for i in range(14)],
        "count": [100],
    }
    stats = {"piper_x": {"observation.state": state_stats, "action": deepcopy(state_stats)}}
    metadata = {"piper_x": {"action_dim": 14, "state_keys": ["observation.state"]}}
    _append_ee_pose_normalization_stats(stats, metadata)
    return RobotProcessorConfig.from_stats(
        stats, tag_metadata=metadata, repo_to_tag={"lerobot_test": "piper_x"}, norm_mode=norm_mode,
    )


@pytest.mark.parametrize("shape", [(14,), (3, 14), (2, 3, 14)])
def test_pose_layout_matches_fk_and_preserves_joint_state(shape):
    joints = np.random.default_rng(1).uniform(-.2, .2, shape).astype(np.float32)
    original = joints.copy()
    state = append_piper_ee_pose(joints)
    assert state.shape == (*shape[:-1], 32)
    assert state.dtype == np.float32
    np.testing.assert_array_equal(joints, original)
    np.testing.assert_array_equal(state[..., :14], joints)
    poses = piper_fk(joints)
    appended = state[..., 14:].reshape(-1, 2, 9)
    np.testing.assert_allclose(appended[..., :3], poses[..., :3, 3], atol=1e-7)
    np.testing.assert_allclose(appended[..., 3:6], poses[..., :3, 0], atol=1e-7)
    np.testing.assert_allclose(appended[..., 6:9], poses[..., :3, 1], atol=1e-7)
    # Grippers don't affect FK or the appended pose.
    joints[..., [6, 13]] = 1
    np.testing.assert_array_equal(append_piper_ee_pose(joints)[..., 14:], state[..., 14:])


@pytest.mark.parametrize("mode", ["min_max", "q01_q99", "q10_q90", "mean_std"])
def test_fixed_bounds_normalization_and_hf_metadata_round_trip(mode):
    config = make_processor_config(mode)
    processor = config.build_processor()
    payload = json.loads(json.dumps(asdict(config)))
    hf_stats = _RobotStats(payload)
    metadata = hf_stats.get_metadata("piper_x")
    assert metadata["append_ee_pose_to_observation"] is True
    assert metadata["ee_pose_frame"] == "base"
    assert metadata["ee_pose_rotation_representation"] == "rotation6d_columns"
    assert metadata["ee_pose_xyz_min"] == PIPER_EE_XYZ_MIN
    assert metadata["ee_pose_xyz_max"] == PIPER_EE_XYZ_MAX
    assert hf_stats.get_state_dim("piper_x") == 32
    assert processor.get_state_dim("lerobot_test") == 32
    assert hf_stats.get_action_dim("piper_x") == 14
    assert len(metadata["state_stats"]["names"]) == 32
    assert metadata["state_stats"]["count"] == [100]
    assert all(len(metadata["state_stats"][key]) == 32 for key in
               ("mean", "std", "min", "max", "q01", "q99", "q10", "q90", "mask"))
    lower = np.array((PIPER_EE_XYZ_MIN + [-1.] * 6) * 2)
    upper = np.array((PIPER_EE_XYZ_MAX + [1.] * 6) * 2)
    states = np.zeros((3, 32), dtype=np.float32)
    states[:, :14] = .4
    states[:, 14:] = np.stack((lower, (lower + upper) / 2, upper))
    normalized = processor.normalize_state(states, "lerobot_test")
    np.testing.assert_allclose(normalized[:, :14], .2, atol=1e-7)
    np.testing.assert_allclose(normalized[:, 14:], np.broadcast_to([[-1.], [0.], [1.]], (3, 18)), atol=1e-7)
    np.testing.assert_allclose(hf_stats.normalize_state(states, "piper_x"), normalized, atol=1e-7)
    np.testing.assert_allclose(processor.unnormalize_state(normalized, "lerobot_test"), states, atol=1e-7)


def test_other_robot_layout_is_rejected():
    with pytest.raises(ValueError, match="only Piper-X"):
        _append_ee_pose_normalization_stats({"other_robot": {}}, {})


@pytest.mark.parametrize("mode", ["joint", "ee"])
def test_append_happens_after_filtering_stride_and_action_labels(monkeypatch, mode):
    from .test_lerobot_delta_action import _make_skipped_wrapper

    wrapper = _make_skipped_wrapper(
        [0, .02, .021, .1, .2, .201, .4, .3, .301, .5, .7],
        action_stride=2, delta_action_mode=mode,
    )
    indices = list(wrapper._sample_indices)
    clipped = wrapper.delta_action_clipped_chunks
    frame = wrapper.dataset[indices[0]]
    monkeypatch.setattr(wrapper, "_get_frame_with_retry", lambda *a, **kw: (frame, 0, indices[0], indices[0], None))
    monkeypatch.setattr(wrapper, "_extract_image", lambda *a: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(wrapper, "_extract_question", lambda *a: "move the arm")
    monkeypatch.setattr(wrapper, "_compute_task_progress", lambda *a: 0.)
    wrapper.style = "robot_action"
    wrapper.state_format = "both"
    baseline = wrapper.get(0, np.random.default_rng(1))
    wrapper.append_ee_pose_to_observation = True
    result = wrapper.get(0, np.random.default_rng(1))
    assert result["state"].shape == (32,)
    np.testing.assert_array_equal(result["state"][:14], baseline["state"])
    np.testing.assert_array_equal(result["state"], append_piper_ee_pose(baseline["state"]))
    np.testing.assert_array_equal(result["action"], baseline["action"])
    assert result["messages"]["state_text"].count("<state_") == 34  # 32 values + boundaries
    assert wrapper._sample_indices == indices
    assert wrapper.delta_action_clipped_chunks == clipped
    wrapper.robot_processor_config = make_processor_config()
    wrapper.robot_processor = wrapper.robot_processor_config.build_processor()
    normalized = wrapper.get(0, np.random.default_rng(1))
    np.testing.assert_allclose(
        normalized["state"], wrapper.robot_processor.normalize_state(result["state"], "lerobot_test"))
    assert normalized["action"].shape == baseline["action"].shape
