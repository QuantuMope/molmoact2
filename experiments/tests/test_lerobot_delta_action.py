import numpy as np

from olmo.data import lerobot_wrapper
from olmo.data.lerobot_wrapper import _build_delta_action
from olmo.data.robot_processing import RobotProcessorConfig


def test_delta_action_clips_joints_and_preserves_absolute_grippers():
    states = np.zeros((3, 14), dtype=np.float32)
    states[1, 0] = 0.2
    states[2, 0] = 0.15
    actions = np.zeros((2, 14), dtype=np.float32)
    actions[:, 6] = [0.25, 0.75]
    actions[:, 13] = [0.5, 1.0]

    delta_action = _build_delta_action(states, actions, 0.1, (6, 13))

    np.testing.assert_allclose(delta_action[:, 0], [0.1, -0.05])
    np.testing.assert_allclose(delta_action[:, 6], actions[:, 6])
    np.testing.assert_allclose(delta_action[:, 13], actions[:, 13])


def test_delta_action_loads_one_more_state_than_actions(monkeypatch):
    class Metadata:
        fps = 30
        features = {
            "observation.state": {},
            "observation.images.front": {},
            "action": {},
        }

    monkeypatch.setattr(lerobot_wrapper, "LeRobotDatasetMetadata", lambda **kwargs: Metadata())
    timestamps, _, action_indices = lerobot_wrapper._resolve_delta_settings(
        "lerobot_test",
        None,
        n_obs_steps=1,
        action_horizon=3,
        state_keys=("observation.state",),
        use_delta_action=True,
    )

    assert action_indices == [0, 1, 2]
    assert timestamps["action"] == [0, 1 / 30, 2 / 30]
    assert timestamps["observation.state"] == [0, 1 / 30, 2 / 30, 3 / 30]
    assert "observation.images.front" not in timestamps


def test_delta_action_stats_normalize_joints_and_round_trip_to_radians():
    names = [f"joint_{i}" for i in range(14)]
    stats = {
        "piper_x": {
            "action": {
                "names": names,
                "min": [-0.1] * 14,
                "max": [0.1] * 14,
                "q01": [-0.1] * 14,
                "q99": [0.1] * 14,
            },
            "observation.state": {
                "min": [-2.0] * 14,
                "max": [2.0] * 14,
                "q01": [-1.0] * 14,
                "q99": [1.0] * 14,
            },
        }
    }
    metadata = {
        "piper_x": {
            "action_key": "action",
            "state_keys": ["observation.state"],
            "action_dim": 14,
            "setup_type": "bimanual piper-x robotic arms",
            "repo_ids": ["lerobot_test"],
        }
    }
    for key in ("min", "q01"):
        stats["piper_x"]["action"][key][6] = 0.0
        stats["piper_x"]["action"][key][13] = 0.0
    for key in ("max", "q99"):
        stats["piper_x"]["action"][key][6] = 1.0
        stats["piper_x"]["action"][key][13] = 1.0
    processor = RobotProcessorConfig.from_stats(
        stats,
        tag_metadata=metadata,
        repo_to_tag={"lerobot_test": "piper_x"},
        norm_mode="q01_q99",
    ).build_processor()
    action = np.zeros((1, 14), dtype=np.float32)
    action[0, 0] = 0.05
    action[0, 6] = 0.5

    normalized = processor.normalize_action(action, "lerobot_test")
    restored = processor.unnormalize_action(normalized, "lerobot_test")

    np.testing.assert_allclose(normalized[0, 0], 0.5)
    np.testing.assert_allclose(restored, action)
