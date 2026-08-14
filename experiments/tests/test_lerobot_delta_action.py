import numpy as np

from olmo.data import lerobot_wrapper
from olmo.data.lerobot_wrapper import LeRobotDatasetWrapper, _build_delta_action
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


def test_static_filter_uses_immediate_delta_action_transition():
    class Dataset:
        repo_id = "lerobot_test"
        episodes = None
        _absolute_to_relative_idx = None

        class Meta:
            total_episodes = 1
            episodes = [{"dataset_from_index": 0, "dataset_to_index": 3}]

        meta = Meta()

        def __len__(self):
            return 3

        def __getitem__(self, index):
            states = (
                [[0.0], [0.0], [0.2]],
                [[0.0], [0.2], [0.2]],
                [[0.2], [0.2], [0.2]],
            )
            return {"observation.state": np.asarray(states[index], dtype=np.float32)}

    wrapper = LeRobotDatasetWrapper.__new__(LeRobotDatasetWrapper)
    wrapper.dataset = Dataset()
    wrapper.metadata_repo_id = "lerobot_test"
    wrapper.state_keys = ("observation.state",)
    wrapper.use_delta_action = True
    wrapper.tag_action_horizon = 1
    wrapper.drop_n_last_frames = 0
    wrapper.static_threshold = 0.1
    wrapper._sample_indices = None
    wrapper._episode_ranges = None
    wrapper._episode_cumulative = None
    wrapper._effective_len = None

    wrapper._build_episode_index()

    assert wrapper._sample_indices == [1]
    np.testing.assert_array_equal(wrapper._filtered_action_indices, [1])
    np.testing.assert_array_equal(wrapper._filtered_next_indices, [2])
    np.testing.assert_array_equal(wrapper._filtered_action_offsets, [0])


def test_filtered_delta_action_chunks_use_filtered_transition_order():
    class HFDataset:
        batch_reads = 0
        scalar_reads = 0
        rows = (
            {"observation.state": np.array([0.0]), "action": np.array([0.0])},
            {"observation.state": np.array([0.2]), "action": np.array([1.0])},
            {"observation.state": np.array([0.2]), "action": np.array([2.0])},
            {"observation.state": np.array([0.5]), "action": np.array([3.0])},
        )

        def select_columns(self, columns):
            return self

        def with_format(self, format):
            return self

        def __getitem__(self, index):
            if isinstance(index, slice):
                index = list(range(*index.indices(len(self.rows))))
            if isinstance(index, list):
                self.batch_reads += 1
                return {
                    key: np.stack([self.rows[row_index][key] for row_index in index])
                    for key in self.rows[0]
                }
            self.scalar_reads += 1
            return self.rows[index]

    class Dataset:
        repo_id = "lerobot_test"
        episodes = None
        _absolute_to_relative_idx = None
        hf_dataset = HFDataset()

        class Meta:
            total_episodes = 1
            episodes = [{"dataset_from_index": 0, "dataset_to_index": 4}]

        meta = Meta()

        def __len__(self):
            return 4

    wrapper = LeRobotDatasetWrapper.__new__(LeRobotDatasetWrapper)
    wrapper.dataset = Dataset()
    wrapper.metadata_repo_id = "lerobot_test"
    wrapper.state_keys = ("observation.state",)
    wrapper.action_keys = ("action",)
    wrapper.use_delta_action = True
    wrapper.tag_action_horizon = 2
    wrapper.delta_joint_limit = 1.0
    wrapper.gripper_indices = ()
    wrapper.drop_n_last_frames = 0
    wrapper.static_threshold = 0.1
    wrapper._sample_indices = None
    wrapper._episode_ranges = None
    wrapper._episode_cumulative = None
    wrapper._effective_len = None

    wrapper._build_episode_index()

    assert wrapper._sample_indices == [0]
    np.testing.assert_array_equal(wrapper._filtered_action_indices, [0, 2])
    np.testing.assert_array_equal(wrapper._filtered_next_indices, [1, 3])
    np.testing.assert_array_equal(wrapper._filtered_action_offsets, [0])
    np.testing.assert_allclose(wrapper._build_filtered_delta_action(0)[:, 0], [0.2, 0.3])
    assert wrapper.dataset.hf_dataset.batch_reads == 2
    assert wrapper.dataset.hf_dataset.scalar_reads == 0
