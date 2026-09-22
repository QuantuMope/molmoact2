from types import SimpleNamespace

import numpy as np
import pytest

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
    wrapper.action_stride = 0
    wrapper.tag_action_horizon = 1
    wrapper.delta_joint_limit = 0.1
    wrapper.gripper_indices = ()
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
    wrapper.action_stride = 0
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


def _make_skipped_wrapper(values, *, action_stride=1, horizon=2, threshold=0.01,
                          use_hf=True, episodes=None, drop_n_last_frames=0,
                          delta_joint_limit=0.1, gripper_values=None):
    states = np.zeros((len(values), 14), dtype=np.float32)
    states[:, 0] = values
    states[:, 1] = -0.2 * states[:, 0]
    if gripper_values is not None:
        states[:, 6] = gripper_values
    actions = np.zeros_like(states)
    actions[:, 6] = np.arange(len(values)) / 10
    actions[:, 13] = 1 - actions[:, 6]
    episodes = episodes or [{"dataset_from_index": 0, "dataset_to_index": len(values)}]

    class HFDataset:
        def select_columns(self, columns):
            return self

        def with_format(self, format):
            return self

        def __getitem__(self, index):
            return {"observation.state": states[index], "action": actions[index]}

    class Dataset:
        repo_id = "lerobot_test"
        meta = SimpleNamespace(episodes=episodes, total_episodes=len(episodes))

        def __len__(self):
            return len(values)

        def __getitem__(self, index):
            end = next(ep["dataset_to_index"] for ep in episodes
                       if ep["dataset_from_index"] <= index < ep["dataset_to_index"])
            return {
                "observation.state": states[[index, min(index + 1, end - 1)]],
                "action": actions[index],
            }

    dataset = Dataset()
    if use_hf:
        dataset.hf_dataset = HFDataset()
    return LeRobotDatasetWrapper(
        dataset=dataset,
        split="train",
        camera_keys=["observation.images.front"],
        camera_keys_alternative=None,
        question_key="task",
        state_keys=["observation.state"],
        action_keys=["action"],
        observation_indices=[0],
        action_indices=list(range(horizon)),
        tag_action_horizon=horizon,
        tag_n_action_steps=1,
        static_threshold=threshold,
        use_delta_action=True,
        action_stride=action_stride,
        delta_joint_limit=delta_joint_limit,
        gripper_indices=(6, 13),
        drop_n_last_frames=drop_n_last_frames,
    )


@pytest.mark.parametrize("use_hf", [True, False])
def test_action_stride_applies_after_filtering_and_clips_each_joint(use_hf):
    wrapper = _make_skipped_wrapper([0, .04, .045, .09, .12, .125, .3], action_stride=2, use_hf=use_hf)

    assert wrapper._sample_indices == [0]
    assert wrapper.static_filter_static_frames == 2
    assert wrapper.delta_action_clipped_chunks == 1
    assert wrapper.delta_action_total_chunks == 1
    action = wrapper._build_filtered_delta_action(0)
    # Filtered states are q[0], q[2], q[3], q[5], q[6]. Stride=2 selects 0 -> 3 -> 6.
    np.testing.assert_allclose(action[:, 0], [.09, .1], atol=1e-7)
    np.testing.assert_allclose(action[:, 1], [-.018, -.042], atol=1e-7)
    np.testing.assert_allclose(action[:, 6], [0, .3])
    np.testing.assert_allclose(action[:, 13], [1, .7])


@pytest.mark.parametrize("use_hf", [True, False])
def test_action_stride_without_static_filtering_keeps_stationary_transitions(use_hf):
    wrapper = _make_skipped_wrapper([0, 0, .04, .04, .08, .1], action_stride=2, threshold=0, use_hf=use_hf)

    assert wrapper._sample_indices == [0, 1]
    assert wrapper.static_filter_static_frames == 0
    np.testing.assert_allclose(wrapper._build_filtered_delta_action(0)[:, 0], [.04, .04])
    np.testing.assert_allclose(wrapper._build_filtered_delta_action(1)[:, 0], [.04, .06])


def test_action_stride_defaults_to_one_and_uses_next_retained_state():
    wrapper = _make_skipped_wrapper([0, .04, .045, .09])
    np.testing.assert_allclose(wrapper._build_filtered_delta_action(0)[:, 0], [.045, .045])


def test_action_stride_does_not_cross_episode_boundaries():
    episodes = [
        {"dataset_from_index": 0, "dataset_to_index": 5},
        {"dataset_from_index": 5, "dataset_to_index": 10},
    ]
    wrapper = _make_skipped_wrapper(
        [0, .02, .04, .06, .08, 10, 9.98, 9.96, 9.94, 9.92],
        action_stride=2,
        episodes=episodes,
    )
    assert wrapper._sample_indices == [0, 5]
    np.testing.assert_allclose(wrapper._build_filtered_delta_action(0)[:, 0], [.04, .04], atol=1e-6)
    np.testing.assert_allclose(wrapper._build_filtered_delta_action(1)[:, 0], [-.04, -.04], atol=1e-6)


def test_action_stride_rejects_episodes_without_a_full_chunk():
    with pytest.raises(ValueError, match="no anchors"):
        _make_skipped_wrapper([0, .02, .04, .06], action_stride=2, threshold=0)


def test_action_stride_respects_dropped_last_frames():
    wrapper = _make_skipped_wrapper(
        [0, .02, .04, .06, .08, .1, .12], action_stride=2, threshold=0, drop_n_last_frames=3,
    )
    assert wrapper._sample_indices == [0]
    np.testing.assert_allclose(wrapper._build_filtered_delta_action(0)[:, 0], [.04, .04])


def test_action_stride_clips_negative_deltas():
    wrapper = _make_skipped_wrapper([0, -.08, -.16, -.24, -.32], action_stride=2, threshold=0)
    action = wrapper._build_filtered_delta_action(0)
    np.testing.assert_allclose(action[:, 0], [-.1, -.1])
    np.testing.assert_allclose(action[:, 1], [.032, .032])


def test_clip_counts_are_per_chunk_not_per_action():
    wrapper = _make_skipped_wrapper([0, .01, .02, .2, .21], threshold=0)
    assert wrapper.delta_action_clipped_chunks == 2
    assert wrapper.delta_action_total_chunks == 3
    wrapper._build_episode_index()
    assert wrapper.delta_action_clipped_chunks == 2
    assert wrapper.delta_action_total_chunks == 3


def test_clip_counts_use_strided_deltas_not_intermediate_transitions():
    wrapper = _make_skipped_wrapper([0, .2, 0, 0, 0, 0], action_stride=2, threshold=0)
    assert wrapper.delta_action_clipped_chunks == 1
    assert wrapper.delta_action_total_chunks == 2


def test_clip_counts_exclude_grippers_and_deltas_exactly_at_limit():
    wrapper = _make_skipped_wrapper(
        [0, .125, .25], threshold=0, delta_joint_limit=.125, gripper_values=[0, 1, 0],
    )
    assert wrapper.delta_action_clipped_chunks == 0
    assert wrapper.delta_action_total_chunks == 1


def test_clip_counts_exclude_short_episodes_and_do_not_cross_boundaries():
    wrapper = _make_skipped_wrapper(
        [0, 1, 10, 10.01, 10.02, 10.03, 10.04],
        threshold=0,
        action_stride=2,
        episodes=[
            {"dataset_from_index": 0, "dataset_to_index": 2},
            {"dataset_from_index": 2, "dataset_to_index": 7},
        ],
    )
    assert wrapper.delta_action_clipped_chunks == 0
    assert wrapper.delta_action_total_chunks == 1


@pytest.mark.parametrize("use_hf", [True, False])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("threshold", [0, .01])
def test_clip_counts_match_raw_chunk_deltas(use_hf, stride, horizon, threshold):
    values = [0, .04, .045, .09, .12, .125, .3, .2, .1, -.1, -.11, -.115, 0, .4, .3, .1]
    wrapper = _make_skipped_wrapper(
        values, action_stride=stride, horizon=horizon, threshold=threshold, use_hf=use_hf,
    )
    clipped_chunks = 0
    for offset in wrapper._filtered_action_offsets:
        current_indices = wrapper._filtered_action_indices[offset:offset + horizon * stride:stride]
        next_indices = wrapper._filtered_next_indices[offset + stride - 1:offset + horizon * stride:stride]
        current = np.stack([wrapper.dataset[i]["observation.state"][0] for i in current_indices])
        following = np.stack([wrapper.dataset[i]["observation.state"][0] for i in next_indices])
        clipped = np.abs(following - current) > wrapper.delta_joint_limit
        clipped[:, wrapper.gripper_indices] = False
        clipped_chunks += int(np.any(clipped))
    assert wrapper.delta_action_clipped_chunks == clipped_chunks
    assert wrapper.delta_action_total_chunks == len(wrapper._sample_indices)
