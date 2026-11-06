import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import datasets
import numpy as np
import pytest

from olmo.data.lerobot_preprocessing_cache import build_or_load, cache_path
from olmo.data.lerobot_wrapper import LeRobotDatasetWrapper


class FrameDataset:
    repo_id = "piper_x_pick_and_place"
    revision = "main"

    def __init__(self, root, episodes=None):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)
        info = self.root / "meta/info.json"
        info.parent.mkdir(exist_ok=True)
        if not info.exists():
            info.write_text('{"total_frames": 16}')
        self.episodes = episodes
        states = np.zeros((16, 14), dtype=np.float32)
        states[:, 0] = np.linspace(0, .1, len(states))
        actions = states.copy()
        actions[:, 6] = np.linspace(0, 1, len(states))
        actions[:, 13] = 1 - actions[:, 6]
        self.hf_dataset = datasets.Dataset.from_dict({"observation.state": states, "action": actions})
        self.meta = SimpleNamespace(total_episodes=2, episodes=[
            {"dataset_from_index": 0, "dataset_to_index": 8},
            {"dataset_from_index": 8, "dataset_to_index": 16},
        ])

    def __len__(self):
        return len(self.hf_dataset)


def make_wrapper(root, **kwargs):
    options = dict(
        split="train", static_threshold=.0005, tag_action_horizon=2, tag_n_action_steps=1,
        use_delta_action=True, delta_action_mode="ee", action_stride=2, adaptive_stride=True,
        gripper_indices=(6, 13),
    )
    options.update(kwargs)
    return LeRobotDatasetWrapper(
        FrameDataset(root), camera_keys=["observation.images.front"], camera_keys_alternative=None,
        question_key="task", state_keys=["observation.state"], action_keys=["action"],
        observation_indices=[0], action_indices=list(range(options["tag_action_horizon"])), **options,
    )


@pytest.mark.parametrize("mode,stride,adaptive", [("ee", 2, True), ("ee", 3, False), ("joint", 2, False)])
def test_cache_reuses_exact_actions_and_statistics(tmp_path, monkeypatch, mode, stride, adaptive):
    root = tmp_path / "dataset"
    original = make_wrapper(root, delta_action_mode=mode, action_stride=stride, adaptive_stride=adaptive)
    cached = original._cached_delta_actions
    original._cached_delta_actions = None
    expected = [original._build_filtered_delta_action(i) for i in range(len(original))]
    original._cached_delta_actions = cached
    path = cache_path(original)
    assert path.is_file()
    assert path.name.startswith(f"piper_x_pick_and_place_train_{mode}_static0.0005_stride{stride}_")

    def fail(self):
        raise AssertionError("cache hit must not preprocess or run FK")

    monkeypatch.setattr(LeRobotDatasetWrapper, "_preprocess_episode_index", fail)
    restored = make_wrapper(root, delta_action_mode=mode, action_stride=stride, adaptive_stride=adaptive)
    assert restored._sample_indices == original._sample_indices
    assert restored.adaptive_stride_counts == original.adaptive_stride_counts
    assert restored.static_filter_static_frames == original.static_filter_static_frames
    assert restored.delta_action_clipped_actions == original.delta_action_clipped_actions
    np.testing.assert_array_equal(restored.delta_ee_statistics["sum"], original.delta_ee_statistics["sum"])
    for i, action in enumerate(expected):
        np.testing.assert_array_equal(restored._build_filtered_delta_action(i), action)


@pytest.mark.parametrize("setting,value", [
    ("static_threshold", .001), ("action_stride", 4), ("adaptive_stride", False),
    ("tag_action_horizon", 3), ("delta_ee_xyz_limit", .02), ("delta_ee_rpy_limit", .04),
    ("split", "validation"),
])
def test_cache_settings_invalidate_selection(tmp_path, setting, value):
    wrapper = make_wrapper(tmp_path / "dataset")
    original = cache_path(wrapper)
    setattr(wrapper, setting, value)
    assert cache_path(wrapper) != original


def test_dataset_rewrite_and_episode_selection_invalidate_cache(tmp_path):
    wrapper = make_wrapper(tmp_path / "dataset")
    original = cache_path(wrapper)
    wrapper.dataset.episodes = [0]
    assert cache_path(wrapper) != original
    wrapper.dataset.episodes = None
    wrapper.dataset.hf_dataset = wrapper.dataset.hf_dataset.add_column("new_version", [1] * 16)
    assert cache_path(wrapper) != original
    wrapper.dataset.hf_dataset = wrapper.dataset.hf_dataset.remove_columns("new_version")
    before = cache_path(wrapper)
    (wrapper.dataset.root / "meta/info.json").write_text('{"total_frames":16,"version":2}')
    assert cache_path(wrapper) != before


def test_partial_cache_rebuilds_without_rename_or_delete(tmp_path, monkeypatch):
    wrapper = make_wrapper(tmp_path / "dataset")
    path = cache_path(wrapper)
    path.write_bytes(b"interrupted write")

    def fail(*args, **kwargs):
        raise AssertionError("bucket writes cannot rename/delete")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail)
        patch.setattr(Path, "rename", fail)
        patch.setattr(Path, "replace", fail)
        wrapper._build_episode_index()
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["_cached_delta_actions"], wrapper._cached_delta_actions)


def test_recorded_cache_preserves_terminal_and_language_boundaries(tmp_path, monkeypatch):
    wrapper = make_wrapper(tmp_path / "dataset")
    wrapper.recorded_demonstrations = True
    wrapper.use_delta_action = False
    for key, values in {
        "action.valid": ([True] * 7 + [False]) * 2,
        "task_index": ([0] * 3 + [1] * 5) * 2,
        "episode_index": [0] * 8 + [1] * 8,
        "frame_index": list(range(8)) * 2,
    }.items():
        wrapper.dataset.hf_dataset = wrapper.dataset.hf_dataset.add_column(key, values)
    wrapper._build_episode_index()
    expected = [3, 2, 1, 4, 3, 2, 1, 0] * 2
    np.testing.assert_array_equal(wrapper._recorded_action_lengths, expected)

    def fail():
        raise AssertionError("recorded cache must not rebuild")

    monkeypatch.setattr(wrapper, "_preprocess_episode_index", fail)
    wrapper._recorded_action_lengths = None
    wrapper._build_episode_index()
    np.testing.assert_array_equal(wrapper._recorded_action_lengths, expected)
    assert len(wrapper) == 14


def concurrent_builder(root):
    # Bypass the constructor's build so the test can count actual cache misses.
    original = LeRobotDatasetWrapper._build_episode_index
    LeRobotDatasetWrapper._build_episode_index = lambda self: None
    wrapper = make_wrapper(root)
    LeRobotDatasetWrapper._build_episode_index = original

    def build():
        with (Path(root) / "builds.txt").open("a") as out:
            out.write("built\n")
        wrapper._preprocess_episode_index()

    build_or_load(wrapper, build)
    assert len(wrapper) > 0


def test_concurrent_processes_build_one_shared_cache(tmp_path):
    root = tmp_path / "dataset"
    FrameDataset(root)
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=concurrent_builder, args=(root,)) for _ in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    assert (root / "builds.txt").read_text().splitlines() == ["built"]
