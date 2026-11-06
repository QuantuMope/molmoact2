from types import SimpleNamespace

import numpy as np
import pytest

from launch_scripts import train_lerobot
from launch_scripts.data_mixtures import TAG_METADATA_BY_TAG
from launch_scripts.lerobot_utils.stats import _apply_tag_metadata_masks, _collect_tagged_stats
from launch_scripts.lerobot_utils.train_plan import (
    _normalize_registered_lerobot_tag_metadata,
    get_lerobot_training_data_plan,
)
from olmo.data.lerobot_wrapper import LeRobotDatasetWrapper
from olmo.data.robot_processing import RobotProcessorConfig


@pytest.mark.parametrize("weights,expected", [((1., 1., 2.), [.25, .25, .5]),
                                             ((.1, .2, .1), [.25, .5, .25]),
                                             ((0., 0., 2.), [1.])])
@pytest.mark.parametrize("name", ["piper_x", "piper_x_molmo2_er", "piper_x_selected_molmo2_er"])
def test_relative_task_weights_preserve_vlm_weight(name, weights, expected):
    plan = get_lerobot_training_data_plan(
        name, pick_and_place_data_weight=weights[0], triple_cube_stack_data_weight=weights[1],
        bimanual_pot_lift_data_weight=weights[2], vlm_mixture_ratio=.3)
    assert [entry.rate for entry in plan.robot_mixture] == pytest.approx(expected)
    assert plan.robot_mixture[-1].datasets[0].dataset_name == "lerobot:piper_x_bimanual_pot_lift_v1"
    scale = 1. if name == "piper_x" else .7
    assert [entry.rate for entry in plan.combined_mixture[:len(expected)]] == pytest.approx(
        [scale * rate for rate in expected])
    if plan.vlm_mixture:
        assert plan.vlm_loader_rate == .3
        assert [entry.rate for entry in plan.combined_mixture[len(expected):]] == pytest.approx(
            [.3 * entry.rate for entry in plan.vlm_mixture])
    else:
        assert sum(entry.rate for entry in plan.combined_mixture) == pytest.approx(1.)


def test_zero_weight_sources_do_not_load_metadata(monkeypatch):
    plan = get_lerobot_training_data_plan("piper_x", pick_and_place_data_weight=0.,
                                        triple_cube_stack_data_weight=2.)
    opened = []

    def metadata(repo_id, **kwargs):
        opened.append(repo_id)
        return SimpleNamespace(features={}, stats={key: {"min": [0.] * 14, "max": [1.] * 14}
                                                   for key in ("action", "observation.state")})

    monkeypatch.setattr("launch_scripts.lerobot_utils.stats.LeRobotDatasetMetadata", metadata)
    _, repo_to_tag, _ = _collect_tagged_stats(
        plan.robot_mixture, root_base="/unused", tag_metadata_by_tag=TAG_METADATA_BY_TAG)
    assert opened == ["piper_x_triple_cube_stack_v1"]
    assert repo_to_tag == {"piper_x_triple_cube_stack_v1": "piper_x_triple_cube_stack"}


def test_default_keeps_original_dataset_and_gripper_preprocessing():
    plan = get_lerobot_training_data_plan("piper_x")
    assert len(plan.robot_mixture) == 1
    assert plan.robot_mixture[0].datasets[0].dataset_name == "lerobot:piper_x_pick_and_place"
    assert TAG_METADATA_BY_TAG["lerobot:piper_x"]["normalize_gripper"] is False


@pytest.mark.parametrize("weights", [(0., 0., 0.), (1., -1., 2.)])
def test_invalid_task_weights(weights):
    with pytest.raises(ValueError, match="data weights"):
        get_lerobot_training_data_plan(
            "piper_x", pick_and_place_data_weight=weights[0], triple_cube_stack_data_weight=weights[1],
            bimanual_pot_lift_data_weight=weights[2])


def test_launcher_passes_float_weights_to_training_plan(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_lerobot.py", "8b", "piper_x",
        "--pick_and_place_data_weight=.5", "--triple_cube_stack_data_weight=.5",
        "--bimanual_pot_lift_data_weight=1", "--use_delta_action=true", "--delta_action_mode=ee"])
    monkeypatch.setattr(train_lerobot, "prepare_torchrun_environment", lambda: None)
    captured = {}

    def capture_plan(name, **kwargs):
        plan = get_lerobot_training_data_plan(name, **kwargs)
        captured["rates"] = [entry.rate for entry in plan.robot_mixture]
        return plan

    class Ready(Exception):
        pass

    def capture_args(args):
        captured["args"] = args
        raise Ready

    monkeypatch.setattr(train_lerobot, "get_lerobot_training_data_plan", capture_plan)
    monkeypatch.setattr(train_lerobot, "_set_lerobot_environment_from_args", capture_args)
    with pytest.raises(Ready):
        train_lerobot.main()
    assert captured["rates"] == pytest.approx([.25, .25, .5])
    assert captured["args"].triple_cube_stack_data_weight == .5


@pytest.mark.parametrize("xyz_limit,rpy_limit", [(.04, .08), (.02, .16)])
def test_recorded_actions_survive_loading_and_match_real_data_units(xyz_limit, rpy_limit):
    get_lerobot_training_data_plan("piper_x", triple_cube_stack_data_weight=1.,
                                   bimanual_pot_lift_data_weight=1.)
    metadata = _normalize_registered_lerobot_tag_metadata(TAG_METADATA_BY_TAG)
    stats = {tag: {key: {"min": [-1.] * 14, "max": [1.] * 14}
                   for key in ("action", "observation.state")} for tag in metadata}
    _apply_tag_metadata_masks(stats, metadata)
    train_lerobot._apply_joint_limit_normalization_stats(stats, metadata)
    train_lerobot._apply_delta_action_stats(stats, metadata, .1, action_stride=4,
        delta_action_mode="ee", delta_ee_xyz_limit=xyz_limit, delta_ee_rpy_limit=rpy_limit)
    repo_to_tag = {"piper_x_pick_and_place": "piper_x", "piper_x_triple_cube_stack_v1": "piper_x_triple_cube_stack",
                   "piper_x_bimanual_pot_lift_v1": "piper_x_bimanual_pot_lift"}
    config = RobotProcessorConfig.from_stats(stats, tag_metadata=metadata, repo_to_tag=repo_to_tag,
                                            norm_mode="min_max")
    processor = config.build_processor()
    labels = np.linspace(-1., 1., 56, dtype=np.float32).reshape(4, 14)
    labels[:, [6, 13]] = [[-1., 1.], [0., -.5], [.5, 0.], [1., -1.]]
    state = np.zeros(14, dtype=np.float32)
    state[[6, 13]] = [.025, .05]
    real_action = labels.copy().reshape(4, 2, 7)
    real_action[..., :6] *= [.04] * 3 + [.08] * 3
    real_action[..., 6] = (real_action[..., 6] + 1.) / 2
    expected = processor.normalize_action(real_action.reshape(4, 14), "piper_x_pick_and_place")

    for repo_id in list(repo_to_tag)[1:]:
        tag = repo_to_tag[repo_id]
        entry = metadata[tag]
        assert entry["use_delta_action"] is False
        assert entry["action_stride"] == 1
        assert entry["adaptive_stride"] is False
        assert stats[tag]["observation.state"]["max"][6] == .05
        row = {"observation.state": state, "action": labels,
               "observation.images.front": np.zeros((8, 8, 3), dtype=np.uint8),
               "task": "put the pot on the tray", "index": 0}

        class Dataset:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                return row

        dataset = Dataset()
        dataset.repo_id = repo_id
        wrapper = LeRobotDatasetWrapper(
            dataset, "train", camera_keys=["observation.images.front"], camera_keys_alternative=None,
            question_key="task", state_keys=["observation.state"], action_keys=["action"],
            observation_indices=[0], action_indices=[0, 1, 2, 3], tag_action_horizon=4,
            tag_n_action_steps=4, normalized_action_limits=entry["normalized_action_limits"],
            style="robot_action", robot_processor=processor, robot_processor_config=config,
            use_annotated_task=False, sample_annotated_task=False)
        result = wrapper.get(0, np.random.default_rng(0))
        np.testing.assert_allclose(result["action"], expected, atol=2e-6)
        np.testing.assert_allclose(result["state"][[6, 13]], [0., 1.], atol=1e-6)
        np.testing.assert_array_equal(row["action"], labels)
        assert result["metadata"]["task"] == "put the pot on the tray"
    assert metadata["piper_x"]["use_delta_action"] is True
    assert metadata["piper_x"]["action_stride"] == 4
