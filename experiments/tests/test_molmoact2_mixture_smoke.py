import pytest

from launch_scripts import train_lerobot
from launch_scripts.data_mixtures import TAG_METADATA_BY_TAG
from launch_scripts.lerobot_utils.train_plan import (
    _normalize_registered_lerobot_tag_metadata,
    _override_action_chunk_size,
    _validate_lerobot_tag_temporal_metadata,
    infer_max_action_dim_from_lerobot_metadata,
    infer_max_action_horizon_from_lerobot_metadata,
)
from launch_scripts.train_lerobot import get_lerobot_training_data_plan


def _rate_sum(mixture):
    return sum(float(item.rate) for item in mixture)


def test_requested_lerobot_mixtures_build_and_are_normalized():
    for name in (
        "pre_post_train",
        "droid",
        "libero",
        "libero_goal",
        "yam",
        "so100_so101",
    ):
        plan = get_lerobot_training_data_plan(name)
        assert plan.combined_mixture
        assert _rate_sum(plan.combined_mixture) == pytest.approx(1.0)


def test_removed_lerobot_mixture_names_are_rejected():
    for name in ("pretrain", "pretrain_v2"):
        with pytest.raises(NotImplementedError):
            get_lerobot_training_data_plan(name)


@pytest.mark.parametrize(
    ("name", "expected_max_action_horizon"),
    [
        ("libero_goal", 10),
        ("droid", 15),
        ("pre_post_train", 30),
    ],
)
def test_max_action_horizon_is_inferred_from_mixture_metadata(name, expected_max_action_horizon):
    plan = get_lerobot_training_data_plan(name)

    assert (
        infer_max_action_horizon_from_lerobot_metadata(
            plan.robot_mixture,
            tag_metadata_by_tag=TAG_METADATA_BY_TAG,
        )
        == expected_max_action_horizon
    )


def test_max_action_dim_is_inferred_from_mixture_metadata():
    plan = get_lerobot_training_data_plan("pre_post_train")

    assert (
        infer_max_action_dim_from_lerobot_metadata(
            plan.robot_mixture,
            tag_metadata_by_tag=TAG_METADATA_BY_TAG,
        )
        == 32
    )


@pytest.mark.parametrize("chunk_size", [1, 8, 64])
def test_action_chunk_size_overrides_mixture_horizons_without_mutating_registry(chunk_size):
    plan = get_lerobot_training_data_plan("pre_post_train")
    original = _normalize_registered_lerobot_tag_metadata(TAG_METADATA_BY_TAG)
    metadata = _override_action_chunk_size(TAG_METADATA_BY_TAG, chunk_size)

    assert infer_max_action_horizon_from_lerobot_metadata(
        plan.robot_mixture, tag_metadata_by_tag=metadata,
    ) == chunk_size
    _validate_lerobot_tag_temporal_metadata(
        plan.robot_mixture, tag_metadata_by_tag=metadata, max_action_horizon=chunk_size,
    )
    for tag, entry in _normalize_registered_lerobot_tag_metadata(metadata).items():
        assert entry["action_horizon"] == chunk_size
        assert entry["n_action_steps"] == min(original[tag]["n_action_steps"], chunk_size)
        assert entry is not original[tag]
    assert _normalize_registered_lerobot_tag_metadata(TAG_METADATA_BY_TAG) == original


@pytest.mark.parametrize("chunk_size, action_stride", [(None, 1), (1, 2), (64, 3)])
@pytest.mark.parametrize("delta_action_mode", ["joint", "ee"])
def test_launcher_parses_action_chunk_and_stride(monkeypatch, chunk_size, action_stride, delta_action_mode):
    argv = ["train_lerobot.py", "8b", "piper_x", "--use_delta_action=true"]
    if delta_action_mode == "ee":
        argv.extend(["--delta_action_mode=ee", "--delta_joint_limit=-1",
                     "--delta_ee_xyz_limit=0.02", "--delta_ee_rpy_limit=0.06"])
    if chunk_size is not None:
        argv.extend([f"--action_chunk_size={chunk_size}", f"--action_stride={action_stride}"])
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(train_lerobot, "prepare_torchrun_environment", lambda: None)
    captured = {}

    class ConfigReady(Exception):
        pass

    def capture_args(args):
        captured.update(vars(args))
        raise ConfigReady

    monkeypatch.setattr(train_lerobot, "_set_lerobot_environment_from_args", capture_args)
    with pytest.raises(ConfigReady):
        train_lerobot.main()

    plan = get_lerobot_training_data_plan("piper_x")
    expected_horizon = chunk_size or infer_max_action_horizon_from_lerobot_metadata(
        plan.robot_mixture, tag_metadata_by_tag=TAG_METADATA_BY_TAG,
    )
    assert captured["action_chunk_size"] == chunk_size
    assert captured["action_stride"] == action_stride
    assert captured["delta_action_mode"] == delta_action_mode
    assert captured["delta_ee_xyz_limit"] == (.02 if delta_action_mode == "ee" else .04)
    assert captured["delta_ee_rpy_limit"] == (.06 if delta_action_mode == "ee" else .08)
    assert captured["max_action_horizon"] == expected_horizon


def test_delta_action_metadata_records_stride_and_keeps_joint_limit():
    metadata = {"piper_x": {"action_dim": 14, "action_horizon": 8}}
    stats = {"piper_x": {"action": {"min": [0.] * 14, "max": [1.] * 14}}}

    train_lerobot._apply_delta_action_stats(stats, metadata, 0.1, action_stride=3)

    assert metadata["piper_x"]["action_stride"] == 3
    assert metadata["piper_x"]["action_horizon"] == 8
    assert metadata["piper_x"]["delta_joint_limit"] == 0.1
    assert stats["piper_x"]["action"]["min"][0] == -0.1
    assert stats["piper_x"]["action"]["max"][0] == 0.1
    assert stats["piper_x"]["action"]["max"][6] == 1.
