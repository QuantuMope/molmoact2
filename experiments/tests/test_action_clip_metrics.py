from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from olmo.train.trainer import (
    Trainer, action_clip_fraction_metrics, delta_ee_statistics_metrics, static_filter_ratio_metrics,
    delta_joint_statistics_metrics,
)


def test_clip_fraction_aggregates_chunk_counts_across_wrapped_datasets():
    loader = SimpleNamespace(dataset=SimpleNamespace(datasets=[
        SimpleNamespace(dataset=SimpleNamespace(delta_action_clipped_chunks=2, delta_action_total_chunks=3)),
        SimpleNamespace(delta_action_clipped_chunks=0, delta_action_total_chunks=7),
        SimpleNamespace(),
    ]))
    assert action_clip_fraction_metrics(loader) == {"train/clip_fraction": .2}


def test_clip_fraction_omits_datasets_without_delta_chunks():
    assert action_clip_fraction_metrics(None) == {}
    assert action_clip_fraction_metrics(SimpleNamespace(dataset=SimpleNamespace())) == {}
    assert action_clip_fraction_metrics(SimpleNamespace(dataset=SimpleNamespace(
        delta_action_clipped_chunks=0, delta_action_total_chunks=0,
    ))) == {}


def test_clip_fraction_reports_zero_when_nothing_is_clipped():
    loader = SimpleNamespace(dataset=SimpleNamespace(
        delta_action_clipped_chunks=0, delta_action_total_chunks=5,
    ))
    assert action_clip_fraction_metrics(loader) == {"train/clip_fraction": 0.0}


def test_static_filter_metrics_still_aggregate_counts():
    loader = SimpleNamespace(dataset=SimpleNamespace(datasets=[
        SimpleNamespace(static_filter_static_frames=2, static_filter_total_frames=4),
        SimpleNamespace(dataset=SimpleNamespace(static_filter_static_frames=1, static_filter_total_frames=6)),
    ]))
    assert static_filter_ratio_metrics("train", loader) == {"train/static_ratio": .3}


def test_clip_fraction_is_written_to_tensorboard_at_step_zero():
    loader = SimpleNamespace(dataset=SimpleNamespace(
        delta_action_clipped_chunks=2, delta_action_total_chunks=3,
        delta_action_clipped_actions=2, delta_action_total_actions=6,
    ))
    writer = Mock()
    trainer = SimpleNamespace(tensorboard_writer=writer)
    Trainer.log_metrics_to_tensorboard(trainer, action_clip_fraction_metrics(loader), step=0)

    writer.add_scalar.assert_any_call("train/clip_fraction", 2 / 3, 0)
    writer.add_scalar.assert_any_call("train/action_clip_fraction", 1 / 3, 0)
    assert writer.add_scalar.call_count == 2
    writer.flush.assert_called_once_with()


def test_action_clip_fraction_aggregates_action_counts_not_dataset_ratios():
    loader = SimpleNamespace(dataset=SimpleNamespace(datasets=[
        SimpleNamespace(dataset=SimpleNamespace(
            delta_action_clipped_actions=2, delta_action_total_actions=6,
        )),
        SimpleNamespace(delta_action_clipped_actions=1, delta_action_total_actions=14),
        SimpleNamespace(delta_action_clipped_actions=0, delta_action_total_actions=0),
        SimpleNamespace(),
    ]))
    assert action_clip_fraction_metrics(loader) == {"train/action_clip_fraction": .15}


def test_action_clip_fraction_reports_zero_and_omits_empty_datasets():
    for total in (0, 5):
        loader = SimpleNamespace(dataset=SimpleNamespace(
            delta_action_clipped_actions=0, delta_action_total_actions=total,
        ))
        assert action_clip_fraction_metrics(loader) == ({"train/action_clip_fraction": 0.} if total else {})


def test_ee_statistics_aggregate_raw_moments_and_log_all_six_axes():
    a = np.arange(12, dtype=np.float64).reshape(2, 6) / 100
    b = -np.arange(18, dtype=np.float64).reshape(3, 6) / 10

    def dataset(values):
        return SimpleNamespace(delta_ee_statistics={
            "count": len(values), "sum": values.sum(axis=0),
            "sum_sq": np.square(values).sum(axis=0), "abs_max": np.abs(values).max(axis=0),
        })

    loader = SimpleNamespace(dataset=SimpleNamespace(datasets=[
        dataset(a), SimpleNamespace(dataset=dataset(b)), SimpleNamespace(),
    ]))
    metrics = delta_ee_statistics_metrics(loader)
    raw = np.concatenate((a, b))
    assert len(metrics) == 18
    for index, axis in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        np.testing.assert_allclose(metrics[f"train/delta_ee/{axis}/mean"], raw[:, index].mean())
        np.testing.assert_allclose(metrics[f"train/delta_ee/{axis}/std"], raw[:, index].std())
        np.testing.assert_allclose(metrics[f"train/delta_ee/{axis}/abs_max"], np.abs(raw[:, index]).max())
    writer = Mock()
    Trainer.log_metrics_to_tensorboard(SimpleNamespace(tensorboard_writer=writer), metrics, 0)
    assert writer.add_scalar.call_count == 18
    assert all(call.args[2] == 0 for call in writer.add_scalar.call_args_list)


def test_ee_statistics_are_absent_for_joint_only_datasets():
    assert delta_ee_statistics_metrics(None) == {}
    assert delta_ee_statistics_metrics(SimpleNamespace(dataset=SimpleNamespace())) == {}


def test_joint_statistics_in_ee_mode_keep_all_twelve_dimensions_and_log_at_step_zero():
    a = np.arange(24, dtype=np.float64).reshape(2, 12) / 100
    b = -np.arange(36, dtype=np.float64).reshape(3, 12) / 10

    def dataset(values):
        return SimpleNamespace(delta_joint_statistics={
            "count": len(values), "sum": values.sum(axis=0),
            "sum_sq": np.square(values).sum(axis=0), "abs_max": np.abs(values).max(axis=0),
        })

    loader = SimpleNamespace(dataset=SimpleNamespace(datasets=[
        dataset(a), SimpleNamespace(dataset=dataset(b)), SimpleNamespace(),
    ]))
    metrics = delta_joint_statistics_metrics(loader)
    raw = np.concatenate((a, b))
    assert len(metrics) == 36
    for index, dim in enumerate([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]):
        np.testing.assert_allclose(metrics[f"train/delta_joint/dim_{dim}/mean"], raw[:, index].mean())
        np.testing.assert_allclose(metrics[f"train/delta_joint/dim_{dim}/std"], raw[:, index].std())
        np.testing.assert_allclose(metrics[f"train/delta_joint/dim_{dim}/abs_max"], np.abs(raw[:, index]).max())
    writer = Mock()
    Trainer.log_metrics_to_tensorboard(SimpleNamespace(tensorboard_writer=writer), metrics, 0)
    assert writer.add_scalar.call_count == 36
    assert all(call.args[2] == 0 for call in writer.add_scalar.call_args_list)


def test_joint_statistics_are_omitted_when_not_collected():
    assert delta_joint_statistics_metrics(None) == {}
    assert delta_joint_statistics_metrics(SimpleNamespace(dataset=SimpleNamespace())) == {}
    assert delta_joint_statistics_metrics(SimpleNamespace(dataset=SimpleNamespace(
        delta_joint_statistics={"count": 0},
    ))) == {}
