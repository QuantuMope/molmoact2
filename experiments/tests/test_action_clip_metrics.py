from types import SimpleNamespace
from unittest.mock import Mock

from olmo.train.trainer import Trainer, action_clip_fraction_metrics, static_filter_ratio_metrics


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
    ))
    writer = Mock()
    trainer = SimpleNamespace(tensorboard_writer=writer)
    Trainer.log_metrics_to_tensorboard(trainer, action_clip_fraction_metrics(loader), step=0)

    writer.add_scalar.assert_called_once_with("train/clip_fraction", 2 / 3, 0)
    writer.flush.assert_called_once_with()
