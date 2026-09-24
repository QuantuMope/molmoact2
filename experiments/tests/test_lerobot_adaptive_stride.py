from dataclasses import asdict
import json

import numpy as np
import pytest

from launch_scripts.train_lerobot import _apply_delta_action_stats
from olmo.data.robot_processing import RobotProcessorConfig
from olmo.hf_model.modeling_molmoact2 import _RobotStats
from olmo.piper_kinematics import adaptive_stride_offsets, piper_delta_ee, piper_fk
from .test_lerobot_delta_action import _make_skipped_wrapper


@pytest.mark.parametrize("use_hf", [True, False])
@pytest.mark.parametrize("threshold", [0., .01])
def test_adaptive_chunks_grippers_and_statistics_match_selected_transitions(use_hf, threshold):
    values = [0, .04, .045, .09, .12, .125, .3, .32, .34, .4, .5, .52]
    wrapper = _make_skipped_wrapper(
        values, use_hf=use_hf, threshold=threshold, action_stride=4,
        delta_action_mode="ee", adaptive_stride=True,
    )
    states = np.stack([wrapper.dataset[i]["observation.state"][0] for i in range(len(values))])
    moving = np.ones(len(values) - 1, dtype=bool)
    if threshold:
        moving = np.any(np.abs(np.diff(states, axis=0)) > threshold, axis=-1)
    retained = np.flatnonzero(moving)
    endpoints = np.append(retained, retained[-1] + 1)
    boundaries = adaptive_stride_offsets(piper_fk(states[endpoints]), 4, .04, .08)
    current = endpoints[boundaries[:-1]]
    following = endpoints[boundaries[1:]]
    grippers = retained[boundaries[1:] - 1]
    np.testing.assert_array_equal(wrapper._filtered_action_indices, current)
    np.testing.assert_array_equal(wrapper._filtered_next_indices, following)
    np.testing.assert_array_equal(wrapper._filtered_gripper_indices, grippers)
    np.testing.assert_array_equal(wrapper._sample_indices, current[:-1])
    assert wrapper.static_filter_static_frames == np.count_nonzero(~moving)
    assert wrapper._filtered_action_stride == 1

    raw = piper_delta_ee(states[current], states[following])
    clipped = np.any(np.abs(raw) > wrapper.delta_action_limits, axis=-1)
    ee_chunks, joint_chunks = [], []
    for sample in range(len(current) - 1):
        chunk = wrapper._build_filtered_delta_action(sample)
        expected = np.clip(raw[sample:sample + 2], -wrapper.delta_action_limits,
                           wrapper.delta_action_limits)
        expected[:, [6, 13]] = np.stack([
            wrapper.dataset[i]["action"][[6, 13]] for i in grippers[sample:sample + 2]
        ])
        np.testing.assert_allclose(chunk, expected)
        ee_chunks.append(raw[sample:sample + 2].reshape(-1, 7)[:, :6])
        joint = states[following[sample:sample + 2]] - states[current[sample:sample + 2]]
        joint_chunks.append(joint.reshape(-1, 2, 7)[..., :6].reshape(-1, 12))
    for stats, chunks in ((wrapper.delta_ee_statistics, ee_chunks),
                          (wrapper.delta_joint_statistics, joint_chunks)):
        data = np.concatenate(chunks).astype(np.float64)
        assert stats["count"] == len(data)
        np.testing.assert_allclose(stats["sum"], data.sum(axis=0), atol=1e-12)
        np.testing.assert_allclose(stats["sum_sq"], np.square(data).sum(axis=0), atol=1e-12)
        np.testing.assert_allclose(stats["abs_max"], np.abs(data).max(axis=0))
    assert wrapper.delta_action_total_chunks == len(current) - 1
    assert wrapper.delta_action_total_actions == (len(current) - 1) * 2
    assert wrapper.delta_action_clipped_actions == sum(clipped[i:i + 2].sum()
                                                      for i in range(len(current) - 1))
    assert wrapper.delta_action_clipped_chunks == sum(clipped[i:i + 2].any()
                                                     for i in range(len(current) - 1))
    assert wrapper.adaptive_stride_limit_violations == clipped.sum()
    assert all(stride == 1 for stride in np.diff(boundaries)[clipped])
    old_counts = wrapper.adaptive_stride_counts.copy()
    wrapper._build_episode_index()
    assert wrapper.adaptive_stride_counts == old_counts
    assert wrapper.adaptive_stride_limit_violations == clipped.sum()


def test_adaptive_stride_keeps_partial_tail_but_requires_full_chunks():
    kwargs = dict(action_stride=4, threshold=0, delta_action_mode="ee", adaptive_stride=True,
                  delta_ee_xyz_limit=1., delta_ee_rpy_limit=1.)
    wrapper = _make_skipped_wrapper([0, .01, .02, .03], horizon=1, **kwargs)
    assert wrapper.adaptive_stride_counts == {3: 1}
    np.testing.assert_array_equal(wrapper._filtered_next_indices, [3])
    np.testing.assert_array_equal(wrapper._filtered_gripper_indices, [2])
    with pytest.raises(ValueError, match="no anchors"):
        _make_skipped_wrapper([0, .01, .02, .03], horizon=2, **kwargs)


def test_adaptive_chunks_do_not_cross_episodes_or_dropped_tail():
    wrapper = _make_skipped_wrapper(
        [0, .03, .06, .09, .12, 1, 1.03, 1.06, 1.09, 1.12],
        episodes=[{"dataset_from_index": 0, "dataset_to_index": 5},
                  {"dataset_from_index": 5, "dataset_to_index": 10}],
        horizon=2, action_stride=4, threshold=0, drop_n_last_frames=1,
        delta_action_mode="ee", adaptive_stride=True, delta_ee_rpy_limit=.04,
    )
    for offset in wrapper._filtered_action_offsets:
        starts = wrapper._filtered_action_indices[offset:offset + 2]
        ends = wrapper._filtered_next_indices[offset:offset + 2]
        np.testing.assert_array_equal(starts // 5, ends // 5)
        assert starts[0] // 5 == starts[-1] // 5
        assert np.all(starts % 5 < 4)
        assert np.all(ends % 5 <= 4)


def test_adaptive_stride_does_not_change_joint_mode():
    kwargs = dict(action_stride=2, threshold=0)
    fixed = _make_skipped_wrapper([0, .02, .04, .06, .08, .1], **kwargs)
    adaptive = _make_skipped_wrapper([0, .02, .04, .06, .08, .1], adaptive_stride=True, **kwargs)
    assert not adaptive.adaptive_stride
    assert adaptive._sample_indices == fixed._sample_indices
    for i in range(len(fixed._sample_indices)):
        np.testing.assert_array_equal(adaptive._build_filtered_delta_action(i),
                                      fixed._build_filtered_delta_action(i))


@pytest.mark.parametrize("mode,enabled,expected", [("ee", True, True), ("ee", False, False),
                                                 ("joint", True, False)])
def test_adaptive_stride_metadata_survives_hf_export(mode, enabled, expected):
    stats = {"piper_x": {"action": {"min": [-1.] * 14, "max": [1.] * 14}}}
    metadata = {"piper_x": {"action_dim": 14, "action_key": "action"}}
    kwargs = {} if enabled else {"adaptive_stride": False}
    _apply_delta_action_stats(stats, metadata, .1, action_stride=4, delta_action_mode=mode, **kwargs)
    config = RobotProcessorConfig.from_stats(
        stats, tag_metadata=metadata, repo_to_tag={"lerobot_test": "piper_x"}, norm_mode="min_max")
    exported = _RobotStats(json.loads(json.dumps(asdict(config)))).get_metadata("piper_x")
    assert exported["adaptive_stride"] is expected
    assert exported["action_stride"] == 4
