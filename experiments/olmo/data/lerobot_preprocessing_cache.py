"""Shared caches for LeRobot frame selection and delta-action preprocessing."""

import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

log = logging.getLogger(__name__)

ARRAY_FIELDS = (
    "_sample_indices", "_episode_ranges", "_episode_cumulative",
    "_filtered_action_indices", "_filtered_next_indices", "_filtered_gripper_indices",
    "_filtered_action_offsets", "_recorded_action_lengths", "_cached_delta_actions",
)
VALUE_FIELDS = (
    "_effective_len", "_filtered_action_stride", "adaptive_stride_counts",
    "adaptive_stride_limit_violations", "static_filter_total_frames", "static_filter_static_frames",
    "delta_action_clipped_chunks", "delta_action_total_chunks",
    "delta_action_clipped_actions", "delta_action_total_actions",
    "delta_ee_statistics", "delta_joint_statistics",
)
SETTING_FIELDS = (
    "split", "state_keys", "action_keys", "observation_delta_indices", "action_delta_indices",
    "drop_n_last_frames", "static_threshold", "tag_action_horizon", "tag_n_action_steps",
    "use_delta_action", "recorded_demonstrations", "normalized_action_limits",
    "active_arm", "delta_joint_limit", "action_stride", "adaptive_stride",
    "delta_action_mode", "delta_ee_xyz_limit", "delta_ee_rpy_limit", "gripper_indices",
)


def _json(value):
    return json.dumps(value, sort_keys=True, default=lambda x: np.asarray(x).tolist())


def cache_path(wrapper):
    dataset = wrapper.dataset
    root = getattr(dataset, "root", None)
    if root is None:
        return None
    root = Path(root).resolve()
    cache_root = Path(os.environ.get(
        "LEROBOT_PREPROCESS_CACHE", root.parent / ".molmoact2_preprocessing_cache"))
    # HF fingerprints track source Parquet changes; marker timestamps also distinguish
    # generated datasets overwritten in place with the same episode/frame counts.
    files = {}
    for name in ("meta/info.json", "meta/stats.json", "meta/episode_groups.json", "RESET.json", "COMPLETE.json"):
        path = root / name
        if path.exists():
            stat = path.stat()
            files[name] = [stat.st_size, stat.st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()]
    source_dir = Path(__file__).resolve().parent
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in (
        Path(__file__), source_dir / "lerobot_wrapper.py", source_dir.parent / "piper_kinematics.py",
    ))).hexdigest()
    settings = {name: getattr(wrapper, name, None) for name in SETTING_FIELDS}
    key = dict(
        version=1, source=source_hash, root=str(root), repo_id=wrapper._get_repo_id(),
        revision=getattr(dataset, "revision", None), files=files,
        fingerprint=getattr(getattr(dataset, "hf_dataset", None), "_fingerprint", None),
        episodes=getattr(dataset, "episodes", None), rows=len(dataset), settings=settings,
    )
    digest = hashlib.sha256(_json(key).encode()).hexdigest()[:16]
    mode = "recorded" if wrapper.recorded_demonstrations else (
        wrapper.delta_action_mode if wrapper.use_delta_action else "absolute")
    name = Path(wrapper._get_repo_id()).name
    prefix = (f"{name}_{wrapper.split}_{mode}_static{wrapper.static_threshold:g}"
              f"_stride{wrapper.action_stride}_{'adaptive' if wrapper.adaptive_stride else 'fixed'}"
              f"_h{wrapper.tag_action_horizon}")
    return cache_root / f"{prefix}_{digest}.npz"


def _load(wrapper, path):
    with np.load(path, allow_pickle=False) as data:
        values = json.loads(data["metadata"].item())
        arrays = {name: data[name] if name in data else None for name in ARRAY_FIELDS}
    for name in ("_sample_indices", "_episode_cumulative"):
        if arrays[name] is not None:
            arrays[name] = arrays[name].tolist()
    if arrays["_episode_ranges"] is not None:
        arrays["_episode_ranges"] = [tuple(x) for x in arrays["_episode_ranges"].tolist()]
    if values.get("adaptive_stride_counts") is not None:
        values["adaptive_stride_counts"] = {int(k): v for k, v in values["adaptive_stride_counts"].items()}
    for name in ("delta_ee_statistics", "delta_joint_statistics"):
        if values.get(name) is not None:
            for stat in ("sum", "sum_sq", "abs_max"):
                values[name][stat] = np.asarray(values[name][stat])
    wrapper.__dict__.update(values)
    wrapper.__dict__.update(arrays)


def build_or_load(wrapper, builder):
    path = cache_path(wrapper)
    if path is None:
        builder()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Checking LeRobot preprocessing cache: %s", path)
    # File locks release on process exit, including failed builds. No NCCL collective
    # waits while another rank preprocesses, which can take longer than its timeout.
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            try:
                _load(wrapper, path)
            except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
                log.warning("Rebuilding incomplete preprocessing cache %s: %s", path, exc)
            else:
                log.info("Loaded LeRobot preprocessing cache: %s", path)
                return
        log.info("Building LeRobot preprocessing cache: %s", path)
        builder()
        arrays = {name: getattr(wrapper, name, None) for name in ARRAY_FIELDS}
        arrays = {name: np.asarray(value) for name, value in arrays.items() if value is not None}
        values = {name: getattr(wrapper, name, None) for name in VALUE_FIELDS}
        # Write directly: the cluster bucket supports truncation, but not rename/delete.
        # An interrupted NPZ has no valid footer and is rebuilt on the next startup.
        with path.open("wb") as target:
            np.savez(target, metadata=np.asarray(_json(values)), **arrays)
        log.info("Saved LeRobot preprocessing cache: %s (%0.1f MB)", path, path.stat().st_size / 1e6)
