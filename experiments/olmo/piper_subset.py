"""Select whole Piper-X demonstrations by arm motion, without decoding videos."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@lru_cache
def left_arm_episode_ids(dataset_root):
    columns = ["episode_index", "stats/observation.state/min", "stats/observation.state/max"]
    episodes = pq.read_table(sorted((Path(dataset_root) / "meta/episodes").rglob("*.parquet")),
                             columns=columns).to_pydict()
    ranges = np.asarray(episodes[columns[2]]) - np.asarray(episodes[columns[1]])
    # The audited dataset has >=0.67 rad active motion and <=0.011 rad idle motion.
    left_only = (ranges[:, :6].max(-1) > .1) & (ranges[:, 7:13].max(-1) <= .1)
    return sorted(np.asarray(episodes["episode_index"])[left_only].tolist())
