#!/usr/bin/env python3
"""CLI tool to download datasets training/eval data"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from olmo.util import prepare_cli_environment


def _flatten_dataset_names(names: List) -> List[str]:
    flattened = []
    for name in names:
        if isinstance(name, tuple):
            name = name[0]
        flattened.append(str(name))
    return flattened


def _download_dataset_without_init(name: str, n_procs: int) -> bool:
    """Download datasets whose constructors require processed files to exist."""
    from olmo.data.dataset import DATA_HOME
    from olmo.data.pixmo_datasets import (
        CoSynPoint,
        PixMoAskModelAnything,
        PixMoCap,
        PixMoCapQa,
        PixMoCount,
        PixMoMultiImageCapQa,
        PixMoMultiPoints,
        PixMoPoints,
    )

    if DATA_HOME:
        os.makedirs(os.path.join(DATA_HOME, "pixmo_datasets"), exist_ok=True)
        _download_pixmo_sidecars(name, DATA_HOME)

    downloaders = {
        "cosyn_point": CoSynPoint.download,
        "pixmo_ask_model_anything": PixMoAskModelAnything.download,
        "pixmo_cap": PixMoCap.download,
        "pixmo_cap_qa": PixMoCapQa.download,
        "pixmo_cap_qa_as_user_qa": PixMoCapQa.download,
        "pixmo_count_train": PixMoCount.download,
        "pixmo_multi_image_qa": PixMoMultiImageCapQa.download,
        "pixmo_multi_image_qa_multi_only_max5": PixMoMultiImageCapQa.download,
        "pixmo_multi_points": PixMoMultiPoints.download,
        "pixmo_points_train": PixMoPoints.download,
        "pixmo_points_high_freq_train": PixMoPoints.download,
    }
    downloader = downloaders.get(name)
    if downloader is None:
        return False
    downloader(n_procs=n_procs)
    return True


def _download_pixmo_sidecars(name: str, data_home: str) -> None:
    sidecars = {
        "pixmo_multi_points": [
            ("pixmo-multi-points-meta-filtered.json", ["allenai/pixmo-multi-points", "allenai/pixmo-points"]),
        ],
        "cosyn_point": [
            ("cosyn-point-data.json", ["allenai/CoSyn-point"]),
        ],
    }
    files = sidecars.get(name)
    if not files:
        return

    from huggingface_hub import hf_hub_download

    pixmo_dir = os.path.join(data_home, "pixmo_datasets")
    for filename, repo_ids in files:
        dst = os.path.join(pixmo_dir, filename)
        if os.path.exists(dst):
            continue

        errors = []
        for repo_id in repo_ids:
            try:
                src = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.abspath(src) != os.path.abspath(dst):
                    import shutil
                    shutil.copy2(src, dst)
                logging.info("Downloaded sidecar %s from %s to %s", filename, repo_id, dst)
                break
            except Exception as exc:
                errors.append(f"{repo_id}: {exc}")
        else:
            raise FileNotFoundError(
                f"Could not download required sidecar {filename}. Tried {', '.join(errors)}"
            )


def _download_dataset_by_name(name: str, n_procs: int) -> None:
    from olmo.data.get_dataset import get_dataset_by_name

    if _download_dataset_without_init(name, n_procs=n_procs):
        return

    errors = []
    for split in ("train", "validation", "test"):
        try:
            dataset = get_dataset_by_name(name, split)
            dataset.__class__.download(n_procs=n_procs)
            return
        except (AssertionError, NotImplementedError, KeyError, ValueError) as exc:
            errors.append(f"{split}: {exc}")
    raise ValueError(f"No downloadable dataset found for {name}. Tried {', '.join(errors)}")


def download_datasets(datasets: List[str], n_procs: int = 8):
    failed_datasets = []
    for i, name in enumerate(datasets, 1):
        logging.info(f"[{i}/{len(datasets)}] Downloading dataset: {name}")
        try:
            _download_dataset_by_name(name, n_procs=n_procs)
            logging.info(f"Successfully downloaded: {name}")
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            logging.error(f"Failed to download {name}: {e}")
            failed_datasets.append((name, str(e)))

    # Summary
    logging.info("\n" + "="*60)
    logging.info(f"Download complete: {len(datasets) - len(failed_datasets)}/{len(datasets)} succeeded")

    if failed_datasets:
        logging.error("\nFailed datasets:")
        for name, error in failed_datasets:
            logging.error(f"  - {name}: {error}")
        return 1
    return 0

DATASET_GROUPS = {
    "pixmo": [
        "pixmo_cap",
        "pixmo_multi_points",
        "pixmo_points_train",
        "pixmo_count_train",
        "cosyn_point"
    ],
    "image_pointing": [
        "pixmo_multi_points",
        "pixmo_points_train",
        "pixmo_count_train",
        "cosyn_point"
    ],
    "video_pointing": [
        "vixmo_points_oversample",
        "academic_points_clip_63s_2fps"
    ],
    "video_tracking": [

        # mot
        "mevis_track",
        "ref_yt_vos_track",
        "lv_vis_track",
        "vicas_track",
        "revos_track",
        "burst_track",
        "ref_davis17_track",
        "yt_vis_track",
        "moca_track",

        "molmo2_video_track",
        "molmo_point_track_any",
        "molmo_point_track_syn",

        # sot
        "webuav_single_point_track",
        "got10k_single_point_track",
        "vasttrack_single_point_track",
        "trackingnet_single_point_track",
        "lvosv1_single_point_track",
        "lvosv2_single_point_track",
        "lasot_single_point_track",
        "uwcot_single_point_track",
        "webuot_single_point_track",
        "latot_single_point_track",
        "tnl2k_single_point_track",
        "tnllt_single_point_track",
    ],
    "demo": [
        "pixmo_ask_model_anything",
        "pixmo_cap",
        "pixmo_cap_qa_as_user_qa",
        "pixmo_multi_image_qa",
        "vixmo_human_qa",
        "vixmo3_top_level_captions_min_3"
    ],
}


def main():
    prepare_cli_environment()

    parser = argparse.ArgumentParser(
        description="Download datasets for Molmo training/evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
  all                    - Download datasets in the built-in groups below
  Individual datasets:   - Any name supported by olmo.data.get_dataset.get_dataset_by_name

Examples:
  # Download a single dataset
  python {sys.argv[0]} text_vqa

  # Download multiple datasets
  python {sys.argv[0]} text_vqa doc_qa chart_qa

  # Download all datasets
  python {sys.argv[0]} all

  # Download dataset group
  python {sys.argv[0]} video_tracking

  # Download with more parallel processes
  python {sys.argv[0]} text_vqa --n-procs 16
"""
    )

    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset name(s) to download, or 'all' for all datasets"
    )

    parser.add_argument(
        "--n-procs",
        type=int,
        default=8,
        help="Number of parallel processes to use for downloading (default: 8)"
    )

    args = parser.parse_args()
    datasets_to_download = {}  # dictionary to preserve insertion order
    for name in args.datasets:
        if name == "all":
            for group in DATASET_GROUPS.values():
                for dataset_name in _flatten_dataset_names(group):
                    datasets_to_download[dataset_name] = None
        elif name in DATASET_GROUPS:
            for dataset_name in _flatten_dataset_names(DATASET_GROUPS[name]):
                datasets_to_download[dataset_name] = None
        else:
            datasets_to_download[str(name)] = None

    return download_datasets(list(datasets_to_download.keys()), n_procs=args.n_procs)


if __name__ == "__main__":
    sys.exit(main())
