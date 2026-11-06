import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from olmo.eval.loss_evaluator import LossDatasetEvaluator, LossMetrics


class InfiniteValidationData(IterableDataset):
    def __iter__(self):
        worker = get_worker_info()
        index = 0 if worker is None else 100 * worker.id
        while True:
            yield {
                "input_ids": torch.tensor([index % 2]),
                "labels": torch.tensor([index % 2]),
                "loss_masks": torch.ones(1),
                "sample_id": torch.tensor(index),
                "worker_pid": torch.tensor(os.getpid()),
            }
            index += 1


class ValidationModel:
    def __init__(self):
        self.metric = 0.
        self.samples = []
        self.pids = set()

    def __call__(self, input_ids, sample_id, worker_pid, **kwargs):
        self.samples.extend(sample_id.tolist())
        self.pids.update(worker_pid.tolist())
        return SimpleNamespace(
            logits=4. * F.one_hot(input_ids, num_classes=2).float(),
            labels=None,
            loss_masks=None,
            metrics={"action_mae": torch.tensor(self.metric)},
        )


def cross_entropy(logits, labels, **kwargs):
    return F.cross_entropy(logits, labels, reduction="none", ignore_index=-100), None


@pytest.mark.parametrize("num_workers", [1, 2])
def test_repeated_evaluation_reuses_workers_and_resets_samples_and_metrics(num_workers):
    loader = DataLoader(
        InfiniteValidationData(), batch_size=2, num_workers=num_workers,
        prefetch_factor=4, persistent_workers=True, multiprocessing_context="spawn",
    )
    evaluator = LossDatasetEvaluator("validation", loader, LossMetrics("cpu"), num_batches=2)
    model = ValidationModel()
    try:
        for cycle in range(20):
            model.metric = float(cycle)
            model.samples.clear()
            metrics = evaluator.run(model, "cpu", torch.bfloat16, loss_fn=cross_entropy)
            workers = loader._iterator._workers
            if cycle == 0:
                initial_pids = [worker.pid for worker in workers]
                initial_samples = model.samples.copy()
            assert [worker.pid for worker in workers] == initial_pids
            assert all(worker.is_alive() for worker in workers)
            assert model.samples == initial_samples
            assert metrics["Accuracy"] == 1.
            assert metrics["CrossEntropyLoss"] == pytest.approx(0.01814996)
            assert metrics["action_mae"] == float(cycle)
        assert model.pids == set(initial_pids)
    finally:
        evaluator.close()
    assert loader._iterator is None
    assert all(not worker.is_alive() for worker in workers)
    evaluator.close()


def distributed_evaluation(rank, rendezvous):
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    loader = DataLoader(
        InfiniteValidationData(), batch_size=2, num_workers=1,
        prefetch_factor=4, persistent_workers=True, multiprocessing_context="spawn",
    )
    evaluator = LossDatasetEvaluator("validation", loader, LossMetrics("cpu"), num_batches=2)
    model = ValidationModel()
    try:
        for cycle in range(10):
            model.metric = float(cycle + rank)
            metrics = evaluator.run(model, "cpu", torch.bfloat16, loss_fn=cross_entropy)
            assert metrics["Accuracy"] == 1.
            assert metrics["action_mae"] == cycle + .5
    finally:
        evaluator.close()
        torch.distributed.destroy_process_group()


def test_repeated_evaluation_reduces_metrics_across_ranks(tmp_path):
    torch.multiprocessing.spawn(
        distributed_evaluation, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2,
    )
