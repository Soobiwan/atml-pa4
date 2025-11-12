"""Data loading and partition helpers for federated experiments."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def build_cifar10_datasets(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Download (if needed) and return CIFAR-10 train/test datasets."""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def get_test_loader(test_dataset: Dataset, batch_size: int = 128, num_workers: int = 2) -> DataLoader:
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _dataset_labels(dataset: Dataset) -> Sequence[int]:
    if hasattr(dataset, "targets"):
        return dataset.targets  # type: ignore[return-value]
    if hasattr(dataset, "labels"):
        return dataset.labels  # type: ignore[return-value]
    raise AttributeError("Dataset does not expose targets/labels for partitioning.")


def dirichlet_partition(labels: Sequence[int], num_clients: int, alpha: float, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    client_indices = [[] for _ in range(num_clients)]
    classes = np.unique(labels)
    for label in classes:
        label_idx = np.where(labels == label)[0]
        rng.shuffle(label_idx)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = (proportions * len(label_idx)).astype(int)
        counts[-1] += len(label_idx) - counts.sum()
        start = 0
        for client_id in range(num_clients):
            take = counts[client_id]
            if take > 0:
                client_indices[client_id].extend(label_idx[start : start + take].tolist())
            start += take
    return [np.array(sorted(indices)) for indices in client_indices]


def build_client_loaders_dirichlet(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    batch_size: int,
    *,
    seed: int = 42,
    num_workers: int = 2,
) -> Tuple[List[DataLoader], List[int]]:
    labels = _dataset_labels(dataset)
    client_indices = dirichlet_partition(labels, num_clients, alpha, seed)
    loaders: List[DataLoader] = []
    sizes: List[int] = []
    for indices in client_indices:
        subset = Subset(dataset, indices.tolist())
        loaders.append(
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        )
        sizes.append(len(indices))
    return loaders, sizes


__all__ = [
    "build_cifar10_datasets",
    "get_test_loader",
    "dirichlet_partition",
    "build_client_loaders_dirichlet",
]
