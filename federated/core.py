"""Core utilities shared by FedAvg-style optimizers."""

from __future__ import annotations

import copy
from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.optim as optim


def client_update(
    model_class: Type[nn.Module],
    model_state: dict,
    client_loader,
    k_epochs: int,
    lr: float,
    device: torch.device,
    *,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> dict:
    """Perform local training for k_epochs and return the updated state_dict."""

    model = model_class().to(device)
    model.load_state_dict(model_state)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(k_epochs):
        for images, labels in client_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())


def server_aggregate_weighted(client_models: Sequence[dict], client_sizes: Sequence[int]) -> dict:
    """Weighted average of client_models using client_sizes as weights."""

    if not client_models:
        raise ValueError("client_models must be non-empty")

    total = float(sum(client_sizes))
    new_global = {}
    for key in client_models[0].keys():
        agg = torch.zeros_like(client_models[0][key])
        for state, size in zip(client_models, client_sizes):
            agg = agg + state[key] * (size / total)
        new_global[key] = agg
    return new_global


def l2_divergence(global_w: dict, client_w_list: Sequence[dict]) -> float:
    """Mean relative L2 distance between client weights and the global model."""

    if not client_w_list:
        return 0.0

    with torch.no_grad():
        global_vec = torch.cat([tensor.detach().cpu().flatten() for tensor in global_w.values()])
        global_norm = global_vec.norm() + 1e-12
        distances = []
        for client_state in client_w_list:
            client_vec = torch.cat([tensor.detach().cpu().flatten() for tensor in client_state.values()])
            distances.append((client_vec - global_vec).norm() / global_norm)
        return float(torch.mean(torch.stack(distances)))


def evaluate_model(model: nn.Module, data_loader, device: torch.device) -> Tuple[float, float]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    avg_loss = loss_sum / len(data_loader)
    return accuracy, avg_loss


__all__ = [
    "client_update",
    "server_aggregate_weighted",
    "l2_divergence",
    "evaluate_model",
]
