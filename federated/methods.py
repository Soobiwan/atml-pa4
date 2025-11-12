"""Federated optimization method implementations."""

from __future__ import annotations

import copy
import math
import random
from typing import Callable, Dict, List, Type

import torch
import torch.nn as nn

from .core import client_update, evaluate_model, l2_divergence, server_aggregate_weighted
from .data_utils import build_client_loaders_dirichlet


def fed_avg(
    *,
    model_class: Type[nn.Module],
    train_dataset,
    test_loader,
    num_clients: int,
    alpha: float,
    batch_size: int,
    num_rounds: int,
    local_epochs: int,
    lr: float,
    client_fraction: float = 1.0,
    device: torch.device | None = None,
    seed: int = 42,
    initial_state: dict | None = None,
) -> Dict[str, object]:
    """Run the FedAvg procedure and return training diagnostics."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = model_class().to(device)
    if initial_state is not None:
        global_model.load_state_dict(copy.deepcopy(initial_state))
    else:
        initial_state = copy.deepcopy(global_model.state_dict())

    client_loaders, client_sizes = build_client_loaders_dirichlet(
        train_dataset,
        num_clients=num_clients,
        alpha=alpha,
        batch_size=batch_size,
        seed=seed,
    )

    rng = random.Random(seed)
    acc_hist: List[float] = []
    drift_hist: List[float] = []

    min_participants = max(1, int(math.ceil(client_fraction * num_clients)))

    for round_idx in range(num_rounds):
        round_state = copy.deepcopy(global_model.state_dict())
        selected_clients = rng.sample(range(num_clients), min_participants)
        selected_clients.sort()

        client_states = []
        selected_sizes = []
        for client_id in selected_clients:
            updated_state = client_update(
                model_class,
                copy.deepcopy(round_state),
                client_loaders[client_id],
                local_epochs,
                lr,
                device,
            )
            client_states.append(updated_state)
            selected_sizes.append(client_sizes[client_id])

        drift_hist.append(l2_divergence(round_state, client_states))
        new_global_state = server_aggregate_weighted(client_states, selected_sizes)
        global_model.load_state_dict(new_global_state)

        acc, loss = evaluate_model(global_model, test_loader, device)
        acc_hist.append(acc)
        print(
            f"Round {round_idx + 1:2d}/{num_rounds} | Acc {acc:6.2f}% | Drift {drift_hist[-1]:.4f}"
        )

    return {
        "accuracy": acc_hist,
        "drift": drift_hist,
        "final_model": global_model,
        "initial_state": initial_state,
        "client_sizes": client_sizes,
    }


FEDERATION_METHODS: Dict[str, Callable[..., Dict[str, object]]] = {
    "fedavg": fed_avg,
}

__all__ = ["fed_avg", "FEDERATION_METHODS"]
