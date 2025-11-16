"""Federated optimization method implementations."""
# pip install torch-optimizer
from __future__ import annotations

import copy
import math
import random
from typing import Callable, Dict, List, Type

import torch
import torch.nn as nn

from .core import client_update, client_update_fedprox, client_update_scaffold, client_update_gh, client_update_sam, evaluate_model, l2_divergence, server_aggregate_weighted
from .data_utils import build_client_loaders_dirichlet
    
def fed_train(
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
    method: str = "fedavg",   # <-- NEW
    mu: float = 0.1,          # <-- for FedProx
    device: torch.device | None = None,
    seed: int = 42,
    client_fraction: float = 1.0,
    initial_state: dict | None = None,
) -> Dict[str, object]:

    assert method in ["fedavg", "fedprox", "scaffold", "gh", "fedsam"]

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = model_class().to(device)
    if initial_state is not None:
        global_model.load_state_dict(copy.deepcopy(initial_state))
    else:
        initial_state = copy.deepcopy(global_model.state_dict())

    # Build clients
    client_loaders, client_sizes = build_client_loaders_dirichlet(
        train_dataset,
        num_clients=num_clients,
        alpha=alpha,
        batch_size=batch_size,
        seed=seed,
    )

    # For Scaffold
    if method == "scaffold":
        c_global = [torch.zeros_like(p) for p in global_model.parameters()]
        c_local = {
            i: [torch.zeros_like(p) for p in global_model.parameters()]
            for i in range(num_clients)
        }

    rng = random.Random(seed)
    acc_hist = []
    drift_hist = []

    min_participants = max(1, int(math.ceil(client_fraction * num_clients)))

    for r in range(num_rounds):

        round_state = copy.deepcopy(global_model.state_dict())
        selected_clients = rng.sample(range(num_clients), min_participants)
        selected_clients.sort()

        client_states = []
        selected_sizes = []

        for cid in selected_clients:

            if method == "fedavg":
                updated = client_update(
                    model_class,
                    copy.deepcopy(round_state),
                    client_loaders[cid],
                    local_epochs,
                    lr,
                    device,
                )

            elif method == "fedprox":
                updated = client_update_fedprox(
                    model_class,
                    copy.deepcopy(round_state),
                    client_loaders[cid],
                    local_epochs,
                    lr,
                    mu,
                    device,
                )

            elif method == "scaffold":
                updated, new_c = client_update_scaffold(
                    model_class,
                    copy.deepcopy(round_state),
                    client_loaders[cid],
                    local_epochs,
                    lr,
                    c_global,
                    c_local[cid],
                    device,
                )
                c_local[cid] = new_c

            elif method == "gh":
                updated = client_update_gh(
                    model_class,
                    copy.deepcopy(round_state),
                    client_loaders[cid],
                    local_epochs,
                    lr,
                    device,
                )

            elif method == "fedsam":
                updated = client_update_sam(
                    model_class,
                    copy.deepcopy(round_state),
                    client_loaders[cid],
                    local_epochs,
                    lr,
                    device,
                )

            client_states.append(updated)
            selected_sizes.append(client_sizes[cid])

        # SCAFFOLD global control variate
        if method == "scaffold":
            for i in range(len(c_global)):
                c_global[i] = sum(c_local[cid][i] for cid in selected_clients) / len(selected_clients)

        # Compute drift
        drift_hist.append(l2_divergence(round_state, client_states))

        # Aggregate
        new_global = server_aggregate_weighted(client_states, selected_sizes)
        global_model.load_state_dict(new_global)

        # Evaluate
        acc, loss = evaluate_model(global_model, test_loader, device)
        acc_hist.append(acc)
        print(f"Round {r+1}/{num_rounds} | Acc {acc:.2f}% | Drift {drift_hist[-1]:.2f}%")

    return {
        "accuracy": acc_hist,
        "drift": drift_hist,
        "final_model": global_model,
        "initial_state": initial_state,
        "client_sizes": client_sizes,
    }

FEDERATION_METHODS = {
    "fedavg": lambda **kw: fed_train(method="fedavg", **kw),
    "fedprox": lambda **kw: fed_train(method="fedprox", **kw),
    "scaffold": lambda **kw: fed_train(method="scaffold", **kw),
    "gh": lambda **kw: fed_train(method="gh", **kw),
    "fedsam": lambda **kw: fed_train(method="fedsam", **kw),
}


__all__ = ["fed_avg", "FEDERATION_METHODS"]
