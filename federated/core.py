"""Core utilities shared by FedAvg-style optimizers."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import List, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.optim as optim

from sam.sam import SAM

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

def client_update_fedprox(model_class, global_weights, loader, epochs, lr, mu, device):
    model = model_class().to(device)
    model.load_state_dict(global_weights)
    global_vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = crit(pred, y)

            local_vec = torch.nn.utils.parameters_to_vector(model.parameters())
            prox = (mu/2) * torch.norm(local_vec - global_vec)**2

            (loss + prox).backward()
            opt.step()

    return copy.deepcopy(model.state_dict())

def client_update_sam(
    model_class: type[nn.Module],
    model_state: dict,
    client_loader,
    k_epochs: int,
    lr: float,
    device: torch.device,
    rho: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> dict:

    model = model_class().to(device)
    model.load_state_dict(model_state)
    model.train()

    base_optimizer = torch.optim.SGD
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        lr=lr,
        rho=rho,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    for _ in range(k_epochs):
        for images, labels in client_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.second_step(zero_grad=True)

    return copy.deepcopy(model.state_dict())

def client_update_scaffold(
    model_class: Type[nn.Module],
    global_state: dict,
    client_loader,
    k_epochs: int,
    lr: float,
    c_global: List[torch.Tensor],
    c_local: List[torch.Tensor],
    device: torch.device,
) -> tuple[dict, List[torch.Tensor]]:

    model = model_class().to(device)
    model.load_state_dict(global_state)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # -------- Track initial global weights θ_t (for control variate update) --------
    global_params_vector = torch.nn.utils.parameters_to_vector(
        [p.detach().clone() for p in model.parameters()]
    )

    # -------------------------------------------------------------------------------
    #   Local Training Loop
    #   SCAFFOLD modifies the gradient at each step: g ← g - c_global + c_local
    # -------------------------------------------------------------------------------
    for _ in range(k_epochs):
        for images, labels in client_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()

            # === SCAFFOLD correction: add (c_local - c_global) to each gradient ===
            for p, cg, cl in zip(model.parameters(), c_global, c_local):
                if p.grad is not None:
                    p.grad.data.add_(cl - cg)

            optimizer.step()

    # -------------------------------------------------------------------------------
    #   Compute new client control variate c_i^{new}
    #   Formula from SCAFFOLD paper:
    #       c_i^{new} = c_i - c + (1 / (K * lr)) * (θ_t - θ_{t+1})
    # -------------------------------------------------------------------------------
    local_params_vector = torch.nn.utils.parameters_to_vector(
        [p.detach().clone() for p in model.parameters()]
    )

    delta = (global_params_vector - local_params_vector) / (k_epochs * lr)

    # Convert flattened Δ into per-parameter tensors
    new_c_local = []
    idx = 0
    for p, cg, cl in zip(model.parameters(), c_global, c_local):
        numel = p.numel()
        new_val = cl - cg + delta[idx:idx + numel].view_as(p)
        new_c_local.append(new_val.detach().clone())
        idx += numel

    return copy.deepcopy(model.state_dict()), new_c_local

def client_update_gh( 
    model_class: Type[nn.Module],
    model_state: dict,
    client_loader,
    k_epochs: int,
    lr: float,
    device: torch.device,
    *,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    grad_clip: float = 50.0,
) -> dict:
    """Single FedGH-style local update (train once and collect prototype statistics)."""

    model = model_class().to(device)
    model.load_state_dict(model_state)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    def _split_outputs(output):
        if isinstance(output, tuple):
            logits = output[0]
            rep = output[1] if len(output) > 1 else None
        else:
            logits, rep = output, None
        return logits, rep

    def _collect_class_prototypes():
        proto_lists = defaultdict(list)
        model.eval()
        with torch.no_grad():
            for images, labels in client_loader:
                images, labels = images.to(device), labels.to(device)
                _, reps = _split_outputs(model(images))
                if reps is None:
                    return {}

                unique_classes = labels.unique()
                for cls in unique_classes:
                    mask = labels == cls
                    cls_reps = reps[mask]
                    if cls_reps.numel() == 0:
                        continue
                    proto_lists[int(cls.item())].append(cls_reps.mean(dim=0).detach())

        prototypes = {}
        for cls, rep_list in proto_lists.items():
            if len(rep_list) == 1:
                prototypes[cls] = rep_list[0]
            else:
                prototypes[cls] = torch.stack(rep_list).mean(dim=0)
        return prototypes

    has_representation = False

    for _ in range(k_epochs):
        for images, labels in client_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, reps = _split_outputs(model(images))
            if reps is not None:
                has_representation = True

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

    if has_representation:
        _ = _collect_class_prototypes()

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
