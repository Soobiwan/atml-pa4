"""Core utilities shared by FedAvg-style optimizers."""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import List, Sequence, Tuple, Type
import sys, os, subprocess, contextlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict


SAM_REPO = Path("external/sam")
if not SAM_REPO.exists():
    SAM_REPO.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Cloning SAM into {SAM_REPO} ...")
    subprocess.run(["git", "clone", "https://github.com/davda54/sam", str(SAM_REPO)], check=True)

if str(SAM_REPO) not in sys.path:
    sys.path.insert(0, str(SAM_REPO))



from sam import SAM

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

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
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
    *,
    grad_clip: float | None = 50.0,
) -> tuple[dict, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Run one SCAFFOLD client update and refresh control variates."""

    if lr <= 0:
        raise ValueError("Learning rate must be positive for SCAFFOLD.")

    model = model_class().to(device)
    model.load_state_dict(global_state)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    initial_params = [param.detach().clone() for param in model.parameters()]
    grad_correction = [(c_g - c_l).detach() for c_g, c_l in zip(c_global, c_local)]

    for _ in range(k_epochs):
        for images, labels in client_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            for param, corr in zip(model.parameters(), grad_correction):
                if param.grad is not None:
                    param.grad.data.add_(corr)
                    torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

    with torch.no_grad():
        y_delta = [param.detach() - init for param, init in zip(model.parameters(), initial_params)]

        coef = 1.0 / (k_epochs * lr)
        new_c_local = []
        c_delta = []
        for c_l, c_g, diff in zip(c_local, c_global, y_delta):
            c_plus = c_l - c_g - coef * diff
            c_delta.append(c_plus - c_l)
            new_c_local.append(c_plus.detach().clone())

    return copy.deepcopy(model.state_dict()), y_delta, new_c_local, c_delta

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



@torch.no_grad()
def flatten_state_dict(state_dict: dict) -> torch.Tensor:
    """
    Flattens a model's state_dict into a single 1D tensor.
    """
    # We move to CPU to avoid GPU memory issues when collecting many models
    return torch.cat([
        p.detach().cpu().flatten() for p in state_dict.values()
    ])

@torch.no_grad()
def unflatten_state_dict(flat_tensor: torch.Tensor, template_state_dict: dict) -> dict:
    """
    Unflattens a 1D tensor back into a state_dict.
    """
    new_state_dict = OrderedDict()
    current_idx = 0
    for key, param in template_state_dict.items():
        numel = param.numel()
        shape = param.shape
        # Ensure we're using the correct device, matching the template
        new_state_dict[key] = flat_tensor[current_idx : current_idx + numel].reshape(shape).to(param.device)
        current_idx += numel
    
    if current_idx != flat_tensor.numel():
        raise ValueError("Flattened tensor size does not match template state_dict")
        
    return new_state_dict

@torch.no_grad()
def server_aggregate_gh(
    global_state: dict, 
    client_states: Sequence[dict], 
    client_sizes: Sequence[int]
) -> dict:
    """
    Performs Gradient Harmonization (FedGH) before weighted averaging.
    """
    if not client_states:
        raise ValueError("client_states must be non-empty")

    # 1. Get global model vector
    global_vec = flatten_state_dict(global_state)
    M = len(client_states)

    # 2. Calculate all client deltas (g_i) as flat vectors
    # g_i = θ_i - θ_g
    client_deltas = [
        flatten_state_dict(state) - global_vec for state in client_states
    ]
    
    # 3. Perform Gradient Harmonization
    # We will modify the client_deltas list in-place
    
    # Pre-compute norms squared for efficiency
    # Add a small epsilon to prevent division by zero
    client_norms_sq = [
        torch.dot(g, g) + 1e-8 for g in client_deltas
    ]

    for i in range(M):
        for j in range(i + 1, M):
            gi = client_deltas[i]
            gj = client_deltas[j]
            
            # Compute dot product
            dot_prod = torch.dot(gi, gj)
            
            # If gi · gj < 0 (conflict)
            if dot_prod < 0:
                # Compute projection of gi onto gj
                # proj_i_on_j = (dot_prod / ||gj||^2) * gj
                proj_i = (dot_prod / client_norms_sq[j]) * gj
                
                # Compute projection of gj onto gi
                # proj_j_on_i = (dot_prod / ||gi||^2) * gi
                proj_j = (dot_prod / client_norms_sq[i]) * gi
                
                # Subtract the conflicting components
                client_deltas[i] = gi - proj_i
                client_deltas[j] = gj - proj_j
                
                # We've modified the deltas, so we must update their norms
                # for subsequent calculations in the inner loop.
                client_norms_sq[i] = torch.dot(client_deltas[i], client_deltas[i]) + 1e-8
                client_norms_sq[j] = torch.dot(client_deltas[j], client_deltas[j]) + 1e-8

    # 4. Perform weighted average on *harmonized* deltas
    total_size = float(sum(client_sizes))
    avg_harmonized_delta = torch.zeros_like(client_deltas[0])
    
    for delta, size in zip(client_deltas, client_sizes):
        avg_harmonized_delta += delta * (size / total_size)

    # 5. Apply the average delta to the global model vector
    # θ_new = θ_g + avg_delta
    new_global_vec = global_vec + avg_harmonized_delta
    
    # 6. Unflatten back into a state_dict
    return unflatten_state_dict(new_global_vec, global_state)

__all__ = [
    "client_update",
    "server_aggregate_weighted",
    "l2_divergence",
    "evaluate_model",
]
