from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable
from typing import Any

import torch


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    optimizer_name = str(training.get("optimizer", "SGD")).lower()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(training.get("learning_rate", 0.005)),
            momentum=float(training.get("momentum", 0.9)),
            weight_decay=float(training.get("weight_decay", 0.0005)),
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(training.get("learning_rate", 1e-4)),
            weight_decay=float(training.get("weight_decay", 0.0005)),
        )
    raise ValueError(f"Unsupported optimizer: {training.get('optimizer')}")


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=device.type == "cuda")


def make_grad_scaler(device: torch.device, enabled: bool):
    use_amp = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def move_targets(targets: Iterable[dict[str, torch.Tensor]], device: torch.device):
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    grad_clip_norm: float | None = None,
    max_steps: int | None = None,
) -> dict[str, float | int]:
    model.train()
    scaler = make_grad_scaler(device, amp)
    total_loss = 0.0
    optimizer_steps = 0
    images_seen = 0
    for images, targets in loader:
        if max_steps is not None and optimizer_steps >= max_steps:
            break
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp and device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(value for value in loss_dict.values())
        if not math.isfinite(float(loss.detach().cpu())):
            detached = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
            raise FloatingPointError(f"Non-finite loss {float(loss.detach().cpu())}: {detached}")
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
        optimizer_steps += 1
        images_seen += len(images)
    return {
        "loss": total_loss / max(optimizer_steps, 1),
        "optimizer_steps": optimizer_steps,
        "images_seen": images_seen,
    }


def select_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

