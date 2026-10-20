"""Parameter masks for PAS-FOD's three nested personalization scopes."""

from __future__ import annotations

from typing import Sequence

import torch

from select_adaptive_scope import SCOPE_PREFIXES


def matches_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def configure_trainable_parameters(
    model: torch.nn.Module, prefixes: Sequence[str]
) -> tuple[list[str], int]:
    selected: list[str] = []
    selected_numel = 0
    for name, parameter in model.named_parameters():
        trainable = matches_prefix(name, prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(name)
            selected_numel += parameter.numel()
    if not selected:
        raise ValueError(f"No model parameters matched prefixes {tuple(prefixes)}")
    return selected, selected_numel


def apply_scope(model: torch.nn.Module, scope: str) -> tuple[list[str], int]:
    """Apply S0 (none), SH (head), or SF (full) to a torchvision-style detector.

    For ``none``, do not construct an optimizer or perform gradient updates.
    Scope decisions must be computed on development data, not test scores.
    """
    if scope not in SCOPE_PREFIXES:
        raise ValueError(f"Unknown scope {scope!r}; expected none, head, or full")
    prefixes = SCOPE_PREFIXES[scope]
    if not prefixes:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return [], 0
    return configure_trainable_parameters(model, prefixes)
