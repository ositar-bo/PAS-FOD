from __future__ import annotations

import contextlib
import gc
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from fedavg import StreamingFedAvg
from training import build_optimizer, move_targets, train_one_epoch


def clone_cpu_state(state: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in state.items())


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=device.type == "cuda")


def _matches_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def summarize_weights(weights: Mapping[str, torch.Tensor]) -> dict[str, float]:
    if not weights:
        return {
            "weight_mean": 1.0,
            "weight_min": 1.0,
            "weight_max": 1.0,
            "weight_near_zero_fraction": 0.0,
            "weight_near_one_fraction": 1.0,
        }
    total = 0
    value_sum = 0.0
    near_zero = 0
    near_one = 0
    minimum = float("inf")
    maximum = float("-inf")
    for value in weights.values():
        tensor = value.detach().float().cpu()
        count = tensor.numel()
        total += count
        value_sum += float(tensor.sum())
        near_zero += int((tensor <= 0.01).sum())
        near_one += int((tensor >= 0.99).sum())
        minimum = min(minimum, float(tensor.min()))
        maximum = max(maximum, float(tensor.max()))
    return {
        "weight_mean": value_sum / max(total, 1),
        "weight_min": minimum,
        "weight_max": maximum,
        "weight_near_zero_fraction": near_zero / max(total, 1),
        "weight_near_one_fraction": near_one / max(total, 1),
    }


class FedALAClient:
    """FedALA client adapted to a torchvision detector.

    The official FedALA update is retained: before ordinary local training, the
    client learns element-wise interpolation weights between its previous local
    parameters and the downloaded global parameters.  Instead of relying on the
    fragile positional ``layer_idx`` used by the reference classifier code, the
    detector adaptation selects its higher layers with explicit state prefixes.
    """

    def __init__(
        self,
        name: str,
        dataset,
        loader_factory: Callable[[int], Any],
        ala_loader_factory: Callable[[int, int], Any],
        model_factory: Callable[[], torch.nn.Module],
        config: dict[str, Any],
        device: torch.device,
        selected_prefixes: Sequence[str],
        eta: float = 1.0,
        rand_percent: float = 20.0,
        threshold: float = 0.01,
        num_pre_loss: int = 3,
        max_start_epochs: int = 5,
        ala_batch_size: int = 2,
        ala_amp: bool = True,
    ) -> None:
        if not 0 < float(rand_percent) <= 100:
            raise ValueError(f"rand_percent must be in (0, 100], got {rand_percent}")
        if float(eta) <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        if int(num_pre_loss) < 2:
            raise ValueError("num_pre_loss must be at least 2")
        if int(max_start_epochs) < 1:
            raise ValueError("max_start_epochs must be positive")
        if int(ala_batch_size) < 1:
            raise ValueError("ala_batch_size must be positive")
        if not selected_prefixes:
            raise ValueError("FedALA requires at least one selected parameter prefix")
        self.name = name
        self.dataset = dataset
        self.loader_factory = loader_factory
        self.ala_loader_factory = ala_loader_factory
        self.model_factory = model_factory
        self.config = config
        self.device = device
        self.selected_prefixes = tuple(selected_prefixes)
        self.eta = float(eta)
        self.rand_percent = float(rand_percent)
        self.threshold = float(threshold)
        self.num_pre_loss = int(num_pre_loss)
        self.max_start_epochs = int(max_start_epochs)
        self.ala_batch_size = int(ala_batch_size)
        self.ala_amp = bool(ala_amp)
        self.weights: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.start_phase = True

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def ala_state_dict(self) -> dict[str, Any]:
        return {
            "weights": clone_cpu_state(self.weights),
            "start_phase": bool(self.start_phase),
            "selected_prefixes": list(self.selected_prefixes),
        }

    def load_ala_state_dict(self, value: Mapping[str, Any]) -> None:
        prefixes = tuple(value.get("selected_prefixes", self.selected_prefixes))
        if prefixes != self.selected_prefixes:
            raise ValueError(
                f"FedALA prefix mismatch for {self.name}: checkpoint={prefixes}, "
                f"configured={self.selected_prefixes}"
            )
        self.weights = clone_cpu_state(value.get("weights", {}))
        self.start_phase = bool(value.get("start_phase", True))

    def _selected_parameters(
        self, model: torch.nn.Module
    ) -> OrderedDict[str, torch.nn.Parameter]:
        selected = OrderedDict(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and _matches_prefix(name, self.selected_prefixes)
        )
        if not selected:
            raise ValueError(
                f"No trainable detector parameters matched FedALA prefixes "
                f"{self.selected_prefixes}"
            )
        return selected

    def _adaptive_local_initialize(
        self,
        model: torch.nn.Module,
        global_state: Mapping[str, torch.Tensor],
        previous_local_state: Mapping[str, torch.Tensor],
        round_index: int,
    ) -> dict[str, float | int | bool]:
        selected = self._selected_parameters(model)
        previous_flags = {
            name: parameter.requires_grad for name, parameter in model.named_parameters()
        }
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in selected)

        local_values: OrderedDict[str, torch.Tensor] = OrderedDict()
        differences: OrderedDict[str, torch.Tensor] = OrderedDict()
        device_weights: OrderedDict[str, torch.Tensor] = OrderedDict()
        before_weights = clone_cpu_state(self.weights)
        with torch.no_grad():
            for name, parameter in selected.items():
                local_value = previous_local_state[name].to(
                    device=self.device, dtype=parameter.dtype, non_blocking=True
                )
                global_value = global_state[name].to(
                    device=self.device, dtype=parameter.dtype, non_blocking=True
                )
                difference = global_value - local_value
                if name not in self.weights:
                    self.weights[name] = torch.ones_like(
                        previous_local_state[name], dtype=torch.float32, device="cpu"
                    )
                weight = self.weights[name].to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                if weight.shape != parameter.shape:
                    raise ValueError(
                        f"ALA weight shape mismatch at {name}: "
                        f"{tuple(weight.shape)} != {tuple(parameter.shape)}"
                    )
                local_values[name] = local_value
                differences[name] = difference
                device_weights[name] = weight
                parameter.copy_(local_value + difference * weight.to(parameter.dtype))

        sample_count = max(1, int(math.ceil(self.rand_percent * self.num_samples / 100.0)))
        loader = self.ala_loader_factory(round_index, sample_count)
        max_epochs = self.max_start_epochs if self.start_phase else 1
        epoch_losses: list[float] = []
        backward_steps = 0
        images_seen = 0
        converged = False
        model.train()
        for _ in range(max_epochs):
            total_loss = 0.0
            epoch_steps = 0
            for images, targets in loader:
                images = [image.to(self.device, non_blocking=True) for image in images]
                targets = move_targets(targets, self.device)
                model.zero_grad(set_to_none=True)
                with _autocast(self.device, self.ala_amp and self.device.type == "cuda"):
                    loss_dict = model(images, targets)
                    loss = sum(value for value in loss_dict.values())
                loss_value = float(loss.detach().cpu())
                if not math.isfinite(loss_value):
                    detached = {
                        key: float(value.detach().cpu()) for key, value in loss_dict.items()
                    }
                    raise FloatingPointError(
                        f"Non-finite FedALA weight-learning loss {loss_value}: {detached}"
                    )
                loss.float().backward()
                with torch.no_grad():
                    for name, parameter in selected.items():
                        if parameter.grad is None:
                            raise RuntimeError(f"Missing FedALA gradient for {name}")
                        updated = device_weights[name] - self.eta * (
                            parameter.grad.detach().float() * differences[name].float()
                        )
                        device_weights[name] = updated.clamp_(0.0, 1.0)
                        parameter.copy_(
                            local_values[name]
                            + differences[name] * device_weights[name].to(parameter.dtype)
                        )
                total_loss += loss_value
                epoch_steps += 1
                backward_steps += 1
                images_seen += len(images)
            if epoch_steps == 0:
                raise RuntimeError(f"Empty FedALA weight-learning loader for {self.name}")
            epoch_losses.append(total_loss / epoch_steps)
            if (
                self.start_phase
                and len(epoch_losses) >= self.num_pre_loss
                and float(torch.tensor(epoch_losses[-self.num_pre_loss:]).std(unbiased=False))
                < self.threshold
            ):
                converged = True
                break

        del loader
        for name, weight in device_weights.items():
            self.weights[name] = weight.detach().cpu().clone()
        self.start_phase = False
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(previous_flags[name])

        delta_square = 0.0
        for name, weight in self.weights.items():
            before = before_weights.get(name)
            if before is None:
                before = torch.ones_like(weight)
            delta_square += float((weight.float() - before.float()).square().sum())
        summary = summarize_weights(self.weights)
        summary.update(
            {
                "ala_loss": epoch_losses[-1],
                "ala_epochs": len(epoch_losses),
                "ala_backward_steps": backward_steps,
                "ala_images_seen": images_seen,
                "ala_weight_delta_l2": math.sqrt(delta_square),
                "ala_converged": converged,
            }
        )
        del local_values, differences, device_weights, before_weights
        return summary

    def train(
        self,
        global_state: Mapping[str, torch.Tensor],
        previous_local_state: Mapping[str, torch.Tensor],
        local_epochs: int,
        round_index: int,
    ) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float | int | bool]]:
        model = self.model_factory()
        model.load_state_dict(global_state, strict=True)
        model.to(self.device)

        if round_index <= 1:
            ala_stats: dict[str, float | int | bool] = {
                "ala_loss": float("nan"),
                "ala_epochs": 0,
                "ala_backward_steps": 0,
                "ala_images_seen": 0,
                "ala_weight_delta_l2": 0.0,
                "ala_converged": False,
                "weight_mean": 1.0,
                "weight_min": 1.0,
                "weight_max": 1.0,
                "weight_near_zero_fraction": 0.0,
                "weight_near_one_fraction": 1.0,
            }
        else:
            ala_stats = self._adaptive_local_initialize(
                model, global_state, previous_local_state, round_index
            )

        optimizer = build_optimizer(model, self.config)
        loader = self.loader_factory(round_index * local_epochs)
        total_loss = 0.0
        total_steps = 0
        total_images = 0
        for _ in range(local_epochs):
            stats = train_one_epoch(
                model,
                loader,
                optimizer,
                self.device,
                amp=bool(self.config["training"].get("amp", False)),
                grad_clip_norm=self.config["training"].get("grad_clip_norm"),
            )
            total_loss += float(stats["loss"]) * int(stats["optimizer_steps"])
            total_steps += int(stats["optimizer_steps"])
            total_images += int(stats["images_seen"])
        del loader
        state = clone_cpu_state(model.state_dict())
        result: dict[str, float | int | bool] = {
            "loss": total_loss / max(total_steps, 1),
            "optimizer_steps": total_steps,
            "images_seen": total_images,
            **ala_stats,
        }
        del optimizer, model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return state, result


class FedALAServer:
    def __init__(
        self,
        initial_state: Mapping[str, torch.Tensor],
        clients: Sequence[FedALAClient],
    ) -> None:
        if not clients:
            raise ValueError("FedALA requires at least one client")
        self.clients = list(clients)
        self.client_names = [client.name for client in clients]
        if len(set(self.client_names)) != len(self.client_names):
            raise ValueError("FedALA client names must be unique")
        self.global_state = clone_cpu_state(initial_state)
        self.client_states = {
            client.name: clone_cpu_state(initial_state) for client in self.clients
        }
        self.round_index = 0

    def run_round(self, local_epochs: int, round_index: int) -> dict[str, Any]:
        if round_index != self.round_index + 1:
            raise ValueError(
                f"FedALA rounds must be sequential: current={self.round_index}, got={round_index}"
            )
        reference = clone_cpu_state(self.global_state)
        aggregator = StreamingFedAvg(reference)
        client_stats: dict[str, dict[str, float | int | bool]] = {}
        for client in self.clients:
            local_state, stats = client.train(
                reference,
                self.client_states[client.name],
                local_epochs,
                round_index,
            )
            self.client_states[client.name] = local_state
            aggregator.add(local_state, client.num_samples)
            client_stats[client.name] = stats
        self.global_state = aggregator.finalize()
        self.round_index = round_index
        total_steps = sum(int(item["optimizer_steps"]) for item in client_stats.values())
        total_images = sum(int(item["images_seen"]) for item in client_stats.values())
        ala_steps = sum(int(item["ala_backward_steps"]) for item in client_stats.values())
        ala_images = sum(int(item["ala_images_seen"]) for item in client_stats.values())
        total_samples = sum(client.num_samples for client in self.clients)
        weighted_loss = sum(
            float(client_stats[client.name]["loss"]) * client.num_samples
            for client in self.clients
        ) / total_samples
        active_stats = [item for item in client_stats.values() if int(item["ala_epochs"]) > 0]
        weight_mean = (
            sum(float(item["weight_mean"]) for item in active_stats) / len(active_stats)
            if active_stats
            else 1.0
        )
        return {
            "loss": weighted_loss,
            "optimizer_steps": total_steps,
            "images_seen": total_images,
            "ala_backward_steps": ala_steps,
            "ala_images_seen": ala_images,
            "weight_mean": weight_mean,
            "clients": client_stats,
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "global_state": self.global_state,
            "client_states": self.client_states,
            "client_ala_states": {
                client.name: client.ala_state_dict() for client in self.clients
            },
        }

    def load_checkpoint_state(self, value: Mapping[str, Any]) -> None:
        checkpoint_names = list(value.get("client_states", {}).keys())
        if checkpoint_names != self.client_names:
            raise ValueError(
                f"FedALA client mismatch: checkpoint={checkpoint_names}, "
                f"configured={self.client_names}"
            )
        self.round_index = int(value["round"])
        self.global_state = clone_cpu_state(value["global_state"])
        self.client_states = {
            name: clone_cpu_state(value["client_states"][name])
            for name in self.client_names
        }
        ala_states = value.get("client_ala_states", {})
        for client in self.clients:
            if client.name not in ala_states:
                raise ValueError(f"Missing FedALA state for client {client.name}")
            client.load_ala_state_dict(ala_states[client.name])
