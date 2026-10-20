from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch


class StreamingFedAvg:
    """Memory-conscious weighted averaging with dtype-safe non-floating handling."""

    def __init__(self, reference: Mapping[str, torch.Tensor]) -> None:
        self.keys = list(reference.keys())
        self.reference = OrderedDict((key, tensor.detach().cpu().clone()) for key, tensor in reference.items())
        self.accumulators: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, tensor in self.reference.items():
            if tensor.is_floating_point() or tensor.is_complex():
                accumulator_dtype = torch.float32 if tensor.dtype in {torch.float16, torch.bfloat16} else tensor.dtype
                self.accumulators[key] = torch.zeros_like(tensor, dtype=accumulator_dtype, device="cpu")
        self.total_weight = 0.0

    def add(self, state: Mapping[str, torch.Tensor], weight: float) -> None:
        if list(state.keys()) != self.keys:
            missing = set(self.keys) - set(state)
            extra = set(state) - set(self.keys)
            raise ValueError(f"state_dict keys differ; missing={sorted(missing)}, extra={sorted(extra)}")
        if weight <= 0:
            raise ValueError(f"FedAvg weight must be positive, got {weight}")
        for key, reference_tensor in self.reference.items():
            tensor = state[key]
            if tensor.shape != reference_tensor.shape or tensor.dtype != reference_tensor.dtype:
                raise ValueError(
                    f"state_dict mismatch at {key}: expected {reference_tensor.shape}/{reference_tensor.dtype}, "
                    f"got {tensor.shape}/{tensor.dtype}"
                )
            if key in self.accumulators:
                self.accumulators[key].add_(tensor.detach().cpu().to(self.accumulators[key].dtype), alpha=float(weight))
        self.total_weight += float(weight)

    def finalize(self) -> OrderedDict[str, torch.Tensor]:
        if self.total_weight <= 0:
            raise RuntimeError("No client states were added")
        result: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, reference_tensor in self.reference.items():
            if key in self.accumulators:
                result[key] = (self.accumulators[key] / self.total_weight).to(reference_tensor.dtype)
            else:
                # Integer/bool buffers (for example num_batches_tracked) have no meaningful
                # weighted mean. Keeping the server/reference value is deterministic and safe.
                result[key] = reference_tensor.clone()
        return result


def weighted_average_state_dicts(
    states: list[Mapping[str, torch.Tensor]], weights: list[float]
) -> OrderedDict[str, torch.Tensor]:
    if not states or len(states) != len(weights):
        raise ValueError("states and weights must be non-empty and have the same length")
    aggregator = StreamingFedAvg(states[0])
    for state, weight in zip(states, weights):
        aggregator.add(state, weight)
    return aggregator.finalize()

