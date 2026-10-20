# PAS-FOD

**Perspective-Adaptive Scope Personalization for Federated Object Detection in Heterogeneous UAV Networks**

Core implementation of PAS-FOD. The method combines client-specific global-local
knowledge fusion with development-set-based personalization-scope selection.

## Core modules

- `fedala.py`: adaptive global-local parameter fusion, client updates, and server aggregation.
- `scopes.py`: the three nested scopes: **S0** (no update), **SH** (ROI head only), and **SF** (full detector).
- `select_adaptive_scope.py`: per-client scope selection from development-set mAP.
- `fedavg.py` and `training.py`: weighted aggregation and local optimization utilities.

The fusion stage uses adaptive local aggregation; the client/server classes retain
the `FedALAClient` and `FedALAServer` API names. This release contains the method's
core modules, not the full benchmark pipeline, datasets, or pretrained checkpoints.

## Installation and tests

Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python -m unittest -v
```

`test_fedala.py` provides a small CPU-only example of fusion-weight learning,
federated rounds, and checkpoint restoration. No dataset download is required.

## Personalization scopes

```python
from scopes import apply_scope

# model: a detector with torchvision Faster R-CNN parameter names.
# selected_scope: "none", "head", or "full", selected on development data.
names, parameter_count = apply_scope(model, selected_scope)
if names:
    # Construct the optimizer using parameters with requires_grad=True,
    # then adapt on the client's local training partition.
    pass
# For "none", retain the model without constructing an optimizer or updating it.
```

`SH` activates `roi_heads.box_head.*` and `roi_heads.box_predictor.*`.
`SF` activates all model parameters. Scope selection maximizes each client's
development mAP over `none`, `head`, and `full`; exact ties follow this order,
favoring the smaller scope. The released selector uses exact maxima, without a
nonzero near-tie tolerance.

The selection CLI accepts three JSON files, each with the schema
`{"per_view": {"HR": {"mAP": ...}, ...}}`, plus a base YAML configuration.
Run `python select_adaptive_scope.py --help` for all arguments. It writes the
selected parameter prefixes and a per-client decision report.

Scope decisions must use an independent development partition. Test-set curves
or final test scores must not determine the selected scope, training duration,
or checkpoint. Personalization remains client-local and does not add a server
communication round.

## License

MIT License. See `LICENSE`.
