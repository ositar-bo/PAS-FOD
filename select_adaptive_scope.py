#!/usr/bin/env python3
"""Select no-update, detector-head, or full-model personalization per client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


HEAD_PREFIXES = ["roi_heads.box_head.", "roi_heads.box_predictor."]
SCOPE_PREFIXES = {"none": [], "head": HEAD_PREFIXES, "full": [""]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--no-update-metrics", type=Path, required=True)
    parser.add_argument("--head-metrics", type=Path, required=True)
    parser.add_argument("--full-metrics", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def read_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "per_view" not in payload:
        raise ValueError(f"Missing per_view metrics in {path}")
    return payload


def main() -> None:
    args = parse_args()
    no_update = read_metrics(args.no_update_metrics)
    head = read_metrics(args.head_metrics)
    full = read_metrics(args.full_metrics)
    clients = list(no_update["per_view"])
    if set(clients) != set(head["per_view"]) or set(clients) != set(full["per_view"]):
        raise ValueError("Development metric files contain different client sets")

    selected: dict[str, list[str]] = {}
    report_rows = []
    for client in clients:
        scores = {
            "none": float(no_update["per_view"][client]["mAP"]),
            "head": float(head["per_view"][client]["mAP"]),
            "full": float(full["per_view"][client]["mAP"]),
        }
        scope = max(scores, key=scores.get)
        selected[client] = list(SCOPE_PREFIXES[scope])
        report_rows.append(
            {
                "client": client,
                "selected_scope": scope,
                "selected_mAP": scores[scope],
                "none_mAP": scores["none"],
                "head_mAP": scores["head"],
                "full_mAP": scores["full"],
            }
        )

    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config["experiment"]["name"] = args.experiment_name
    config["experiment"]["seed"] = args.seed
    personalization = config.setdefault("head_personalization", {})
    personalization["method_name"] = "FedALA+AdaptiveScope"
    personalization["split"] = "full"
    personalization["source_checkpoint"] = str(args.source_checkpoint)
    personalization["trainable_prefixes"] = list(HEAD_PREFIXES)
    personalization["client_trainable_prefixes"] = selected
    personalization["selection_rule"] = (
        "highest per-client mAP among no-update, head-only, and full-update "
        "on deterministic grouped development validation"
    )
    config.setdefault("output", {})["results_dir"] = args.results_dir
    config["output"]["checkpoint_dir"] = args.checkpoint_dir

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    args.output_report.write_text(
        json.dumps(
            {
                "selection_metric": "COCO mAP@[.50:.95]",
                "selection_split": "grouped development validation",
                "seed": args.seed,
                "rows": report_rows,
                "selected_prefixes": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report_rows, indent=2))
    print(f"config={args.output_config}")


if __name__ == "__main__":
    main()
