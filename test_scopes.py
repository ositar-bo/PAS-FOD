from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
import yaml

from fedavg import weighted_average_state_dicts
from scopes import apply_scope


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.rpn = torch.nn.Linear(2, 2)
        self.roi_heads = torch.nn.Module()
        self.roi_heads.box_head = torch.nn.Linear(2, 2)
        self.roi_heads.box_predictor = torch.nn.Linear(2, 1)


class ScopeMaskTest(unittest.TestCase):
    def test_no_update_freezes_all_parameters(self):
        model = TinyDetector()
        names, count = apply_scope(model, "none")
        self.assertEqual((names, count), ([], 0))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_head_scope_only_enables_roi_detection_parameters(self):
        model = TinyDetector()
        names, count = apply_scope(model, "head")
        self.assertTrue(names)
        self.assertGreater(count, 0)
        self.assertTrue(all(n.startswith("roi_heads.") for n in names))
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("roi_heads."))

    def test_full_scope_enables_every_parameter(self):
        model = TinyDetector()
        apply_scope(model, "none")
        names, count = apply_scope(model, "full")
        self.assertEqual(names, [name for name, _ in model.named_parameters()])
        self.assertEqual(count, sum(p.numel() for p in model.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_unknown_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_scope(TinyDetector(), "invalid")

    def test_unmatched_head_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_scope(torch.nn.Linear(2, 1), "head")


class AggregationTest(unittest.TestCase):
    def test_weighted_parameters_and_integer_buffers(self):
        first = OrderedDict(weight=torch.tensor([1.0]), counter=torch.tensor(2))
        second = OrderedDict(weight=torch.tensor([5.0]), counter=torch.tensor(99))
        result = weighted_average_state_dicts([first, second], [1, 3])
        torch.testing.assert_close(result["weight"], torch.tensor([4.0]))
        self.assertEqual(result["counter"].item(), 2)


class SelectionTest(unittest.TestCase):
    def test_development_maxima_and_exact_ties(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base.yaml"
            base.write_text("experiment: {}\nhead_personalization: {}\n", encoding="utf-8")
            # Synthetic development scores used only to test the selector.
            scores = {
                "none": {"HR": 0.2, "MR": 0.3, "LR": 0.5},
                "head": {"HR": 0.5, "MR": 0.3, "LR": 0.4},
                "full": {"HR": 0.4, "MR": 0.3, "LR": 0.6},
            }
            for scope, clients in scores.items():
                (root / f"{scope}.json").write_text(json.dumps({
                    "per_view": {name: {"mAP": value} for name, value in clients.items()}
                }), encoding="utf-8")
            config = root / "selected.yaml"
            report = root / "report.json"
            command = [sys.executable, str(Path(__file__).with_name("select_adaptive_scope.py")),
                "--base-config", str(base),
                "--no-update-metrics", str(root / "none.json"),
                "--head-metrics", str(root / "head.json"),
                "--full-metrics", str(root / "full.json"),
                "--source-checkpoint", "checkpoints/model.pth",
                "--seed", "42", "--experiment-name", "pas_fod",
                "--results-dir", "results/pas_fod", "--checkpoint-dir", "checkpoints/pas_fod",
                "--output-config", str(config), "--output-report", str(report)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            decisions = {row["client"]: row["selected_scope"] for row in json.loads(report.read_text())["rows"]}
            self.assertEqual(decisions, {"HR": "head", "MR": "none", "LR": "full"})
            selected = yaml.safe_load(config.read_text())
            self.assertEqual(selected["head_personalization"]["client_trainable_prefixes"]["MR"], [])


if __name__ == "__main__":
    unittest.main()
