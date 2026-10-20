from __future__ import annotations

import sys
import unittest
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from fedala import FedALAClient, FedALAServer, summarize_weights


class ToyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.tensor([1.0]), {"y": torch.tensor([1.0])}


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class ToyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lower = torch.nn.Linear(1, 1, bias=False)
        self.head = torch.nn.Linear(1, 1, bias=False)

    def forward(self, images, targets):
        values = torch.stack(images)
        expected = torch.stack([target["y"] for target in targets])
        prediction = self.head(self.lower(values))
        return {"loss": (prediction - expected).square().mean()}


def make_client(name="toy"):
    dataset = ToyDataset()
    config = {
        "training": {
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "amp": False,
            "grad_clip_norm": None,
        }
    }

    def loader_factory(offset):
        return DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate)

    def ala_loader_factory(round_index, sample_count):
        return DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate)

    return FedALAClient(
        name=name,
        dataset=dataset,
        loader_factory=loader_factory,
        ala_loader_factory=ala_loader_factory,
        model_factory=ToyDetector,
        config=config,
        device=torch.device("cpu"),
        selected_prefixes=["head."],
        eta=0.1,
        rand_percent=100,
        threshold=0.01,
        num_pre_loss=2,
        max_start_epochs=1,
        ala_batch_size=2,
        ala_amp=False,
    )


class FedALAWeightTest(unittest.TestCase):
    def test_weight_summary(self):
        summary = summarize_weights(
            OrderedDict(a=torch.tensor([0.0, 0.5, 1.0]))
        )
        self.assertAlmostEqual(summary["weight_mean"], 0.5)
        self.assertAlmostEqual(summary["weight_near_zero_fraction"], 1 / 3)
        self.assertAlmostEqual(summary["weight_near_one_fraction"], 1 / 3)

    def test_higher_layer_weights_are_learned_and_bounded(self):
        client = make_client()
        model = ToyDetector()
        global_state = OrderedDict(
            (name, value.detach().clone()) for name, value in model.state_dict().items()
        )
        local_state = OrderedDict(
            (name, value.detach().clone()) for name, value in model.state_dict().items()
        )
        global_state["lower.weight"].fill_(1.0)
        global_state["head.weight"].fill_(0.0)
        local_state["lower.weight"].fill_(3.0)
        local_state["head.weight"].fill_(2.0)
        model.load_state_dict(global_state)
        stats = client._adaptive_local_initialize(
            model, global_state, local_state, round_index=2
        )
        self.assertEqual(stats["ala_epochs"], 1)
        self.assertIn("head.weight", client.weights)
        self.assertTrue(torch.isfinite(client.weights["head.weight"]).all())
        self.assertGreaterEqual(float(client.weights["head.weight"].min()), 0.0)
        self.assertLessEqual(float(client.weights["head.weight"].max()), 1.0)
        self.assertFalse(torch.equal(client.weights["head.weight"], torch.ones(1, 1)))
        torch.testing.assert_close(model.lower.weight, global_state["lower.weight"])


class FedALAServerTest(unittest.TestCase):
    def test_round_and_checkpoint_restore(self):
        clients = [make_client("a"), make_client("b")]
        initial = ToyDetector().state_dict()
        server = FedALAServer(initial, clients)
        stats = server.run_round(local_epochs=1, round_index=1)
        self.assertEqual(server.round_index, 1)
        self.assertEqual(stats["ala_backward_steps"], 0)
        restored = FedALAServer(initial, [make_client("a"), make_client("b")])
        restored.load_checkpoint_state(server.checkpoint_state())
        self.assertEqual(restored.round_index, 1)
        for key in server.global_state:
            torch.testing.assert_close(restored.global_state[key], server.global_state[key])


if __name__ == "__main__":
    unittest.main()
