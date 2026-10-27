import json
import tempfile
from pathlib import Path
import unittest

import torch

from common.runtime import Run, optimizer_step


class Checkpointing(unittest.TestCase):
    def train(self, output, stop, resume=False, checkpoints=(1, 2, 3)):
        torch.manual_seed(17)
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1 / (s + 1))
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        run = Run(output, {"seed": 17, "updates": 3}, [7], "test", resume)
        progress = run.restore({"m": model}, {"m": optimizer}, {"m": scheduler}, scaler)
        for index in range(progress["update"], stop):
            optimizer.zero_grad()
            loss = model(torch.randn(4, 3)).square().mean()
            loss.backward()
            norm, skipped = optimizer_step(list(model.parameters()), optimizer, scaler, 1.)
            scheduler.step()
            run.log({"update": index + 1, "loss": float(loss.detach())})
            if index + 1 in checkpoints:
                run.save({"m": model}, {"m": optimizer}, {"m": scheduler}, scaler, {"update": index + 1, "cursor": index + 1})
        return model.state_dict(), scheduler.get_last_lr()

    def test_resume_matches_uninterrupted_optimizer_scheduler_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            full, full_lr = self.train(Path(directory) / "full", 3)
            partial = Path(directory) / "resumed"
            self.train(partial, 2, checkpoints=(1,))  # update 2 is logged but never checkpointed
            resumed, lr = self.train(partial, 3, True)
            for key in full:
                torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)
            self.assertEqual(full_lr, lr)
            updates = [json.loads(line)["update"] for line in (partial / "train.jsonl").read_text().splitlines()]
            self.assertEqual(updates, [1, 2, 3])
            sizes = json.loads((partial / "run.json").read_text())["checkpoints"]
            self.assertEqual([c["update"] for c in sizes], [1, 2, 3])

    def test_resume_rejects_changed_config(self):
        with tempfile.TemporaryDirectory() as directory:
            self.train(Path(directory), 1)
            with self.assertRaises(ValueError):
                Run(directory, {"seed": 18, "updates": 3}, [7], "test", True)


if __name__ == "__main__":
    unittest.main()
