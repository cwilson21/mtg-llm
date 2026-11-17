import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import EpochArtifacts, lora_options


class TrainingTests(unittest.TestCase):
    def args(self, **overrides):
        return SimpleNamespace(**({"command": "sft", "from_adapter": None, "fresh_base": True,
                                   "lora_rank": None, "lora_alpha": None, "lora_targets": None} | overrides))

    def test_fresh_sft_uses_rank32_across_linear_layers(self):
        self.assertEqual(lora_options(self.args()),
                         {"r": 32, "lora_alpha": 64, "target_modules": "all-linear"})
        self.assertEqual(lora_options(self.args(command="train"))["target_modules"], ["q_proj", "v_proj"])

    def test_adapter_cannot_silently_override_requested_rank(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            lora_options(self.args(from_adapter=Path("old")))
        with self.assertRaisesRegex(ValueError, "retains its rank"):
            lora_options(self.args(from_adapter=Path("old"), fresh_base=False, lora_rank=32))
        self.assertIsNone(lora_options(self.args(from_adapter=Path("old"), fresh_base=False)))
        with self.assertRaisesRegex(ValueError, "positive"):
            lora_options(self.args(lora_rank=0))

    def test_each_checkpoint_keeps_its_own_provenance_and_metrics(self):
        with TemporaryDirectory() as temporary:
            args = SimpleNamespace(output_dir=temporary)
            record = {"base_model": "example/base", "revision": "revision", "parent_adapter": None,
                      "lora": {"rank": 32}, "rules": {"corpus_sha256": "sha"}}
            callback = EpochArtifacts(record)
            state = SimpleNamespace(global_step=0, epoch=0, log_history=[], is_world_process_zero=True)
            for epoch in (1, 2):
                state.global_step = epoch * 10
                state.epoch = epoch
                state.log_history.append({"eval_loss": 1 / epoch, "epoch": epoch})
                checkpoint = Path(temporary) / f"checkpoint-{state.global_step}"
                checkpoint.mkdir()
                callback.on_evaluate(args, state, None)
                callback.on_save(args, state, None)
                saved = json.loads((checkpoint / "run.json").read_text())
                self.assertEqual(saved["optimizer_steps"], epoch * 10)
                self.assertEqual(saved["validation_metrics"]["epoch"], epoch)
                self.assertEqual(saved["base_model"], "example/base")
            first = json.loads((Path(temporary) / "checkpoint-10/run.json").read_text())
            self.assertEqual(len(first["epoch_evaluations"]), 1)
            self.assertEqual(len(json.loads((Path(temporary) / "epoch_metrics.json").read_text())), 2)

    def test_multiple_choice_targets_match_choices_in_expanded_curriculum(self):
        root = Path(__file__).resolve().parents[1]
        for split in ("train", "validation"):
            for line in (root / "qa" / f"rank32-{split}.jsonl").read_text(encoding="utf8").splitlines():
                row = json.loads(line)
                if "choices" in row:
                    self.assertTrue(row["answer"].startswith(f"ANSWER: {row['correct_choice']}\n"), row["id"])
                    self.assertIn(row["choices"][row["correct_choice"]], row["answer"], row["id"])


if __name__ == "__main__":
    unittest.main()
