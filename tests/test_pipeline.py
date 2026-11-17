import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rules import RuleIndex, retrieved_rule_ids
from smoke_tests import score_answer
from supervised import encode_example, load_datasets


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(f"<{m['role']}>{m['content']}</end>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode("utf-8"))


class PipelineTests(unittest.TestCase):
    def test_choice_parser_accepts_correct_formats_and_rejects_ambiguity(self):
        case = {"correct_choice": "D", "rules": ["302.6"]}
        for answer in ("ANSWER: D\nExplanation", "D. No\nExplanation", "D"):
            self.assertEqual(score_answer(case, answer)["status"], "correct_choice")
        self.assertEqual(score_answer(case, "ANSWER: C\nActually D")["status"], "wrong_choice")
        self.assertEqual(score_answer(case, "ANSWER: D or A")["status"], "unparsed")

    def test_supervised_labels_mask_prompt_and_keep_complete_answer(self):
        row = {"id": "example", "question": "Question?", "answer": "The ruling. [CR 100.1]",
               "rules": ["100.1"]}
        index = SimpleNamespace(by_id={"100.1": object()})
        builder = lambda question, hits, cards: [{"role": "user", "content": question}]
        tokenizer = CharacterTokenizer()
        encoded = encode_example(row, index, tokenizer, builder, [], 200)
        start = next(i for i, label in enumerate(encoded["labels"]) if label != -100)
        self.assertEqual(encoded["labels"][:start], [-100] * start)
        self.assertEqual(encoded["labels"][start:], encoded["input_ids"][start:])
        self.assertEqual(bytes(encoded["input_ids"][start:]).decode(), row["answer"] + "</end>")
        with self.assertRaisesRegex(ValueError, "preserve the full answer"):
            encode_example(row, index, tokenizer, builder, [], 10)

    def test_train_validation_overlap_is_rejected(self):
        row = {"id": "one", "question": "A unique question?", "answer": "Answer [CR 100.1]",
               "rules": ["100.1"], "cards": []}
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("train.jsonl", "val.jsonl")]
            for path in paths:
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "distinct questions"):
                load_datasets(*paths, SimpleNamespace(by_id={"100.1": object()}))

    def test_reserved_smoke_question_cannot_be_used_for_training(self):
        root = Path(__file__).resolve().parents[1]
        reserved = json.loads((root / "smoke_questions.json").read_text(encoding="utf-8"))["cases"][0]
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("train.jsonl", "val.jsonl")]
            for number, path in enumerate(paths):
                row = {"id": str(number), "question": reserved["question"] if not number else "Different question?",
                       "answer": "Answer [CR 100.1]", "rules": ["100.1"], "cards": []}
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reserved evaluation"):
                load_datasets(*paths, SimpleNamespace(by_id={"100.1": object()}))

    def test_scenario_groups_cannot_span_training_and_validation(self):
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("train.jsonl", "val.jsonl")]
            for number, path in enumerate(paths):
                row = {"id": str(number), "group": "same-scenario", "question": f"Variant number {number}?",
                       "answer": "Answer [CR 100.1]", "rules": ["100.1"], "cards": []}
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scenario group"):
                load_datasets(*paths, SimpleNamespace(by_id={"100.1": object()}))

    def test_reserved_question_is_rejected_after_choices_are_appended(self):
        root = Path(__file__).resolve().parents[1]
        reserved = json.loads((root / "smoke_questions.json").read_text(encoding="utf-8"))["cases"][0]
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("train.jsonl", "val.jsonl")]
            for number, path in enumerate(paths):
                question = reserved["question"] + "\nA. First choice\nB. Second choice" if not number else "Unrelated question?"
                row = {"id": str(number), "question": question,
                       "answer": "Answer [CR 100.1]", "rules": ["100.1"], "cards": []}
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reserved evaluation"):
                load_datasets(*paths, SimpleNamespace(by_id={"100.1": object()}))

    def test_explicit_rule_reference_is_first_with_lexical_fallback(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "passages.json"
            data = {"manifest": {}, "passages": [
                {"id": "100.1", "heading": "General", "text": "Rules apply to games."},
                {"id": "117.4", "heading": "Priority", "text": "Consecutive passes resolve the top spell."},
            ]}
            path.write_text(json.dumps(data), encoding="utf-8")
            index = RuleIndex(path)
            self.assertEqual(index.search("Explain CR 117.4")[0].id, "117.4")
            self.assertEqual(index.search("passes", limit=0), [])

    def test_semantic_cache_checks_corpus_and_deduplicates_chunks(self):
        import numpy as np
        from retrieval import SemanticIndex

        with TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npz"
            np.savez_compressed(
                path, vectors=np.array([[1., 0.], [.9, .1], [0., 1.]]),
                ids=np.array(["100.1", "100.1", "117.4"]),
                metadata=json.dumps({"source_sha256": "current", "revision": "test"}),
            )
            with self.assertRaisesRegex(ValueError, "different rules"):
                SemanticIndex(path, "changed")
            index = SemanticIndex(path, "current")
            index.encoder = SimpleNamespace(encode=lambda texts: np.array([[1., 0.]]))
            self.assertEqual(index.search("question", 2), ["100.1", "117.4"])

    def test_context_preserves_related_subrules_and_reference_provenance(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "passages.json"
            entries = [
                {"id": identifier, "heading": "Test", "text": text}
                for identifier, text in (
                    ("100.1", "The main rule."),
                    ("100.1a", "The first exception."),
                    ("100.1b", "The second exception."),
                    ("100.10", "A different rule."),
                )
            ]
            path.write_text(json.dumps({"manifest": {}, "passages": entries}), encoding="utf-8")
            index = RuleIndex(path)
            context = index.rule_context(index.by_id["100.1b"])
            self.assertEqual(set(retrieved_rule_ids([context])), {"100.1", "100.1a", "100.1b"})
            self.assertIn("[CR 100.1a] The first exception.", context.text)
            self.assertNotIn("A different rule.", context.text)

    def test_cached_glossary_alias_retrieves_operative_rule_without_network(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "passages.json"
            entries = [
                {"id": "Glossary: APNAP Order", "heading": "Glossary",
                 "text": "See Active Player, Nonactive Player Order."},
                {"id": "603.3b", "heading": "Triggered abilities",
                 "text": "Multiple triggered abilities go on the stack in APNAP order. Each player orders their own."},
                {"id": "117.4", "heading": "Priority", "text": "The top spell resolves after consecutive passes."},
            ]
            path.write_text(json.dumps({"manifest": {}, "passages": entries}), encoding="utf-8")
            with patch("rules.urlopen", side_effect=AssertionError("Lookup must use the local JSON")):
                index = RuleIndex(path)
                hits = index.search("Multiple triggered abilities trigger simultaneously. Which resolves first?")
            self.assertIn("603.3b", retrieved_rule_ids(hits))

    def test_generation_uses_cache_and_only_last_logits_when_supported(self):
        import torch
        from transformers import BatchEncoding
        from mtg import reply

        class Tokenizer:
            pad_token_id = 0

            def apply_chat_template(self, *args, **kwargs):
                return "prompt"

            def __call__(self, *args, **kwargs):
                return BatchEncoding({"input_ids": torch.tensor([[1, 2]])})

            def decode(self, tokens, **kwargs):
                return "answer"

        class Model:
            device = "cpu"

            def forward(self, input_ids=None, logits_to_keep=0):
                pass

            def generate(self, **kwargs):
                self.options = kwargs
                return torch.tensor([[1, 2, 3]])

        class OtherModel(Model):
            def forward(self, input_ids=None):
                pass

        for model, optimized in ((Model(), True), (OtherModel(), False)):
            self.assertEqual(reply(model, Tokenizer(), "Question?", [], []), "answer")
            self.assertTrue(model.options["use_cache"])
            self.assertEqual("logits_to_keep" in model.options, optimized)


if __name__ == "__main__":
    unittest.main()
