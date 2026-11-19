"""Validated question-answer datasets with assistant-only causal LM labels."""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re

from cards import DATABASE, CardIndex, open_cards
from rules import ID_IN_TEXT


def normalize_question(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def read_examples(path, index):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not all(
            isinstance(row.get(field), str) and row[field].strip() for field in ("id", "question", "answer")
        ):
            raise ValueError(f"{path}:{line_number}: provide id, question and answer strings.")
        references = row.get("rules")
        source = row.get("source")
        sourced_context = (row.get("track") in {"commander-brackets", "full-deck-review"}
                           and isinstance(source, str) and source.strip())
        if not isinstance(references, list) or (not references and not sourced_context) or any(
            not isinstance(rule, str) or rule not in index.by_id for rule in references
        ):
            raise ValueError(f"{row['id']}: supply valid numbered rule references.")
        cited = set(ID_IN_TEXT.findall(row["answer"]))
        if not set(references) <= cited:
            raise ValueError(f"{row['id']}: the answer must cite its supporting rules.")
        row.setdefault("cards", [])
        if not isinstance(row["cards"], list) or not all(isinstance(name, str) for name in row["cards"]):
            raise ValueError(f"{row['id']}: cards must be a list of names.")
        if "group" in row and (not isinstance(row["group"], str) or not row["group"].strip()):
            raise ValueError(f"{row['id']}: group must be a nonempty string when supplied.")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no examples.")
    return rows


def load_datasets(train_file, eval_file, index):
    training = read_examples(train_file, index)
    validation = read_examples(eval_file, index)
    all_rows = training + validation
    ids = [row["id"] for row in all_rows]
    questions = [normalize_question(row["question"]) for row in all_rows]
    if len(set(ids)) != len(ids) or len(set(questions)) != len(questions):
        raise ValueError("Training and validation must have unique IDs and distinct questions.")
    train_groups = {row["group"] for row in training if "group" in row}
    validation_groups = {row["group"] for row in validation if "group" in row}
    if train_groups & validation_groups:
        raise ValueError("A scenario group appears in both training and validation.")
    reserved = set()
    root = Path(__file__).parent
    for name in ("smoke_questions.json", "smoke_deck_questions.json", "evaluation.json", "evaluation_cards.json"):
        data = json.loads((root / name).read_text(encoding="utf-8"))
        cases = data["cases"] if isinstance(data, dict) else data
        reserved.update(normalize_question(case["question"]) for case in cases)
    if any(question == heldout or question.startswith(heldout + " ")
           for question in questions for heldout in reserved):
        raise ValueError("A training or validation question duplicates a reserved evaluation question.")
    if any(row["cards"] for row in all_rows):
        if not DATABASE.is_file():
            raise ValueError("Card examples need cards.py prepare first.")
        with open_cards() as db:
            cards = CardIndex(db)
            for row in all_rows:
                if row["cards"]:
                    cards.for_question(row["question"], row["cards"])
    print(f"Validated {len(training)} training and {len(validation)} validation examples.")
    return training, validation


def chat_template_options(tokenizer):
    template = getattr(tokenizer, "chat_template", None)
    return {"enable_thinking": False} if isinstance(template, str) and "enable_thinking" in template else {}


def encode_example(row, index, tokenizer, message_builder, cards, max_length):
    # Teacher evidence is curated in the training data. Evaluation retrieval never
    # receives an expected answer or reference list.
    hits = [index.by_id[reference] for reference in row["rules"]]
    messages = message_builder(row["question"], hits, cards)
    options = chat_template_options(tokenizer)
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **options)
    assistant = {"role": "assistant", "content": row["answer"]}
    full = tokenizer.apply_chat_template(
        messages + [assistant], tokenize=True, add_generation_prompt=False, **options,
    )
    if full[:len(prompt)] != prompt and [message["role"] for message in messages] == ["system", "user"]:
        # NeMo's template includes the system text only when the user turn is last.
        # Fold it into that turn and verify the result against the inference prompt.
        user = {"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}
        full = tokenizer.apply_chat_template([user, assistant], tokenize=True, add_generation_prompt=False, **options)
    if full[:len(prompt)] != prompt:
        raise ValueError(f"{row['id']}: the chat template's training and generation prefixes differ.")
    if len(full) > max_length:
        raise ValueError(f"{row['id']} needs {len(full)} tokens. Increase --length to preserve the full answer.")
    if len(full) <= len(prompt):
        raise ValueError(f"{row['id']} has no assistant tokens.")
    return {
        "input_ids": full, "attention_mask": [1] * len(full),
        "labels": [-100] * len(prompt) + full[len(prompt):],
    }


def prepare_datasets(args, index, tokenizer, message_builder):
    training, validation = load_datasets(args.train_file, args.eval_file, index)
    datasets = []
    with open_cards() if DATABASE.is_file() else nullcontext() as db:
        card_index = CardIndex(db) if db else None
        for rows in (training, validation):
            encoded = []
            for row in rows:
                cards = card_index.for_question(row["question"], row["cards"]) if row["cards"] else []
                encoded.append(encode_example(row, index, tokenizer, message_builder, cards, args.length))
            datasets.append(encoded)
    metadata = {
        "train_sha256": hashlib.sha256(args.train_file.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(args.eval_file.read_bytes()).hexdigest(),
        "training_examples": len(training), "validation_examples": len(validation),
        "assistant_tokens": sum(sum(token != -100 for token in row["labels"]) for row in datasets[0]),
        "label_masking": "assistant answer and end-of-message tokens only",
        "cards": card_index.manifest if card_index and any(row["cards"] for row in training) else None,
    }
    return datasets[0], datasets[1], metadata
