"""Fine-tune a Qwen model on the rules, ask questions, and compare answers."""

import argparse
from contextlib import nullcontext
import hashlib
from itertools import batched
import json
import os
from pathlib import Path
import sqlite3
import sys

from cards import DATABASE, CardIndex, format_card, open_cards
from mana import MANA_GUIDE, computed_costs
from rules import DATA, RuleIndex, format_passage, write_text


os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM = (
    "Answer Magic: The Gathering rules questions. Use the provided rules passages and card Oracle text. "
    "State the outcome, explain the relevant steps, and cite supporting numbered "
    "rules as [CR 117.4]. If the passages or facts are insufficient, say what "
    "information is missing. Do not invent rules or card text."
)


def doctor():
    import accelerate
    import peft
    import torch
    import transformers

    print(f"Python {sys.version.split()[0]}")
    print(f"PyTorch {torch.__version__}, CUDA build {torch.version.cuda}")
    print(f"Transformers {transformers.__version__}, PEFT {peft.__version__}, Accelerate {accelerate.__version__}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"{torch.cuda.get_device_name()}: {free / 2**30:.1f} / {total / 2**30:.1f} GiB free")
        print(f"Native bfloat16: {torch.cuda.is_bf16_supported(including_emulation=False)}")
    else:
        print("CUDA unavailable. CPU training requires --device cpu and is slower.")


def runtime(requested):
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Run setup.cmd and check doctor, or use --device cpu.")
    bf16 = requested == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    return torch.device(requested), torch.bfloat16 if bf16 else torch.float32


def load_base(model_id, dtype, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype, use_safetensors=True,
    )
    return model, tokenizer


def token_blocks(tokens, size):
    # A shared boundary token lets causal LM loss score each subsequent token once.
    return [
        tokens[start:start + size]
        for start in range(0, len(tokens) - 1, size - 1)
    ]


class Blocks:
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        tokens = self.blocks[index]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}


def train(args):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed

    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError(f"{args.output} already contains files. Choose a fresh --output directory.")
    if args.with_cards and not DATABASE.is_file():
        raise ValueError("Card index is missing. Run cards.py prepare first.")
    index = RuleIndex()
    raw = (DATA / "corpus.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != index.manifest["corpus_sha256"]:
        raise ValueError("Corpus and search index differ. Run rules.py prepare again.")
    device, dtype = runtime(args.device)
    set_seed(42)
    parent = None
    if args.from_adapter:
        parent = json.loads((args.from_adapter / "run.json").read_text(encoding="utf-8"))
    base_model = parent["base_model"] if parent else args.model
    model, tokenizer = load_base(base_model, dtype, parent["revision"] if parent else None)
    revision = model.config._commit_hash
    model.config.use_cache = False
    token_ids = tokenizer(raw.decode("utf-8"), add_special_tokens=False, verbose=False)["input_ids"]
    token_ids.append(tokenizer.eos_token_id)
    rule_blocks = token_blocks(token_ids, args.length)
    mana_tokens = tokenizer(MANA_GUIDE, add_special_tokens=False, verbose=False)["input_ids"]
    mana_tokens.append(tokenizer.eos_token_id)
    mana_blocks = token_blocks(mana_tokens, args.length)
    card_manifest = None
    card_tokens = 0
    if args.with_cards:
        with open_cards() as conn:
            cards = CardIndex(conn)
            card_manifest = cards.manifest
            card_blocks = []
            delimiter = tokenizer.eos_token
            for batch in batched(cards.training_texts(), 128):
                encoded = tokenizer(
                    delimiter.join(batch) + delimiter, add_special_tokens=False, verbose=False,
                )["input_ids"]
                card_tokens += len(encoded)
                card_blocks.extend(token_blocks(encoded, args.length))
        # Include every rule block and replay it as needed so the model keeps
        # seeing rules during the much larger card stage.
        replay_count = max(len(rule_blocks), (len(card_blocks) + 2) // 3)
        blocks = Blocks(card_blocks + [rule_blocks[i % len(rule_blocks)] for i in range(replay_count)])
        print(f"Card stage: {len(card_blocks):,} card blocks and {replay_count:,} rules blocks")
    else:
        blocks = Blocks(rule_blocks)
    # Repeat this concise reference throughout either training stage.
    mana_replays = max(len(mana_blocks), (len(blocks) + 19) // 20)
    blocks.blocks.extend(mana_blocks[i % len(mana_blocks)] for i in range(mana_replays))
    print(f"{len(token_ids):,} source rules tokens; {card_tokens:,} source card tokens")
    if parent:
        model = PeftModel.from_pretrained(model, args.from_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ))
    model.print_trainable_parameters()
    settings = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        max_steps=args.steps,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16,
        use_cpu=device.type == "cpu",
        optim="adamw_torch",
        dataloader_num_workers=0,
        dataloader_pin_memory=device.type == "cuda",
        logging_steps=10 if args.steps < 0 else 1,
        save_strategy="no",
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model, args=settings, train_dataset=blocks, processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    write_text(args.output / "run.json", json.dumps({
        "base_model": base_model, "revision": revision, "rules": index.manifest,
        "cards": card_manifest, "parent_adapter": str(args.from_adapter) if parent else None,
        "tokens": len(token_ids) + card_tokens + len(mana_tokens), "sequence_length": args.length,
        "mana_reference_sha256": hashlib.sha256(MANA_GUIDE.encode("utf-8")).hexdigest(),
        "optimizer_steps": trainer.state.global_step,
    }, indent=2))
    print(f"Adapter saved to {args.output.resolve()}")


def load_for_answers(args):
    from peft import PeftModel

    device, dtype = runtime(args.device)
    if args.adapter:
        record = json.loads((args.adapter / "run.json").read_text(encoding="utf-8"))
        model, tokenizer = load_base(record["base_model"], dtype, record["revision"])
        model = PeftModel.from_pretrained(model, args.adapter)
    else:
        record = None
        model, tokenizer = load_base(args.model, dtype)
    return model.to(device).eval(), tokenizer, record


def reply(model, tokenizer, question, hits, cards):
    import torch

    excerpts = "\n\n".join(map(format_passage, hits)) or "(No matching passages.)"
    messages = [
        {"role": "system", "content": SYSTEM + "\n\n" + MANA_GUIDE},
        {"role": "user", "content": f"Rules passages:\n{excerpts}\n\nQuestion: {question}"},
    ]
    if cards:
        card_text = "\n\n".join(map(format_card, cards))
        messages[-1]["content"] = f"Scryfall card Oracle text:\n{card_text}\n\n" + messages[-1]["content"]
        calculations = computed_costs(question, cards)
        if calculations:
            messages[-1]["content"] = (
                "Verified cost calculations. Use these totals when answering.\n"
                + "\n".join(calculations) + "\n\n" + messages[-1]["content"]
            )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
    ).to(model.device)
    with torch.inference_mode():
        result = model.generate(
            **inputs, max_new_tokens=240, do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(result[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def answer(args):
    with open_cards() if DATABASE.is_file() else nullcontext() as conn:
        run_answers(args, CardIndex(conn) if conn else None)


def run_answers(args, card_index):
    index = RuleIndex()
    cases = None
    if args.command == "evaluate":
        cases = json.loads(Path(__file__).with_name("evaluation.json").read_text(encoding="utf-8"))
        if card_index:
            cases += json.loads(Path(__file__).with_name("evaluation_cards.json").read_text(encoding="utf-8"))
    if args.command == "ask" and args.card and card_index is None:
        raise ValueError("Card index is missing. Run cards.py prepare first.")
    cards = card_index.for_question(args.question, args.card) if card_index and cases is None else []
    model, tokenizer, record = load_for_answers(args)
    if record and record["rules"]["corpus_sha256"] != index.manifest["corpus_sha256"]:
        print("The adapter was trained with a different rules edition than the current index.")
    if record and record.get("cards") and (
        not card_index or record["cards"]["source_sha256"] != card_index.manifest["source_sha256"]
    ):
        print("The adapter's training snapshot differs from the available card lookup data.")
    if cases is None:
        hits = index.search(args.question)
        calculations = computed_costs(args.question, cards)
        if calculations:
            print("\n".join(calculations), "\n")
        print(reply(model, tokenizer, args.question, hits, cards))
        print("\nRetrieved references:", ", ".join(p.id for p in hits) or "none")
        if cards:
            print("Scryfall cards:", ", ".join(card["name"] for card in cards))
            print("Card snapshot:", card_index.manifest["snapshot"])
        print("Rules edition:", index.manifest["effective"] or "unknown")
        return
    rows = []
    for case in cases:
        hits = index.search(case["question"])
        cards = card_index.for_question(case["question"]) if card_index else []
        with model.disable_adapter() if record else nullcontext():
            base = reply(model, tokenizer, case["question"], hits, cards)
        tuned = reply(model, tokenizer, case["question"], hits, cards) if record else None
        retrieved = [p.id for p in hits]
        expected = case["rules"]
        rows.append({
            **case, "retrieved": retrieved,
            "cards_retrieved": [card["name"] for card in cards],
            "computed_costs": computed_costs(case["question"], cards),
            "expected_rules_retrieved": [rule for rule in expected if rule in retrieved],
            "base_answer": base, "adapter_answer": tuned,
        })
        print(f"Evaluated {case['id']}", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_text(args.report, json.dumps({
        "rules_edition": index.manifest["effective"],
        "card_snapshot": card_index.manifest if card_index else None,
        "base_model": record["base_model"] if record else args.model,
        "adapter": str(args.adapter) if record else None,
        "cases": rows,
    }, ensure_ascii=False, indent=2))
    print(f"Review the generated answers against the reference answers in {args.report.resolve()}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check the Python dependencies and GPU.")
    training = commands.add_parser("train", help="Train a LoRA adapter on the imported rules.")
    training.add_argument("--length", type=int, default=512, choices=(256, 512, 768, 1024))
    training.add_argument("--steps", type=int, default=-1, help="Set a positive optimizer-step limit for a smoke run.")
    training.add_argument("--epochs", type=int, default=1)
    training.add_argument("--output", type=Path, default=Path("runs/mtg-rules"))
    training.add_argument("--with-cards", action="store_true", help="Mix in Scryfall cards prepared by cards.py.")
    training.add_argument("--from-adapter", type=Path, help="Continue training from an existing adapter.")
    ask = commands.add_parser("ask", help="Ask a rules question with retrieved passages.")
    ask.add_argument("question")
    ask.add_argument("--card", action="append", default=[], help="Include an exact card or face name. Repeat as needed.")
    evaluation = commands.add_parser("evaluate", help="Generate answers for the included evaluation scenarios.")
    evaluation.add_argument("--report", type=Path, default=Path("runs/evaluation.json"))
    for cmd in (training, ask, evaluation):
        cmd.add_argument("--model", default=BASE_MODEL)
        cmd.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    for cmd in (ask, evaluation):
        cmd.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if args.command == "train" and (args.epochs < 1 or args.steps == 0 or args.steps < -1):
        parser.error("Use --epochs >= 1 and either --steps -1 or a positive step count.")
    if args.command == "train" and args.from_adapter and args.model != BASE_MODEL:
        parser.error("--from-adapter supplies the base model. Omit --model.")
    try:
        if args.command == "doctor":
            doctor()
        elif args.command == "train":
            train(args)
        else:
            answer(args)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
