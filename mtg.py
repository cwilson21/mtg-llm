"""Fine-tune a language model on the rules, ask questions, and compare answers."""

import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
from importlib.util import find_spec
import inspect
from itertools import batched
import json
import os
from pathlib import Path
import sqlite3
import sys

from cards import DATABASE, CardIndex, format_card, open_cards
from mana import MANA_GUIDE, computed_costs
from rules import DATA, RuleIndex, format_passage, retrieved_rule_ids, write_text
from supervised import chat_template_options


os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
NEMO_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
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


def load_base(model_id, dtype, revision=None, *, quantized_device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    options = {}
    if quantized_device is not None:
        import torch
        from transformers import BitsAndBytesConfig

        if find_spec("bitsandbytes") is None:
            raise ValueError("4-bit loading requires bitsandbytes. Install it with python -m pip install bitsandbytes==0.50.2.")
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        options["device_map"] = {"": quantized_device.index if quantized_device.index is not None
                                 else torch.cuda.current_device()}
    tokenizer_options = {"fix_mistral_regex": True} if model_id == NEMO_MODEL else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, **tokenizer_options)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype, use_safetensors=True, **options,
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
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
    from training import EpochArtifacts, lora_options, training_record

    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError(f"{args.output} already contains files. Choose a fresh --output directory.")
    options = lora_options(args)
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
    load_in_4bit = args.load_in_4bit or bool(parent and parent.get("load_in_4bit"))
    if load_in_4bit and device.type != "cuda":
        raise ValueError("4-bit training requires --device cuda.")
    model, tokenizer = load_base(
        base_model, dtype, parent["revision"] if parent else None,
        quantized_device=device if load_in_4bit else None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    revision = model.config._commit_hash
    model.config.use_cache = False
    evaluation = None
    supervised_record = None
    if args.command == "sft":
        from supervised import prepare_datasets
        message_builder = partial(answer_messages, closed_book=True) if args.closed_book else answer_messages
        blocks, evaluation, supervised_record = prepare_datasets(args, index, tokenizer, message_builder)
        supervised_record["closed_book"] = args.closed_book
    token_ids = []
    if args.command == "sft" and args.full_rules_corpus:
        source_tokens = tokenizer(raw.decode("utf-8"), add_special_tokens=False, verbose=False)["input_ids"]
        corpus_tokens = [tokenizer.eos_token_id, *source_tokens, tokenizer.eos_token_id]
        corpus_blocks = token_blocks(corpus_tokens, args.length)
        blocks.extend({
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
            "labels": [-100, *tokens[1:]],
        } for tokens in corpus_blocks)
        supervised_record["rules_corpus"] = {
            "sha256": index.manifest["corpus_sha256"],
            "source_tokens": len(source_tokens),
            "blocks": len(corpus_blocks),
            "loss_tokens": sum(len(tokens) - 1 for tokens in corpus_blocks),
        }
        print(f"Full rules corpus: {len(source_tokens):,} source tokens across {len(corpus_blocks):,} blocks")
    if args.command == "train":
        token_ids = tokenizer(raw.decode("utf-8"), add_special_tokens=False, verbose=False)["input_ids"]
        token_ids.append(tokenizer.eos_token_id)
    rule_blocks = token_blocks(token_ids, args.length)
    mana_tokens = tokenizer(MANA_GUIDE, add_special_tokens=False, verbose=False)["input_ids"]
    mana_tokens.append(tokenizer.eos_token_id)
    mana_blocks = token_blocks(mana_tokens, args.length)
    card_manifest = None
    if supervised_record:
        card_manifest = supervised_record.get("cards") or (parent or {}).get("cards")
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
    elif args.command == "train":
        blocks = Blocks(rule_blocks)
    # Repeat this concise reference throughout either training stage.
    if args.command == "train":
        mana_replays = max(len(mana_blocks), (len(blocks) + 19) // 20)
        blocks.blocks.extend(mana_blocks[i % len(mana_blocks)] for i in range(mana_replays))
    if args.command == "train":
        print(f"{len(token_ids):,} source rules tokens; {card_tokens:,} source card tokens")
    else:
        print(f"Supervised assistant-answer tokens: {supervised_record['assistant_tokens']:,}")
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        if model.config.model_type == "qwen3" and dtype == torch.bfloat16:
            # PEFT upcasts these large frozen vocabulary matrices. Keep them in the
            # model's compute dtype to leave room for long-sequence backpropagation.
            model.get_input_embeddings().to(dtype=dtype)
            model.get_output_embeddings().to(dtype=dtype)
    if parent:
        model = PeftModel.from_pretrained(model, args.from_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, lora_dropout=0.05, **options,
        ))
    model.print_trainable_parameters()
    settings = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
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
        save_strategy="epoch" if evaluation else "no",
        eval_accumulation_steps=1,
        prediction_loss_only=True,
        report_to="none",
        seed=42,
        eval_strategy="epoch" if evaluation else "no",
    )
    if args.command == "sft":
        from transformers import DataCollatorForSeq2Seq
        collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    adapter_config = model.peft_config["default"]
    targets = adapter_config.target_modules
    record = {
        "base_model": base_model, "revision": revision, "rules": index.manifest,
        "cards": card_manifest, "parent_adapter": str(args.from_adapter) if parent else None,
        "tokens": (sum(len(row["input_ids"]) for row in blocks) if supervised_record
                   else len(token_ids) + card_tokens + len(mana_tokens)),
        "sequence_length": args.length,
        "mana_reference_sha256": hashlib.sha256(MANA_GUIDE.encode("utf-8")).hexdigest(),
        "training_mode": args.command, "supervised": supervised_record,
        "lora": {"rank": adapter_config.r, "alpha": adapter_config.lora_alpha,
                 "target_modules": sorted(targets) if isinstance(targets, set) else targets},
        "learning_rate": args.learning_rate, "requested_epochs": args.epochs,
        "save_strategy": settings.save_strategy.value, "eval_strategy": settings.eval_strategy.value,
        "precision": str(dtype), "load_in_4bit": load_in_4bit, "seed": 42,
    }
    trainer = Trainer(
        model=model, args=settings, train_dataset=blocks, processing_class=tokenizer,
        data_collator=collator, eval_dataset=evaluation,
        callbacks=[EpochArtifacts(record)],
    )
    trainer.train()
    trainer.save_model(str(args.output))
    write_text(args.output / "run.json", json.dumps(training_record(record, trainer.state), indent=2))
    print(f"Adapter saved to {args.output.resolve()}")


def load_for_answers(args):
    from peft import PeftModel

    device, dtype = runtime(args.device)
    if args.adapter:
        record = json.loads((args.adapter / "run.json").read_text(encoding="utf-8"))
        model_id, revision = record["base_model"], record["revision"]
    else:
        record = None
        model_id, revision = args.model, None
    load_in_4bit = args.load_in_4bit or bool(record and record.get("load_in_4bit"))
    if load_in_4bit and device.type != "cuda":
        raise ValueError("4-bit inference requires --device cuda.")
    quantized_device = device if load_in_4bit else None
    model, tokenizer = load_base(model_id, dtype, revision, quantized_device=quantized_device)
    if record is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
    if quantized_device is None:
        model = model.to(device)
    return model.eval(), tokenizer, record


def answer_messages(question, hits, cards, *, closed_book=False):
    if closed_book:
        if cards:
            card_text = "\n\n".join(format_card(card, include_mana_hints=False) for card in cards)
            question = f"Scryfall card Oracle text:\n{card_text}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": "Answer Magic: The Gathering rules questions. "
             "State the outcome and explain the relevant steps. When a numbered rule applies, cite it "
             "using [CR <rule number>]. Use the card text supplied in the question when available."},
            {"role": "user", "content": question},
        ]
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
    return messages


def reply(model, tokenizer, question, hits, cards, *, closed_book=False, max_new_tokens=240):
    import torch

    prompt = tokenizer.apply_chat_template(
        answer_messages(question, hits, cards, closed_book=closed_book),
        tokenize=False, add_generation_prompt=True,
        **chat_template_options(tokenizer),
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
    ).to(model.device)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    generation_options = {"use_cache": True}
    if "logits_to_keep" in inspect.signature(base.forward).parameters:
        generation_options["logits_to_keep"] = 1
    with torch.inference_mode():
        result = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id,
            **generation_options,
        )
    return tokenizer.decode(result[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def answer(args):
    with open_cards() if DATABASE.is_file() else nullcontext() as conn:
        run_answers(args, CardIndex(conn) if conn else None)


def run_answers(args, card_index):
    cases = None
    if args.command == "evaluate":
        cases = json.loads(Path(__file__).with_name("evaluation.json").read_text(encoding="utf-8"))
        if card_index:
            cases += json.loads(Path(__file__).with_name("evaluation_cards.json").read_text(encoding="utf-8"))
    if args.command == "ask" and args.card and card_index is None:
        raise ValueError("Card index is missing. Run cards.py prepare first.")
    cards = card_index.for_question(args.question, args.card) if card_index and cases is None else []
    model, tokenizer, record = load_for_answers(args)
    closed_book = args.closed_book or bool(record and (record.get("supervised") or {}).get("closed_book"))
    index = None if closed_book else RuleIndex()
    if index and record and record["rules"]["corpus_sha256"] != index.manifest["corpus_sha256"]:
        print("The adapter was trained with a different rules edition than the current index.")
    if record and record.get("cards") and (
        not card_index or record["cards"]["source_sha256"] != card_index.manifest["source_sha256"]
    ):
        print("The adapter's training snapshot differs from the available card lookup data.")
    if cases is None:
        hits = index.search(args.question, cards=cards) if index else []
        calculations = [] if closed_book else computed_costs(args.question, cards)
        if calculations:
            print("\n".join(calculations), "\n")
        print(reply(model, tokenizer, args.question, hits, cards, closed_book=closed_book))
        if index:
            print("\nRetrieved references:", ", ".join(retrieved_rule_ids(hits)) or "none")
        if cards:
            print("Scryfall cards:", ", ".join(card["name"] for card in cards))
            print("Card snapshot:", card_index.manifest["snapshot"])
        if index:
            print("Rules edition:", index.manifest["effective"] or "unknown")
        return
    rows = []
    for case in cases:
        cards = card_index.for_question(case["question"]) if card_index else []
        hits = index.search(case["question"], cards=cards) if index else []
        with model.disable_adapter() if record else nullcontext():
            base = reply(model, tokenizer, case["question"], hits, cards, closed_book=closed_book)
        tuned = reply(model, tokenizer, case["question"], hits, cards, closed_book=closed_book) if record else None
        retrieved = retrieved_rule_ids(hits)
        expected = case["rules"]
        rows.append({
            **case, "retrieved": retrieved,
            "cards_retrieved": [card["name"] for card in cards],
            "computed_costs": [] if closed_book else computed_costs(case["question"], cards),
            "expected_rules_retrieved": [rule for rule in expected if rule in retrieved],
            "base_answer": base, "adapter_answer": tuned,
        })
        print(f"Evaluated {case['id']}", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_text(args.report, json.dumps({
        "rules_edition": index.manifest["effective"] if index else None,
        "closed_book": closed_book,
        "card_snapshot": card_index.manifest if card_index else None,
        "retrieval": index.retrieval_info() if index else None,
        "base_model": record["base_model"] if record else args.model,
        "load_in_4bit": args.load_in_4bit or bool(record and record.get("load_in_4bit")),
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
    training.add_argument("--learning-rate", type=float, default=5e-5)
    sft = commands.add_parser("sft", help="Supervised training on reviewed questions and answers.")
    sft.add_argument("--train-file", type=Path, default=Path("qa/train.jsonl"))
    sft.add_argument("--eval-file", type=Path, default=Path("qa/validation.jsonl"))
    sft.add_argument("--validate", action="store_true", help="Validate datasets without loading a model.")
    sft.add_argument("--closed-book", action="store_true", help="Train on questions and card facts without rule or mana hints.")
    sft.add_argument("--full-rules-corpus", action="store_true",
                     help="Mix the complete prepared rules corpus into the supervised epoch.")
    sft.add_argument("--from-adapter", type=Path)
    sft.add_argument("--fresh-base", action="store_true", help="Reject --from-adapter; initialize a new adapter.")
    sft.add_argument("--output", type=Path, default=Path("runs/qwen3-sft"))
    sft.add_argument("--length", type=int, default=2048, choices=(512, 1024, 2048))
    sft.add_argument("--epochs", type=int, default=1)
    sft.add_argument("--steps", type=int, default=-1)
    sft.add_argument("--learning-rate", type=float, default=2e-5)
    sft.set_defaults(with_cards=False)
    for cmd in (training, sft):
        cmd.add_argument("--lora-rank", type=int, help="Fresh adapter rank (SFT: 32; corpus: 8).")
        cmd.add_argument("--lora-alpha", type=int, help="Fresh adapter scaling alpha (default: twice rank).")
        cmd.add_argument("--lora-targets", choices=("all-linear", "qv"),
                         help="Fresh adapter layers (SFT: all-linear; corpus: qv).")
    ask = commands.add_parser("ask", help="Ask a rules question.")
    ask.add_argument("question")
    ask.add_argument("--card", action="append", default=[], help="Include an exact card or face name. Repeat as needed.")
    evaluation = commands.add_parser("evaluate", help="Generate answers for the included evaluation scenarios.")
    evaluation.add_argument("--report", type=Path, default=Path("runs/evaluation.json"))
    for cmd in (training, sft, ask, evaluation):
        cmd.add_argument("--model", default=BASE_MODEL)
        cmd.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        cmd.add_argument("--load-in-4bit", action="store_true", help="Load the base weights in 4-bit NF4 on CUDA.")
    for cmd in (ask, evaluation):
        cmd.add_argument("--adapter", type=Path)
        cmd.add_argument("--closed-book", action="store_true", help="Answer without rule retrieval or mana hints.")
    args = parser.parse_args()
    if args.command != "doctor" and args.load_in_4bit and args.device != "cuda":
        parser.error("--load-in-4bit requires --device cuda.")
    if args.command in {"train", "sft"} and (args.epochs < 1 or args.steps == 0 or args.steps < -1 or args.learning_rate <= 0):
        parser.error("Use --epochs >= 1 and either --steps -1 or a positive step count.")
    if args.command in {"train", "sft"} and args.from_adapter and args.model != BASE_MODEL:
        parser.error("--from-adapter supplies the base model. Omit --model.")
    if args.command in {"train", "sft"}:
        from training import lora_options
        try:
            lora_options(args)
        except ValueError as error:
            parser.error(str(error))
    try:
        if args.command == "doctor":
            doctor()
        elif args.command == "sft" and args.validate:
            from supervised import load_datasets
            load_datasets(args.train_file, args.eval_file, RuleIndex())
        elif args.command in {"train", "sft"}:
            train(args)
        else:
            answer(args)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
