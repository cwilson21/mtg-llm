"""LoRA settings and self-contained epoch checkpoint provenance."""

import json
from pathlib import Path

from transformers import TrainerCallback
from rules import write_text


def lora_options(args):
    if getattr(args, "fresh_base", False) and args.from_adapter:
        raise ValueError("--fresh-base cannot be combined with --from-adapter.")
    if args.from_adapter:
        if any(getattr(args, field, None) is not None
               for field in ("lora_rank", "lora_alpha", "lora_targets")):
            raise ValueError("An existing adapter retains its rank and layers. Use --fresh-base for new settings.")
        return None
    rank = args.lora_rank if args.lora_rank is not None else (32 if args.command == "sft" else 8)
    alpha = args.lora_alpha if args.lora_alpha is not None else 2 * rank
    if rank < 1 or alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive integers.")
    targets = args.lora_targets or ("all-linear" if args.command == "sft" else "qv")
    return {"r": rank, "lora_alpha": alpha,
            "target_modules": "all-linear" if targets == "all-linear" else ["q_proj", "v_proj"]}


def training_record(record, state):
    evaluations = [entry for entry in state.log_history if "eval_loss" in entry]
    return {**record, "optimizer_steps": state.global_step, "epoch": state.epoch,
            "validation_metrics": evaluations[-1] if evaluations else None,
            "epoch_evaluations": evaluations}


class EpochArtifacts(TrainerCallback):
    def __init__(self, record):
        self.record = record

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            write_text(Path(args.output_dir) / "epoch_metrics.json",
                       json.dumps(training_record(self.record, state)["epoch_evaluations"], indent=2))

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_text(checkpoint / "run.json", json.dumps(training_record(self.record, state), indent=2))
