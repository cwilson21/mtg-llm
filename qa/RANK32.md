# Fresh rank-32 experiment

This run initializes a new adapter on `Qwen/Qwen3-4B-Instruct-2507`, the original model used by this project. It loads no previous MTG adapter. This is the instruction model used as our foundation; `Qwen/Qwen3-4B-Base` is a separate pretrained model and is not selected by these commands.

The configuration is LoRA rank 32, alpha 64, dropout 0.05, targeting `all-linear`: Q/K/V/O attention projections and gate/up/down MLP projections in every transformer block. PEFT excludes the output vocabulary head from `all-linear`. The foundation weights stay frozen in BF16 on supported CUDA GPUs, with trainable LoRA adapters.

## Data

| Split | Retained round-2 | New examples | Total |
| --- | ---: | ---: | ---: |
| Training | 184 | 316 | 500 |
| Validation | 54 | 78 | 132 |

The additions include 38 authored scenarios, 79 mana-cost applications, and 80 Oracle-derived card examples. Each new scenario has a free-form answer and a multiple-choice version. The 197 scenarios produce 394 additional examples. New training multiple-choice answers are balanced at 40/40/39/39 across A/B/C/D; validation has 10/10/10/9.

New rule applications cover ward, protection, flying, first strike, damage marking, empty libraries, mana abilities, crew, flashback, kicker, undying, copying, battle cry, and identifying insufficient information. Arithmetic answers show the calculation and distinguish costs, mana value, and X in different zones. Oracle examples use 80 additional single-face cards, checking calculated mana value against Scryfall's `cmc`.

Related scenario variants stay in the same split; the new Oracle cards are disjoint by name across splits and exclude cards detected in the reserved evaluation questions. Validation shares concepts and templates with training, and the retained original curriculum includes parameter variations across splits. This is development validation, not an independent estimate of general MTG judging ability. The existing smoke/evaluation questions and their answers are not included in training.

The source edition is November 6, 2026, with the November 17 Scryfall snapshot. `rank32-manifest.json` records source and dataset hashes. Rebuild with `python qa/build_rank32.py` from the project directory after preparing the local rules and cards. The generator uses local data only. The authored explanations have not had independent judge review. Additional data and higher rank do not guarantee better accuracy.

## Windows commands

Run in the project directory containing your `.venv`, `data`, and `mtg.py`. Choose a new output directory for each experiment.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python mtg.py sft --train-file qa/rank32-train.jsonl --eval-file qa/rank32-validation.jsonl --validate
& $python mtg.py sft --fresh-base --model Qwen/Qwen3-4B-Instruct-2507 --lora-rank 32 --lora-alpha 64 --lora-targets all-linear --train-file qa/rank32-train.jsonl --eval-file qa/rank32-validation.jsonl --length 2048 --learning-rate 2e-5 --epochs 3 --output runs/qwen3-rank32
& $python smoke_tests.py --adapter runs/qwen3-rank32 --compare-base --output runs/smoke-rank32
```

## Ubuntu commands

With the project's CUDA-enabled Python environment activated and local rules/cards prepared:

```bash
python mtg.py sft --train-file qa/rank32-train.jsonl --eval-file qa/rank32-validation.jsonl --validate
python mtg.py sft --fresh-base --model Qwen/Qwen3-4B-Instruct-2507 --lora-rank 32 --lora-alpha 64 --lora-targets all-linear --train-file qa/rank32-train.jsonl --eval-file qa/rank32-validation.jsonl --length 2048 --learning-rate 2e-5 --epochs 3 --output runs/qwen3-rank32
python smoke_tests.py --adapter runs/qwen3-rank32 --compare-base --output runs/smoke-rank32
```

## Epochs and evaluation

Every SFT epoch calculates validation **answer-token loss** and saves a complete Trainer checkpoint: adapter, tokenizer, optimizer, scheduler, RNG, Trainer state, and `run.json`. Nothing prunes earlier checkpoints. `epoch_metrics.json` collects the validation losses as the run progresses. The root adapter saved when training finishes is the final epoch; there is no automatic best-epoch substitution.

For this 500-example dataset with batch size 1 and gradient accumulation 4, three full epochs produce `checkpoint-125`, `checkpoint-250`, and `checkpoint-375`. Each is directly usable with `ask`, `evaluate`, and `smoke_tests.py`. A positive `--steps` caps a smoke training run and can end at a fractional epoch; omit it for the full experiment.

Validation loss does not measure multiple-choice ruling accuracy. After training, optionally compare all three checkpoints on the development suite:

```powershell
foreach ($step in 125, 250, 375) {
    & $python smoke_tests.py --adapter "runs/qwen3-rank32/checkpoint-$step" --output "runs/smoke-rank32-$step"
}
```

Review `results.html` for explanations as well as selected choices. Using this suite to select a checkpoint makes it a development suite; use separate unseen scenarios for a final accuracy claim.

Fresh SFT now defaults to rank 32 and all-linear; corpus `train` retains rank 8 with Q/V projections. `--from-adapter` preserves the saved adapter's rank and layers and rejects attempts to override them. `--fresh-base` rejects `--from-adapter`. The explicit commands above avoid inheriting an earlier adapter accidentally.

The complete dataset tokenizes to at most 1,641 training tokens and 993 validation tokens with the selected tokenizer; examples exceeding `--length` fail instead of silently losing answers. Gradient checkpointing, batch size 1, and loss-only evaluation limit GPU memory. Rules and cards are read locally; they are not fetched on each training iteration.

## Local verification

On Windows with the pinned dependencies, an RTX 4060 Ti 16 GB completed a fresh rank-32 run using the four longest training examples and two longest validation examples for two epochs. It trained 66,060,288 adapter parameters, evaluated twice, and saved `checkpoint-1` and `checkpoint-2`, including optimizer state and matching epoch provenance. Peak allocated CUDA memory was 11.51 GiB; peak reserved memory was 15.24 GiB. This short run checks execution and checkpoint behavior; it does not establish improved ruling accuracy. All 15 regression tests passed.

The second epoch checkpoint also loaded through the application's normal adapter loader and generated an answer. A one-step corpus-training regression retained rank 8 on Q/V projections and saved its final adapter successfully.
