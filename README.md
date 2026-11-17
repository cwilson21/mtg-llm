# MTG LLM

Train a local language model on the Magic Comprehensive Rules and Scryfall card text, ask questions with retrieved rules and Oracle text, and evaluate answers against an expected-answer suite.

The current model is `Qwen/Qwen3-4B-Instruct-2507` with bfloat16 LoRA training. The training pipeline has completed a three-step CUDA smoke run on an NVIDIA RTX 4060 Ti with 16 GB of VRAM. A full training run's ruling accuracy needs evaluation.

The Linux target is **Ubuntu 26.04 LTS**. Canonical lists standard security maintenance through May 2031, and NVIDIA's CUDA Linux installation guide includes Ubuntu 26.04 for amd64 systems. Linux runtime validation is pending. [Ubuntu lifecycle](https://ubuntu.com/about/release-cycle?product=ubuntu&release=ubuntu&version=26.04+LTS), [NVIDIA Linux support](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

The proposed future 12B profile requires 4-bit QLoRA support. The code currently implements the 4B LoRA profile.

## Install

Use Python 3.13 and a compatible NVIDIA driver. Check GPU availability with `nvidia-smi`.

On Windows, open PowerShell in the repository directory.

```powershell
.\setup.cmd
$python = ".\.venv\Scripts\python.exe"
```

On Linux, create a Python 3.13 environment and install the same pinned training dependencies.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python mtg.py doctor
```

The Linux commands below use `python` from the activated environment. In PowerShell, replace `python` with `& $python`.

## Prepare the data

Import the English November 6, 2026 Comprehensive Rules from Wizards.

```bash
python rules.py prepare
python rules.py lookup "117.4"
```

Download the requested Scryfall snapshot and import its local path.

```bash
python cards.py prepare --file /path/to/all-cards-20261117102242.jsonl.gz
python cards.py lookup "Walking Ballista"
```

The snapshot contains 552,011 printing records. Importing selects 117,392 English records with Oracle IDs and deduplicates them into 39,289 cards. It preserves all card faces and selects the latest released English printing for each Oracle ID, with printing ID as a tie-breaker.

Card training text includes names, mana costs, types, Oracle text, power, toughness, loyalty, defense, mana values and computed X-cost examples. Training also replays rules and mana guidance. Data imports record their source hashes for reproducibility.

Generated data, model caches, trained adapters and run reports are stored locally under the paths in `.gitignore`.

[Official rules downloads](https://magic.wizards.com/en/rules), [Scryfall snapshot](https://data.scryfall.io/all-cards/all-cards-20261117102242.jsonl.gz)

## Train

First check a short combined rules-and-cards run.

```bash
python mtg.py train --with-cards --steps 3 --output runs/qwen3-smoke
```

Train for one full pass through the prepared corpus.

```bash
python mtg.py train --with-cards --epochs 1 --output runs/qwen3-full
```

For staged training, train on rules first and then continue the resulting adapter with card data.

```bash
python mtg.py train --output runs/qwen3-rules
python mtg.py train --with-cards --from-adapter runs/qwen3-rules --output runs/qwen3-rules-cards
```

Choose a fresh output directory for every run. The current trainer saves the adapter when training finishes. `--from-adapter` loads that adapter's original base model and starts fresh optimizer state. Changing base models requires a new adapter.

## Ask and evaluate

```bash
python mtg.py ask --adapter runs/qwen3-full --card "Walking Ballista" "With X=3, what is its mana value on the stack?"
python mtg.py evaluate --adapter runs/qwen3-full --report runs/qwen3-evaluation.json
python smoke_tests.py --adapter runs/qwen3-full --compare-base
```

Open `runs/post-training-smoke/results.html` to review the 30-question smoke suite. The JSON report contains expected answers, generated answers, selected choices, reference rules, retrieved cards and model/data provenance. Automatic scoring checks the selected answer choice. Review explanations and citations manually.

The tests exercise the application with its retrieved context and mana helpers. Expected answer keys are used for scoring and are excluded from prompts. Questions and answer keys are excluded from the training input.

The following command validates question structure, rule IDs and card names without loading a model.

```bash
python smoke_tests.py --validate
```

[Smoke-test instructions](SMOKE_TESTS.md), [Questions and expected answers](SMOKE_QUESTIONS.md)

## Mana

`{W}` means white, `{U}` blue, `{B}` black, `{R}` red and `{G}` green. Numerical costs are generic and can be paid with any types of mana. `{C}` specifically requires colorless mana. For mana value, X is zero outside the stack and takes its chosen value on the stack.

```bash
python mana.py "{2}{G}"
python mana.py "{X}{X}" --x 3
```

Questions automatically detect multiword card names. Use repeated `--card` arguments to choose exact names, including single-word names. The application includes up to six selected cards per question.

## Practical limits

Training uses next-token prediction over rulebook and card text. Evaluate the resulting rulings before relying on them. Current data covers the Comprehensive Rules and the imported Oracle snapshot. Tournament policy and individual card rulings require additional sources.
