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
python rules.py index
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

The importer writes the numbered rules and glossary to `data/passages.json`, along with the edition, source URL and hashes. It also writes `data/corpus.txt` for corpus training. Each process loads this local data once. Training steps and questions use the local data and in-memory search index. Downloading a new rules edition happens only when `rules.py prepare` is explicitly run.

The default source is the November 6, 2026 English TXT. The supplied June 6, 2025 PDF is a historical edition. The training examples and tests in this repository target the imported 2026 edition. [June 2025 rules PDF](https://media.wizards.com/2025/downloads/MagicCompRules%2020250606.pdf)

[Official rules downloads](https://magic.wizards.com/en/rules), [Scryfall snapshot](https://data.scryfall.io/all-cards/all-cards-20261117102242.jsonl.gz)

## Rule retrieval

`rules.py index` builds `data/rule_vectors.npz` and caches two small CPU models. The first build downloads their weights from Hugging Face. Subsequent questions reuse those models and the local rules. The current Python dependencies already support them.

Search combines BM25 keywords and MiniLM semantic similarity, adds keywords and text from selected Oracle records, and reranks candidate passages with a passage-relevance model. Official glossary references anchor the relevant rule families. Bounded groups of related subrules preserve definitions, exceptions and cross-references together.

The final context uses up to 12 blocks and approximately 12,000 rule-text characters. Reports list all rule IDs actually included in those blocks, including grouped subrules. Directly requested rule IDs are prioritized. If no semantic index exists, lexical retrieval remains available. When the rules data changes, rebuild the semantic index before using it.

Embedding and reranking use the CPU, leaving GPU memory for the language model. The cache records the rules-file hash and model revisions. Test reports record which retrieval configuration was used. The embedding and reranking models are separate from the MTG answer model.

[MiniLM embedding model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [Passage reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)

## Card database

The JSONL archive is the import format. Runtime card lookup already uses a read-only SQLite database with parameterized queries. The application supplies exact Oracle text through named card lookups.

SQLite suits this local application. MySQL becomes useful for a centrally hosted card service, several applications or shared database administration. A future MySQL backend can expose the same typed lookup functions. Updating card data then updates lookup results without retraining the model. SQL generation is separate from teaching MTG rule application. [SQLite deployment guidance](https://www.sqlite.org/whentouse.html)

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

## Supervised question-and-answer training

Use the supervised stage to train rule application and answer style after the corpus stage.

```bash
python mtg.py sft --validate
python mtg.py sft --from-adapter runs/qwen3-full --steps 3 --output runs/qwen3-sft-smoke
python mtg.py sft --from-adapter runs/qwen3-full --epochs 1 --output runs/qwen3-sft
python smoke_tests.py --adapter runs/qwen3-sft --compare-base --output runs/smoke-sft
```

The included starter curriculum contains 88 training and 22 validation examples, with rule citations. It covers mana costs, hybrid and Phyrexian mana, X, trample, indestructible, priority, legal targets, APNAP ordering, Commander, counters and other state-based actions. Many validation examples are parameter variants of the same concepts. The separate smoke suite is a development evaluation and is excluded from supervised training.

Training examples are JSONL records in `qa/train.jsonl` and `qa/validation.jsonl`. Each record supplies `id`, `question`, `answer`, `rules` and an optional `cards` list. Answers cite their supporting rules. You can provide your own files through `--train-file` and `--eval-file`.

Each training prompt includes the example's supporting rule passages and selected cards. Only assistant-answer and end-of-message tokens contribute to the supervised loss. User prompts and source passages are masked. Evaluation never receives an expected answer or its reference list as retrieval input.

The command checks that training and validation have distinct IDs and questions, that citations and card names resolve, and that no question duplicates a reserved smoke or evaluation question. Complete examples must fit the chosen `--length`; an overlong example produces an error without truncating its answer.

Defaults are one epoch, learning rate `2e-5`, sequence length 2048, and batch size 1. Use `--learning-rate` and `--length` to change them. Run metadata records data hashes, supervised token counts and validation loss. A three-step CUDA run continuing the trained Qwen3-4B adapter completed successfully. Full supervised ruling-accuracy validation is pending.

To train from the original base model, omit `--from-adapter`. To measure the effect of the supervised stage, compare its report with an existing-adapter report generated using the same retrieval configuration. The `--compare-base` option compares against the original base model.

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

The corpus stage uses next-token prediction over rulebook and card text. The supervised stage uses question-and-answer examples. Evaluate the resulting rulings before relying on them. Current data covers the Comprehensive Rules and the imported Oracle snapshot. Tournament policy and individual card rulings require additional sources.

## Validation

```bash
python -m unittest discover -s tests -v
```

Nine regression tests cover answer parsing, complete assistant-only labels, dataset overlap, reserved evaluation questions, semantic cache validation, chunk deduplication, subrule provenance, cached lookup behavior and memory-conscious generation.

On the existing 30-question development suite, expected supporting references present in retrieved context increased from 29 of 50 to 40 of 50. The earlier search returned eight individual passages. The new search allows up to twelve bounded blocks, including related subrules. This measures evidence coverage on the development questions. Full answer accuracy after supervised training still needs a new evaluation run. The three-step supervised smoke adapter still answered a zero-toughness question incorrectly despite receiving the relevant rules.
