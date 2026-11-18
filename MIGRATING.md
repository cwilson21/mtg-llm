# Moving to headless Ubuntu

Use Ubuntu Server **26.04 LTS** (26.04.1 installation media is available). Ubuntu has no 26.06 release. See the [official releases](https://wiki.ubuntu.com/Releases).

Headless operation can leave more GPU memory available to training. It does not guarantee that the full BF16 rank-32 workload will fit comfortably or eliminate the slowdown; measure memory use and sustained iteration times after installing the NVIDIA driver and rebuilding the environment.

## Preserve local training artifacts

GitHub contains the code, QA datasets, and instructions. These working folders are intentionally ignored by Git and need to be copied to another disk or machine before replacing the Windows installation:

- `data/`: prepared rules JSON, corpus, card SQLite database, and retrieval index.
- `runs/`: adapters, epoch checkpoints, metadata, and smoke reports. In particular, retain `runs/qwen3-rank32/checkpoint-125` from the completed first epoch and any later checkpoints.
- Optionally `.cache/huggingface/`: cached foundation models and retrieval models, to avoid downloading them again.
- Optionally the original Scryfall `all-cards-20261117102242.jsonl.gz` archive from Downloads, for rebuilding the card database.

Rebuild `.venv` on Linux using the [installation instructions](README.md#install). A Windows virtual environment cannot be reused as the Linux environment. Relative project paths remain the same.

## Get the project

```bash
git clone https://github.com/cwilson21/mtg-llm.git
cd mtg-llm
```

Copy the preserved `data/`, `runs/`, and optional `.cache/` folders into this directory. Install the CUDA-enabled Python environment described in the README, then activate it and check the setup:

```bash
source .venv/bin/activate
nvidia-smi
python mtg.py doctor
python mtg.py sft --train-file qa/rank32-train.jsonl --eval-file qa/rank32-validation.jsonl --validate
```

If local data was not copied, follow [Prepare the data](README.md#prepare-the-data) before validation and training. The [rank-32 guide](qa/RANK32.md#ubuntu-commands) contains the fresh training command. Use a new output directory if the previous run was copied, because the trainer refuses to overwrite existing runs.

The existing CLI does not expose Trainer checkpoint resumption. `--from-adapter` starts a new training schedule from saved adapter weights; it does not restore the optimizer, scheduler, or original epoch position. Keep the complete checkpoints so that these states remain available.

To evaluate a copied first-epoch adapter before starting another experiment:

```bash
python smoke_tests.py --adapter runs/qwen3-rank32/checkpoint-125 --compare-base --output runs/smoke-rank32-epoch1-linux
```
