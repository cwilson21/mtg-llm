# Closed-book Commander curriculum

Build from the repo root using the prepared Comprehensive Rules index and Scryfall SQLite snapshot.

```bash
.venv/bin/python qa/build_commander_curriculum.py
```

The builder writes `train.jsonl`, `validation.jsonl`, and `manifest.json` to
`data/commander-curriculum`. Specify an empty directory with `--output` for
another build. By default it samples 2,400 Oracle profiles across card types.
Use `--cards 0` to generate profiles from the whole local snapshot. The builder
deduplicates normalized card names and excludes unusually long records. The
manifest tracks source snapshots, dataset hashes, and counts by track.

## Build with the bracket catalogs

The card importer retains Scryfall's `game_changer` field. If an older database
omits it, reimport the original Scryfall archive with `cards.py prepare --file`.
Fetch the EDHREC snapshot once, then build from that saved file.

```bash
.venv/bin/python qa/fetch_commander_combo_data.py
.venv/bin/python qa/build_commander_curriculum.py \
  --combo-snapshot data/edhrec-early-combos.json \
  --output data/commander-curriculum-keywords
```

The downloader follows the continuation links, checks page counts and combo IDs,
and records every fetched page in the snapshot. A listed early-game pair sets
a Bracket 4 floor. Bracket 5 also depends on competitive intent and legal play.

## Training and evaluation

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/python mtg.py sft --closed-book --full-rules-corpus \
  --from-adapter runs/falcon3-10b-sft-closed-book \
  --train-file data/commander-curriculum-keywords/train.jsonl \
  --eval-file data/commander-curriculum-keywords/validation.jsonl \
  --epochs 1 --output runs/falcon3-10b-commander
```

For a fresh Qwen3-8B QLoRA run, use its explicit model ID and a fresh output path.
The first run downloads the base weights. Training and inference both select
Qwen's non-thinking chat format so their assistant-answer prefixes agree.

```bash
.venv/bin/python mtg.py sft --model Qwen/Qwen3-8B --load-in-4bit \
  --fresh-base --lora-rank 16 --closed-book --full-rules-corpus \
  --train-file data/commander-curriculum-keywords/train.jsonl \
  --eval-file data/commander-curriculum-keywords/validation.jsonl \
  --epochs 1 --output runs/qwen3-8b-commander
```

The curriculum replays the existing supervised data and adds the following tracks.

- **Rules recall** targets numbered Comprehensive Rules text, grouped by rule family.
- **Oracle recall** targets card profiles from card names. Oracle text is omitted
  from the prompt for these examples. The Scryfall snapshot contains cards from
  multiple environments. A recall example supplies no claim of Commander legality.
- **Gameplay** targets choices under stated constraints: combat math, mana
  sequencing, priority, threat assessment, mulligans, and playing through
  interaction and sweepers.
- **Commander deckbuilding** targets mana-source estimates, legality concepts,
  packages and card comparisons for 16 commanders, ramp timing, mana curves,
  cut priorities, and testing for resilience.
- **Commander brackets** teaches the supplied Commander Brackets Beta chart.
  It covers the intended play experience, feature checks by bracket, and
  pregame discussions. The optional catalogs add the complete Game Changer
  list from the imported Scryfall snapshot and the listed EDHREC early-game
  pairs. Check current event policy and the banned list when applying them.
- **Keywords and mechanics** covers the named entries in rules 701 and 702
  and their numbered subrules, plus distinctions between keyword abilities,
  ability words, and flavor words. Applied cases cover manifest, manifest
  dread, cloak, Zimone's landfall, trample, haste, and vigilance.
- **Phyrexian mana** contrasts actual payment with printed mana value,
  including hybrid Phyrexian mana, X, discounts, and life-payment limits.
- **Rules recovery** uses paired choices and explanations for basic mana,
  generic payment, X, APNAP, zero toughness, failed draws, and Commander zone changes.
- **Full-deck review** teaches complete 100-card list audits and focused
  reviews of the same lists. The targets cover named synergies, ramp and
  curve, recovery, bracket caveats, and specific swaps with tradeoffs.
  Brago's list is reserved for validation and the Zimone smoke case remains held out.

The review build uses the same generator with
`--output data/commander-curriculum-review`. Use that build for the next
continuation of the trained Qwen adapter. Review the standard smoke score
and the depth and factual accuracy of the deck review independently.

Named-card strategy questions can supply Oracle text and printed characteristics.
Closed-book questions omit retrieved rule passages and mana hints.
With `--full-rules-corpus`, the epoch also trains directly on the complete
prepared rules text, including the glossary. The run metadata records its
checksum, token count, and block count. The generator
keeps each new scenario family together in one split and runs the repository's
duplicate, reference, and reserved-question checks. Existing examples may still
cover related rules, cards, and concepts.

For evaluation, compare the established 30-question closed-book score and review
fresh gameplay and deckbuilding cases by track. Use actual playtest results and
deck audits to assess strategic quality. The authored deckbuilding targets are
starting packages and decision criteria. Complete deck legality, card prices,
current ban lists, and matchup performance require separate evidence.
