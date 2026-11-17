# Expanded supervised dataset

The second curriculum adds 64 authored scenarios in 32 paired groups. Each scenario has a free-form version and a multiple-choice version. The combined files retain the original curriculum.

| Split | Total examples | Original examples | New free-form | New multiple-choice |
| --- | ---: | ---: | ---: | ---: |
| Training | 184 | 88 | 48 | 48 |
| Validation | 54 | 22 | 16 | 16 |

The new training multiple-choice answers have twelve correct answers at each position A, B, C and D. Validation has four at each position. Each answer starts with the selected letter, states the matching option text, then gives a short explanation with rule citations.

The paired scenarios change conditions such as whose turn it is, who controls a hexproof creature, whether damage or a toughness reduction occurred, whether an attacker has trample, and whether a card is a token. Other pairs teach invariants, such as mana value staying the same when a Phyrexian cost is paid with life.

New material includes source-independent activated abilities, intervening-if clauses, optional triggers, hexproof, shroud, sacrifice, deathtouch with trample, first strike and double strike, haste, vigilance, prevention with lifelink, Aura and Equipment behavior, equip timing, counters and replacement effects. Named-card examples use the imported Oracle records.

All new paired conditions and format variants from one group stay in the same split. The optional `group` field is checked by the supervised loader. Training and validation still share MTG concepts, and the retained original data includes parameter variations across its splits. The smoke and evaluation questions remain reserved, including versions with answer choices appended.

The examples target the November 6, 2026 Comprehensive Rules and the imported November 17, 2026 Scryfall snapshot. Rule references, card names, complete tokenization and answer-letter consistency are validated against the local setup. Independent judge review is pending.

## Next experiment

Let the current matched-retrieval comparison finish before starting another GPU job. Keep its report for comparison.

The following commands start a fresh supervised run from the earlier corpus adapter. They use one epoch and a learning rate of `1e-5` as a controlled next experiment. The completed `runs/qwen3-sft` adapter and previous reports remain available.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python .\mtg.py sft --train-file qa/round2-train.jsonl --eval-file qa/round2-validation.jsonl --validate
& $python .\mtg.py sft --from-adapter runs/qwen3-full --train-file qa/round2-train.jsonl --eval-file qa/round2-validation.jsonl --learning-rate 1e-5 --epochs 1 --output runs/qwen3-sft-round2
& $python .\smoke_tests.py --adapter runs/qwen3-sft-round2 --output runs/smoke-sft-round2
```

Compare `runs/smoke-sft-round2/results.html` with `runs/smoke-current-retrieval/results.html`. Review the selected answers, explanations, contradictions and repetition. The effect of this expanded dataset on model accuracy has yet to be measured.

On Linux, activate the virtual environment and replace `& $python` with `python`.
