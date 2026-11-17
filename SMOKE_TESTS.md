# Post-training smoke tests

The suite contains 30 questions with four answer choices, an expected answer, and rule references. It covers mana symbols, generic and colorless costs, X, hybrid and Phyrexian mana, priority, legal targets, state-based actions, combat, Commander and named cards.

The questions are in `smoke_questions.json`. You can read that file before running the model, or add your own cases using the same fields.

## Run after training finishes

In your existing project folder, open a second PowerShell window after the full training command finishes and saves `runs/qwen3-full`.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python .\smoke_tests.py --adapter runs/qwen3-full
Invoke-Item .\runs\post-training-smoke\results.html
```

The HTML report shows the expected choice, expected explanation, generated answer, retrieved rules and selected cards for every question. The JSON report at `runs/post-training-smoke/results.json` contains the same results plus the model revision, training record, source-data manifests and question-file hash.

Each completed question is saved immediately, so an interrupted test run leaves its completed answers available. Re-running with the same output folder replaces that report. Use `--output` to keep multiple reports.

## Start with a few questions

```powershell
& $python .\smoke_tests.py --adapter runs/qwen3-full --limit 5 --output runs/smoke-first-five
```

To check only mana questions, use the mana group. The available groups are `mana`, `priority`, `state`, `combat`, `commander` and `cards`. Repeat `--group` to select several.

```powershell
& $python .\smoke_tests.py --adapter runs/qwen3-full --group mana --output runs/smoke-mana
```

To compare the trained adapter with the original model, run both with identical context.

```powershell
& $python .\smoke_tests.py --adapter runs/qwen3-full --compare-base --output runs/smoke-comparison
```

This loads the base and adapter together once and disables the adapter for each base answer. It performs two generations per question.

## Read the scores

- `correct_choice` means the generated answer selected the expected letter.
- `wrong_choice` means it selected another letter.
- `unparsed` means the response did not begin with the requested `ANSWER: A` format, so its choice needs manual review.

Review explanations against the expected answer, even when the selected letter matches. Check for contradictions, incorrect numbers, missing conditions and unsupported citations. The automatic score covers the answer choice. Explanation review remains marked `pending` in the JSON.

The suite exercises the application's normal question-answering path, including retrieved rule passages, card Oracle text, mana guidance and computed cost totals. Its score measures the model with those tools available. Expected choices and expected explanations are used for scoring and reports. They are excluded from the model's prompt. Choice text is shown as part of each multiple-choice question.

The expected rulings target the November 6, 2026 Comprehensive Rules and your November 17, 2026 Scryfall snapshot. Review the answers when replacing either source. A retrieved or cited rule number alone does not establish that the explanation is correct.

## Validate while training is running

This checks all questions, expected choices, rule IDs and card names without loading a model or using CUDA.

```powershell
& $python .\smoke_tests.py --validate
```

The add-on has passed this validation against your imported rules and cards. Report generation, choice scoring and comparison behavior were also checked using simulated model responses. Model answers from the currently running full training have yet to be evaluated.
