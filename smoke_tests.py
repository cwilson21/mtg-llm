"""Run MTG questions with expected answers after training finishes."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sqlite3
import sys

from cards import DATABASE, CardIndex, open_cards
from mana import computed_costs
from mtg import BASE_MODEL, load_for_answers, reply
from rules import RuleIndex, write_text


def load_suite(path):
    suite = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    if not suite.get("cases"):
        raise ValueError("The question file must have a nonempty cases list.")
    for case in suite["cases"]:
        for field in ("id", "group", "question", "choices", "correct_choice", "expected_answer", "rules", "cards"):
            if field not in case:
                raise ValueError(f"A case is missing {field!r}.")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in seen:
            raise ValueError(f"Case IDs must be unique nonempty strings: {case['id']!r}")
        seen.add(case["id"])
        if set(case["choices"]) != set("ABCD") or case["correct_choice"] not in case["choices"]:
            raise ValueError(f"{case['id']}: provide choices A-D and a correct_choice.")
        if not all(isinstance(text, str) and text.strip() for text in (
            case["group"], case["question"], case["expected_answer"], *case["choices"].values()
        )):
            raise ValueError(f"{case['id']}: question, group, answer and choices must contain text.")
        if not all(isinstance(items, list) and all(isinstance(item, str) for item in items)
                   for items in (case["cards"], case["rules"])):
            raise ValueError(f"{case['id']}: cards and rules must be lists of strings.")
    return suite


def question_prompt(case):
    choices = "\n".join(f"{letter}. {case['choices'][letter]}" for letter in "ABCD")
    return (
        case["question"] + "\n\n" + choices
        + "\n\nChoose the single best answer. Start with exactly ANSWER: A, ANSWER: B, "
        "ANSWER: C, or ANSWER: D on the first line. Then explain briefly and cite "
        "applicable rules. Use only one answer letter."
    )


def score_answer(case, answer):
    match = re.match(r"\s*ANSWER:[ \t]*([ABCD])[ \t]*(?:\r?\n|$)", answer, re.IGNORECASE)
    choice = match.group(1).upper() if match else None
    status = "unparsed" if choice is None else (
        "correct_choice" if choice == case["correct_choice"] else "wrong_choice"
    )
    cited = set(re.findall(r"\b[1-9]\d{2}\.\d+[a-z]*", answer.lower()))
    return {
        "answer": answer, "selected_choice": choice, "status": status,
        "expected_rules_cited": [rule for rule in case["rules"] if rule in cited],
        "explanation_review": "pending",
    }


def summary(rows, label):
    results = [row[label] for row in rows if label in row]
    return {
        "answered": len(results),
        "correct_choice": sum(result["status"] == "correct_choice" for result in results),
        "wrong_choice": sum(result["status"] == "wrong_choice" for result in results),
        "unparsed": sum(result["status"] == "unparsed" for result in results),
    }


def report_html(report):
    escape = lambda text: html.escape(str(text))
    pieces = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>MTG smoke test results</title>",
        "<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1040px;margin:32px auto;padding:0 20px}"
        "article{border-top:1px solid #bbb;padding:22px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "table{border-collapse:collapse}td,th{padding:6px 12px;border:1px solid #bbb;text-align:left}"
        ".correct_choice{color:#176138}.wrong_choice,.unparsed{color:#a12424}</style>",
        "<h1>MTG smoke test results</h1>",
        f"<p>Adapter {escape(report['adapter'])}. Completed {len(report['cases'])} "
        f"of {report['selected_cases']} questions.</p>",
        "<p>The automatic score checks the selected answer letter. Review the explanation and citations "
        "against the expected answer. A matching citation alone does not validate the reasoning.</p>",
        "<table><tr><th>Model</th><th>Correct choice</th><th>Wrong choice</th><th>Unparsed</th></tr>",
    ]
    for label, totals in report["summary"].items():
        pieces.append(
            f"<tr><td>{escape(label)}</td><td>{totals['correct_choice']}</td>"
            f"<td>{totals['wrong_choice']}</td><td>{totals['unparsed']}</td></tr>"
        )
    pieces.append("</table>")
    for row in report["cases"]:
        pieces.extend([
            f"<article><h2>{escape(row['id'])}</h2><p>{escape(row['question'])}</p>",
            "<ul>" + "".join(f"<li>{letter}. {escape(row['choices'][letter])}</li>" for letter in "ABCD") + "</ul>",
            f"<h3>Expected {row['correct_choice']}</h3><p>{escape(row['expected_answer'])}</p>",
            f"<p>Reference rules {escape(', '.join(row['rules']) or 'none')}. "
            f"Retrieved rules {escape(', '.join(row['retrieved_rules']))}. "
            f"Cards {escape(', '.join(row['retrieved_cards']) or 'none')}.</p>",
        ])
        for label in ("base", "adapter"):
            if label in row:
                result = row[label]
                pieces.append(
                    f"<h3>{label.title()} <span class=\"{result['status']}\">{escape(result['status'])}</span></h3>"
                    f"<pre>{escape(result['answer'])}</pre>"
                )
        pieces.append("</article>")
    return "\n".join(pieces) + "</html>"


def save_report(report, output):
    report["summary"] = {
        label: summary(report["cases"], label) for label in ("adapter", "base")
        if label == "adapter" or report["compare_base"]
    }
    output.mkdir(parents=True, exist_ok=True)
    write_text(output / "results.json", json.dumps(report, ensure_ascii=False, indent=2))
    write_text(output / "results.html", report_html(report))


def run(args, suite, cases, index, card_index):
    # Validate every selected reference before allocating model memory.
    for case in cases:
        unknown = [rule for rule in case["rules"] if rule not in index.by_id]
        if unknown:
            raise ValueError(f"{case['id']}: missing rules {', '.join(unknown)}")
        if case["cards"]:
            if card_index is None:
                raise ValueError("The selected cases need data/cards.sqlite3. Run cards.py prepare.")
            card_index.for_question(case["question"], case["cards"])
    if args.validate:
        print(f"Validated {len(cases)} questions, expected choices, rule IDs and card names.")
        return
    if not (args.adapter / "run.json").is_file():
        raise ValueError(f"{args.adapter} has no run.json yet. Run this after training finishes.")
    model, tokenizer, record = load_for_answers(args)
    report = {
        "suite": suite["suite"], "suite_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "adapter": str(args.adapter.resolve()), "training_run": record,
        "rules": index.manifest, "cards": card_index.manifest if card_index else None,
        "expected_rules_edition": suite.get("rules_edition"),
        "compare_base": args.compare_base, "selected_cases": len(cases),
        "context": "Application lookup with retrieved rules, card text and mana helpers",
        "cases": [],
    }
    if index.manifest["effective"] != suite.get("rules_edition"):
        print("The imported rules edition differs from the suite. Review expected answers for this edition.")
    save_report(report, args.output)
    for position, case in enumerate(cases, 1):
        hits = index.search(case["question"])
        cards = card_index.for_question(case["question"], case["cards"]) if card_index else []
        prompt = question_prompt(case)
        row = {
            **case, "retrieved_rules": [passage.id for passage in hits],
            "retrieved_cards": [card["name"] for card in cards],
            "expected_rules_retrieved": [rule for rule in case["rules"] if any(p.id == rule for p in hits)],
            "computed_costs": computed_costs(case["question"], cards),
        }
        if args.compare_base:
            with model.disable_adapter():
                row["base"] = score_answer(case, reply(model, tokenizer, prompt, hits, cards))
        row["adapter"] = score_answer(case, reply(model, tokenizer, prompt, hits, cards))
        report["cases"].append(row)
        save_report(report, args.output)
        print(f"{position}/{len(cases)} {case['id']}: {row['adapter']['status']}", flush=True)
    print(json.dumps(report["summary"], indent=2))
    print(f"Open {(args.output / 'results.html').resolve()}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=Path("runs/qwen3-full"))
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("smoke_questions.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/post-training-smoke"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--group", action="append", help="Select a group. Repeat to include multiple groups.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected questions.")
    parser.add_argument("--compare-base", action="store_true", help="Also run the original model with identical context.")
    parser.add_argument("--validate", action="store_true", help="Check the suite and local data without loading the model.")
    args = parser.parse_args()
    args.model = BASE_MODEL
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    try:
        suite = load_suite(args.questions)
        groups = {case["group"] for case in suite["cases"]}
        if args.group and set(args.group) - groups:
            raise ValueError(f"Choose groups from {', '.join(sorted(groups))}.")
        cases = [case for case in suite["cases"] if not args.group or case["group"] in args.group]
        if args.limit:
            cases = cases[:args.limit]
        index = RuleIndex()
        with open_cards() if DATABASE.is_file() else nullcontext() as conn:
            run(args, suite, cases, index, CardIndex(conn) if conn else None)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
