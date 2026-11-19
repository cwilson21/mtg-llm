"""Build a compact closed-book revision set with new cases and prior-task replay."""

import argparse
from collections import Counter
import json
from pathlib import Path

from build_commander_curriculum import Curriculum, fingerprint
from mana import COLORS, mana_value
from rules import RuleIndex
from supervised import load_datasets


def add_scenarios(data):
    number = 0

    def add(family, question, correct, explanation, distractors, rules, review=False):
        nonlocal number
        number += 1
        key = f"{family}-{number}"
        group = f"{family}-heldout" if review else family
        answer = f"{correct} {explanation}"
        data.add("rules-focus", group, key, question, answer, rules, validation=review)
        position = number % 4
        choices = list(distractors)
        choices.insert(position, correct)
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", choices, strict=True))
        data.add("rules-focus", group, f"{key}-mc",
                 f"{question}\n\n{options}\n\nChoose the single best answer. Start with exactly ANSWER: "
                 "and its letter. Then explain briefly.",
                 f"ANSWER: {'ABCD'[position]}\n\n{answer}", rules, validation=review)

    lands = dict(zip(COLORS, ("Plains", "Island", "Swamp", "Mountain", "Forest"), strict=True))
    for pair in (("B", "U"), ("U", "B"), ("B", "R"), ("B", "G"), ("R", "B"), ("B", "W"),
                 ("U", "R"), ("G", "W"), ("W", "B"), ("R", "G")):
        wanted, other = pair
        add("colored-mana",
            f"You have an untapped {lands[wanted]} and {lands[other]}. An activation has a printed cost "
            f"of {{{wanted}}}. Which of those lands can supply the payment, and what type of mana does it produce?",
            f"{lands[wanted]}, {COLORS[wanted]} mana.",
            f"The inherent ability of a {lands[wanted]} adds {{{wanted}}}. A {lands[other]} adds {{{other}}}.",
            (f"{lands[other]}, {COLORS[other]} mana.", "Either land, because colored requirements are interchangeable.",
             "Neither land, because a basic land only adds colorless mana."), ["107.4a", "305.6"],
            review=pair in {("W", "B"), ("R", "G")})
    for symbol in COLORS:
        add("colored-mana",
            f"Read this payment symbol without abbreviating its color: {{{symbol}}}. "
            "What color is required, and which basic land type has the matching intrinsic mana ability?",
            f"{COLORS[symbol].capitalize()}; {lands[symbol]}.",
            f"The symbol {{{symbol}}} is {COLORS[symbol]} mana. The basic land type {lands[symbol]} adds it.",
            tuple(f"{COLORS[s].capitalize()}; {lands[s]}." for s in COLORS if s != symbol)[:3],
            ["107.4a", "305.6"])
    for amount in range(1, 9):
        available = "{C}" * amount + "{W}"
        add("generic",
            f"Your mana pool contains {available}. Can you cast a spell costing {{{amount}}}{{W}} with that "
            "pool? Assume normal payment with no changes.",
            "Yes, the colorless mana pays the generic portion.",
            f"Pay {amount} colorless mana toward the numeric requirement and pay white for {{W}}. "
            "Generic costs accept any type, including colorless.",
            ("No, all the mana must be white.", "No, colorless mana cannot pay a numeric generic cost.",
             "Yes, because the white mana can pay twice."), ["107.4a", "107.4b", "107.4c"],
            review=amount == 8)
        add("generic",
            f"You have {amount} blue mana plus one green mana, and a spell costs {{{amount}}}{{C}}. "
            "Can you pay the entire printed cost?",
            "No. The {C} symbol requires a colorless mana.",
            "Blue or green can pay the generic part, but neither pays a specifically colorless requirement.",
            ("Yes, any mana can pay both parts.", "No, all of the mana must be colorless.",
             "Yes, green is colorless for the purpose of costs."), ["107.4b", "107.4c"],
            review=amount == 8)
    for i in range(18):
        symbol = "BURG W".replace(" ", "")[i % 5]
        x = (i * 3 + 2) % 8
        repeats = 1 + i % 2
        fixed = 1 + i % 3
        cost = "{X}" * repeats + f"{{{symbol}}}" + (f"{{{fixed}}}" if i % 2 else "")
        off = mana_value(cost)
        on = mana_value(cost, x)
        add("x-zone",
            f"A spell has mana cost {cost} and is cast with X={x}. What is its mana value on the stack and "
            f"what was its value as a card in {'exile' if i % 2 else 'its controller’s library'}?",
            f"{on} on the stack; {off} before casting.",
            f"The fixed symbols total {off}. On the stack each of the {repeats} X symbols contributes "
            f"{x}. Outside the stack they contribute zero.",
            (f"{off} on the stack; {on + 1} before casting.",
             f"{on + 1} on the stack; {off} before casting.",
             f"{on} on the stack; {on + 1} before casting."), ["202.3", "202.3e"],
            review=i in {16, 17})
    for power, toughness in ((1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9)):
        add("zero-toughness",
            f"An indestructible {power}/{toughness} creature with no marked damage gets "
            f"+1/-{toughness} until end of turn. Where is it after the next state-based-action check?",
            "In its owner’s graveyard.",
            "Its toughness is zero. That state-based action puts it in the graveyard without destroying it. "
            "Indestructible only prevents destruction.",
            ("On the battlefield because it is indestructible.", "In exile.", "On the battlefield until cleanup."),
            ["704.5f", "702.12b"], review=power == 7)
        add("zero-toughness",
            f"A {power}/{toughness} creature with indestructible is dealt {toughness} ordinary damage. "
            "Its toughness is otherwise unchanged. What happens at the next state-based-action check?",
            "It stays on the battlefield.",
            f"It still has toughness {toughness}. Marked damage is not a toughness reduction. "
            "Indestructible prevents the destruction normally caused by lethal damage.",
            ("It goes to the graveyard for zero toughness.", "It is exiled.", "It is destroyed for lethal damage."),
            ["120.3e", "704.5g", "702.12b"], review=power == 7)
    for left in range(7):
        for extra in (0, 1):
            if left + extra == 0:
                continue
            failed = extra == 1
            requested = left + extra
            correct = ("Finish resolving the spell, then lose at the next state-based-action check." if failed else
                       "Finish resolving the spell and stay in the game.")
            explanation = (
                f"The first {left} draws succeed and another draw is attempted from an empty library. "
                "The remaining spell instruction happens before the state-based check." if failed else
                "Each attempted draw succeeds. The empty library afterward does not cause a loss by itself."
            )
            options = ("Lose as soon as the library becomes empty.", "Lose before the spell can finish resolving.",
                       "Skip the instruction after the draw.") if not failed else (
                "Lose during the failed draw and stop resolving the spell.", "Finish resolving and lose at end of turn.",
                "Survive because a draw cannot be attempted from an empty library.")
            add("draw-timing",
                f"You start resolving 'Draw {requested} {'card' if requested == 1 else 'cards'}, then gain 2 life' "
                f"with {left} {'card' if left == 1 else 'cards'} in your library. No other effect applies. "
                "What happens to you and when?",
                correct, explanation, options, ["121.2", "121.4", "704.4", "704.5b"], review=left == 6)
    for verb in ("sacrificed to pay a cost", "destroyed by lethal damage",
                 "put into the graveyard for having zero toughness", "destroyed by a resolving sweeper"):
        add("commander-zone",
            f"Your nontoken commander is {verb} and its destination is its owner's graveyard. "
            "Trace the death event and any later choice about the command zone.",
            "It dies first, then its owner may move it to the command zone as a state-based action.",
            "A move from the battlefield to the graveyard is dying. The optional later move neither replaces nor undoes it.",
            ("It goes straight to the command zone without dying.", "It dies only if the owner declines the command zone.",
             "Its current controller must return it to the command zone immediately."), ["700.4", "903.9a"])
    for destination in ("exile", "its owner's hand", "the top of its owner's library"):
        exile = destination == "exile"
        add("commander-zone",
            f"A resolving spell would put a nontoken commander from the battlefield into {destination}. "
            "With no other change, does it die, and when can the owner choose the command zone?",
            ("It does not die. The owner can move it after it reaches exile, at a state-based-action check." if exile else
             "It does not die. Its owner may replace the move with a move to the command zone."),
            ("Exile is not the graveyard, and the command-zone option follows the move to exile." if exile else
             "The hand or library option is a replacement effect. Neither outcome sends it to the graveyard."),
            ("It dies and must move to the command zone.", "It dies only if it reaches the proposed destination.",
             "The choice is made before casting the spell."), ["700.4", "903.9a" if exile else "903.9b"],
            review="library" in destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/commander-curriculum-review"))
    parser.add_argument("--output", type=Path, default=Path("data/qwen3-8b-rules-focus"))
    args = parser.parse_args()
    data = Curriculum()
    data.rows = {"train": [], "validation": []}
    data.counts = {split: Counter() for split in data.rows}
    data.questions.clear()
    replay = {"legacy": 40, "rules-recovery": 52, "mechanics-practice": 16, "phyrexian-mana": 24,
              "gameplay": 20, "commander-brackets": 12, "deckbuilding": 12, "full-deck-review": 15}
    for split in data.rows:
        rows = data.read_jsonl(args.source / f"{split}.jsonl")
        for track, limit in replay.items():
            pool = [row for row in rows if row.get("track", "legacy") == track
                    and (track != "commander-brackets" or not row["group"].startswith(
                        ("commander-v1-commander-brackets-early-combo-", "commander-v1-commander-brackets-game-changer-")))]
            chosen = sorted(pool, key=lambda row: fingerprint(row["id"]))[:limit]
            data.rows[split].extend(chosen)
            data.counts[split][track] = len(chosen)
    add_scenarios(data)
    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in data.rows.items():
        with (args.output / f"{split}.jsonl").open("x", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    load_datasets(args.output / "train.jsonl", args.output / "validation.jsonl", RuleIndex(semantic=False))
    for split, counts in data.counts.items():
        print(split, dict(counts))


if __name__ == "__main__":
    main()
