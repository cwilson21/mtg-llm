"""Generate closed-book SFT data from the local rules, Oracle DB, and authored cases."""

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from math import comb
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cards import CardIndex, format_card, open_cards
from commander_brackets import SOURCE as BRACKET_SOURCE, add_bracket_catalogs, add_brackets
from commander_curriculum import COMMANDERS, DECK_DECISIONS, PLAY_CASES
from full_deck_curriculum import add_full_deck_reviews
from keyword_curriculum import add_keyword_curriculum
from rules_recovery_curriculum import add_rules_recovery
from rules import RuleIndex
from supervised import load_datasets, normalize_question


def fingerprint(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Curriculum:
    def __init__(self):
        self.rows = {
            "train": self.read_jsonl(ROOT / "qa/falcon3-followup-train.jsonl"),
            "validation": self.read_jsonl(ROOT / "qa/rank32-validation.jsonl"),
        }
        self.counts = {split: Counter(legacy=len(rows)) for split, rows in self.rows.items()}
        self.questions = {normalize_question(row["question"]) for rows in self.rows.values() for row in rows}

    @staticmethod
    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def add(self, track, family, key, question, answer, rules, cards=(), validation=False, *, source=None):
        split = "validation" if validation else "train"
        normalized = normalize_question(question)
        if normalized in self.questions:
            raise ValueError(f"Duplicate curriculum question: {question}")
        self.questions.add(normalized)
        row = {
            "id": f"commander-v1-{track}-{key}",
            "group": f"commander-v1-{track}-{family}",
            "track": track,
            "question": question,
            "answer": (answer + " " + " ".join(f"[CR {rule}]" for rule in rules)).rstrip(),
            "rules": list(rules),
            "cards": list(cards),
        }
        if source:
            row["source"] = source
        self.rows[split].append(row)
        self.counts[split][track] += 1


def add_rules(data, index):
    for passage in index.passages:
        family = re.match(r"\d+\.\d+", passage.id)
        if not family or len(passage.text) > 5000:
            continue
        family = family.group()
        data.add("rules", family, passage.id,
                 f"State the operative text of Magic Comprehensive Rule {passage.id} ({passage.heading}).",
                 passage.text, [passage.id], validation=int(fingerprint(family)[:8], 16) % 16 == 0)


def stratified_cards(db, limit):
    buckets = defaultdict(list)
    for (payload,) in db.execute("SELECT card_json FROM cards ORDER BY oracle_id"):
        card = json.loads(payload)
        key = normalize_question(card["name"])
        if not key or not card.get("type_line"):
            continue
        # Prefer a single normalized card name if the snapshot contains colliding names.
        type_line = card["type_line"].split(" —", 1)[0]
        kind = next((kind for kind in ("Land", "Creature", "Planeswalker", "Instant", "Sorcery",
                                      "Artifact", "Enchantment", "Battle") if kind in type_line), "other")
        buckets[kind].append((fingerprint(card["oracle_id"]), key, card))
    # Interleave broad types instead of following print order.
    queues = [enumerate(sorted(bucket, key=lambda item: item[0])) for bucket in buckets.values()]
    seen = set()
    selected = heapq.merge(*queues, key=lambda item: item[0])
    for _, (_, key, card) in selected:
        if key in seen:
            continue
        seen.add(key)
        yield card
        if limit and len(seen) == limit:
            break


def add_cards(data, card_index, limit):
    for card in stratified_cards(card_index.db, limit):
        name = card["name"]
        family = card["oracle_id"]
        holdout = int(fingerprint(family)[:8], 16) % 20 == 0
        text = format_card(card, include_mana_hints=False)
        if len(text) > 4500:
            continue
        # A recall target comes from Oracle; the card is intentionally absent from the prompt.
        data.add("oracle", family, family, f"Give the Oracle profile for the Magic card named {name}.",
                 text, ["109.3"], validation=holdout)


def add_play(data):
    held_out = {"recovery"}
    for i, (family, question, answer, rules, cards) in enumerate(PLAY_CASES):
        data.add("gameplay", family, str(i), question, answer, rules, cards, family in held_out)

    # Quantities vary the correct result, including scenarios where blocking is optional,
    # mandatory for survival, or insufficient even against an ordinary trampler.
    for power in range(4, 10):
        for toughness in range(2, 6):
            for life in (3, 6, 9):
                damage = max(0, power - toughness)
                if damage >= life:
                    decision = "Blocking is insufficient to survive the maximum legal assignment."
                elif power >= life:
                    decision = "Blocking is necessary and sufficient to survive this combat."
                else:
                    decision = "Both lines survive this combat; decide whether the creature is worth preserving."
                question = (
                    f"You are at {life} life. An opponent attacks you with a {power}/{power} trampler. "
                    f"Your only blocker is an undamaged 1/{toughness}. No other effects apply. "
                    "Compare your survival if you take the attack or block, assuming the attacker maximizes player damage."
                )
                answer = (
                    f"Unblocked you take {power}, leaving {life - power} life before state-based actions. "
                    f"Blocking reduces the maximum damage to {damage} because the trampler must first assign "
                    f"lethal damage to the blocker before assigning excess to you. {decision}"
                )
                family = f"trample-power-{power}"
                data.add("gameplay", family, f"race-{power}-{toughness}-{life}", question, answer,
                         ["702.19b", "104.3b"], validation=power == 8)

    for active_seat in range(4):
        for empty_seat in range(4):
            seats = ("Amber", "Beck", "Cora", "Devi")
            active, empty = seats[active_seat], seats[empty_seat]
            placement = [seats[(active_seat + offset) % 4] for offset in range(4)]
            placement.remove(empty)
            data.add(
                "gameplay", f"apnap-{active}", f"apnap-{active}-{empty}",
                f"Clockwise turn order is Amber, Beck, Cora, Devi. On {active}'s turn a creature dies. "
                f"Everyone except {empty} controls exactly one ordinary trigger waiting to go on the stack. "
                "With all players passing and no further effects, give the resolution order.",
                f"First to last: {', '.join(reversed(placement))}. The triggers were placed in APNAP order "
                f"starting from the active seat and skipping the player without a trigger: {', '.join(placement)}. "
                "Each consecutive pass sequence resolves the next stack object from the top.",
                ["101.4", "603.3b", "117.4"], validation=active_seat == 3,
            )

    for life in (4, 6, 8, 10):
        for attackers in ((2, 3), (2, 5), (3, 4), (4, 5)):
            for blocker in (False, True):
                first, second = attackers
                damage = first if blocker else first + second
                lethal = damage >= life
                board = ("one untapped 7/7 that can block either attacker"
                         if blocker else "no creatures that can block")
                result = ("That is lethal this combat." if lethal else
                          "The opponent survives this combat.")
                data.add(
                    "gameplay", f"lethal-at-{life}", f"lethal-{life}-{first}-{second}-{int(blocker)}",
                    f"Your opponent is at {life} life and has {board}. You can attack with a vanilla "
                    f"{first}/{first} and {second}/{second}; nobody has tricks. If you attack with both "
                    "and the opponent blocks to minimize life loss, how much damage gets through, and is it lethal?",
                    (f"{damage} damage. " +
                     ("The 7/7 blocks the larger attacker. Without trample, only the smaller attacker hits. "
                      if blocker else "Both attackers are unblocked and deal their combined power. ") + result),
                    ["509.1a", "510.1c", "510.1b", "104.3b"], validation=life == 8,
                )

    for generic in (0, 1, 2, 3):
        for blue in (1, 2, 3):
            for islands in (2, 3, 4):
                forests = 2
                can_hold = islands >= blue + 2 and islands + forests >= generic + blue + 2
                cost = (f"{{{generic}}}" if generic else "") + "{U}" * blue
                ruling = ("Yes, use Forests for generic mana first and reserve two Islands."
                          if can_hold else "No, this configuration cannot pay both spells in sequence.")
                data.add(
                    "gameplay", f"reserve-blue-{blue}", f"reserve-{generic}-{blue}-{islands}",
                    f"During your main phase you have {islands} untapped basic Islands and {forests} "
                    f"untapped basic Forests. Can you cast a spell costing {cost} and still leave {{U}}{{U}} "
                    "to cast Counterspell later this turn? No other mana or cost changes apply.",
                    f"{ruling} The sequence needs {blue + 2} blue mana and {generic + blue + 2} total mana. "
                    "Compare those requirements with the available Islands and total lands.",
                    ["305.6", "107.4a", "107.4b", "601.2f"], validation=blue == 3,
                )


def add_deckbuilding(data, card_index):
    held_out = {"Sythis, Harvest's Hand", "Brago, King Eternal", "Borborygmos Enraged", "Zada, Hedron Grinder"}
    for i, (commander, plan, guidance, package, interaction) in enumerate(COMMANDERS):
        for name in (commander, *package):
            if len(card_index.get(name)) != 1:
                raise ValueError(f"Expected one Oracle record for {name!r}")
        cards = (commander, *package)
        heldout = commander in held_out
        data.add("deckbuilding", str(i), f"package-{i}",
                 f"Design a starting Commander package for {commander} around {plan}. "
                 f"Assess the roles of {', '.join(package)} and describe what the rest of the deck needs.",
                 f"{guidance} {interaction} Treat this as a package inside a 100-card deck including the commander.",
                 ["903.5a", "903.5c"], cards, heldout)
        data.add("deckbuilding", str(i), f"resilience-{i}",
                 f"Audit a {commander} Commander brew built around {plan}. It repeatedly stalls when the commander "
                 "costs four extra mana to recast. Propose changes and a way to test them.",
                 f"Keep the plan coherent: {guidance} {interaction} Use cheap independent development and redundant "
                 "engines so the deck can operate without paying the full recast cost. Goldfish with the commander "
                 "unavailable, then test against live opponents and track first meaningful play and recovery after interaction.",
                 ["903.8", "903.5a"], cards, heldout)
        data.add("deckbuilding", str(i), f"slot-review-{i}",
                 f"You are cutting a {commander} draft down from 107 cards. The plan is {plan}. "
                 f"You are considering cutting {', '.join(package)} or unrelated high-cost cards. "
                 "Explain how to prioritize the cuts and what game evidence could reverse the decision.",
                 f"{interaction} Keep enough low-cost mana and interaction to execute the plan. "
                 "Cut expensive cards that duplicate existing jobs or that only help when the deck is already ahead. "
                 "Challenge each package card with actual playtest evidence: count where it is castable on time, "
                 "where it advances the plan, and where another role would have changed the outcome. "
                 "Make seven net cuts and verify that the resulting total includes the commander.",
                 ["903.5a"], cards, heldout)

    for i, (commander, candidates, question, answer, rules) in enumerate(DECK_DECISIONS):
        data.add(
            "deckbuilding", str(i), f"candidate-fit-{i}",
            f"For a Commander deck led by {commander}, compare {candidates[0]} and {candidates[1]}. {question}",
            answer, rules, (commander, *candidates), commander in held_out,
        )

    for target in (4, 5, 6, 7, 8):
        for ramp_turn, ramp_cost in ((2, 2), (3, 2), (3, 3), (4, 3), (4, 4)):
            earliest = max(ramp_turn + 1, target - 1)
            accelerated = earliest < target
            comparison = (
                f"The ramp advances the commander from turn {target} to turn {earliest}."
                if accelerated else
                f"Cast the commander on turn {target} using land drops alone. Committing to the ramp spell "
                f"first would allow the commander on turn {earliest}, so it does not accelerate this goal."
            )
            data.add(
                "deckbuilding", f"curve-target-{target}", f"ramp-{target}-{ramp_turn}-{ramp_cost}",
                f"Your commander costs {target} mana. You play one untapped, one-mana-producing land each turn "
                f"and have all required colors. On turn {ramp_turn} you first get access to a {ramp_cost}-mana "
                "ramp spell that puts an extra basic land onto the battlefield tapped. There is no other "
                "acceleration or interaction. If you want the earliest possible commander, does casting the ramp help?",
                comparison + " A tapped land first contributes on a later turn. "
                "Evaluate ramp slots against the turn and colors they can actually advance.",
                ["305.1", "601.2f"], validation=target == 7,
            )

    # Explicitly model lands as sources in a sample with no mulligans, tutors or extra draw.
    for sources in (10, 14, 18, 22, 26, 30, 34, 38):
        for draws in (7, 9, 11):
            probability = (1 - comb(99 - sources, draws) / comb(99, draws)) * 100
            data.add(
                "deckbuilding", f"source-count-{sources}", f"source-{sources}-{draws}",
                f"Your one-commander deck has {sources} sources of an early required color among the 99 shuffled cards. "
                f"If you see {draws} distinct cards from that deck with no mulligans or selection, "
                "what is the probability of seeing at least one source? What does that estimate leave out?",
                f"About {probability:.1f}%. Subtract the probability of zero sources, "
                f"C({99 - sources},{draws})/C(99,{draws}), from one. To plan actual casting, separately consider "
                "whether those sources enter untapped, produce the needed color on schedule, and provide enough total mana.",
                ["903.5a", "103.5"], validation=sources in (14, 34),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/commander-curriculum")
    parser.add_argument("--cards", type=int, default=2400, help="Oracle sample size. Use 0 for the entire snapshot.")
    parser.add_argument("--combo-snapshot", type=Path, help="Include EDHREC combos and Scryfall Game Changer flags.")
    args = parser.parse_args()
    if args.cards < 0:
        parser.error("--cards must be nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose an empty output directory")
    index = RuleIndex(semantic=False)
    data = Curriculum()
    add_rules(data, index)
    add_play(data)
    add_brackets(data)
    keyword_coverage = add_keyword_curriculum(data, index)
    add_rules_recovery(data)
    with open_cards() as db:
        cards = CardIndex(db)
        add_cards(data, cards, args.cards)
        add_deckbuilding(data, cards)
        bracket_catalogs = None
        early_combos = ()
        if args.combo_snapshot:
            combo_snapshot = json.loads(args.combo_snapshot.read_text(encoding="utf-8"))
            bracket_catalogs = add_bracket_catalogs(data, cards, combo_snapshot,
                                                   combo_digest=digest(args.combo_snapshot))
            early_combos = combo_snapshot["combos"]
        add_full_deck_reviews(data, cards, early_combos)
        card_manifest = cards.manifest
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {split: args.output / f"{split}.jsonl" for split in data.rows}
    for split, path in paths.items():
        with path.open("x", encoding="utf-8") as handle:
            for row in data.rows[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    load_datasets(paths["train"], paths["validation"], index)
    manifest = {
        "rules": index.manifest, "cards": card_manifest, "requested_oracle_profiles": args.cards,
        "commander_brackets_source": BRACKET_SOURCE,
        "commander_bracket_catalogs": bracket_catalogs,
        "keyword_coverage": keyword_coverage,
        "counts": {split: dict(counts) for split, counts in data.counts.items()},
        "sha256": {path.name: digest(path) for path in paths.values()},
    }
    summary = json.dumps(manifest, indent=2) + "\n"
    (args.output / "manifest.json").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
