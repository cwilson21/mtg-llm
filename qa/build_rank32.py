"""Rebuild the rank-32 curriculum using the prepared local rules and Oracle DB.

Run from the project directory: python qa/build_rank32.py
No network, model, or held-out answer keys are used to generate examples.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cards import CardIndex, open_cards
from mana import COLORS, mana_value
from rules import RuleIndex


def build():
    data = {split: [json.loads(line) for line in
                   (ROOT / "qa" / f"round2-{split}.jsonl").read_text(encoding="utf8").splitlines()]
            for split in ("train", "validation")}
    counts = Counter()

    def add(group, question, correct, wrong, explanation, rules, cards=(), validation=False):
        split = "validation" if validation else "train"
        assert len(wrong) == 3 and len(set([correct, *wrong])) == 4
        number = counts[group]
        counts[group] += 1
        common = {"group": f"rank32-{group}", "rules": rules, "cards": list(cards),
                  "source": "Authored rule application or deterministic Oracle/mana derivation; see RANK32.md."}
        citations = " ".join(f"[CR {rule}]" for rule in rules)
        identity = f"r32-{group}-{number}"
        data[split].append({**common, "id": identity + "-free", "format": "freeform",
                            "question": question, "answer": f"{correct} {explanation} {citations}"})
        position = sum(row.get("format") == "multiple_choice" and row["id"].startswith("r32-")
                       for row in data[split]) % 4
        choices = list(wrong)
        choices.insert(position, correct)
        data[split].append({**common, "id": identity + "-mc", "format": "multiple_choice",
                            "correct_choice": "ABCD"[position], "choices": dict(zip("ABCD", choices)),
                            "question": question + "\n\n" + "\n".join(
                                f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)) +
                            "\n\nChoose one answer. Start with ANSWER: and its letter. Then explain and cite the rules.",
                            "answer": f"ANSWER: {'ABCD'[position]}\n\n{correct} {explanation} {citations}"})

    # Each entry explicitly states its assumptions. Related cases stay together.
    scenarios = [
        ("ward", "An opponent targets your ward {2} creature with a legal destroy spell and declines to pay when ward resolves. What happens?", "The destroy spell is countered.", ["Your creature is sacrificed.", "The spell resolves normally.", "The target becomes illegal immediately."], "Ward triggers on targeting and counters the spell unless its controller pays.", ["702.21a"]),
        ("ward", "You target your own ward {2} creature with a legal spell. Must you pay ward?", "No, ward does not trigger for your own spell.", ["Yes, always pay {2}.", "Your spell is automatically countered.", "Ward prevents all targeting."], "Ward triggers for spells or abilities controlled by an opponent.", ["702.21a"]),
        ("ward", "An opponent's uncounterable spell legally targets your ward {2} creature. They decline to pay ward. Does ward counter that spell?", "No, the spell cannot be countered.", ["Yes, ward overrides uncounterable.", "Ward changes the spell's target.", "The creature is exiled instead."], "Ward still triggers, but the prohibition on countering wins.", ["702.21a", "101.2"]),
        ("protection", "Your creature has protection from red. Can you target it with your own red instant?", "No, protection prohibits that targeting.", ["Yes, protection only affects opponents.", "Yes, if the instant is helpful.", "Yes, if you pay an extra {2}."], "The quality of the spell matters regardless of its controller.", ["702.16b"]),
        ("protection", "A red source would deal 4 damage to a creature with protection from red. No effect stops prevention. How much damage is dealt?", "Zero damage is dealt.", ["Four damage is dealt.", "Two damage is dealt.", "Protection destroys the source."], "Protection prevents damage from a source of the stated quality.", ["702.16e"]),
        ("protection", "A creature with protection from white is affected by a white spell saying 'Destroy all creatures.' It does not target or deal damage. Does protection save the creature?", "No, it is destroyed.", ["Yes, protection prevents all white effects.", "Yes, every destroy effect targets.", "It returns to hand instead."], "Protection's restrictions do not stop a nontargeted destruction instruction.", ["702.16b", "702.16e", "115.10a"]),
        ("evasion", "An untapped ground creature with no relevant abilities tries to block a flying attacker. Is the block legal?", "No, it needs flying or reach.", ["Yes, all creatures can block flyers.", "Yes, if its toughness is higher.", "Only if it entered this turn."], "Flying restricts which creatures may block the attacker.", ["702.9b"]),
        ("evasion", "An untapped flying creature with no blocking restrictions tries to block a nonflying attacker. Is that legal?", "Yes, flying can block a ground attacker.", ["No, flying can block only flying.", "No, it needs reach too.", "Only during its controller's turn."], "Flying does not restrict the creature to blocking other flyers.", ["702.9b"]),
        ("defender", "A creature has both defender and haste, with no other relevant effects. Can haste let it attack?", "No, defender still prohibits attacking.", ["Yes, haste removes defender.", "Yes, but only once.", "Only if it entered this turn."], "Haste removes the control-duration restriction, while defender's prohibition remains.", ["702.3b", "702.10b", "101.2"]),
        ("first-strike-duel", "An attacking 2/2 with first strike is blocked by an ordinary 2/2. Neither has other abilities or damage. What survives combat?", "The first-strike attacker survives; the blocker dies.", ["Both die simultaneously.", "Only the blocker survives.", "Both survive without damage."], "The attacker deals lethal damage in the first damage step; the blocker dies before it can deal normal combat damage.", ["702.7b", "704.5g"]),
        ("first-strike-duel", "An attacking 2/2 with first strike is blocked by an ordinary 3/3. No other effects apply. What survives combat?", "The blocker survives; the attacker dies.", ["The attacker survives; the blocker dies.", "Both die.", "Neither deals any damage."], "Two first-strike damage is not lethal to the blocker. It then deals three damage in the normal damage step.", ["702.7b", "704.5g"]),
        ("damage-marking", "An ordinary 5/5 creature takes 2 damage. No other effects apply. What are its power and toughness afterward?", "It remains 5/5 with 2 damage marked.", ["It becomes 5/3.", "It becomes 3/3.", "It becomes 3/5."], "Ordinary damage is marked on a creature; it does not reduce toughness.", ["120.3e"]),
        ("damage-marking", "An ordinary 5/5 has 2 damage already marked and then takes 3 more this turn. What happens at the next state-based-action check?", "It is destroyed for lethal marked damage.", ["It becomes a 0/0.", "Only the most recent 3 damage matters.", "Nothing happens until the end step."], "Marked damage accumulates; five damage is lethal for toughness five.", ["120.3e", "704.5g"]),
        ("draw-empty", "Your library becomes empty, but you have not attempted to draw from it. Does that fact alone make you lose?", "No, an empty library alone does not cause a loss.", ["Yes, immediately.", "Yes, at the next end step regardless of draws.", "You must shuffle your graveyard into it."], "The state-based loss requires an attempted draw from an empty library.", ["121.4", "704.5b"]),
        ("draw-empty", "A resolving instruction says draw three cards, but your library has only two. With no replacement effects, what happens?", "You draw two cards and lose at the next state-based-action check.", ["You draw nothing because the whole instruction fails.", "You draw two and never lose.", "You shuffle the graveyard automatically."], "Draws happen individually; the third draw is attempted from an empty library.", ["121.2", "121.4"]),
        ("draw-wording", "An effect puts the top card of your library into your hand without saying draw. Does an ability that triggers whenever you draw a card trigger?", "No, this action is not a draw.", ["Yes, every library-to-hand move is a draw.", "Only if the card is a land.", "Only during the draw step."], "Moving cards into hand counts as drawing only when the instruction uses draw.", ["121.5"]),
        ("mana-ability", "You activate a basic Forest's normal mana ability. Can an opponent respond to that ability before the mana is added?", "No, it resolves immediately without using the stack.", ["Yes, every activated ability uses the stack.", "Yes, if the opponent has an instant.", "Only the active player can respond."], "This is an activated mana ability.", ["305.6", "605.1a", "605.3b"]),
        ("mana-ability", "A hypothetical activated ability says 'Add {G}. Target creature gets +1/+1 until end of turn.' Is it an activated mana ability?", "No, it has a target and uses the stack normally.", ["Yes, every ability that adds mana is a mana ability.", "Yes, because the mana is green.", "Only if it costs no mana."], "An ability with a target fails the mana-ability criteria.", ["605.1a", "605.5a"]),
        ("crew", "You just gained control of an untapped 2/2 with no haste. Can you tap it to pay crew 2 on a Vehicle you control?", "Yes, it can pay the crew cost.", ["No, summoning sickness forbids all tapping.", "Only if the Vehicle has haste.", "Only on an opponent's turn."], "Crew taps creatures as its cost; it does not activate their own ability with a tap symbol.", ["302.6", "702.122a"]),
        ("crew", "You control a Vehicle with crew 3 and two untapped creatures of powers 1 and 2. Can you tap both to crew it?", "Yes, their combined power is 3.", ["No, one creature must have power 3.", "No, crew requires three creatures.", "Only if both have haste."], "Crew permits any number of other untapped creatures with sufficient total power.", ["702.122a"]),
        ("land-turn", "An effect permits an extra land play, but it is your opponent's turn. Can you play a land from hand because of that extra allowance alone?", "No, you cannot play a land on another player's turn.", ["Yes, extra allowances override whose turn it is.", "Yes, if the land is basic.", "Only if you pay {1}."], "The turn restriction applies even when an effect gives additional land plays.", ["305.2", "305.3"]),
        ("flashback", "You cast an instant using flashback. An opponent counters that spell. Where does its card go?", "It is exiled.", ["It returns to the graveyard.", "It returns to hand.", "It goes on top of the library."], "Paying flashback exiles the card instead of putting it elsewhere any time it leaves the stack.", ["702.34a"]),
        ("flashback", "You cast an instant using flashback and it resolves normally. Where does its card go after resolution?", "It is exiled.", ["It returns to the graveyard.", "It returns to hand.", "It remains on the stack."], "The flashback replacement applies when the card would leave the stack after resolution too.", ["702.34a"]),
        ("kicker", "A card costs {2}{R} and has kicker {1}{R}. You cast it kicked with no other cost changes. What are the total cost and mana value?", "Pay {3}{R}{R}; mana value is 3.", ["Pay {3}{R}{R}; mana value is 5.", "Pay {1}{R}; mana value is 2.", "Pay {2}{R}; mana value is 5."], "Kicker is an additional cost and does not change the printed mana cost used for mana value.", ["702.33a", "601.2f", "202.3"]),
        ("undying", "A nontoken creature with undying dies with no +1/+1 counters on it. Its card stays in the graveyard and its trigger resolves. What happens?", "It returns with a +1/+1 counter.", ["It returns with no counters.", "It stays in the graveyard.", "It returns with a -1/-1 counter."], "Undying checks for the absence of a +1/+1 counter when it dies and returns it with one.", ["702.93a"]),
        ("undying", "A nontoken creature with undying dies while it has a +1/+1 counter on it. Does undying trigger?", "No, undying does not trigger.", ["Yes, it always returns.", "Yes, but it returns without a counter.", "Only if its controller pays {1}."], "The intervening-if condition is false when the creature dies.", ["702.93a", "603.4"]),
        ("copy-permanent", "A printed 2/2 has two +1/+1 counters and a temporary +3/+3 effect. Another creature enters as a copy of it, with no other effects on the copy. What are the copy's power and toughness?", "It is 2/2 with no copied counters.", ["It is 7/7.", "It is 4/4 with two copied counters.", "It is 5/5."], "Counters and ordinary temporary power/toughness effects are not copiable values.", ["707.2"]),
        ("copy-spell", "An effect copies a spell already on the stack. No effect tells a player to cast the copy. Does that copy trigger 'whenever you cast a spell'?", "No, putting this copy on the stack is not casting.", ["Yes, every spell on the stack was cast.", "Only if its mana value is positive.", "Only if it has new targets."], "Copying a spell puts the copy on the stack without casting it.", ["707.10"]),
        ("copy-spell", "A spell with {X} in its cost was cast with X=6. An effect copies that spell on the stack, with no new X choice specified. What is X for the copy?", "X is 6 for the copy.", ["X is 0 because no mana was paid.", "The copy's controller must choose X again.", "X is always 1 for copies."], "A spell copy copies casting choices including the chosen value of X.", ["707.2", "707.10"]),
        ("changeling", "A creature card with changeling is in your graveyard. Is it an Elf card there?", "Yes, changeling gives every creature type in every zone.", ["No, changeling works only on the battlefield.", "Only if it was an Elf when it died.", "Only if it is green."], "Changeling is a characteristic-defining ability that works everywhere.", ["702.73a"]),
        ("battle-cry", "You attack with a 2/2 with battle cry and a separate 3/3. No other effects apply. After battle cry resolves, what are their powers?", "The battle-cry creature has power 2; the other has power 4.", ["Their powers are 3 and 4.", "Their powers are 2 and 3.", "Their powers are 3 and 3."], "Battle cry gives +1/+0 to each other attacking creature, excluding its source.", ["702.91a"]),
        ("battle-cry", "A 2/2 with battle cry attacks alone. No other effects apply. What is its power after its battle-cry trigger resolves?", "Its power remains 2.", ["Its power becomes 3.", "Its power becomes 4.", "Battle cry forces it to leave combat."], "There are no other attacking creatures to receive the bonus.", ["702.91a"]),
        ("life-damage", "An effect says 'Target opponent loses 3 life.' That opponent has an effect preventing all damage. How much life do they lose?", "They lose 3 life.", ["They lose no life.", "They gain 3 life.", "The spell is countered."], "Direct life loss is not damage, so damage prevention does not apply.", ["120.3a"]),
        ("lifelink-source", "An opponent controls a creature with lifelink. It deals 3 damage to you. Who gains life from lifelink?", "The opponent gains 3 life.", ["You gain 3 life.", "Both players gain 3 life.", "Nobody gains life when a player is damaged."], "Lifelink causes the damage source's controller to gain life.", ["120.3f"]),
        ("sick-nontap", "A creature just entered under your control with no haste. Its ability costs only {1}, with no tap or untap symbol. Can you activate it while you have priority and can pay?", "Yes, summoning sickness does not prohibit that activation.", ["No, no abilities can be used that turn.", "Only during combat.", "Only after the creature attacks."], "The control-duration restriction applies to attacking and the creature's abilities with tap or untap symbols in their costs.", ["302.6", "117.1b"]),
        ("pay-cost", "An activated ability costs 'Sacrifice this creature.' You have activated it legally and paid. Can an opponent respond by destroying that same creature on the battlefield?", "No, the creature is already sacrificed.", ["Yes, costs are paid when the ability resolves.", "Yes, sacrifice uses the stack separately.", "Only if they control the creature."], "The cost has already moved the creature to the graveyard; the ability exists independently on the stack.", ["113.7a", "701.21a"]),
        ("missing-facts", "A creature blocks another creature. No power, toughness, abilities, or damage information is supplied. Which creature dies?", "There is insufficient information to determine that.", ["The blocker always dies.", "The attacker always dies.", "Both always die."], "Lethal damage depends on toughness and marked damage, while abilities can change the result. Supply those facts.", ["704.5g"]),
        ("missing-facts", "A spell is said to target a creature, but neither its text nor the creature's restrictions are supplied. Is its target legal?", "There is insufficient information to decide legality.", ["All creature targets are always legal.", "All creature targets are always illegal.", "Only the active player may target creatures."], "The targeting requirements and applicable restrictions must be known to evaluate legality.", ["608.2b"]),
    ]
    heldout = {"defender", "draw-wording", "land-turn", "changeling", "life-damage", "sick-nontap"}
    for group, question, correct, wrong, explanation, rules in scenarios:
        add(group, question, correct, wrong, explanation, rules, validation=group in heldout)

    # Arithmetic applications require explicit steps instead of repeating raw text.
    for color_index, (symbol, color) in enumerate(COLORS.items()):
        for generic in range(1, 9):
            cost = f"{{{generic}}}{{{symbol}}}{{{symbol}}}"
            total = mana_value(cost)
            add(f"cost-{symbol}", f"A hypothetical spell has cost {cost}. With no cost modifiers, state its mana value and the colored mana that must be paid.",
                f"Mana value {total}; two {color} mana plus {generic} generic mana are required.",
                [f"Mana value {generic}; only generic mana is needed.", f"Mana value {total}; all mana must be {color}.", f"Mana value {total + 1}; three {color} mana are required."],
                f"The number contributes {generic}, and each {symbol} symbol contributes one. Generic mana may be paid using any types.",
                ["107.4a", "107.4b", "202.3"], validation=color_index == 4)
    for repeated in (1, 2, 3):
        for x in (1, 2, 4, 5, 7):
            cost = "{X}" * repeated + "{2}{U}"
            total = mana_value(cost, x)
            add(f"x-{repeated}", f"A hypothetical spell costs {cost} and is on the stack with X={x}. Compute its mana value, then its mana value after the card moves to hand.",
                f"Its mana value is {total} on the stack and 3 in hand.",
                [f"Its mana value is {total} in both zones.", "Its mana value is 3 in both zones.", f"Its mana value is {total + 1} on the stack and 0 in hand."],
                f"On the stack, {repeated} occurrence(s) of X contribute {repeated} times {x}, plus 2 + 1, giving {total}. In hand all X symbols count as zero.",
                ["107.3a", "202.3e"], validation=repeated == 3)
    for cost in ("{W/U}", "{2/B}", "{G/P}", "{C}", "{R/G}", "{U/P}"):
        for generic in (1, 3, 5, 7):
            printed = f"{{{generic}}}" + cost
            total = mana_value(printed)
            rules = ["202.3", "202.3f"] if "/" in cost and "P" not in cost else ["202.3", "202.3g"] if "P" in cost else ["202.3", "107.4c"]
            add("special-" + re.sub(r"\W", "", cost), f"A hypothetical card's sole mana cost is {printed}. In your hand, what is its mana value?",
                f"Its mana value is {total}.", [f"Its mana value is {v}." for v in (total + 1, total + 2, max(0, total - 1))],
                f"The numerical symbol contributes {generic}; the remaining symbol contributes {total - generic}. Payment choices do not change the printed-cost calculation.",
                rules, validation=cost == "{U/P}")

    # Oracle-derived examples use one ordinary cost and verified cmc only.
    # Stable SHA ordering avoids relying on DB insertion order. Names already used
    # by either retained split or reserved evaluations are excluded wholesale.
    excluded = {name.casefold() for rows in data.values() for row in rows for name in row.get("cards", [])}
    for filename in ("smoke_questions.json", "evaluation.json", "evaluation_cards.json"):
        content = json.loads((ROOT / filename).read_text(encoding="utf8"))
        cases = content.get("cases", []) if isinstance(content, dict) else content
        for case in cases:
            excluded.update(name.casefold() for name in case.get("cards", []))
    with open_cards() as db:
        index = CardIndex(db)
        for filename in ("smoke_questions.json", "evaluation.json", "evaluation_cards.json"):
            content = json.loads((ROOT / filename).read_text(encoding="utf8"))
            for case in content.get("cases", []) if isinstance(content, dict) else content:
                excluded.update(card["name"].casefold() for card in index.for_question(case["question"]))
        records = [json.loads(row[0]) for row in db.execute("SELECT card_json FROM cards")]
        records.sort(key=lambda card: hashlib.sha256(card["oracle_id"].encode()).hexdigest())
        selected = []
        for card in records:
            if card["name"].casefold() in excluded or card.get("faces") or not card.get("mana_cost"):
                continue
            if any(word in card.get("type_line", "") for word in ("Land", "Battle")):
                continue
            try:
                value = mana_value(card["mana_cost"])
            except ValueError:
                continue
            if value != card.get("cmc") or "{X}" in card["mana_cost"]:
                continue
            selected.append(card)
            if len(selected) == 80:
                break
        assert len(selected) == 80
        for number, card in enumerate(selected):
            value = int(card["cmc"])
            rules = ["202.3"]
            if "/P}" in card["mana_cost"]:
                rules.append("202.3g")
            elif "/" in card["mana_cost"]:
                rules.append("202.3f")
            add("oracle-" + hashlib.sha256(card["name"].encode()).hexdigest()[:12],
                f"Using the supplied Oracle record for {card['name']}, give its printed mana cost and mana value in hand.",
                f"Its printed mana cost is {card['mana_cost']} and its mana value is {value}.",
                [f"Its printed mana cost is {card['mana_cost']} and its mana value is {value + 1}.",
                 f"Its printed mana cost is {card['mana_cost']} and its mana value is {value + 2}.",
                 "It has no mana cost and no mana value."],
                "Use the supplied Oracle mana cost and Scryfall cmc. Mana value comes from the cost's symbols, independent of what is paid to cast it.",
                rules, [card["name"]], validation=number >= 64)
        source = index.manifest
    # Validate rule IDs, named cards, groups and the held-out question boundary.
    from supervised import load_datasets
    paths = []
    for split, rows in data.items():
        path = ROOT / "qa" / f"rank32-{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")
        paths.append(path)
    load_datasets(*paths, RuleIndex())
    manifest = {"rules": RuleIndex().manifest, "cards": source,
                "counts": {split: len(rows) for split, rows in data.items()},
                "authored_scenarios": len(scenarios), "new_application_scenarios": sum(counts.values()),
                "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (ROOT / "qa" / "rank32-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    build()
