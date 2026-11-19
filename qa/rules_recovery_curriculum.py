"""Contrast scenarios for the Qwen closed-book rules smoke failures."""

from mana import COLORS, mana_value


# These scenarios vary the rule-bearing facts instead of reusing smoke questions.
PAIRS = (
    ("mana-basics", "You have a Swamp and an Island that can each be tapped this turn. Which pays for a spell with only {B} in its cost?",
     "The Swamp can pay it. Its intrinsic ability adds black mana, {B}. The Island adds blue, {U}.",
     ("Either land can supply {B}.", "Only the Island can pay it.", "Neither land can pay a colored symbol."), ("107.4a", "305.6")),
    ("mana-basics", "You have a Swamp and an Island that can each be tapped this turn. Which pays for a spell with only {U} in its cost?",
     "The Island can pay it. Its intrinsic ability adds blue mana, {U}. The Swamp adds black, {B}.",
     ("Only the Swamp can pay it.", "Either land can supply {U}.", "Neither land can pay a colored symbol."), ("107.4a", "305.6")),
    ("generic-payment", "A spell costs {2}{R}. You have {C}{U}{R} available. Can you pay the cost?",
     "Yes. Spend {R} on the red symbol and use the colorless and blue mana to pay the two generic mana.",
     ("No, the generic portion requires two colorless mana.", "No, the spell costs three red mana.", "No, blue mana cannot pay generic costs."), ("107.4a", "107.4b", "107.4c")),
    ("generic-payment", "A spell costs {1}{C}. You have {U}{G} available. Can you pay?",
     "No. Either mana can pay {1}, but {C} explicitly requires a colorless mana.",
     ("Yes, any two mana pay this.", "Yes, green mana can substitute for {C}.", "No, this cost requires both mana to be colorless."), ("107.4b", "107.4c")),
    ("generic-payment", "A spell costs {3}{U}. You have three colorless mana and one blue mana. Is that sufficient, and what is the mana value?",
     "Yes. Spend the three colorless mana for the generic three and the blue mana for {U}. The spell has mana value four.",
     ("No, generic costs cannot be paid with colorless mana.", "Yes, but the mana value is one.", "No, the cost requires four blue mana."), ("107.4a", "107.4b", "107.4c", "202.3")),
    ("apnap", "On Bea's turn two ordinary damage triggers await the stack, one controlled by Bea and the other by Cy. In this two-player game whose trigger is placed first, and whose resolves first if both players pass?",
     "Bea puts her trigger on first, and Cy's resolves first. The active player's trigger is lower on the stack. The top object resolves first after all players pass.",
     ("Bea places hers and resolves hers first.", "Cy places his and resolves his first.", "They resolve simultaneously."), ("603.3b", "117.4")),
    ("apnap", "On Cy's turn two ordinary damage triggers await the stack, one controlled by Bea and the other by Cy. In this two-player game whose trigger is placed first, and whose resolves first if both players pass?",
     "Cy puts his trigger on first, and Bea's resolves first. The active player's trigger is lower on the stack. The top object resolves first after all players pass.",
     ("Cy places his and resolves his first.", "Bea places hers and resolves hers first.", "They resolve simultaneously."), ("603.3b", "117.4")),
    ("toughness", "A 4/4 with indestructible gets -2/-4 until end of turn and has no damage marked. What happens when state-based actions are checked?",
     "It is put into its owner's graveyard. Its toughness is zero. This state-based action does not destroy it, so indestructible does not prevent it.",
     ("It survives because indestructible prevents all moves to the graveyard.", "It is exiled.", "It waits until cleanup to leave."), ("704.5f", "702.12b")),
    ("toughness", "A 4/4 with indestructible instead has four ordinary damage marked and no toughness modifiers. What happens when state-based actions are checked?",
     "It stays on the battlefield. It still has positive toughness and indestructible prevents destruction from lethal damage.",
     ("It goes to the graveyard for having zero toughness.", "It is put in exile.", "It is destroyed because damage equals toughness."), ("120.3e", "704.5g", "702.12b")),
    ("failed-draw", "Your library contains one card. A resolving spell instructs you to draw two cards and then gain four life. No effect changes what happens. When do you gain life and when do you lose?",
     "You draw one card, fail to draw the second, and gain the life during resolution. You then lose at the next state-based-action check after the spell finishes.",
     ("You lose before gaining life.", "You gain life and survive until your next draw step.", "The spell stops before drawing any cards."), ("121.2", "121.4", "704.4", "704.5b")),
    ("failed-draw", "Your library contains two cards. A resolving spell instructs you to draw two cards and then gain four life. With no other draw attempts or effects, do you lose after resolution?",
     "No. Both draws succeed. Your library ends empty, but the state-based action requires an attempt to draw from an empty library.",
     ("You lose during the second draw.", "You lose after gaining life because the library is empty.", "You lose before the first draw."), ("121.2", "704.5b")),
    ("commander-zone", "Your nontoken commander is sacrificed to pay a cost and actually reaches the graveyard. You plan to move it to the command zone. Does a separate permanent see it die?",
     "Yes. It went from the battlefield to the graveyard. Its owner can move it to the command zone at the subsequent state-based-action check, after the death event.",
     ("No, the move to the command zone replaces the sacrifice.", "No, sacrificing is never dying.", "Yes, but only if it stays in the graveyard for the rest of the turn."), ("700.4", "903.9a")),
    ("commander-zone", "A replacement effect sends a destroyed commander to exile instead of its owner's graveyard. It later moves from exile to the command zone at a state-based-action check. Does a dies trigger see it?",
     "No. It went directly from the battlefield to exile and never entered a graveyard. Its owner can move it from exile to the command zone at the state-based-action check.",
     ("Yes, moving to exile counts as dying.", "Yes, the later command-zone move triggers dies.", "No, but commanders must stay exiled forever."), ("700.4", "903.9a")),
    ("commander-zone", "A spell tries to put your commander into your hand. If you choose the command zone, does it briefly enter your hand first?",
     "No. For hand or library moves, the owner can replace the move with a move to the command zone. That differs from the later state-based action after a graveyard or exile move.",
     ("Yes, it enters your hand before the state-based action.", "Yes, but only on your own turn.", "No, because a commander can never go to hand."), ("903.9a", "903.9b")),
)


def add_rules_recovery(data):
    def add_pair(family, key, question, answer, choices, rules, validation=False):
        data.add("rules-recovery", family, f"{key}-free", question, answer, rules, validation=validation)
        index = sum(map(ord, key)) % 4
        options = list(choices)
        options.insert(index, answer.split(". ", 1)[0].rstrip(".") + ".")
        option_text = "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", options, strict=True))
        data.add("rules-recovery", family, f"{key}-mc",
                 f"{question}\n\n{option_text}\n\nSelect one answer. Start with ANSWER: and the letter, then explain.",
                 f"ANSWER: {'ABCD'[index]}\n\n{answer}", rules, validation=validation)

    for i, (family, question, answer, choices, rules) in enumerate(PAIRS):
        add_pair(family, f"contrast-{i}", question, answer, choices, rules)

    basics = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
    for symbol, color in COLORS.items():
        land = basics[symbol]
        add_pair("symbol-mapping", f"symbol-{symbol}",
                 f"You are choosing a basic land to produce {{{symbol}}} with its intrinsic mana ability. Which land and what color?",
                 f"{land}, {color} mana. The symbol {{{symbol}}} represents {color} mana, which a {land} can produce.",
                 tuple(f"{basics[key]}, {COLORS[key]} mana." for key in basics if key != symbol)[:3],
                 ("107.4a", "305.6"))

    for triple in (("W", 1, 2), ("B", 1, 5), ("R", 2, 3), ("G", 2, 6), ("U", 2, 4),
                   ("W", 2, 7), ("B", 1, 0), ("G", 1, 3)):
        color, xs, chosen = triple
        cost = "{X}" * xs + f"{{{color}}}" * 2
        on_stack = mana_value(cost, chosen)
        off_stack = mana_value(cost)
        heldout = xs == 2 and color == "U"
        add_pair("x-separate-review" if heldout else f"x-{xs}", f"x-{color}-{xs}-{chosen}",
                 f"A card costs {cost}. Give its mana value in the graveyard and on the stack as a spell cast with X={chosen}.",
                 f"{off_stack} in the graveyard and {on_stack} on the stack. "
                 f"Off the stack each X is zero. On the stack count {chosen} for each X and add two for the colored symbols.",
                 (f"0 in the graveyard and {on_stack} on the stack.",
                  f"{off_stack} in both zones." if chosen else f"{off_stack} in the graveyard and 0 on the stack.",
                  f"{on_stack + 1} in the graveyard and {on_stack + 1} on the stack."),
                 ("202.3", "202.3e"), validation=heldout)

    reviews = (
        ("An effect reduces a spell's cost of {2}{B} by {1}. With {U}{B} available, can you cast it and what is its mana value?",
         "Yes, and the mana value remains three. Spend blue for the remaining generic one and black for {B}.",
         ("No, the generic mana must be colorless.", "Yes, and its mana value is two.", "No, the reduction removes the black requirement."), ("107.4a", "107.4b", "202.3", "601.2f")),
        ("An opponent's commander is exiled while your own commander is returned to your hand. When can each owner choose the command zone?",
         "The exiled commander's owner can choose it at the state-based-action check; the other owner can replace the move to hand.",
         ("Both choices replace the original moves.", "Both commanders wait for the next state-based-action check.", "The player who caused the moves chooses."), ("903.9a", "903.9b")),
    )
    for i, (question, answer, choices, rules) in enumerate(reviews):
        add_pair("mixed-application-review", f"review-{i}", question, answer, choices, rules, validation=True)
