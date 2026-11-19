"""Commander matchmaking examples based on the user-supplied beta chart."""

import json

from fetch_commander_combo_data import LISTING

SOURCE = "User-supplied Commander Brackets Beta matchmaking chart"
BRACKETS = (
    (1, "Exhibition", "an ultra-casual, theme-first play experience"),
    (2, "Core", "a deck around the strength of an average contemporary preconstructed deck"),
    (3, "Upgraded", "a deck stronger than an average preconstructed deck"),
    (4, "Optimized", "high-power Commander"),
    (5, "cEDH", "high power tuned for competitive play and its metagame"),
)

# Each audit assumes the deck is otherwise suitable for the requested bracket.
FEATURES = (
    ("land-denial", "a plan to deny mana from lands across the table on a mass scale", (
        "Conflicts with the no-mass-land-denial guidance.",
        "Conflicts with the no-mass-land-denial guidance.",
        "Conflicts with the no-mass-land-denial guidance.",
        "The bracket adds no mass-land-denial limit. Disclose the plan to the pod.",
        "The bracket adds no mass-land-denial limit. Its competitive usefulness depends on the metagame.",
    )),
    ("extra-turn", "one standalone extra-turn effect with no line for chaining it", (
        "Conflicts with the guidance against extra turns.",
        "The chart excludes chaining extra turns. An isolated extra turn does not conflict with that condition.",
        "The chart excludes chaining extra turns. An isolated extra turn does not conflict with that condition.",
        "The bracket adds no restriction on extra turns.",
        "The bracket adds no restriction on extra turns.",
    )),
    ("turn-chain", "a repeatable line for chaining extra turns", (
        "Conflicts with the guidance against extra turns.",
        "Conflicts with the guidance against chaining extra turns.",
        "Conflicts with the guidance against chaining extra turns.",
        "The bracket adds no restriction on chaining extra turns.",
        "The bracket adds no restriction on chaining extra turns.",
    )),
    ("early-combo", "a reliable early-game two-card infinite combo", (
        "Conflicts with the two-card infinite combo restriction.",
        "Conflicts with the two-card infinite combo restriction.",
        "Conflicts with the chart's allowance for late-game two-card infinite combos only.",
        "The bracket adds no two-card infinite combo restriction.",
        "The bracket adds no two-card infinite combo restriction.",
    )),
    ("late-combo", "a two-card infinite combo that is genuinely a late-game line", (
        "Conflicts with the two-card infinite combo restriction even though the line is late.",
        "Conflicts with the two-card infinite combo restriction even though the line is late.",
        "Fits the late-game two-card infinite combo guidance. Tell the pod what makes the line late.",
        "The bracket adds no restriction on late-game two-card infinite combos.",
        "The bracket adds no restriction on late-game two-card infinite combos.",
    )),
    ("three-game-changers", "exactly three cards designated Game Changers for the chart", (
        "Conflicts with the zero-Game-Changers guidance.",
        "Conflicts with the zero-Game-Changers guidance.",
        "Fits the allowance of up to three Game Changers. The rest of the deck still matters.",
        "The chart adds no cap on Game Changers.",
        "The chart adds no cap on Game Changers.",
    )),
    ("four-game-changers", "four cards designated Game Changers for the chart", (
        "Conflicts with the zero-Game-Changers guidance.",
        "Conflicts with the zero-Game-Changers guidance.",
        "Exceeds the allowance of up to three Game Changers.",
        "The chart adds no cap on Game Changers.",
        "The chart adds no cap on Game Changers.",
    )),
    ("tutor-density", "a dense suite of tutors that reliably finds the same powerful line", (
        "Conflicts with the few-tutors guidance and may undermine the ultra-casual experience.",
        "Conflicts with the few-tutors guidance and may play above the intended preconstructed benchmark.",
        "The chart gives no separate numeric tutor cap here. Consider how consistency changes the speed and accessibility of the lines the tutors find.",
        "The chart adds no tutor cap. Evaluate the power and play pattern.",
        "The chart adds no tutor cap. Evaluate how the package performs against the expected competitive field.",
    )),
)

# Hold out complete scenario families, including answers that require combining chart conditions.
CASES = (
    ("fit", "A slow, deliberately underpowered deck showcases cards from a favorite storyline. It has few tutors and none of the chart's flagged effects. What bracket would you suggest?",
     "Bracket 1, Exhibition. The theme-first, ultra-casual play experience is the key signal. Mention the theme and usual pace in the pregame conversation."),
    ("fit", "A deck plays roughly like an average contemporary precon. It has few tutors, no Game Changers, no mass land denial, no extra-turn chains and no two-card infinite combo. Where does it fit?",
     "Bracket 2, Core. Its play strength matches the preconstructed benchmark. Check the actual list and play pattern if the deck's strength is uncertain."),
    ("fit", "A tuned tribal deck wins through better mana and efficient combat synergies. It reliably outpaces average precons, but has no Game Changers or infinite combo. Does it have to stay at Core?",
     "No. Bracket 3, Upgraded, is a reasonable fit if the deck is stronger than Core but still plays like an Upgraded deck. Zero Game Changers and zero combos are compatible with Bracket 3."),
    ("fit", "A high-power pet deck uses powerful tutors and an early combo. Its pilot wants optimized games with friends and has tuned around that playgroup rather than a competitive field. Which bracket best fits that intent?",
     "Bracket 4, Optimized. The deck is high power. Bracket 5 additionally describes a competitive, metagame-focused objective, so strong tutors and an early combo alone do not establish cEDH intent."),
    ("fit", "A legal high-power deck is built to win against tournament-minded Commander opponents. Its win lines and interaction slots are chosen for the expected competitive field. Which bracket is the natural queue?",
     "Bracket 5, cEDH. The strategy is optimized around a competitive Commander metagame."),
    ("insufficient", "All you know about a deck is that it has zero Game Changers. Can you assign its bracket?",
     "No. Zero fits multiple brackets, including a cEDH deck that happens to run none. Ask about typical power and speed, tutors, combos, extra turns, mass land denial, and competitive intent."),
    ("insufficient", "Someone says their deck is a precon. Does the product label by itself prove Bracket 2?",
     "No. The Core benchmark is the strength of an average contemporary precon. Ask how this particular list plays and whether its contents trigger any chart guidance."),
    ("insufficient", "An Upgraded player claims their two-card infinite combo is late game because both pieces cost four mana. Is mana value alone sufficient?",
     "No. Ask when both pieces can normally be assembled with the deck's ramp and tutors, what the line requires, and whether it appears unusually early in real games. The chart supplies no numeric turn boundary."),
    ("legality", "A deck is otherwise a good fit for Optimized, but it contains a card banned in Commander. Does Bracket 4 make the deck legal?",
     "No. The Bracket 4 column adds no bracket-specific restrictions and retains the banned list. Ordinary Commander construction requirements still apply."),
    ("legality", "Does the cEDH bracket permit an ordinary deck to ignore the 100-card total or to use off-color cards?",
     "No. Brackets describe matchmaking. Commander deck-construction rules and the applicable banned list still apply."),
    ("legality", "A card is identified as a Game Changer for the supplied chart. Does that designation by itself mean it is banned?",
     "No. Game Changer is a matchmaking designation here, with limits for Brackets 1 through 3. Check the applicable Commander banned list separately."),
    ("tradeoffs", "Your Upgraded deck has four Game Changers but plays a fair late game and has no turn chains or mass land denial. What is the smallest stated adjustment to fit the chart's Game Changer allowance?",
     "Replace at least one of the four with a non-Game-Changer. That addresses the up-to-three allowance. Reassess the resulting deck's overall play strength and any remaining restrictions."),
    ("tradeoffs", "A Core deck has an isolated extra-turn spell. An upgrade would add a repeatable way to recur it and chain turns. What changed for matchmaking?",
     "The isolated use and the repeatable chain are different. The chain conflicts with Core and Upgraded guidance. Remove the repeatable line for either bracket or disclose the change and find a higher-power pod."),
    ("tradeoffs", "You are moving a deck from Upgraded toward Core. It has two Game Changers and a late-game two-card infinite combo, but no mass land denial or repeated extra turns. What needs to change?",
     "Remove both Game Changers and break the two-card infinite combo by changing at least one piece. Then revisit tutor density and overall strength relative to an average precon."),
    ("tradeoffs", "An Exhibition deck meets the tutor, combo, and Game Changer guidance but has a single extra-turn spell. Can its pilot assume that avoiding loops is enough?",
     "No. Exhibition excludes extra turns, including a single isolated use. Replace that effect to fit Exhibition or discuss whether the deck and pod fit another bracket."),
    ("paired-audit", "A strong combat deck has a one-time extra turn, two Game Changers, and a genuinely late two-card infinite combo. It has no mass land denial or turn chain. Which bracket is the best starting point if it plays stronger than average precons?",
     "Bracket 3, Upgraded. The one-time turn does not form a chain, two Game Changers are within the allowance, and the two-card line is late. Explain the line and typical pace during matchmaking."),
    ("paired-audit", "A slow theme deck runs four Game Changers and a mass land denial spell. The pilot chooses Bracket 1 based on pace alone. What should the pod clarify?",
     "Its pace may be Exhibition-like, but both features conflict with Exhibition and exceed the guidance for Core and Upgraded as well. Replace those effects to fit a lower bracket or disclose the mismatch and agree on the game the group wants."),
    ("paired-audit", "Two legal high-power decks can combo early. One is a personal high-power brew for a local group. The other is built around expected competitive Commander matchups. Must they both be Bracket 5?",
     "No. The personal high-power brew can fit Bracket 4, Optimized. A competitive metagame-focused deck fits Bracket 5, cEDH. Their combo speed alone does not distinguish them."),
    ("paired-audit", "A deck submitted for Core has no Game Changers and no mass land denial. It consistently uses tutors to assemble a late two-card infinite combo. Is late timing enough?",
     "No. Core excludes two-card infinite combos even late and calls for few tutors. Removing or breaking the combo and reducing tutor density can help it fit. An Upgraded deck can include a genuinely late line if the overall experience fits."),
    ("paired-audit", "A nominal Upgraded deck has three Game Changers and no infinite combo, but repeatedly locks most opponents out of their lands. Does staying within the Game Changer allowance settle the bracket?",
     "No. Mass land denial independently conflicts with Upgraded. Remove that plan to aim for Bracket 3 or discuss an Optimized game if the resulting play style is high power."),
    ("policy-limits", "Does the beta chart specify that few tutors means exactly two or fewer?",
     "No. The chart gives qualitative few-tutors guidance for Exhibition and Core. Describe density, what the tutors search for, and how they affect repeated play patterns."),
    ("policy-limits", "What turn does the supplied chart say a two-card infinite combo becomes late game?",
     "The supplied chart gives no exact turn. Explain how early the deck can actually assemble the line and agree on expectations before playing."),
    ("policy-limits", "Can you name every current Game Changer using just this chart?",
     "No. The chart provides bracket allowances, but it does not list individual Game Changers. Use the list for the policy version or event in question before classifying named cards."),
    ("policy-limits", "Does switching from destroying many lands to another effect that broadly keeps players from using their lands necessarily avoid the chart's concern?",
     "No. The chart says mass land denial. Discuss the actual effect on others' access to land mana and confirm how your group or event applies the category."),
)


def add_brackets(data):
    def add(family, key, question, answer, validation=False):
        data.add("commander-brackets", family, key, question, answer, [],
                 validation=validation, source=SOURCE)

    for number, name, description in BRACKETS:
        expected = f"Bracket {number}, {name}, is intended for {description}."
        add("definitions", f"definition-{number}",
            f"What play experience is Commander Bracket {number} meant to match under the beta chart?",
            expected)
        add("definitions", f"definition-inverse-{number}",
            f"Which Commander bracket is aimed at {description} under the beta chart?", expected)
        changer_limit = ("The Game Changer allowance is zero." if number < 3 else
                         "The Game Changer allowance is up to three." if number == 3 else
                         "The chart adds no Game Changer cap.")
        tutors = " The chart calls for few tutors." if number < 3 else ""
        # The full checklist uses explicit wording rather than feature-specific pronouns.
        if number <= 3:
            turns = "No extra turns." if number == 1 else "No chaining extra turns."
            combos = "No two-card infinite combos." if number < 3 else "Two-card infinite combos are for late game."
            detail = f"No mass land denial. {turns} {combos} {changer_limit}{tutors}"
        else:
            detail = "The chart adds no bracket-specific restrictions. Commander legality and the banned list still apply."
        add("definitions", f"checklist-{number}",
            f"Summarize the deck guidance for Commander Bracket {number}, {name}, under the beta chart.",
            f"{expected} {detail}")

    for family, feature, answers in FEATURES:
        for (number, name, _), answer in zip(BRACKETS, answers, strict=True):
            add(family, f"{family}-{number}",
                f"Under the Commander Brackets Beta chart, audit {feature} for an otherwise suitable "
                f"Bracket {number} ({name}) deck. How does that feature affect bracket fit?",
                answer + " Confirm overall play strength and Commander legality separately.")

    for number, (family, question, answer) in enumerate(CASES):
        add(family, f"scenario-{number:02}", question, answer, family in {"paired-audit", "policy-limits"})


def add_bracket_catalogs(data, card_index, snapshot, *, combo_digest):
    flags = {}
    for (payload,) in card_index.db.execute("SELECT card_json FROM cards ORDER BY oracle_id"):
        card = json.loads(payload)
        if type(card.get("game_changer")) is not bool:
            raise ValueError("Reimport the Scryfall snapshot with game_changer metadata before building the catalogs")
        if card["name"] in flags and flags[card["name"]][1] != card["game_changer"]:
            raise ValueError(f"Conflicting Game Changer designations for {card['name']}")
        flags[card["name"]] = (card["oracle_id"], card["game_changer"])

    combos = snapshot["combos"]
    if snapshot["listing"] != LISTING or sum(page["count"] for page in snapshot["pages"]) != len(combos):
        raise ValueError("The combo snapshot does not match the EDHREC listing and page totals")
    if len(combos) != len({frozenset(row["cards"]) for row in combos}):
        raise ValueError("The combo snapshot contains repeated card pairs")

    combo_cards = set()
    for row in combos:
        first, second = row["cards"]
        for name in (first, second):
            cards = [card for card in card_index.get(name)
                     if card["layout"] not in {"art_series", "token", "double_faced_token"}]
            if len(cards) != 1:
                raise ValueError(f"Expected one Scryfall record for {name!r} in combo {row['id']}")
            combo_cards.add(cards[0]["name"])
        outcomes = ", ".join(row["outcomes"])
        count = row["non_card_prerequisites"]
        prerequisites = (
            f" The entry has {count} non-card prerequisite{'s' if count != 1 else ''}."
            if count else ""
        )
        data.add(
            "commander-brackets", f"early-combo-{row['id']}", f"early-combo-{row['id']}",
            f"Under EDHREC's early-game two-card combo classification, how does including "
            f"{first} and {second} together affect a Commander deck's bracket, and what can the combo do?",
            f"EDHREC lists this pair as an early-game two-card combo. Its bracket floor is 4, Optimized. "
            f"It is also compatible with Bracket 5 if the deck is competitively tuned and legal. "
            f"Listed outcomes: {outcomes}.{prerequisites}", [], source=row["source"],
        )

    game_changers = [name for name, (_, flag) in flags.items() if flag]
    if not game_changers:
        raise ValueError("The imported Scryfall snapshot contains no Game Changers")
    negatives = sorted(combo_cards - set(game_changers), key=str.casefold)[:len(game_changers)]
    card_source = f"Scryfall All Cards snapshot {card_index.manifest['snapshot']}, game_changer field"
    for name in sorted(game_changers + negatives, key=str.casefold):
        oracle_id, flag = flags[name]
        ruling = (
            "Yes. It uses one Game Changer slot in Upgraded and conflicts with the zero-Game-Changers "
            "guidance for Exhibition and Core." if flag else
            "No. It uses zero Game Changer slots. Its role in a listed combo can still affect bracket fit."
        )
        data.add("commander-brackets", f"game-changer-{oracle_id}", f"game-changer-{oracle_id}",
                 f"Does {name} count as a Game Changer for Commander in the training Scryfall snapshot?",
                 ruling, [], source=card_source)
    return {
        "game_changers": len(game_changers), "negative_game_changer_examples": len(negatives),
        "early_game_two_card_combos": len(combos),
        "edhrec_listing": snapshot["listing"], "edhrec_fetched_utc": snapshot["fetched_utc"],
        "edhrec_pages": snapshot["pages"], "edhrec_snapshot_sha256": combo_digest,
    }
