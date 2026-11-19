"""Authored whole-deck reviews with computed counts from the local Oracle snapshot."""

from collections import Counter

from cards import format_card


UTILITY = (
    "Sol Ring", "Arcane Signet", "Mind Stone", "Thought Vessel", "Wayfarer's Bauble",
    "Commander's Sphere", "Swiftfoot Boots", "Lightning Greaves", "Skullclamp",
    "Solemn Simulacrum", "Meteor Golem", "Hedron Archive",
)

DECKS = (
    {
        "key": "krenko", "commander": "Krenko, Mob Boss", "lands": {"Mountain": 36},
        "rules": ["702.10c", "117.1b"],
        "spells": """
Goblin Warchief|Goblin Chieftain|Goblin Matron|Goblin Ringleader|Goblin Recruiter|Goblin King
Skirk Prospector|Goblin Instigator|Mogg War Marshal|Goblin Sledder|Goblin Chirurgeon
Siege-Gang Commander|Pashalik Mons|Goblin Trashmaster|Goblin Cratermaker
Witty Roastmaster|Impact Tremors|Goblin Bombardment|Dualcaster Mage|Twinflame
Dragon Fodder|Krenko's Command|Hordeling Outburst|Goblin Rally|Empty the Warrens|Beetleback Chief
Goblin Bushwhacker|Reckless Bushwhacker|Foundry Street Denizen|Legion Loyalist|Goblin Guide
Goblin Motivator|Fanatical Firebrand|Torch Courier|Goblin Tunneler|Goblin Smuggler
Abrade|Chaos Warp|Blasphemous Act|Vandalblast|Lightning Bolt|Wild Magic Surge
Faithless Looting|Thrill of Possibility|Demand Answers|Reforge the Soul
Battle Hymn|Brightstone Ritual|Burn at the Stake|Quest for the Goblin Lord|Ruby Medallion
""",
        "analysis": (
            "Plan. Build a Goblin board, activate Krenko to multiply it, and finish with combat or entry and sacrifice damage. "
            "Goblin Warchief grants Krenko haste and discounts Goblin spells; hold up the option to tap Krenko before exposure "
            "to another turn cycle. Goblin Matron can find a haste enabler or a missing answer. Impact Tremors deals damage "
            "on token entries; Goblin Bombardment converts expendable tokens into damage and can respond to a wipe.\n"
            "Mana and recovery. Cheap Goblins are genuine setup. Sol Ring and Mind Stone can accelerate Krenko, "
            "whereas Battle Hymn and Brightstone Ritual require a board and are better treated as burst mana. "
            "Keep the required red sources for Krenko and red-heavy follow-ups. If Krenko is removed, Hordeling Outburst "
            "and Siege-Gang Commander still provide bodies; Goblin Ringleader and Skullclamp can rebuild resources.\n"
            "Swaps. Cut Hedron Archive for Wily Goblin to make the ramp slot advance the Goblin count at two mana; "
            "the Treasure is one-use, so retain some recurring ramp. Cut Meteor Golem for Goblin Assault to add "
            "earlier repeatable fodder; retain Chaos Warp and Wild Magic Surge to preserve broad permanent interaction. "
            "Goblin Assault forces Goblins to attack if able, so skip that swap if you need to keep them back.\n"
            "Matchmaking. Recommend Bracket 4 with the early combo intact. With another legal creature target, "
            "cast Twinflame and respond with Dualcaster Mage. Dualcaster's entry copies Twinflame and the copy can make "
            "a hasty Dualcaster token. Repeat the copying trigger and then stop to finish the stack. Disclose the combo and "
            "cut a piece if the goal is Bracket 3. The presence of this line does not establish cEDH intent."
        ),
    },
    {
        "key": "meren", "commander": "Meren of Clan Nel Toth", "lands": {"Forest": 18, "Swamp": 18},
        "rules": ["700.4", "603.2"],
        "spells": """
Viscera Seer|Carrion Feeder|Woe Strider|Yahenni, Undying Partisan|Evolutionary Leap
Sakura-Tribe Elder|Llanowar Elves|Elvish Mystic|Fyndhorn Elves|Wood Elves
Satyr Wayfinder|Stitcher's Supplier|Grisly Salvage|Glowspore Shaman|Golgari Findbroker
Reclamation Sage|Caustic Caterpillar|Acidic Slime|Ravenous Chupacabra|Shriekmaw
Plaguecrafter|Fleshbag Marauder|Merciless Executioner|Spore Frog|Haywire Mite
Poison-Tip Archer|Blood Artist|Zulaport Cutthroat|Bastion of Remembrance|Syr Konrad, the Grim
Elvish Visionary|Wall of Blossoms|Grim Haruspex|Midnight Reaper|Village Rites
Victimize|Unearth|Animate Dead|Journey to Eternity|Living Death
Beast Within|Putrefy|Go for the Throat|Golgari Charm|Return to Nature
Cultivate|Kodama's Reach|Harmonize|Overwhelming Stampede|End-Raze Forerunners|Izoni, Thousand-Eyed
""",
        "analysis": (
            "Plan. Pair disposable creatures with sacrifice outlets, gain experience from creature deaths, and use Meren "
            "to recur the best body on your end step. Sakura-Tribe Elder ramps and leaves a recur-able creature card; "
            "Viscera Seer lets you control when your creatures die; Plaguecrafter can trade with multiple opponents and be "
            "reused when experience supports it. Blood Artist and Zulaport Cutthroat make repeated deaths matter.\n"
            "Mana and recovery. Elf ramp is fast but vulnerable to wipes. The 18/18 basic split can miss a turn-one green "
            "source or delay access to both colors; early fixing is more important than a high count of late mana rocks. "
            "Victimize and Living Death provide recovery when Meren is too expensive, while Harmonize and the drain "
            "enchantments can still function through commander removal. Graveyard hate attacks the shared resource; save "
            "Beast Within or Reclamation Sage for the permanent that shuts off your plan when practical.\n"
            "Swaps. Replace one Swamp with Command Tower for a source that gives either color, trading a basic land "
            "for better access to green. Replace Hedron Archive with "
            "Golgari Signet to develop both colors earlier; the Signet needs another mana to activate.\n"
            "Matchmaking. Bracket 3 is a reasonable starting estimate for the repeatable sacrifice and recursion engine. "
            "Disclose repeated edicts and Living Death. Check the deck's actual speed and the group's tolerance for those "
            "lines before settling on a pod."
        ),
    },
    {
        "key": "tatyova", "commander": "Tatyova, Benthic Druid", "lands": {"Forest": 20, "Island": 14, "Evolving Wilds": 1, "Terramorphic Expanse": 1},
        "rules": ["603.2", "601.2h", "305.2"],
        "spells": """
Explore|Growth Spiral|Rampant Growth|Nature's Lore|Three Visits|Farseek|Sakura-Tribe Elder
Harrow|Cultivate|Kodama's Reach|Into the North|Roiling Regrowth|Springbloom Druid
Arboreal Grazer|Llanowar Elves|Elvish Mystic|Coiling Oracle|Wood Elves|Farhaven Elf
Azusa, Lost but Seeking|Dryad of the Ilysian Grove|Llanowar Scout|Walking Atlas|Druid Class
Elvish Rejuvenator|Oracle of Mul Daya|Tireless Provisioner|Tireless Tracker|Rampaging Baloths
Scute Swarm|Retreat to Kazandu|Retreat to Coralhelm|Zendikar's Roil|Grazing Gladehart
Meloku the Clouded Mirror|Avenger of Zendikar|Overwhelming Stampede|End-Raze Forerunners
Ramunap Excavator|Regrowth|Eternal Witness|Broken Bond|Return to Nature|Reclamation Sage
Beast Within|Pongify|Rapid Hybridization|Negate|Counterspell|Harmonize|Fact or Fiction
""",
        "analysis": (
            "Plan. Put lands onto the battlefield early to reach five mana. Once Tatyova is in play, each of your land "
            "entries provides life and a card. Evolving Wilds can yield a trigger on entry and another when sacrificed for a "
            "basic. Harrow can produce two triggers at instant speed if both basics are found, but sacrifices a land as a cost. "
            "Ramunap Excavator lets you reuse fetch lands from the graveyard within your land-play allowance. Rampaging Baloths "
            "and Scute Swarm convert the same land entries into threats.\n"
            "Mana and recovery. Land ramp helps the commander and the engine. Explore and Azusa need lands in hand to add "
            "mana. Early green access is important, but double blue for Counterspell also deserves a check. Tireless Tracker "
            "and Fact or Fiction help keep cards flowing without Tatyova. Heavy finishers are strongest after you have established "
            "the land engine; preserve interaction before committing all your mana.\n"
            "Swaps. Replace Into the North with Search for Tomorrow for acceleration that can find one of the listed ordinary "
            "basic lands; the submitted list contains no snow land. Replace Meteor Golem with Simic Charm for cheaper protection "
            "or tempo, while keeping Beast Within for broad removal. This reduces the ability to permanently answer some targets, "
            "so retain the Golem if durable answers are a bigger problem than tempo.\n"
            "Matchmaking. Bracket 3 is plausible for the density of ramp and repeatable landfall value if the list plays at "
            "that pace. Review Retreat to Coralhelm and land-putting creatures for loop prerequisites rather than assuming "
            "any two synergistic cards form an early infinite combo. Disclose the fastest real line and confirm the relevant list."
        ),
    },
    {
        "key": "brago", "commander": "Brago, King Eternal", "lands": {"Plains": 17, "Island": 17, "Command Tower": 1, "Azorius Chancery": 1},
        "rules": ["603.2", "702.18a"],
        "spells": """
Wall of Omens|Spirited Companion|Mulldrifter|Cloudblazer|Inspiring Overseer|Skyscanner
Sea Gate Oracle|Watcher for Tomorrow|Aether Channeler|Reflector Mage|Man-o'-War
Loran of the Third Path|Cathar Commando|Skyclave Apparition|Inquisitor Exarch|Loyal Warhound
Kor Cartographer|Archaeomancer|Mission Briefing|Eternal Dragon|Sun Titan|Karmic Guide
Ephemerate|Cloudshift|Momentary Blink|Ghostly Flicker|Displace|Eerie Interlude
Restoration Angel|Flickerwisp|Felidar Guardian|Charming Prince|Soulherder|Teleportation Circle
Counterspell|Arcane Denial|Negate|Dovin's Veto|Disenchant|Generous Gift
Swords to Plowshares|Path to Exile|Supreme Verdict|Day of Judgment|Austere Command
Omen of the Sea|Azorius Signet|Talisman of Progress|Whispersilk Cloak|Access Tunnel|Aerial Extortionist
""",
        "analysis": (
            "Plan. Brago must deal combat damage to a player to blink targeted nonland permanents you control. "
            "Wall of Omens and Cloudblazer give cards on entry. Aether Channeler can draw, make a blocker, or bounce a nonland "
            "permanent other than itself. Mana rocks that reenter untapped can be tapped for mana again when legal. "
            "Whispersilk Cloak can help Brago connect; shroud also prevents your own targeting, so choose blink targets accordingly.\n"
            "Mana and recovery. Mind Stone and Azorius Signet provide early development; Kor Cartographer is slower but gives "
            "land ramp and blink value. Protect both colors for Brago and interaction. The commander can be answered before "
            "damage, so use Teleportation Circle or Soulherder to keep some entries recurring independently. "
            "Be careful about blinking Skyclave Apparition because leaving can give its opponent a token.\n"
            "Swaps. Replace Eternal Dragon with Watcher of the Spheres if early evasive development matters more than a late "
            "recursive card and Plainscycling. Replace Meteor Golem with Touch the Spirit Realm for cheaper artifact or creature "
            "interaction with a blink mode, while retaining Generous Gift for wider coverage.\n"
            "Matchmaking. Start the discussion around Bracket 3 for the tuned repeatable blink engine. Check whether "
            "mana-rock reuse and repeatable blink form a loop with the actual deck. A value blink "
            "engine can be strong without being cEDH. Explain its normal setup, interaction, and finish to the pod."
        ),
        "validation": True,
    },
)


def add_full_deck_reviews(data, card_index, early_combos=()):
    for spec in DECKS:
        commander = spec["commander"]
        names = [name.strip() for line in spec["spells"].strip().splitlines() for name in line.split("|")]
        entries = Counter(names + list(UTILITY) + [commander])
        entries.update(spec["lands"])
        if sum(entries.values()) != 100 or any(entries[name] != 1 for name in set(entries) - set(spec["lands"])):
            raise ValueError(f"{commander}: expected a 100-card singleton list with basic-land exceptions")

        records = {}
        for name in entries:
            cards = [card for card in card_index.get(name) if name in card["name"].split(" // ")
                     and card["layout"] not in {"token", "double_faced_token", "art_series"}]
            if len(cards) != 1:
                raise ValueError(f"Expected one main Oracle record for {name}")
            records[name] = cards[0]
        # Lands are identified by Oracle type, so nonbasic utility lands count too.
        lands = {name for name, card in records.items() if "Land" in card["type_line"].split(" —")[0].split()}
        curve = Counter()
        for name in set(entries) - lands - {commander}:
            curve[records[name]["cmc"]] += entries[name]
        land_total = sum(entries[name] for name in lands)
        changers = sorted(name for name, card in records.items() if card.get("game_changer"))
        main = "\n".join(
            f"{entries[name]} {name}" if name in lands else f"{entries[name]} {name} | MV {records[name]['cmc']:g}"
            for name in sorted(set(entries) - {commander}, key=str.casefold)
        )
        deck = (
            f"Commander (1):\n{format_card(records[commander], include_mana_hints=False)}\n\nMain deck:\n{main}"
        )
        question = (
            "Review this complete Commander deck. Consider only the commander and main deck shown. "
            "Give a bracket assessment with caveats, explain at least three commander synergies, assess early ramp "
            "and the nonland mana curve, explain how the deck functions through disruption, and give two concrete "
            "add-and-cut recommendations with tradeoffs. Apply rules where relevant and explain their effect.\n\n"
            f"{deck}"
        )
        curve_text = ", ".join(f"{value:g}: {count}" for value, count in sorted(curve.items()))
        inventory = (
            f"Inventory. {commander} plus 99 main-deck cards, including {land_total} lands. "
            f"Nonland mana values excluding the commander (value: count): {curve_text}. "
            f"Game Changers flagged in the local snapshot: {', '.join(changers) or 'none'}."
        )
        matches = [row["cards"] for row in early_combos if set(row["cards"]) <= entries.keys()]
        combo_audit = ("The EDHREC snapshot contains these pairs in the list: "
                       + "; ".join(" + ".join(pair) for pair in matches)
                       + ". A listed early-game pair has a Bracket 4 floor."
                       if matches else
                       "The EDHREC snapshot has no complete listed pair in this list. Other lines may exist.")
        if not early_combos:
            combo_audit = "An exact bracket needs a separate check of the deck's combo lines."
        plan, mana, swaps, matchmaking = spec["analysis"].splitlines()
        answer = f"{inventory}\n{plan}\n{mana}\n{swaps}\n{matchmaking}\n{combo_audit}"
        validation = spec.get("validation", False)
        source = "Authored deck and review; local Oracle and EDHREC snapshots when available"
        data.add("full-deck-review", spec["key"], spec["key"], question, answer, ["903.5a", *spec["rules"]],
                 validation=validation, source=source)
        for key, request, section, rules in (
            ("engine", "Identify at least three actual cards and describe how they support the commander's plan.",
             plan, spec["rules"]),
            ("curve", "Audit land count and the nonland curve. Distinguish early development from conditional mana and explain how the deck recovers.",
             f"{inventory}\n{mana}", []),
            ("swaps", "Give concrete changes with named additions and cuts. Explain the gain and tradeoff.", swaps, []),
            ("bracket", "Assess the deck's bracket and give the pilot a concise pregame disclosure.",
             f"{inventory}\n{matchmaking}\n{combo_audit}", []),
        ):
            data.add("full-deck-review", spec["key"], f"{spec['key']}-{key}",
                     f"{request}\n\n{deck}", section, rules, validation=validation, source=source)
