"""Closed-book mechanic explanations and rule-application examples."""

from collections import defaultdict
from itertools import batched
import re

from mana import COLORS, mana_value


SCENARIOS = (
    ("face-down", "From your library you manifest an Island and cloak a Forest. Neither has been turned face up. Describe both permanents and say whether either can tap for mana.",
     "The manifested card is a face-down 2/2 creature without abilities. The cloaked card is also a face-down 2/2 creature but has ward {2}. Neither is a land while face down, so neither has its basic land mana ability.", ("701.40a", "701.58a", "305.6"), ()),
    ("face-down", "You control a cloaked artifact card that isn't a creature card. Can you turn it face up simply by paying its printed mana cost? What if an effect explicitly turns it face up?",
     "You can't use the cloak special action for a noncreature card. A separate effect can turn it face up if the card can legally be face up on the battlefield. It then has its normal characteristics.", ("701.58b", "701.58g", "708.8"), ()),
    ("face-down", "A spell tells you to manifest dread with exactly one card left in your library. With nothing replacing the action, what happens to that card? Is this a draw?",
     "Look at the one available card and manifest it. There is no other looked-at card to put into the graveyard. Manifest dread is not a draw, so this action does not itself cause a loss for attempting to draw from an empty library.", ("701.62a", "609.3", "121.5", "704.5b"), ()),
    ("face-down", "You manifest dread, looking at a land and a creature card, and want the creature face up eventually using only the normal manifest special action. How should you choose?",
     "Manifest the creature and put the land into your graveyard. Later, with priority, reveal that the manifested card is a creature card and pay its mana cost to turn it face up if it has one. You cannot use that procedure for the land.", ("701.62a", "701.40b"), ()),
    ("face-down", "Your opponent casts a legal removal spell on your manifested creature. You have priority and can pay the creature card's mana cost to turn it face up. Can the opponent respond to the actual flip? Is the spell automatically stopped?",
     "Turning it face up this way is a special action and does not use the stack. The same permanent is still its target. When the spell resolves, check legality again using the current characteristics; the mere face-up change does not stop it.", ("701.40b", "116.2b", "608.2b", "708.8"), ()),
    ("face-down", "One effect says to cloak the top card of your library. Another says you may cast a card with disguise face down. Which action casts a spell?",
     "Casting with disguise puts a face-down spell onto the stack. Cloaking directly puts the card onto the battlefield face down as a 2/2 with ward {2}, without casting it.", ("701.58a", "702.168a", "601.2"), ()),
    ("face-down", "You manifest an instant card. A later effect attempts to turn it face up. Can you cast the instant as a result?",
     "No. Reveal the instant and leave it face down. This did not cast it or turn it face up, so turn-face-up triggers do not trigger.", ("701.40g", "601.2"), ()),
    ("face-down", "You control a face-down creature represented by a card with an enters trigger and a turn-face-up trigger. You turn the card face up without moving it. Which trigger fires?",
     "The turn-face-up trigger can trigger. The existing permanent did not enter the battlefield again, so its enters trigger does not trigger from this action.", ("708.8", "603.2"), ()),
    ("face-down", "You cloak a creature card with no mana cost. Can you turn it face up for zero using cloak's normal special action?",
     "No. No mana cost is not a cost of zero. The cloak procedure requires that the underlying creature card have a mana cost you can pay.", ("701.58b", "118.6"), ()),
    ("face-down", "A cloaked creature is targeted by an opponent's spell, putting ward on the stack. You then turn it face up and its printed face has no ward. Does that undo the ward trigger?",
     "No. Ward had already triggered when the opponent targeted the cloaked creature. The trigger on the stack is independent of the later change to the permanent's abilities.", ("701.58a", "702.21a", "113.7a"), ()),
    ("zimone-engine", "You control Zimone, Mystery Unraveler. On an opponent's turn you make your first land enter by cracking a fetch land that was already on the battlefield. No other land enters this turn. Does this both manifest and flip the card?",
     "No. Sacrificing the fetch land does not itself create a land entry. The fetched land causes one Zimone trigger; its first resolution that turn manifests dread. A further land entry and a later resolution that turn are needed to turn a permanent face up with Zimone.", ("603.2", "701.62a"), ("Zimone, Mystery Unraveler",)),
    ("zimone-engine", "Zimone, Mystery Unraveler is in play, and no land has entered under your control this turn. Harrow resolves and puts two basic lands onto the battlefield at the same time. How do Zimone's triggers resolve?",
     "Each land causes its own trigger. The first Zimone ability to resolve this turn manifests dread; the second may turn a permanent you control face up, including a permanent manifested by the first if it can turn face up.", ("603.2c", "603.3", "701.62a", "701.40g"), ("Zimone, Mystery Unraveler", "Harrow")),
    ("zimone-engine", "Zimone, Mystery Unraveler has already manifested dread on your turn. During the next player's turn two lands enter under your control one at a time. What happens when those triggers resolve?",
     "The first resolution on that new turn manifests dread. The later one can turn a permanent face up. The distinction resets each turn, including opponents' turns.", ("603.2", "701.62a"), ("Zimone, Mystery Unraveler",)),
    ("zimone-engine", "An opponent counters the first Zimone, Mystery Unraveler landfall trigger of the turn. A later land enters, and that trigger resolves. Which branch happens?",
     "It manifests dread. The countered ability did not resolve, so this is the first time the ability has resolved this turn.", ("603.2", "701.6a", "701.62a"), ("Zimone, Mystery Unraveler",)),
    ("zimone-engine", "You manifested an artifact card that is not a creature. Can a later Zimone, Mystery Unraveler landfall resolution turn it face up? Does its enters ability trigger?",
     "Yes, Zimone's effect can turn that permanent face up without the creature-card restriction of the normal manifest special action. It remains the same battlefield object, so the artifact did not enter again.", ("701.40b", "701.40g", "708.8"), ("Zimone, Mystery Unraveler",)),
    ("haste", "You take control of an untapped opponent's creature in your main phase. It has haste and a {T} ability. Compare it with an identical creature without haste under the same effect.",
     "You may attack with the one with haste or activate its own tap-symbol ability while otherwise legal. A change of control means the other creature normally fails the continuous-control requirement for both. Haste does not untap anything, so using the tap ability first can leave the creature unable to attack.", ("302.6", "702.10b", "702.10c", "508.1a"), ()),
    ("haste", "You played a creature with defender and haste this turn. Then an effect taps it before combat. Does either haste or untapping it let it attack while it still has defender?",
     "No. Haste removes the continuous-control timing restriction. It does not override defender or let a tapped creature attack. Even if you untap it, defender still prohibits the attack.", ("702.3b", "702.10b", "508.1a"), ()),
    ("vigilance", "A creature with vigilance attacks while untapped. After it is declared, can an effect tap it, and does tapping pull it out of combat?",
     "Vigilance means declaring it as an attacker does not tap it. Other effects can still tap it. Tapping a creature after it attacks does not remove it from combat or prevent its combat damage.", ("702.20b", "506.4b"), ()),
    ("vigilance", "A tapped creature gains vigilance before combat. Can it attack now? What changes if another effect untaps it before attackers?",
     "It cannot attack while tapped. If untapped in time and all other attack requirements are met, it can attack without tapping. Vigilance does not untap it or give haste.", ("508.1a", "702.20b", "302.6"), ()),
    ("vigilance", "A 3/3 with vigilance has a useful tap-symbol ability. You have controlled it since your turn began. To both attack and activate its ability this combat, what is the ordinary sequence?",
     "Leave it untapped and declare it as an attacker. Vigilance keeps it untapped, so you can activate its tap-symbol ability when you have priority afterward. Using the ability first normally taps it and prevents it from being declared.", ("508.1a", "508.2", "702.20b", "117.1b"), ()),
    ("trample", "A trampling 10/10 is blocked by a 2/4 with indestructible and two damage already marked, and a fresh 3/3. What is the maximum damage it can assign to the defending player?",
     "Five. Assign two more to the damaged 2/4 and three to the fresh 3/3. The remaining five may go to the player. Indestructible changes whether the 2/4 is destroyed, not the lethal assignment threshold.", ("702.19b", "702.12b"), ()),
    ("trample", "Your 5/5 with trample and deathtouch attacks and is blocked by two undamaged 4/4 creatures. No other damage is assigned. How much can you assign to the defending player?",
     "At most three. One assigned damage from the attacker counts as lethal to each blocker due to deathtouch. After assigning one to each, up to three can be assigned to the player.", ("702.2c", "702.19b"), ()),
    ("trample", "A 7/7 with trample is declared blocked by a creature that is sacrificed before combat damage. The trampler stays in combat and is attacking a player. Does it assign damage to that player?",
     "Yes. It remains blocked, but trample allows it to assign all seven to the player when no blockers remain at assignment.", ("509.1h", "702.19d"), ()),
    ("trample", "A 7/7 with trample blocks an attacking 2/2. Can the blocker send five excess damage to the attacking player's life total?",
     "No. Trample modifies an attacking creature's combat damage assignment. The blocker assigns its damage to the creature it blocked.", ("702.19a", "510.1d"), ()),
    ("trample", "An 8/8 with trample is blocked by a fresh 3/3 that will prevent all damage dealt to it. What is the maximum the attacker can assign to the player?",
     "Five. The attacker still assigns at least three to the blocker before assigning excess to the player. Damage prevention changes damage dealt, not how much counts as lethal for the assignment.", ("702.19b",), ()),
    ("phyrexian-special", "At one life, you want to cast a spell whose only cost is {G/P}. With one usable green mana, what payments are possible, and what mana value will the spell have?",
     "Pay one green mana. You cannot pay two life from one life. The Phyrexian symbol contributes one to mana value regardless of the permitted payment.", ("107.4f", "119.4", "202.3g"), ()),
    ("phyrexian-special", "At exactly two life, can you pay two life for {U/P} to cast a spell, and what happens before anyone can respond if nothing can prevent you from losing?",
     "Yes, the payment is legal and leaves you at zero life. You lose when state-based actions are checked before priority. Paying life still leaves that symbol contributing one mana value.", ("107.4f", "119.4", "704.5a", "202.3g"), ()),
    ("phyrexian-special", "An ordinary spell costs {3}{W/U/P}. No cost changes apply and you can afford any payment. List the options for the last symbol and calculate mana value.",
     "Pay one white mana, one blue mana, or two life for {W/U/P}, in addition to the generic three. The symbol contributes one, giving mana value four. Its colored component is both white and blue regardless of payment.", ("107.4f", "202.3", "202.3g"), ()),
    ("phyrexian-special", "A spell with printed cost {2}{2/B}{G/P} is cast by paying one black for {2/B}, two life for {G/P}, and the generic two. Compare mana spent and mana value.",
     "You spend three mana and two life. Mana value is five: two from the printed generic amount, two from the largest component of {2/B}, and one from {G/P}. The chosen payments do not replace those values.", ("107.4f", "202.3", "202.3f", "202.3g"), ()),
    ("phyrexian-special", "You cast a card with mana cost {X}{R/P} while choosing X=5 and paying life for the red Phyrexian symbol. What is its mana value in your hand before casting and on the stack?",
     "One in hand and six on the stack. X is zero off the stack and five for that spell on the stack. The Phyrexian symbol contributes one in both cases.", ("202.3e", "202.3g"), ()),
    ("phyrexian-special", "A spell costs {4}{U/P} and an effect reduces its total generic cost by {2}. You pay life for the Phyrexian symbol. What do you actually pay and what is the mana value?",
     "You pay two generic mana and two life. Mana value remains five because the printed {4} and {U/P} still contribute four and one, respectively.", ("107.4f", "601.2f", "202.3", "202.3g"), ()),
    ("phyrexian-special", "Can a mono-green Commander deck include a card with cost {W/G/P} just because it can pay green or life for that symbol?",
     "No. A hybrid Phyrexian symbol includes both component colors in its color identity. The white component is outside a mono-green commander's identity regardless of the intended payment.", ("107.4f", "903.4", "903.5c"), ()),
    ("phyrexian-special", "Mutagenic Growth is cast by paying two life. What are its color and mana value on the stack, absent other effects?",
     "It is green and has mana value one. Paying life for the green Phyrexian symbol changes neither the color of the symbol nor its contribution to mana value.", ("107.4f", "202.2", "202.3g"), ("Mutagenic Growth",)),
)

VALIDATION = (
    ("An opponent casts a spell targeting your manifested creature and another spell targeting your cloaked creature. Neither underlying card grants any extra abilities. Which choice of target normally triggers ward?",
     "Targeting your cloaked creature triggers its ward {2}. Manifest by itself grants no ward.", ("701.40a", "701.58a", "702.21a")),
    ("You manifest dread an enchantment card, then later a resolving ability turns it face up. It has 'When this enchantment enters, draw a card.' Do you draw from that ability when it turns face up?",
     "No. The enchantment was already on the battlefield as a face-down creature. Turning it face up does not make it enter again.", ("701.62a", "701.40a", "708.8", "603.2")),
    ("You play a 4/4 that has both haste and vigilance. It is untapped and can attack normally. What do the two abilities each change if you declare it as an attacker this turn?",
     "Haste permits it to attack despite entering under your control this turn. Vigilance means it does not tap when declared as an attacker.", ("302.6", "702.10b", "702.20b")),
    ("Two attackers are blocked separately by 3/3 creatures. The first attacker is a 5/5 with trample and the second is a 5/5 without trample. Both blockers leave combat before damage. How much damage can each attacker assign to the player?",
     "The trampling attacker can assign five to the defending player. The other attacker stays blocked and assigns no damage when its blocker is gone.", ("509.1h", "510.1c", "702.19d")),
    ("A spell costs {5}{B/P}{G/P}. Its caster pays one black mana and two life for the colored symbols, and a cost reduction removes three generic mana. What is its mana value, and what did the caster spend?",
     "Mana value is seven. The caster spent two generic mana, one black mana, and two life. The reduction and life payment change the total paid while the mana value still uses all printed symbols.", ("107.4f", "202.3", "202.3g", "601.2f")),
)


def add_keyword_curriculum(data, index):
    mechanics = {(p.id, p.text): [] for p in index.passages
                 if re.fullmatch(r"70[12]\.\d+", p.id) and not p.id.endswith(".1")}
    children = defaultdict(list)
    for passage in index.passages:
        match = re.fullmatch(r"(70[12]\.\d+)[a-z]+", passage.id)
        if match:
            children[match[1]].append(passage)
    for rule, term in mechanics:
        parts = children[rule]
        if not parts:
            raise ValueError(f"Missing operative subrules for {term} ({rule})")
        kind = "keyword action" if rule.startswith("701.") else "keyword ability"
        # Preserve every operative subrule, with short batches for long entries.
        for start, chunk in enumerate(batched(parts, 3), 1):
            refs = [part.id for part in chunk]
            question = (f"How does the Magic {kind} {term} work? Start with its core rules."
                        if start == 1 else
                        f"Explain the details covered by {', '.join(refs)} for the Magic {kind} {term}.")
            answer = f"{term} is a {kind}. " + "\n\n".join(part.text for part in chunk)
            data.add("keyword-reference", rule, f"{rule}-{start}", question, answer, refs)
    for ref, term in (("207.2c", "ability words"), ("207.2d", "flavor words")):
        data.add("keyword-reference", ref, ref,
                 f"How are {term} different from keyword abilities when playing Magic?",
                 index.by_id[ref].text, [ref])
    for ref, question in (
        ("701.1", "What makes a keyword action different from other verbs on Magic cards?"),
        ("702.1", "What general rules apply across Magic keyword abilities?"),
    ):
        parts = [index.by_id[ref], *children[ref]]
        data.add("keyword-reference", ref, ref, question,
                 "\n\n".join(part.text for part in parts), [part.id for part in parts])

    for i, (family, question, answer, rules, cards) in enumerate(SCENARIOS):
        data.add("mechanics-practice", family, f"scenario-{i}", question, answer, rules, cards)
    for i, (question, answer, rules) in enumerate(VALIDATION):
        data.add("mechanics-practice", "mixed-review", f"review-{i}", question, answer, rules, validation=True)

    standard = tuple(f"{color}/P" for color in COLORS)
    hybrid = ("W/U/P", "W/B/P", "U/B/P", "U/R/P", "B/R/P",
              "B/G/P", "R/G/P", "R/W/P", "G/W/P", "G/U/P")
    for symbol_index, symbol in enumerate((*standard, *hybrid)):
        is_hybrid = symbol in hybrid
        for generic, count, life_paid in ((0, 2, 1), (2, 1, 1), (3, 2, 0), (4, 3, 2)):
            cost = (f"{{{generic}}}" if generic else "") + f"{{{symbol}}}" * count
            total = mana_value(cost)
            mana_paid = generic + count - life_paid
            life_description = f"pay life for {life_paid} of them" if life_paid else "pay no life"
            key = f"{symbol.replace('/', '-')}-{generic}-{count}"
            scenario = (f"A hypothetical spell has printed cost {cost}. You {life_description} and use "
                        "colored mana for any remaining Phyrexian symbols. No other costs or modifiers apply.")
            answer = (
                f"Spend {mana_paid} mana and pay {2 * life_paid} life. Mana value is {total}: "
                f"{generic} from generic mana plus {count} from the printed Phyrexian symbols. Each "
                f"{'hybrid ' if is_hybrid else ''}Phyrexian symbol contributes one regardless of payment.")
            refs = ["107.4f", "202.3", "202.3g"]
            heldout = symbol in {"R/W/P", "U/R/P"}
            data.add("phyrexian-mana", symbol, key,
                     scenario + " How much mana and life do you pay, and what is its mana value?",
                     answer, refs, validation=heldout)
            position = (generic + symbol_index) % 4
            choices = [total - 2, total - 1, total + 1]
            choices.insert(position, total)
            options = "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", choices, strict=True))
            data.add("phyrexian-mana", symbol, f"{key}-choice",
                     f"{scenario} What is the spell's mana value?\n\n{options}\n\n"
                     "Choose one answer. Start with ANSWER: and its letter. Then explain.",
                     f"ANSWER: {'ABCD'[position]}\n\n{answer}", refs, validation=heldout)
    return {
        "actions": sum(rule.startswith("701.") for rule, _ in mechanics),
        "abilities": sum(rule.startswith("702.") for rule, _ in mechanics),
    }
