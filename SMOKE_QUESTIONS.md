# Questions and expected answers

These questions target the November 6, 2026 Comprehensive Rules and the imported Scryfall snapshot.

## blue-mana

Which color is {U}, and which basic land type normally produces it?

- A. Black, Swamp
- B. Blue, Island
- C. White, Plains
- D. Green, Forest

**Expected choice B.** {U} represents blue mana. An Island has an intrinsic ability to tap for blue mana.

Reference rules 107.4a, 305.6.

## black-mana

Which color is {B}, and which basic land type normally produces it?

- A. Blue, Island
- B. Red, Mountain
- C. Black, Swamp
- D. Green, Forest

**Expected choice C.** {B} represents black mana. A Swamp has an intrinsic ability to tap for black mana.

Reference rules 107.4a, 305.6.

## generic-mana

Can one white mana, one colorless mana and one green mana pay {2}{G}?

- A. Yes
- B. No, all three mana must be green
- C. No, generic mana must be colorless
- D. Only if a Forest produced all three

**Expected choice A.** Yes. White and colorless can pay the generic {2}, while green pays {G}.

Reference rules 107.4a, 107.4b.

## colorless-mana

Can green mana pay a {C} requirement under the normal mana rules?

- A. Yes, green pays every type of mana
- B. Yes, if a Forest produced it
- C. Yes, if {C} is in an activation cost
- D. No, {C} requires colorless mana

**Expected choice D.** {C} requires colorless mana. Generic numerical requirements can be paid with any type of mana.

Reference rules 107.4b, 107.4c.

## ordinary-mana-value

What is the mana value of an ordinary spell with printed mana cost {3}{U}{U}?

- A. 2
- B. 3
- C. 5
- D. It depends on which lands paid for it

**Expected choice C.** Its mana value is 5, from three generic mana plus two blue mana symbols.

Reference rules 202.3.

## x-mana-value

An ordinary spell has printed cost {X}{U}. With X=4, what is its mana value on the stack and in hand, respectively?

- A. 5 on the stack, 1 in hand
- B. 4 on the stack, 0 in hand
- C. 5 in both zones
- D. 1 in both zones

**Expected choice A.** On the stack X is 4, so the mana value is 5. In hand X is treated as 0, so the mana value is 1.

Reference rules 107.3a, 202.3e.

## ballista-x

Walking Ballista is cast with X=3. How much mana does its printed cost require and what is its mana value on the stack?

- A. Three generic mana, mana value 3
- B. Six generic mana, mana value 6
- C. Three colorless mana, mana value 6
- D. Six generic mana, mana value 0

**Expected choice B.** Its printed cost is {X}{X}. Each X is 3, so it requires six generic mana and has mana value 6 on the stack.

Reference rules 107.3a, 107.4b, 202.3e.

## hybrid-value

An ordinary spell has printed mana cost {2/B}{2/B}{2/B}. You pay three black mana to cast it. What is its mana value?

- A. 3
- B. 0
- C. It depends on the colors spent
- D. 6

**Expected choice D.** Its mana value is 6. Each {2/B} symbol contributes its largest component, which is 2.

Reference rules 107.4e, 202.3f.

## phyrexian-value

You pay 2 life to satisfy the {B/P} in an ordinary spell's printed mana cost {1}{B/P}. What is the spell's mana value?

- A. 0
- B. 1
- C. 2
- D. 3

**Expected choice C.** Its mana value is 2. The generic symbol contributes 1 and the Phyrexian symbol contributes 1, regardless of paying life.

Reference rules 107.4f, 202.3g.

## one-spell-resolves

Two spells are on the stack in a two-player game. Both players pass in succession. What happens next?

- A. Only the top spell resolves, then the active player receives priority
- B. Both spells resolve before anyone receives priority
- C. The lower spell resolves first
- D. The phase ends immediately

**Expected choice A.** The top spell resolves. The active player then receives priority. A further round of consecutive passes is needed to resolve the next spell.

Reference rules 117.4, 117.3b.

## empty-stack

Both players pass in succession while the stack is empty. What happens?

- A. The active player draws a card
- B. All permanents untap
- C. The active player gets priority again in the same step
- D. The current step or phase ends

**Expected choice D.** Consecutive passes with an empty stack end the current step or phase.

Reference rules 117.4.

## all-targets-illegal

An instant says 'Destroy target creature. Draw a card.' The creature leaves the battlefield before resolution. Does the caster draw a card?

- A. Yes, because drawing has no target
- B. No, the spell's only target is illegal so it does not resolve
- C. Yes, after choosing a new target
- D. Only if the creature was destroyed

**Expected choice B.** The caster does not draw. With its only target illegal, the instant does not resolve and none of its instructions happen.

Reference rules 608.2b.

## some-targets-legal

A spell says 'Tap target artifact and untap target creature.' The artifact leaves the battlefield. The creature remains a legal target. What happens?

- A. Nothing resolves
- B. The caster chooses a new artifact
- C. The spell resolves and untaps the creature
- D. The creature is tapped instead

**Expected choice C.** The spell resolves because at least one target remains legal. It untaps the creature and cannot affect the departed artifact.

Reference rules 608.2b.

## simultaneous-triggers

During Alice's turn, one ordinary enters-the-battlefield ability controlled by Alice and one controlled by Bob trigger simultaneously. Which resolves first if neither player takes further actions?

- A. Bob's ability
- B. Alice's ability
- C. Both simultaneously
- D. Alice chooses which player's ability resolves first

**Expected choice A.** Alice is the active player and places her trigger on the stack first. Bob places his above it, so Bob's resolves first.

Reference rules 603.3b, 117.4.

## zero-toughness

A 2/2 creature with indestructible gets -2/-2 until end of turn. There are no other effects. What happens when state-based actions are checked?

- A. It stays because it has indestructible
- B. It is exiled
- C. It becomes a 0/1
- D. It is put into its owner's graveyard

**Expected choice D.** Having zero toughness puts it into its owner's graveyard. This action does not destroy it, so indestructible does not prevent it.

Reference rules 704.5f, 702.12b.

## last-card

A player draws the last card in their library and makes no further draw attempts. Do they lose merely because their library is now empty?

- A. Yes, immediately
- B. No, an empty library alone does not cause that loss
- C. Yes, at the next end step
- D. Yes, when they next get priority

**Expected choice B.** An empty library alone does not cause a loss. The relevant state-based action checks for an attempted draw from an empty library.

Reference rules 704.5b.

## failed-draw

During a spell's resolution a player attempts to draw from an empty library. There are no effects preventing a loss. When do they lose?

- A. Before the draw instruction begins
- B. Only at the end of the turn
- C. When state-based actions are checked after the spell finishes resolving
- D. Immediately in the middle of the spell's resolution

**Expected choice C.** The player loses at the next state-based-action check, normally after the resolving spell finishes.

Reference rules 704.5b, 704.3, 704.4.

## legend-rule

Alice and Bob each control one legendary permanent with the same name. No player controls a second copy. Does the legend rule make either player put one into a graveyard?

- A. No
- B. Yes, both players
- C. Yes, only Alice
- D. Yes, only the nonactive player

**Expected choice A.** The legend rule checks multiple legendary permanents with the same name controlled by the same player.

Reference rules 704.5j.

## blink-counters

A nontoken creature with a +1/+1 counter is exiled and immediately returned. No effect grants new counters. Does the returned creature keep that counter?

- A. Yes, because the physical card is the same
- B. Yes, until end of turn
- C. Only if its owner controlled it
- D. No

**Expected choice D.** The counter ceases to exist when the object changes zones. The returned permanent is a new object.

Reference rules 122.2, 400.7.

## trample-indestructible

A 6/6 with trample attacks a player and is blocked by a 2/2 with indestructible. No damage is already marked and no other effects apply. What is the maximum damage assignable to the player?

- A. 0
- B. 4
- C. 6
- D. 2

**Expected choice B.** At least 2 must be assigned to the blocker, leaving at most 4 for the player. Indestructible does not increase lethal damage for trample assignment.

Reference rules 702.19b, 702.12b.

## combat-damage-stack

Can a player cast an instant between assigning combat damage and dealing that damage in the same combat damage step?

- A. Yes, after each creature assigns damage
- B. Yes, if they control the attacker
- C. No
- D. Only if a creature has trample

**Expected choice C.** Combat damage is a turn-based action. Players receive no opportunity to act between its assignment and dealing.

Reference rules 510.2.

## lifelink-amount

An unblocked 3/3 creature with lifelink deals 3 combat damage to a player. Nothing changes damage or life gain. How much life does its controller gain?

- A. 3
- B. 1
- C. 0
- D. The defending player's current life total

**Expected choice A.** Lifelink causes the source's controller to gain life equal to the damage dealt, which is 3.

Reference rules 702.15b.

## summoning-tap

A creature without haste enters under your control during your turn. Its activated ability has {T} in its cost. Can you activate that ability this turn under the normal rules?

- A. Yes, if it is a mana ability
- B. Yes, if it has power 0
- C. Yes, after combat
- D. No

**Expected choice D.** The creature has not been under your control continuously since your most recent turn began. The restriction applies to an ability with the tap symbol in its cost.

Reference rules 302.6.

## commander-dies

Your commander creature is destroyed, with no replacement effects. You want it in the command zone. Does it enter the graveyard and die first?

- A. No, it goes directly to the command zone
- B. Yes, then its owner may move it to the command zone during state-based actions
- C. No, commanders cannot die
- D. Only if every opponent agrees

**Expected choice B.** It enters the graveyard and dies. Its owner may subsequently move it to the command zone during the appropriate state-based-action check.

Reference rules 903.9a, 700.4.

## commander-hand

A spell would return your commander to your hand. Can you have it move directly to the command zone instead?

- A. No, it must enter the hand first
- B. Only after it reaches the graveyard
- C. Yes, its owner can apply the commander replacement effect
- D. Only during its owner's turn

**Expected choice C.** The owner may replace a move to hand or library with a move to the command zone.

Reference rules 903.9b.

## commander-tax

Your commander has printed mana cost {2}{G} and has already been cast from the command zone twice this game by you. With no cost modifiers, what does the third command-zone cast cost and what is its mana value?

- A. {6}{G}, mana value 3
- B. {6}{G}, mana value 7
- C. {4}{G}, mana value 5
- D. {2}{G}, mana value 3

**Expected choice A.** Two prior command-zone casts add {4} to the cost, producing {6}{G}. The additional cost does not change the printed mana cost, so mana value remains 3.

Reference rules 903.8, 202.3.

## bolt-indestructible

Lightning Bolt resolves targeting Darksteel Myr. No other effects apply. What happens to the Myr?

- A. It is exiled
- B. It loses indestructible
- C. It is destroyed immediately
- D. It remains on the battlefield with 3 damage marked

**Expected choice D.** Lightning Bolt deals 3 damage. Darksteel Myr has indestructible, so lethal damage does not destroy it.

Reference rules 702.12b, 704.5g.

## path-target-gone

Path to Exile's only target leaves the battlefield before resolution. Does that creature's controller get to search for a basic land?

- A. Yes
- B. No
- C. Only if it went to the graveyard
- D. Only if it was a commander

**Expected choice B.** Path to Exile does not resolve because its only target is illegal. Its search instruction is not performed.

Reference rules 608.2b.

## blood-moon-depths

Blood Moon is on the battlefield when Dark Depths enters. No other effects apply. How many ice counters does it enter with?

- A. Ten
- B. One
- C. Zero
- D. Its controller chooses

**Expected choice C.** It enters with zero ice counters. Blood Moon removes Dark Depths' printed abilities when determining how it would enter, so its enters-with-counters replacement effect does not apply.

Reference rules 305.7, 614.12.

## ice-spell

You cast the Ice half of Fire // Ice targeting an opponent's untapped land. The target stays legal. Who draws the card?

- A. You, as the controller of the Ice spell
- B. The land's controller
- C. Both players
- D. Nobody, because the land started untapped

**Expected choice A.** Ice taps target permanent and instructs its own controller to draw a card. The land's controller does not draw from Ice.

Reference rules 608.2c.
