"""Explain Magic mana symbols and ordinary printed mana costs."""

import argparse
import re


COLORS = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}
MANA_GUIDE = """Mana reference
[CR 107.4a] {W} means white, {U} blue, {B} black, {R} red, and {G} green.
[CR 305.6] A Plains can tap for {W}, an Island for {U}, a Swamp for {B},
a Mountain for {R}, and a Forest for {G}. Land types and mana colors are distinct.
[CR 107.4b] {1}, {2}, and other numbers are generic costs payable with that many
mana of any types, including colorless mana. Different types may be mixed.
[CR 107.4c] {C} specifically requires colorless mana.
[CR 107.3a, CR 107.4b] In a cost, {X} normally represents a chosen nonnegative
integer of generic mana, subject to the card's instructions. Each occurrence of
{X} has the same chosen value. {X}{X} with X=3 requires six generic mana.
[CR 202.3, CR 202.3e] Mana value counts the symbols in the printed mana cost.
Treat X as zero outside the stack and as its chosen value on the stack.
{2}{G} has mana value 3 and requires two generic mana plus one green mana.
{X}{U} with X=4 has mana value 5 on the stack and mana value 1 in hand.
[CR 202.3f] Use the largest component of a hybrid symbol for mana value.
{W/U} requires one white or blue mana and contributes one to mana value.
{2/B} requires two generic mana or one black mana and contributes two.
[CR 107.4f] {B/P} requires one black mana or 2 life and contributes one to mana value.
Paying a different amount through a discount or additional cost generally does
not change a spell's mana value.
"""


def describe_symbol(symbol):
    if symbol in COLORS:
        return f"one {COLORS[symbol]} mana"
    if symbol.isascii() and symbol.isdigit():
        return f"{int(symbol)} generic mana, payable with mana of any types"
    if symbol == "C":
        return "one colorless mana"
    if symbol == "S":
        return "one mana of any type produced by a snow source"
    if symbol in {"X", "Y", "Z"}:
        return f"the chosen value of {symbol} in generic mana, subject to the card's instructions"
    parts = symbol.split("/")
    if len(parts) > 1:
        if parts[-1] == "P" and all(part in COLORS for part in parts[:-1]):
            return "one " + " or ".join(COLORS[part] for part in parts[:-1]) + " mana, or 2 life"
        if all(part in COLORS or part == "C" or part == "2" for part in parts):
            return " or ".join(describe_symbol(part) for part in parts)
    return "special symbol; check the card text and Comprehensive Rules"


def cost_explanation(cost):
    symbols = dict.fromkeys(re.findall(r"\{([^{}]+)\}", cost))
    return ". ".join(f"{{{symbol}}} = {describe_symbol(symbol)}" for symbol in symbols)


def mana_value(cost, x=0):
    """Value for a single ordinary cost, with the supplied value of X."""
    if x < 0:
        raise ValueError("X must be nonnegative.")
    symbols = re.findall(r"\{([^{}]+)\}", cost)
    if not symbols or "".join(f"{{{symbol}}}" for symbol in symbols) != cost:
        raise ValueError("Supply a single brace-delimited cost, such as {2}{G}.")
    total = 0
    for symbol in symbols:
        parts = symbol.split("/")
        valid_hybrid = (
            len(parts) == 2 and all(part in COLORS or part in {"C", "2"} for part in parts)
            or len(parts) in {2, 3} and parts[-1] == "P" and all(part in COLORS for part in parts[:-1])
        )
        if len(parts) > 1 and not valid_hybrid:
            raise ValueError(f"Special symbol {{{symbol}}} needs card-specific interpretation.")
        values = []
        for part in parts:
            if part in COLORS or part in {"C", "S"}:
                values.append(1)
            elif part.isascii() and part.isdigit():
                values.append(int(part))
            elif part == "X":
                values.append(x)
            elif part != "P":
                raise ValueError(f"Special symbol {{{symbol}}} needs card-specific interpretation.")
        if not values:
            raise ValueError(f"Special symbol {{{symbol}}} needs card-specific interpretation.")
        total += max(values)
    return total


def computed_costs(question, cards):
    choices = re.findall(r"\bX\s*=\s*(\d+)\b", question, re.IGNORECASE)
    x_values = sorted({0, *(int(value) for value in choices)})
    facts = []
    for card in cards:
        for record in (card, *card.get("faces", [])):
            cost = record.get("mana_cost")
            if not cost:
                continue
            for value in x_values:
                try:
                    total = mana_value(cost, value)
                except ValueError:
                    continue
                facts.append(
                    f"{record['name']}, printed cost {cost}, X={value}: symbol total {total}. "
                    "This is the mana value of a spell cast with this single mana cost. "
                    "Outside the stack X counts as zero. Apply layout-specific rules to split cards and other faces."
                )
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cost", help="A single printed cost, such as {2}{G}.")
    parser.add_argument("--x", type=int, default=0, help="Value of X, normally zero outside the stack.")
    args = parser.parse_args()
    try:
        value = mana_value(args.cost, args.x)
    except ValueError as error:
        parser.error(str(error))
    print(cost_explanation(args.cost))
    print(f"Mana value with X={args.x}: {value}")


if __name__ == "__main__":
    main()
