"""Import Scryfall's All Cards JSONL snapshot and look up Oracle text."""

import argparse
from contextlib import closing
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unicodedata

from mana import cost_explanation, mana_value


SNAPSHOT = "all-cards-20261117102242.jsonl.gz"
DATABASE = Path("data/cards.sqlite3")
WORD = re.compile(r"\w+(?:'\w+)*")
CARD_FIELDS = ("name", "mana_cost", "cmc", "type_line", "oracle_text", "power", "toughness", "loyalty", "defense")


def words(text):
    tokens = WORD.findall(unicodedata.normalize("NFKC", text).casefold().replace("’", "'"))
    return [token.removesuffix("'s") for token in tokens]


def key(text):
    return " ".join(words(text))


def import_cards(source, destination=DATABASE):
    # Build a new index next to the existing one. A failed or interrupted import
    # leaves the previous snapshot available.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="cards-", dir=destination.parent) as temporary:
        staging = Path(temporary) / destination.name
        with closing(sqlite3.connect(staging)) as db, db:
            db.executescript("""
                CREATE TABLE cards (
                    oracle_id TEXT PRIMARY KEY, released TEXT NOT NULL,
                    printing_id TEXT NOT NULL, card_json TEXT NOT NULL
                );
                CREATE TABLE names (name TEXT NOT NULL, oracle_id TEXT NOT NULL,
                    PRIMARY KEY (name, oracle_id));
                CREATE TABLE manifest (data TEXT NOT NULL);
            """)
            source_rows = english_rows = 0
            with gzip.open(source, "rt", encoding="utf-8-sig") as records:
                for line_no, line in enumerate(records, 1):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSONL record at line {line_no}: {error.msg}") from error
                    if not isinstance(obj, dict) or obj.get("object") != "card":
                        raise ValueError(f"Expected a Scryfall card object at line {line_no}.")
                    source_rows += 1
                    if obj.get("lang") != "en" or not obj.get("oracle_id") or not obj.get("name"):
                        continue
                    english_rows += 1
                    card = {field: obj[field] for field in CARD_FIELDS if field in obj}
                    card["oracle_id"] = obj["oracle_id"]
                    card["layout"] = obj.get("layout")
                    card["keywords"] = obj.get("keywords", [])
                    if obj.get("card_faces"):
                        card["faces"] = [
                            {field: face[field] for field in CARD_FIELDS if field in face}
                            for face in obj["card_faces"]
                        ]
                    db.execute("""
                        INSERT INTO cards VALUES (?, ?, ?, ?)
                        ON CONFLICT(oracle_id) DO UPDATE SET
                            released=excluded.released, printing_id=excluded.printing_id,
                            card_json=excluded.card_json
                        WHERE (excluded.released, excluded.printing_id) >
                              (cards.released, cards.printing_id)
                    """, (obj["oracle_id"], obj.get("released_at") or "", obj.get("id") or "",
                          json.dumps(card, ensure_ascii=False)))
            for oracle_id, card_json in db.execute("SELECT oracle_id, card_json FROM cards"):
                card = json.loads(card_json)
                names = {card["name"], *(face["name"] for face in card.get("faces", []) if face.get("name"))}
                db.executemany("INSERT OR IGNORE INTO names VALUES (?, ?)",
                               [(key(name), oracle_id) for name in names if key(name)])
            count = db.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
            if not count:
                raise ValueError("The archive contained no English cards with Oracle IDs.")
            with source.open("rb") as handle:
                checksum = hashlib.file_digest(handle, "sha256").hexdigest()
            manifest = {
                "snapshot": source.name, "source_sha256": checksum,
                "printing_rows": source_rows, "english_rows_with_oracle_id": english_rows,
                "unique_oracle_cards": count,
            }
            db.execute("INSERT INTO manifest VALUES (?)", (json.dumps(manifest),))
        staging.replace(destination)
    return manifest


class CardIndex:
    def __init__(self, db):
        self.db = db
        self.manifest = json.loads(db.execute("SELECT data FROM manifest").fetchone()[0])

    def get(self, name):
        matches = [json.loads(row[0]) for row in self.db.execute("""
            SELECT card_json FROM names JOIN cards USING (oracle_id) WHERE names.name=?
            ORDER BY released DESC, printing_id DESC
        """, (key(name),))]
        full_names = [card for card in matches if key(card["name"]) == key(name)]
        return full_names or matches

    def for_question(self, question, requested=()):
        if len(requested) > 6:
            raise ValueError("Specify up to six cards.")
        cards = {}
        for name in requested:
            matches = self.get(name)
            if not matches:
                raise ValueError(f"No card or face with the name {name!r} in this snapshot.")
            for card in matches:
                if len(cards) == 6:
                    return list(cards.values())
                cards.setdefault(card["oracle_id"], card)
        if requested:
            return list(cards.values())
        tokens = words(question)
        cursor = 0
        while cursor < len(tokens) and len(cards) < 6:
            for length in range(min(12, len(tokens) - cursor), 1, -1):
                candidates = self.get(" ".join(tokens[cursor:cursor + length]))
                if candidates:
                    for card in candidates[:6 - len(cards)]:
                        cards.setdefault(card["oracle_id"], card)
                    cursor += length
                    break
            else:
                cursor += 1
        return list(cards.values())

    def training_texts(self):
        for (payload,) in self.db.execute("SELECT card_json FROM cards ORDER BY oracle_id"):
            yield format_card(json.loads(payload))


def open_cards(path=DATABASE):
    return closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True))


def format_card(card):
    def fields(record):
        lines = []
        for label, field in (
            ("Mana cost", "mana_cost"), ("Type", "type_line"), ("Oracle text", "oracle_text"),
            ("Power", "power"), ("Toughness", "toughness"), ("Loyalty", "loyalty"), ("Defense", "defense"),
        ):
            if record.get(field):
                lines.append(f"{label}: {record[field]}")
        if "cmc" in record:
            lines.append(f"Scryfall mana value (cmc): {record['cmc']}")
        if record.get("mana_cost"):
            lines.append(f"Mana symbols: {cost_explanation(record['mana_cost'])}")
            if "{X}" in record["mana_cost"]:
                for value in (0, 1, 3, 5):
                    try:
                        total = mana_value(record["mana_cost"], value)
                    except ValueError:
                        break
                    lines.append(f"Printed-cost symbol total with X={value}: {total}")
        return lines

    lines = [f"Card: {card['name']}", *fields(card)]
    for face in card.get("faces", []):
        lines.extend([f"Face: {face.get('name', 'Unnamed face')}", *fields(face)])
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--file", type=Path, default=Path.home() / "Downloads" / SNAPSHOT)
    lookup = commands.add_parser("lookup")
    lookup.add_argument("name")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            if not args.file.is_file():
                raise ValueError(f"Archive not found at {args.file}. Pass --file with the downloaded JSONL.GZ path.")
            print(json.dumps(import_cards(args.file), indent=2))
        else:
            with open_cards() as db:
                matches = CardIndex(db).get(args.name)
            if not matches:
                raise ValueError(f"No card or face with the name {args.name!r}.")
            print("\n\n".join(map(format_card, matches)))
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
