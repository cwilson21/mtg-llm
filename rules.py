"""Import the official TXT and retrieve passages for rules questions."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from urllib.request import urlopen


DEFAULT_URL = "https://media.wizards.com/2026/downloads/MagicCompRules%2020261106.txt"
DATA = Path("data")
RULE_ID = r"[1-9]\d{2}\.\d+[a-z]*"
RULE_LINE = re.compile(rf"^({RULE_ID})\.?[ \t]+(.+)$")
SECTION = re.compile(r"^[1-9]\d{2}\.[ \t]+(.+)$")
CHAPTERS = {
    "1. Game Concepts", "2. Parts of a Card", "3. Card Types", "4. Zones",
    "5. Turn Structure", "6. Spells, Abilities, and Effects", "7. Additional Rules",
    "8. Multiplayer Rules", "9. Casual Variants",
}
ID_IN_TEXT = re.compile(rf"(?<![\w.]){RULE_ID}(?!\w|\.\d)", re.IGNORECASE)
WORDS = re.compile(r"[a-z0-9]+")
STOP_WORDS = set(
    "a an and are as at be by can could did do does for from had has have how i if in "
    "is it its me my of on or our s should so than that the their them then there "
    "these they this to was were what when where which who will with would you your".split()
)


@dataclass(frozen=True)
class Passage:
    id: str
    heading: str
    text: str


def decode_rules(raw):
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def parse_rules(text):
    lines = [line.strip() for line in text.splitlines()]
    first = next((i for i, line in enumerate(lines) if re.match(r"100\.1\.[ \t]", line)), None)
    if first is None:
        raise ValueError("Could not find the first numbered rule. Use Wizards' English TXT.")
    start = next((i for i in range(first - 1, -1, -1) if lines[i] == "1. Game Concepts"), first)
    glossary = lines.index("Glossary", first)
    credits = lines.index("Credits", glossary)
    numbered = []
    heading = ""
    rule_id = None
    body = []
    for line in lines[start:glossary]:
        rule = RULE_LINE.fullmatch(line)
        section = SECTION.fullmatch(line)
        if rule or section or line in CHAPTERS:
            if rule_id:
                numbered.append(Passage(rule_id, heading, "\n".join(body).strip()))
            if rule:
                rule_id, content = rule.groups()
                body = [content]
            else:
                heading = line
                rule_id = None
        elif rule_id:
            body.append(line)
    if rule_id:
        numbered.append(Passage(rule_id, heading, "\n".join(body).strip()))
    terms = []
    for entry in re.split(r"\n{2,}", "\n".join(lines[glossary + 1:credits]).strip()):
        title, separator, definition = entry.partition("\n")
        if separator:
            terms.append(Passage(f"Glossary: {title}", "Glossary", definition))
    corpus = "\n".join(lines[start:credits]).strip() + "\n"
    return numbered, terms, corpus


def write_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def prepare(args):
    if args.file:
        raw = args.file.read_bytes()
        source = str(args.file.resolve())
    else:
        with urlopen(args.url, timeout=45) as response:
            raw = response.read()
            source = response.geturl()
    text = decode_rules(raw)
    numbered, glossary, corpus = parse_rules(text)
    ids = [p.id for p in numbered]
    if (
        len(ids) < 1_000 or len(ids) != len(set(ids))
        or {rule_id[0] for rule_id in ids} != set("123456789")
        or len(glossary) < 100
    ):
        raise ValueError("The TXT appears incomplete or its structure changed.")
    effective = re.search(r"effective as of ([^\r\n]+?)\.?\s*$", text, re.I | re.M)
    manifest = {
        "source": source,
        "effective": effective.group(1) if effective else None,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "numbered_rules": len(numbered),
        "glossary_entries": len(glossary),
    }
    DATA.mkdir(exist_ok=True)
    write_text(DATA / "corpus.txt", corpus)
    write_text(DATA / "passages.json", json.dumps(
        {"manifest": manifest, "passages": [vars(p) for p in numbered + glossary]},
        ensure_ascii=False,
    ))
    print(json.dumps(manifest, indent=2))


def terms_for(text):
    terms = WORDS.findall(text.lower())
    # Match common singular/plural variants such as target/targets.
    return [
        term[:-1] if len(term) > 4 and term.endswith("s") and not term.endswith("ss") else term
        for term in terms if term not in STOP_WORDS
    ]


class RuleIndex:
    def __init__(self, path=DATA / "passages.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.manifest = data["manifest"]
        self.passages = [Passage(**item) for item in data["passages"]]
        self.by_id = {p.id.lower(): p for p in self.passages}
        self.documents = [Counter(terms_for(f"{p.id} {p.heading} {p.text}")) for p in self.passages]
        self.lengths = [sum(doc.values()) for doc in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.frequency = Counter(word for doc in self.documents for word in doc)

    def search(self, question, limit=8):
        explicit = dict.fromkeys(rule.lower() for rule in ID_IN_TEXT.findall(question))
        picked = [self.by_id[rule] for rule in explicit if rule in self.by_id]
        keys = set(terms_for(question))
        count = len(self.documents)
        weights = {
            key: math.log1p((count - self.frequency[key] + 0.5) / (self.frequency[key] + 0.5))
            for key in keys
        }
        scored = []
        for index, (doc, length) in enumerate(zip(self.documents, self.lengths)):
            norm = 1.2 * (0.25 + 0.75 * length / self.average_length)
            score = sum(weights[key] * (doc[key] * 2.2) / (doc[key] + norm)
                        for key in sorted(keys & doc.keys()))
            if score and self.passages[index].id.lower() not in explicit:
                scored.append((-score, index))
        picked.extend(self.passages[index] for _, index in sorted(scored)[:limit])
        return picked[:limit]


def format_passage(passage):
    label = passage.id if passage.id.startswith("Glossary: ") else f"CR {passage.id}"
    return f"[{label}] {passage.heading}\n{passage.text}"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    import_cmd = commands.add_parser("prepare", help="Import the Comprehensive Rules TXT.")
    source = import_cmd.add_mutually_exclusive_group()
    source.add_argument("--file", type=Path)
    source.add_argument("--url", default=DEFAULT_URL)
    lookup = commands.add_parser("lookup", help="Search the imported rules without a model.")
    lookup.add_argument("question")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args)
        else:
            index = RuleIndex()
            print(f"Rules edition: {index.manifest['effective'] or 'unknown'}")
            print("\n\n".join(map(format_passage, index.search(args.question))))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
