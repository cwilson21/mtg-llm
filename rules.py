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
RULE_REFERENCE = re.compile(r"\brules?\s+([1-9]\d{2}(?:\.\d+[a-z]*)?)", re.I)
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
    members: tuple[str, ...] = ()


def retrieved_rule_ids(passages):
    return list(dict.fromkeys(identifier for p in passages for identifier in (p.id, *p.members)))


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
    return [
        term[:-1] if len(term) > 4 and term.endswith("s") and not term.endswith("ss") else term
        for term in terms if term not in STOP_WORDS
    ]


def topic_terms(text):
    result = set()
    for term in WORDS.findall(text.lower()):
        if term in STOP_WORDS:
            continue
        if term.endswith("ies"):
            term = term[:-3] + "y"
        elif len(term) > 5 and term.endswith("ed"):
            term = term[:-2]
        elif len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
            term = term[:-1]
        result.add(term)
    return result


class RuleIndex:
    def __init__(self, path=DATA / "passages.json", semantic=True):
        raw = path.read_bytes()
        data = json.loads(raw)
        self.source_hash = hashlib.sha256(raw).hexdigest()
        self.manifest = data["manifest"]
        self.passages = [Passage(**item) for item in data["passages"]]
        self.by_id = {p.id.lower(): p for p in self.passages}
        self.documents = [Counter(terms_for(f"{p.id} {p.heading} {p.text}")) for p in self.passages]
        self.lengths = [sum(doc.values()) for doc in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.frequency = Counter(word for doc in self.documents for word in doc)
        self.glossary = [p for p in self.passages if p.id.startswith("Glossary: ")]
        self.semantic = None
        cache = path.with_name("rule_vectors.npz")
        if semantic and cache.is_file():
            from retrieval import SemanticIndex
            self.semantic = SemanticIndex(cache, self.source_hash)

    def retrieval_info(self):
        return {
            "mode": "hybrid-reranked" if self.semantic else "lexical",
            "passages_sha256": self.source_hash,
            "semantic": dict(self.semantic.metadata) if self.semantic else None,
        }

    def rule_context(self, passage, max_characters=3000):
        match = re.match(r"[1-9]\d{2}\.\d+", passage.id)
        if not match:
            return passage
        parent = match.group()
        family = [passage] + [
            p for p in self.passages if p.id != passage.id
            and (p.id == parent or p.id.startswith(parent) and p.id[len(parent):].isalpha())
        ]
        selected = []
        length = 0
        for member in family:
            if selected and length + len(member.text) > max_characters:
                continue
            selected.append(member)
            length += len(member.text)
        return Passage(
            passage.id, passage.heading,
            "\n\n".join(f"[CR {p.id}] {p.text}" for p in selected),
            tuple(p.id for p in selected),
        )

    def search(self, question, limit=12, cards=()):
        if limit < 1:
            return []
        explicit = dict.fromkeys(rule.lower() for rule in ID_IN_TEXT.findall(question))
        picked = [self.by_id[rule] for rule in explicit if rule in self.by_id]
        # Mechanics from actual Oracle records connect named-card questions to rules.
        mechanics = " ".join(keyword for card in cards for keyword in card.get("keywords", []))
        query = mechanics + ". " + question if mechanics else question
        oracle = []
        for card in cards:
            for face in (card, *card.get("faces", [])):
                oracle.append(f"{face.get('type_line', '')} {face.get('oracle_text', '')}"[:400])
        if oracle:
            query += " Oracle text: " + " ".join(oracle)
        stats = re.findall(r"(?<![-+\d])(\d+)\s*/\s*(\d+)", question)
        changes = re.findall(r"([+-]\d+)\s*/\s*([+-]\d+)", question)
        if len(stats) == len(changes) == 1:
            toughness = int(stats[0][1]) + int(changes[0][1])
            query += f" Computed creature toughness after the modifier is {toughness}."
            if toughness <= 0:
                query += " Creature toughness zero or less."
        concepts = topic_terms(query)
        if {"trigger", "ability"} <= concepts and any(
            term in query.lower() for term in ("simultaneous", "together", "same event")
        ):
            query += " Multiple triggered abilities controlled by different players. APNAP order of abilities on the stack."
        if "exil" in query.lower() and "return" in query.lower():
            query += " Counters retained when an object moves from one zone to another."
        keys = set(terms_for(query))
        count = len(self.documents)
        weights = {
            key: math.log1p((count - self.frequency[key] + 0.5) / (self.frequency[key] + 0.5))
            for key in keys
        }
        scores = {}
        for index, (doc, length) in enumerate(zip(self.documents, self.lengths)):
            norm = 1.2 * (0.25 + 0.75 * length / self.average_length)
            score = sum(weights[key] * (doc[key] * 2.2) / (doc[key] + norm)
                        for key in sorted(keys & doc.keys()))
            if score:
                scores[self.passages[index].id] = score

        ranked = sorted(scores, key=lambda identifier: (-scores[identifier], identifier))
        if self.semantic:
            dense = self.semantic.search(query, 40)
            # Reciprocal-rank fusion keeps exact terms and semantic matches useful.
            fused = Counter()
            for ranking, weight in ((ranked[:40], 1.0), (dense, 2.0)):
                for rank, identifier in enumerate(ranking, 1):
                    fused[identifier] += weight / (10 + rank)
            ranked = sorted(fused, key=lambda identifier: (-fused[identifier], identifier))
        # Use explicit mechanic names in the official glossary to bring in their
        # substantive definitions. Generic timing words cannot replace these.
        topics = []
        query_terms = topic_terms(query)
        for passage in self.glossary:
            title = passage.id.removeprefix("Glossary: ")
            normalized = topic_terms(title)
            if normalized and normalized <= query_terms:
                specificity = sum(weights.get(term, 0) for term in normalized) / math.sqrt(len(normalized))
                topics.append((-specificity, passage))
        anchors = []
        definitions = []
        for _, passage in sorted(topics, key=lambda item: (item[0], item[1].id))[:3]:
            references = RULE_REFERENCE.findall(passage.text)
            if not references:
                # Glossary aliases such as APNAP Order point to another term.
                # Find operative rules that explicitly use that term as well.
                title_terms = topic_terms(passage.id.removeprefix("Glossary: "))
                matches = [
                    p for p in self.passages if not p.id.startswith("Glossary: ")
                    and title_terms <= topic_terms(p.text)
                ]
                if matches:
                    roots = set()
                    for match in sorted(matches, key=lambda p: (-scores.get(p.id, 0), p.id)):
                        root = re.match(r"[1-9]\d{2}\.\d+", match.id).group()
                        if root not in roots:
                            definitions.append(self.rule_context(match))
                            roots.add(root)
                        if len(roots) == 3:
                            break
            for reference in references:
                family = [
                    p for p in self.passages if (
                        p.id == reference or p.id.startswith(reference + ".")
                        or p.id.startswith(reference) and p.id[len(reference):].isalpha()
                    )
                    and len(p.text.split()) >= 8
                ]
                anchors.extend(family)
                if family:
                    candidates = family[:1] if "." in reference else sorted(
                        family, key=lambda p: (-scores.get(p.id, 0), p.id),
                    )[:3]
                    definitions.extend(self.rule_context(p) for p in candidates)
        if self.semantic:
            candidates = dict.fromkeys(ranked[:60] + [p.id for p in anchors])
            ranked = self.semantic.rerank(query, [self.by_id[key.lower()] for key in candidates])
        picked.extend(p for p in definitions if p.id not in retrieved_rule_ids(picked))
        for passage in definitions[:6]:
            for reference in ID_IN_TEXT.findall(passage.text):
                if reference in self.by_id and reference not in retrieved_rule_ids(picked):
                    picked.append(self.rule_context(self.by_id[reference]))
                    break
        picked.extend(self.by_id[identifier.lower()] for identifier in ranked)
        result = []
        seen = set()
        characters = 0
        for passage in picked:
            if passage.id in seen or result and characters + len(passage.text) > 12000:
                continue
            result.append(passage)
            seen.update(retrieved_rule_ids([passage]))
            characters += len(passage.text)
            if len(result) == limit:
                break
        return result


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
    commands.add_parser("index", help="Build the CPU semantic retrieval index.")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "index":
            from retrieval import build_index
            build_index(RuleIndex(semantic=False))
        else:
            index = RuleIndex()
            print(f"Rules edition: {index.manifest['effective'] or 'unknown'}")
            print("\n\n".join(map(format_passage, index.search(args.question))))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
