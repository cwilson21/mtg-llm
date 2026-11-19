"""Fetch and normalize EDHREC's paginated early-game two-card combo listing."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen


LISTING = "https://edhrec.com/combos/early-game-2-card-combos"
PAGE_ROOT = "https://json.edhrec.com/pages/"
PAGE_PATH = re.compile(r"combos/early-game-2-card-combos-(\d+)\.json")


def fetch(url):
    request = Request(url, headers={"User-Agent": "mtg-llm curriculum builder"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def initial_page(html):
    # The first page is rendered into a Next.js data payload. Subsequent pages
    # are public JSON documents referenced by the payload's "more" field.
    for match in re.finditer(r"self\.__next_f\.push\((.*?)\)</script>", html):
        frame = json.loads(match[1])
        if len(frame) < 2 or not isinstance(frame[1], str):
            continue
        for line in frame[1].splitlines():
            if '"combosList":' in line:
                return json.loads(line.split(":", 1)[1])[3]["page"]["combosList"]
    raise ValueError("EDHREC's initial combo page did not expose the expected list")


def normalize(combo_id, names, vote, results, prerequisites, url):
    if vote["bracket"] != "4-5" or not names or len(set(names)) != 2 or len(names) != 2:
        raise ValueError(f"Unexpected bracket vote or card count for {combo_id}: {names}")
    return {
        "id": combo_id, "cards": sorted(names, key=str.casefold),
        "source": url, "outcomes": results,
        "non_card_prerequisites": prerequisites,
    }


def collect():
    first = initial_page(fetch(LISTING).decode("utf-8"))
    combos = [
        normalize(combo["id"], [card["name"] for card in combo["cards"]],
                  combo["vote"], combo["results"], combo["nonCardPrerequisiteCount"], LISTING)
        for combo in first["combos"]
    ]
    pages = [{"url": LISTING, "count": len(combos)}]
    next_page = first["more"]
    total_pages = None
    while next_page:
        match = PAGE_PATH.fullmatch(next_page)
        if not match or int(match[1]) != len(pages):
            raise ValueError(f"Unexpected EDHREC next page: {next_page!r}")
        url = PAGE_ROOT + next_page
        page = json.loads(fetch(url))
        if total_pages is None:
            total_pages = page["total_pages"]
        if total_pages != page["total_pages"]:
            raise ValueError("EDHREC page count changed during download")
        if not page["cardlists"]:
            raise ValueError(f"Empty EDHREC continuation page: {url}")
        for entry in page["cardlists"]:
            combo = entry["combo"]
            combos.append(normalize(combo["comboId"], [card["name"] for card in entry["cardviews"]],
                                    combo["comboVote"], combo["results"],
                                    combo["nonCardPrerequisiteCount"], url))
        pages.append({"url": url, "count": len(page["cardlists"])})
        next_page = page.get("more")
    if total_pages != len(pages):
        raise ValueError(f"Expected {total_pages} EDHREC pages and downloaded {len(pages)}")
    if len({combo["id"] for combo in combos}) != len(combos):
        raise ValueError("Duplicate combo IDs across EDHREC pages")
    return {
        "listing": LISTING, "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "pages": pages, "combos": combos,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/edhrec-early-combos.json"))
    args = parser.parse_args()
    snapshot = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved {len(snapshot['combos'])} combos from {len(snapshot['pages'])} pages to {args.output}")


if __name__ == "__main__":
    main()
