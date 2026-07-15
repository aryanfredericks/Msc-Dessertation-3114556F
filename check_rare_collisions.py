import argparse
import time
import requests
from pathlib import Path


def parse_spans_file(path: Path) -> dict[str, list[str]]:
    """Parses the '# TypeName (N unique spans)' + one-per-line format
    written by extract_rare_entities.py."""
    sections: dict[str, list[str]] = {}
    current = None
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# "):
                current = line[2:].split(" (")[0]
                sections[current] = []
            elif line.strip() and current:
                sections[current].append(line)
    return sections


def cellosaurus_exact_match(text: str, retries: int = 3) -> tuple[bool, list[str]]:
    """Mirrors lookup_cellosaurus() but also returns what it matched on,
    for inspection."""
    url = "https://api.cellosaurus.org/search/cell-line"
    params = {"q": text, "fields": "id,name", "format": "json", "rows": 5}
    text_lower = text.strip().lower()

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            cell_lines = data.get("Cellosaurus", {}).get("cell-line-list", [])
            matched_ids = []
            for cl in cell_lines:
                for name_entry in cl.get("name-list", []):
                    if name_entry.get("value", "").strip().lower() == text_lower:
                        matched_ids.append(cl.get("id", "?"))
            return (len(matched_ids) > 0, matched_ids)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"    [error] lookup failed for '{text}': {e}")
                return (False, [])
    return (False, [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans_file", default="outputs/rare_entity_analysis/organism_cellline_spans.txt")
    ap.add_argument("--sleep", type=float, default=0.3, help="delay between API calls, be polite to Cellosaurus")
    args = ap.parse_args()

    sections = parse_spans_file(Path(args.spans_file))
    organism_spans = sections.get("OrganismTaxon", [])

    if not organism_spans:
        print(f"[error] no OrganismTaxon spans found in {args.spans_file}")
        return

    print(f"[check] testing {len(organism_spans)} unique OrganismTaxon spans against Cellosaurus...\n")

    collisions = []
    for i, span in enumerate(organism_spans, 1):
        matched, ids = cellosaurus_exact_match(span)
        status = "COLLISION" if matched else "clear"
        print(f"  [{i}/{len(organism_spans)}] '{span}' -> {status}" + (f" ({', '.join(ids)})" if matched else ""))
        if matched:
            collisions.append((span, ids))
        time.sleep(args.sleep)

    print(f"\n[summary] {len(collisions)} / {len(organism_spans)} gold OrganismTaxon spans "
          f"also get a Cellosaurus exact match")
    if collisions:
        print("\nThese spans would be misrouted to CellLine by resolve_rare_entity()'s "
              "sequential ordering, since Cellosaurus is checked first:")
        for span, ids in collisions:
            print(f"    '{span}' -> Cellosaurus id(s): {', '.join(ids)}")
    else:
        print("\nNo collisions found - the sequential-lookup-order hypothesis is NOT "
              "supported by this data. OrganismTaxon's low recall likely traces back "
              "to the span extraction stage instead (spans never generated as "
              "candidates in the first place), consistent with the earlier diagnosis.")


if __name__ == "__main__":
    main()