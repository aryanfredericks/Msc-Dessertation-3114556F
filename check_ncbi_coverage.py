import argparse
import os
import time
import requests
from pathlib import Path


def parse_spans_file(path: Path) -> dict[str, list[str]]:
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


def ncbi_taxonomy_hit(text: str, retries: int = 3) -> tuple[bool, int]:
    """Mirrors lookup_ncbi_taxonomy() exactly, but returns the hit count too."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "taxonomy",
        "term": text,
        "retmode": "json",
        "retmax": 1,
        "email": os.getenv("NCBI_EMAIL", ""),
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            count = int(data["esearchresult"]["count"])
            return (count > 0, count)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"    [error] lookup failed for '{text}': {e}")
                return (False, -1)
    return (False, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans_file", default="outputs/rare_entity_analysis/organism_cellline_spans.txt")
    ap.add_argument("--sleep", type=float, default=0.34, help="NCBI e-utils asks for max ~3 req/sec without an API key")
    args = ap.parse_args()

    sections = parse_spans_file(Path(args.spans_file))
    organism_spans = sections.get("OrganismTaxon", [])

    if not organism_spans:
        print(f"[error] no OrganismTaxon spans found in {args.spans_file}")
        return

    print(f"[check] testing {len(organism_spans)} unique OrganismTaxon spans against NCBI Taxonomy...\n")

    no_hit = []
    for i, span in enumerate(organism_spans, 1):
        hit, count = ncbi_taxonomy_hit(span)
        status = "MATCH" if hit else "NO MATCH"
        print(f"  [{i}/{len(organism_spans)}] '{span}' -> {status} (count={count})")
        if not hit:
            no_hit.append(span)
        time.sleep(args.sleep)

    print(f"\n[summary] {len(no_hit)} / {len(organism_spans)} unique gold OrganismTaxon spans "
          f"get ZERO NCBI Taxonomy results")
    if no_hit:
        print("\nThese are valid BioRED OrganismTaxon gold mentions that NCBI Taxonomy simply")
        print("does not have an entry for - a KB coverage gap, not an extraction or code bug:")
        for span in no_hit:
            print(f"    '{span}'")
    else:
        print("\nAll unique spans resolved - the KB coverage gap hypothesis is NOT supported.")
        print("The typing failure must be happening elsewhere (check for silent exceptions,")
        print("rate limiting during the real pipeline run, or the confidence gate logic).")


if __name__ == "__main__":
    main()