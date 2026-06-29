import os
import time
import requests
from typing import Optional

_cache: dict[str, tuple[Optional[str], float, str]] = {}

def lookup_cellosaurus(text: str, retries: int = 3) -> bool:
    url = "https://api.cellosaurus.org/search/cell-line"
    params = {"q": text, "fields": "id", "format": "json", "rows": 5}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            cell_lines = data.get("Cellosaurus", {}).get("cell-line-list", [])
            text_lower = text.strip().lower()
            for cl in cell_lines:
                for name_entry in cl.get("name-list", []):
                    if name_entry.get("value", "").strip().lower() == text_lower:
                        return True
            return False
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"[cellosaurus] lookup failed for '{text}': {e}")
                return False

def lookup_ncbi_taxonomy(text: str, retries: int = 3) -> bool:
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
            return count > 0
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"[ncbi_taxonomy] lookup failed for '{text}': {e}")
                return False

def resolve_rare_entity(text: str) -> tuple[Optional[str], float, str]:
    if text in _cache:
        return _cache[text]
    if lookup_cellosaurus(text):
        result = ("CellLine", 1.0, "cellosaurus_exact_match")
    elif lookup_ncbi_taxonomy(text):
        result = ("OrganismTaxon", 1.0, "ncbi_taxonomy_count")
    else:
        result = (None, 0.0, "not_resolved")
    _cache[text] = result
    return result