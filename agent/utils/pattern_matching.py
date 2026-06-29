import re
from typing import Optional

# HGVS protein-level: p.Val600Glu, p.V600E, p.Arg554*, p.Leu858Arg
HGVS_PROTEIN = re.compile(
    r'\bp\.'
    r'(?:[A-Z][a-z]{2}|\*)' 
    r'\d+'                    
    r'(?:[A-Z][a-z]{2}|[A-Z]|\*|fs|del|dup|ins)?'
    r'\b',
    re.IGNORECASE
)

# HGVS DNA/cDNA-level: c.1799T>A, c.del, c.35insG
HGVS_CDNA = re.compile(
    r'\bc\.\d+(?:[+-]\d+)?'
    r'(?:[ACGT]+>[ACGT]+|del[ACGT]*|ins[ACGT]*|dup[ACGT]*)?'
    r'\b',
    re.IGNORECASE
)

# HGVS genomic: g.12345A>T
HGVS_GENOMIC = re.compile(
    r'\bg\.\d+[ACGT]>[ACGT]\b',
    re.IGNORECASE
)

# rsIDs: rs1801133, rs334
RSID = re.compile(r'\brs\d{3,}\b')

# Short amino acid substitution (most common in BioRED): A118G, V600E, R553X
# format: single-letter AA + position + single-letter AA or stop
AA_SHORT = re.compile(r'\b[ACDEFGHIKLMNPQRSTVWY]\d{1,4}[ACDEFGHIKLMNPQRSTVWY*]\b')

# Deletion/insertion shorthand: del508, DelF508, ins6
INDEL = re.compile(r'\b(?:del|ins|dup)[A-Z]?\d+\b', re.IGNORECASE)

ALL_PATTERNS = [
    ("HGVS_PROTEIN", HGVS_PROTEIN),
    ("HGVS_CDNA",    HGVS_CDNA),
    ("HGVS_GENOMIC", HGVS_GENOMIC),
    ("RSID",         RSID),
    ("AA_SHORT",     AA_SHORT),
    ("INDEL",        INDEL),
]


def match_sequence_variant(text: str) -> Optional[str]:
    """Return the matched pattern name if text is a SequenceVariant, else None.
    
    Checks the span text against all HGVS/rsID/AA patterns in priority order.
    Returns the name of the first matching pattern for traceability, or None.
    """
    t = text.strip()
    for name, pattern in ALL_PATTERNS:
        if pattern.fullmatch(t) or pattern.search(t):
            return name   # return pattern name so you can log which rule fired
    return None