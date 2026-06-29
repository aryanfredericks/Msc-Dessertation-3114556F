"""Tier 3 - prompt construction for single-call LLM NER.

The type descriptions are this tier's "label phrasing knob" (the analogue of
Tier 2's LABEL_PHRASES): a crisp gloss of what each type includes/excludes
measurably changes results. Tune the wordings on dev, never test.
"""
import json

from biored import CANONICAL_TYPES, load_biored_documents

TYPE_DESCRIPTIONS = {
    "GeneOrGeneProduct": "genes, proteins, and gene products (e.g. BRCA1, p53, TNF-alpha)",
    "DiseaseOrPhenotypicFeature": (
        "diseases, disorders, syndromes, signs, symptoms, and phenotypic "
        "abnormalities (e.g. breast cancer, hypertension, hearing loss)"
    ),
    "ChemicalEntity": "chemicals, drugs, and small molecules (e.g. aspirin, glucose, cisplatin)",
    "SequenceVariant": (
        "genetic / sequence variants and mutations, including HGVS notation "
        "(p.Val600Glu, c.1799T>A), rsIDs (rs1801133), and protein- or DNA-level changes"
    ),
    "OrganismTaxon": "organisms and species (e.g. human, Homo sapiens, mouse, Escherichia coli)",
    "CellLine": "cultured cell lines (e.g. HeLa, HEK293, MCF-7)",
}


def build_system_prompt():
    lines = [
        "You are a biomedical named entity recognition system.",
        "Identify every entity mention in the text and classify each into exactly "
        "one of these types:",
        "",
    ]
    for t in CANONICAL_TYPES:
        lines.append(f"- {t}: {TYPE_DESCRIPTIONS[t]}")
    lines += [
        "",
        "Rules:",
        "- Return EVERY mention, including repeated ones.",
        '- The "text" field must be copied verbatim as a substring of the input '
        "(exact characters and case).",
        '- "type" must be exactly one of the type names listed above.',
        "- Do not include character offsets. Do not explain.",
        "",
        "Respond with ONLY a JSON object of this exact form:",
        '{"entities": [{"text": "<surface string>", "type": "<TypeName>"}]}',
    ]
    return "\n".join(lines)


def build_doc_text(passages):
    """Place each passage at its absolute offset so string search yields gold-aligned offsets.

    passages: list[(offset, text)]. Gaps between passages are filled with spaces.
    """
    end = 0
    for off, text in passages:
        end = max(end, off + len(text))
    buf = [" "] * end
    for off, text in passages:
        for i, ch in enumerate(text):
            buf[off + i] = ch
    return "".join(buf)


def reconstruct_doc_text(doc):
    return build_doc_text([(p.offset, p.text) for p in doc.passages])


def build_fewshot_messages(example_json, shots):
    """Few-shot ICL turns drawn from train/dev (NEVER test).

    Picks the `shots` shortest documents to keep the prompt compact, and formats
    each as (user = document text, assistant = gold entities JSON).
    """
    docs, _, _ = load_biored_documents(example_json)
    chosen = sorted(docs, key=lambda d: len(reconstruct_doc_text(d)))[:shots]

    msgs = []
    for d in chosen:
        text = reconstruct_doc_text(d)
        seen, ents = set(), []
        for p in d.passages:
            for e in p.entities:
                key = (e.text, e.type)
                if key in seen:
                    continue
                seen.add(key)
                ents.append({"text": e.text, "type": e.type})
        msgs.append({"role": "user", "content": text})
        msgs.append(
            {
                "role": "assistant",
                "content": json.dumps({"entities": ents}, ensure_ascii=False),
            }
        )
    return msgs