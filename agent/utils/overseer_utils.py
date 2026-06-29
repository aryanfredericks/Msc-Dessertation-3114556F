def build_requery_prompt(
    doc_text: str,
    requery_cases: list[dict]  # [{"span_text", "context", "hint"}]
) -> str:
    cases_str = "\n".join(
        f"- span: \"{c['span_text']}\" | context: \"{c['context']}\" | hint: {c['hint']}"
        for c in requery_cases
    )
    return f"""Document (excerpt for context):
{doc_text[:1000]}

Spans to classify:
{cases_str}"""

REQUERY_SYSTEM_PROMPT = """You are an expert biomedical named entity annotator.

You will be given a list of spans from a biomedical document that could not be 
confidently typed by automated tools. For each span, you are given the surrounding 
context and any partial information from previous classifiers.

Classify each span into exactly one of these types, or null if it is not a 
biomedical entity:
- GeneOrGeneProduct: genes, proteins, gene products (e.g. BRCA1, p53, TNF-alpha)
- DiseaseOrPhenotypicFeature: diseases, syndromes, phenotypes (e.g. breast cancer, hypertension)
- ChemicalEntity: chemicals, drugs, small molecules (e.g. aspirin, cisplatin)
- SequenceVariant: genetic variants, mutations, HGVS notation (e.g. V600E, rs1801133)
- OrganismTaxon: organisms, species (e.g. human, mouse, E. coli)
- CellLine: cultured cell lines (e.g. HeLa, HEK293)

Rules:
- Use the context to disambiguate
- Return null if the span is genuinely not a biomedical entity
- Do not guess — if uncertain, return null

Respond ONLY with this exact JSON, no other text:
{
  "entities": [
    {"span_text": "the span text", "entity_type": "TypeName or null", "reasoning": "brief reason"}
  ]
}"""