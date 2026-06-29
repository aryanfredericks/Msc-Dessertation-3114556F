import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from biored import CANONICAL_TYPES

COMMON_TYPES = {
    "GeneOrGeneProduct",
    "DiseaseOrPhenotypicFeature", 
    "ChemicalEntity"
}

def predict_span_type(
    span_text: str,
    passage_text: str,
    passage_offset: int,
    doc_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification,
    device: str,
    max_length: int = 512,
) -> tuple[str | None, float]:
    span_start_in_passage = passage_text.find(span_text)
    if span_start_in_passage == -1:
        return None, 0.0

    
    enc = tokenizer(
        passage_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    offset_mapping = enc["offset_mapping"][0].tolist()
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    probs = torch.softmax(logits[0], dim=-1)
    pred_ids = logits[0].argmax(-1).tolist()
    id2label = model.config.id2label

    span_end_in_passage = span_start_in_passage + len(span_text)
    span_labels: list[tuple[str, float]] = []
    for idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == 0 and tok_end == 0:
            continue 
        if tok_start >= span_start_in_passage and tok_end <= span_end_in_passage:
            label = id2label[pred_ids[idx]]
            conf = probs[idx][pred_ids[idx]].item()
            span_labels.append((label, conf))

    if not span_labels:
        return None, 0.0
    
    type_votes: dict[str, list[float]] = {}
    for label, conf in span_labels:
        if label == "O":
            continue
        entity_type = label[2:]  # strip B- or I-
        if entity_type not in COMMON_TYPES:
            continue
        type_votes.setdefault(entity_type, []).append(conf)

    if not type_votes:
        return None, 0.0

    best_type = max(type_votes, key=lambda t: sum(type_votes[t]) / len(type_votes[t]))
    best_conf = sum(type_votes[best_type]) / len(type_votes[best_type])
    return best_type, best_conf