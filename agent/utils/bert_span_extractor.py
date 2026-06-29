import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from biored import encode_for_inference, decode_chunk


def extract_spans_with_bert(
    doc,
    doc_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification,
    device: str,
    max_length: int = 256,
    stride: int = 128,
) -> list[str]:
    
    id2label = model.config.id2label
    chunks = encode_for_inference([doc], tokenizer, max_length, stride)

    found: dict[tuple[int, int], str] = {}  # (start, end) -> surface string

    for chunk in chunks:
        L = len(chunk["input_ids"])
        input_ids = torch.tensor(
            [chunk["input_ids"]], dtype=torch.long
        ).to(device)
        attn = torch.tensor(
            [chunk["attention_mask"]], dtype=torch.long
        ).to(device)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids, attention_mask=attn
            ).logits

        pred_ids = logits.argmax(-1)[0].tolist()
        lab_strs = [id2label[int(x)] for x in pred_ids[:L]]

        ents = decode_chunk(
            lab_strs,
            chunk["offset_mapping"],
            chunk["sequence_ids"],
            chunk["passage_offset"]
        )

        for e in ents:
            key = (e["start"], e["end"])
            if key in found:
                continue  # dedupe across sliding window overlap
            surface = doc_text[e["start"]:e["end"]].strip()
            if surface:
                found[key] = surface

    return list(found.values())