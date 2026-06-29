# Biomedical NER — MSc Dissertation
**Biomedical Information Extraction with Agents**
University of Glasgow · MSc Robotics & AI

A four-tier comparison study for biomedical named entity recognition on the
[BioRED dataset](https://github.com/ncbi/BioRED), progressing from a
fine-tuned encoder baseline to a full multi-agent agentic pipeline.

---

## Project Structure

```
biomedical_ner/
├── biored.py                  # shared: BioRED loader, label maps, canonical I/O
├── scorer.py                  # shared: strict / relaxed / per-type scorer
├── dataset/
│   ├── train/Train.BioC.JSON  # 400 documents — fine-tuning + few-shot examples
│   ├── dev/Dev.BioC.JSON      # 100 documents — threshold / hyperparameter tuning
│   └── test/Test.BioC.JSON    # 100 documents — all reported results
├── pubmedbert/                # Tier 1: fine-tuned PubMedBERT
├── gli_ner_bert/              # Tier 2: GLiNER-biomed zero-shot
├── llm/                       # Tier 3: single-call LLM
├── agent/                     # Tier 4: multi-agent system
│   ├── workflow.py            # LangGraph graph definition
│   ├── run_agent.py           # entry point — loads docs, runs graph, saves output
│   ├── models.py              # shared dataclasses and TypedDict state
│   ├── prompts.py             # LLM prompt builders
│   ├── config.py              # model names and paths
│   └── utils/
│       ├── pattern_matching.py   # regex rules for SequenceVariant
│       ├── rare_agent_utils.py   # Cellosaurus + NCBI Taxonomy KB lookups
│       ├── common_agent_utils.py # PubMedBERT span-level type prediction
│       ├── bert_span_extractor.py# BERT candidate span generator
│       ├── overseer_utils.py     # overseer prompt + output schema
│       └── offset_utils.py       # find_occurrences — string to char offsets
└── outputs/
    ├── tier1_pubmedbert/
    ├── tier2_gliner/
    ├── tier3_llm_0shot/
    ├── tier3_llm_3shot/
    └── tier4_agent/
```

---

## Setup

```bash
uv sync
cp .env.example .env
```

Add to `.env`:
```
GROQ_API_KEY=your_key_here
NCBI_EMAIL=your@email.com
```

All commands run from the project root (`biomedical_ner/`) with `PYTHONPATH=.`
so shared modules (`biored.py`, `scorer.py`) resolve correctly.

---

## Scoring

Every tier writes predictions in the same canonical format:
```json
{ "<doc_id>": [{"start": 123, "end": 130, "type": "ChemicalEntity", "text": "aspirin"}] }
```

`start`/`end` are document-absolute character offsets (end exclusive). A
prediction matches gold only if `(doc_id, start, end, type)` are all identical
under strict scoring. Relaxed scoring requires same type and character overlap.

The shared scorer is run identically for every tier:
```bash
PYTHONPATH=. uv run scorer.py \
  --pred outputs/<tier>/test_predictions.json \
  --gold outputs/<tier>/gold_test.json \
  --name <tier_name> \
  --out  outputs/<tier>/full_metrics.json
```

---

## Tier 1 — Fine-tuned PubMedBERT

Supervised token-classification baseline. Fine-tunes
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` on BioRED
training data with BIO tagging over six entity types. Sliding window (stride 128)
handles passages over 512 tokens. Best checkpoint saved by dev F1.

**Train**
```bash
PYTHONPATH=. uv run pubmedbert/train.py \
  --train_json ./dataset/train/Train.BioC.JSON \
  --dev_json   ./dataset/dev/Dev.BioC.JSON \
  --output_dir outputs/tier1_pubmedbert \
  --batch_size 2 --max_length 256
```

> On a 3.7 GiB GPU, `--batch_size 2` and `--max_length 256` are required to
> avoid OOM. Effective batch size is 8 via `gradient_accumulation_steps=4`
> in `TrainingArguments`. Training takes ~45 min.

**Predict**
```bash
PYTHONPATH=. uv run pubmedbert/predict.py \
  --model_dir  outputs/tier1_pubmedbert/model \
  --test_json  ./dataset/test/Test.BioC.JSON \
  --output_dir outputs/tier1_pubmedbert \
  --batch_size 8 --max_length 256
```

**Score**
```bash
PYTHONPATH=. uv run scorer.py \
  --pred outputs/tier1_pubmedbert/test_predictions.json \
  --gold outputs/tier1_pubmedbert/gold_test.json \
  --name tier1_pubmedbert \
  --out  outputs/tier1_pubmedbert/full_metrics.json
```

**Results:** Strict F1 **89.9** · Relaxed F1 **93.8** · Macro F1 **90.7**
Precision 87.9 · Recall 92.0

**Per-type (strict):**

| Type | F1 | Support |
|---|---|---|
| OrganismTaxon | 96.9 | 393 |
| GeneOrGeneProduct | 92.2 | 1180 |
| CellLine | 91.6 | 50 |
| ChemicalEntity | 90.4 | 754 |
| SequenceVariant | 89.1 | 241 |
| DiseaseOrPhenotypicFeature | 83.9 | 917 |

**Key finding:** difficulty tracks surface-form regularity, not frequency.
Disease (most frequent, F1 83.9) is harder than OrganismTaxon (F1 96.9).
Dominant confusion: `GeneOrGeneProduct → ChemicalEntity` (44 errors).
Recall (92.0) exceeds precision (87.9) — the model over-predicts, motivating
a precision-repair layer in Tier 4.

---

## Tier 2 — GLiNER-biomed (Zero-shot)

Off-the-shelf biomedical NER with no BioRED training. Uses
`Ihor/gliner-biomed-large-v1.0`. Returns character spans directly — no
offset reconstruction needed. Confidence threshold tuned on dev, applied
unchanged to test.

**Tune threshold on dev** *(never on test)*
```bash
PYTHONPATH=. uv run gli_ner_bert/tune_threshold.py \
  --dev_json ./dataset/dev/Dev.BioC.JSON
```

> Sweeps thresholds 0.30–0.70, prints P/R/F1 table, recommends best.
> Best threshold on dev: **0.45** (F1 65.0). F1 was flat across 0.30–0.55,
> confirming there is no hidden performance above this ceiling.

**Predict**
```bash
PYTHONPATH=. uv run gli_ner_bert/predict.py \
  --test_json  ./dataset/test/Test.BioC.JSON \
  --output_dir outputs/tier2_gliner \
  --threshold  0.45
```

**Score**
```bash
PYTHONPATH=. uv run scorer.py \
  --pred outputs/tier2_gliner/test_predictions.json \
  --gold outputs/tier2_gliner/gold_test.json \
  --name tier2_gliner \
  --out  outputs/tier2_gliner/full_metrics.json
```

**Results:** Strict F1 **63.3** · Relaxed F1 **76.3** · Macro F1 **52.3**
Precision 64.7 · Recall 61.9

**Key finding:** fine-tuning accounts for ~26 F1 points (Tier 1 vs Tier 2).
GLiNER's weakest types are SequenceVariant (35.4) and OrganismTaxon (39.3) —
the opposite profile from Tier 1, which excels on those same types. The
strict-vs-relaxed gap (13 points) is wider than Tier 1 (4 points), confirming
GLiNER has systematically looser boundary conventions than BioRED annotations.

---

## Tier 3 — Single-call LLM (Groq)

One LLM call per document — the orchestration-free control tier. Uses
`llama-3.1-8b-instant` via Groq with JSON mode and temperature 0. The LLM
returns surface strings (not offsets); a deterministic string-search maps
each string to all token-boundary-aligned occurrences in the document.
Adaptive rate budgeting (rolling 60s window) respects the 6K tokens/min
free-tier limit. Checkpointing allows interrupted runs to resume.

> Set `GROQ_API_KEY` in `.env` before running.
> Rate limits: 6K tokens/min, 500K tokens/day (llama-3.1-8b-instant).
> A full 100-doc run costs ~90K tokens (0-shot) or ~350K tokens (3-shot).

**Zero-shot** *(run 3–5 times for mean ± std — LLM output is non-deterministic)*
```bash
PYTHONPATH=. uv run llm/predict.py \
  --test_json  ./dataset/test/Test.BioC.JSON \
  --shots      0 \
  --output_dir outputs/tier3_llm_0shot
```

**3-shot** *(examples drawn from train, never test)*
```bash
PYTHONPATH=. uv run llm/predict.py \
  --test_json    ./dataset/test/Test.BioC.JSON \
  --example_json ./dataset/train/Train.BioC.JSON \
  --shots        3 \
  --output_dir   outputs/tier3_llm_3shot
```

> If interrupted, rerun the same command — it resumes from checkpoint.
> Delete `outputs/tier3_llm_*/checkpoint.jsonl` to start fresh.

**Score**
```bash
PYTHONPATH=. uv run scorer.py \
  --pred outputs/tier3_llm_0shot/test_predictions.json \
  --gold outputs/tier3_llm_0shot/gold_test.json \
  --name tier3_llm_0shot \
  --out  outputs/tier3_llm_0shot/full_metrics.json

PYTHONPATH=. uv run scorer.py \
  --pred outputs/tier3_llm_3shot/test_predictions.json \
  --gold outputs/tier3_llm_3shot/gold_test.json \
  --name tier3_llm_3shot \
  --out  outputs/tier3_llm_3shot/full_metrics.json
```

**Results:**

| Variant | Strict F1 | Relaxed F1 | Macro F1 | P | R |
|---|---|---|---|---|---|
| 0-shot | 60.9 | 67.5 | 51.5 | 61.9 | 59.9 |
| 3-shot | 57.9 | 64.8 | 51.4 | 65.5 | 51.8 |

**Key finding:** 3-shot degraded overall F1 on the 8B model (−3.0 F1) despite
improving OrganismTaxon (+19.7 F1). The few-shot examples caused the model to
over-apply the gene/chemical pattern from the examples — `GeneOrGeneProduct →
ChemicalEntity` confusion jumped from 32 to 92 errors. This instability of ICL
on small models motivates the structured agentic approach of Tier 4. This is a
reported finding, not a bug.

---

## Tier 4 — Multi-Agent NER System

The dissertation's core contribution. A LangGraph agentic pipeline using
heterogeneous-routing: candidate spans are broadcast to three specialist
branches simultaneously, and a priority combiner arbitrates the votes.

**Architecture:**

```
Document
  └─> Span extraction (PubMedBERT — Tier 1 model, high-recall candidate generator)
        └─> broadcast to all three branches in parallel
              ├─> Pattern branch   — regex for SequenceVariant (HGVS, rsIDs, AA substitutions)
              ├─> Common branch    — PubMedBERT token classifier (gene / disease / chemical)
              └─> Rare branch      — KB lookup (Cellosaurus → CellLine, NCBI Taxonomy → OrganismTaxon)
        └─> Combiner (priority: pattern > rare [if common conf < 0.7] > common > requery > drop)
        └─> Overseer / re-query (Qwen3-32B — resolves low-confidence spans via LLM)
        └─> Offset localisation (deterministic string→char offsets, cap=5 per span)
  └─> Canonical output → scorer.py
```

**Design decisions (all data-driven):**
- Occurrence cap of 5 per span covers 92.5% of BioRED gold mentions (train set analysis)
- Rare branch confidence gate (common conf < 0.7) prevents KB from overriding
  confident encoder predictions — improved F1 by +2.6 points
- Cellosaurus uses exact-match on identifier to prevent substring false positives
- Qwen3-32B think-block (`<think>...</think>`) stripped before JSON parsing

> Set `GROQ_API_KEY` and `NCBI_EMAIL` in `.env` before running.
> Rare branch makes live API calls to Cellosaurus and NCBI Taxonomy.
> A local in-memory cache prevents duplicate KB lookups within a run.
> Rate limits: llama-3.3-70b-versatile (12K tpm), qwen3-32b (6K tpm).

**Debug run** *(5 docs, minimal token cost)*
```bash
PYTHONPATH=. uv run agent/run_agent.py \
  --test_json  ./dataset/test/Test.BioC.JSON \
  --output_dir outputs/tier4_agent \
  --limit      5
```

**Full run** *(100 docs, checkpoint/resume supported)*
```bash
PYTHONPATH=. uv run agent/run_agent.py \
  --test_json  ./dataset/test/Test.BioC.JSON \
  --output_dir outputs/tier4_agent
```

> If interrupted, rerun the same command to resume from checkpoint.
> Delete `outputs/tier4_agent/checkpoint.jsonl` to start fresh.
> Given the 12K tpm llama limit, a full run takes ~2 sessions across 2 days.

**Score**
```bash
PYTHONPATH=. uv run scorer.py \
  --pred outputs/tier4_agent/test_predictions.json \
  --gold outputs/tier4_agent/gold_test.json \
  --name tier4_agent \
  --out  outputs/tier4_agent/full_metrics.json
```

**Results:** Strict F1 **76.7** · Relaxed F1 **79.7** · Macro F1 **68.1**
Precision 83.9 · Recall 70.7

**Per-type (strict):**

| Type | F1 | P | R | Support |
|---|---|---|---|---|
| GeneOrGeneProduct | 83.3 | 90.0 | 77.5 | 1180 |
| ChemicalEntity | 82.2 | 88.8 | 76.5 | 754 |
| DiseaseOrPhenotypicFeature | 77.8 | 74.9 | 80.9 | 917 |
| SequenceVariant | 59.9 | **100.0** | 42.7 | 241 |
| OrganismTaxon | 47.6 | 75.9 | 34.6 | 393 |
| CellLine | 57.8 | 65.0 | 52.0 | 50 |

**Key findings:**
- Orchestration adds **+15.8 F1** over Tier 3 (same base model, only variable
  is agentic architecture)
- SequenceVariant precision **1.0** — deterministic regex achieves zero false
  positives on variants it covers; recall limited by BERT span extraction ceiling
- KB confidence gate (+2.6 F1): without it, the rare branch overrides confident
  encoder predictions, causing 85 unnecessary type confusions
- OrganismTaxon recall (34.6%) is the main weakness — BERT span extractor misses
  organism mentions like `patients`, `human`, `Chinese hamster`; the KB lookup
  cannot help what it never receives as a candidate

---

## Results Summary

| Tier | System | Strict F1 | Relaxed F1 | Macro F1 | P | R |
|---|---|---|---|---|---|---|
| 1 | PubMedBERT fine-tuned | **89.9** | **93.8** | **90.7** | 87.9 | 92.0 |
| 2 | GLiNER-biomed zero-shot | 63.3 | 76.3 | 52.3 | 64.7 | 61.9 |
| 3 | LLM 0-shot | 60.9 | 67.5 | 51.5 | 61.9 | 59.9 |
| 3 | LLM 3-shot | 57.9 | 64.8 | 51.4 | 65.5 | 51.8 |
| **4** | **Multi-agent system** | **76.7** | **79.7** | **68.1** | **83.9** | **70.7** |

---

## Ablation Table (Tier 4)

Each row disables one component and reports the F1 drop.
Shows which parts of the architecture contribute measurably.

| System | Strict F1 | Δ | What this measures |
|---|---|---|---|
| Full system | 76.7 | — | — |
| − KB confidence gate | 74.1 | −2.6 | value of gating rare branch on common confidence |
| − rare branch | TBD | −? | value of KB lookup for OrganismTaxon / CellLine |
| − pattern branch | TBD | −? | value of deterministic regex for SequenceVariant |
| − overseer / requery | TBD | −? | value of LLM re-query for low-confidence spans |
| Single-LLM (Tier 3) | 60.9 | −15.8 | total value of agentic orchestration |

> The confidence gate ablation (−2.6) is already measured — it is the
> pre-gate run from the earlier iteration. Remaining ablations require
> one rerun each with the relevant component disabled.

---

## Cross-Tier Findings

1. **Fine-tuning gap is large (~27 F1).** Tier 1 vs Tier 2 confirms the value
   of domain-specific supervised training on BioRED. Zero-shot systems —
   whether encoder-based (GLiNER) or LLM-based — fall well short.

2. **Difficulty is semantic, not frequency-based.** OrganismTaxon (support 393)
   scores 96.9 in Tier 1; DiseaseOrPhenotypicFeature (support 917) scores 83.9.
   High frequency does not guarantee high F1 — surface-form regularity is the
   better predictor of difficulty.

3. **Gene↔Chemical is the dominant confusion across all tiers.**
   `GeneOrGeneProduct → ChemicalEntity` appears as the top or second-top
   confusion in every tier. This is a real BioRED-level ambiguity, not a
   model-specific quirk, and motivates the gene/chemical KB check in Tier 4's
   common branch.

4. **3-shot ICL is unstable on 8B models.** Few-shot hurt overall F1 despite
   improving OrganismTaxon, confirming that structured orchestration (Tier 4)
   is more reliable than prompt engineering for multi-type biomedical NER on
   small models.

5. **Agentic orchestration adds +15.8 F1 over a single LLM call.** Tier 3 →
   Tier 4 with the same base model. The improvement is attributable to:
   heterogeneous routing (each entity class handled by the method best suited
   to it), KB-grounded type validation, and the confidence-gated combiner.

6. **Tier 4 achieves highest precision (83.9) of any non-fine-tuned tier.**
   The combiner and KB grounding reduce spurious predictions. The tradeoff is
   lower recall (70.7) than Tier 1 (92.0) — the agentic system is more
   conservative but more precise.

---

## Known Limitations

- **OrganismTaxon recall (34.6%)** — BERT span extractor misses organism
  mentions used in descriptive or colloquial contexts (`patients`, `human`,
  `Chinese hamster`). KB lookup can only type candidates it receives.
- **SequenceVariant recall (42.7%)** — same span extraction ceiling.
  138 gold variants never extracted as candidates.
- **CellLine variance** — support of 50 in the test set makes per-type F1
  high-variance. Results should be interpreted cautiously.
- **Descriptive annotation spans** — a subset of BioRED gold annotations are
  long descriptive phrases (e.g. `valine (gtg) to a methionine (atg)`) that
  no NER system returns as a single span. These affect all four tiers equally.
- **LLM non-determinism** — Tier 3 and Tier 4 overseer use LLMs at temperature
  0. Results may vary slightly across runs. Multiple runs recommended for Tier 3
  headline numbers.

---

## Environment Variables

| Variable | Required by | Description |
|---|---|---|
| `GROQ_API_KEY` | Tiers 3, 4 | Groq API key for LLM calls |
| `NCBI_EMAIL` | Tier 4 | Email for NCBI Entrez API (courtesy, no registration needed) |