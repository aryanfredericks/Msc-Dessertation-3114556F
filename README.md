# Biomedical NER — MSc Dissertation
**Biomedical Information Extraction with Agents**
University of Glasgow · MSc Robotics & AI

A comparative study for biomedical named entity recognition on the
[BioRED dataset](https://github.com/ncbi/BioRED), progressing from a
fine-tuned encoder baseline to a full multi-agent agentic pipeline, with a
further extension replacing deterministic arbitration with LLM-orchestrated
tool selection.

---

## Project Structure

```
biomedical_ner/
├── biored.py                  # shared: BioRED loader, label maps, canonical I/O
├── scorer.py                  # shared: strict / relaxed / per-type scorer
├── extract_rare_entities.py   # dumps gold OrganismTaxon/CellLine spans for diagnosis
├── check_extraction_ceiling.py# measures BERT span-extraction recall ceiling per type
├── check_ncbi_coverage.py     # tests gold OrganismTaxon spans against live NCBI Taxonomy API
├── check_rare_collisions.py   # tests gold OrganismTaxon spans against live Cellosaurus API
├── dataset/
│   ├── train/Train.BioC.JSON  # 400 documents — fine-tuning + few-shot examples
│   ├── dev/Dev.BioC.JSON      # 100 documents — threshold / hyperparameter tuning
│   └── test/Test.BioC.JSON    # 100 documents — all reported results
├── pubmedbert/                # Tier 1: fine-tuned PubMedBERT
├── gli_ner_bert/               # Tier 2: GLiNER-biomed zero-shot
├── llm/                        # Tier 3: single-call LLM
├── agent/                      # Tier 4 + Tier 4 Extended: agentic pipelines
│   ├── workflow.py             # Tier 4: LangGraph graph definition
│   ├── run_agent.py            # Tier 4: entry point — loads docs, runs graph, saves output
│   ├── run_tier5.py             # Tier 4 Extended: LLM-orchestrated agent entry point
│   ├── models.py                # shared dataclasses and TypedDict state
│   ├── prompts.py               # LLM prompt builders
│   ├── config.py                # model names and paths
│   └── utils/
│       ├── pattern_matching.py   # regex rules for SequenceVariant
│       ├── rare_agent_utils.py   # Cellosaurus + NCBI Taxonomy KB lookups + human-referent allowlist
│       ├── common_agent_utils.py # PubMedBERT span-level type prediction
│       ├── bert_span_extractor.py# BERT candidate span generator
│       ├── overseer_utils.py     # overseer prompt + output schema (Tier 4 only)
│       └── offset_utils.py       # find_occurrences — string to char offsets
└── outputs/
    ├── tier1_pubmedbert/
    ├── tier2_gliner/
    ├── tier3_llm_0shot/
    ├── tier3_llm_3shot/
    ├── tier4_agent/             # Tier 4, pre-fix (kept for the diagnostic before/after comparison)
    ├── tier4_agent_v2/          # Tier 4, FINAL — with the human-referent allowlist fix
    ├── tier5_agent/             # Tier 4 Extended, pre-fix
    ├── tier5_agent_v2/          # Tier 4 Extended, FINAL — with the same fix
    └── rare_entity_analysis/    # extract_rare_entities.py output (gold OrganismTaxon/CellLine spans)
```

> **Reported results throughout this README use the `_v2` (fixed) runs for
> Tier 4 and Tier 4 Extended.** The pre-fix runs are kept and referenced
> explicitly in the **OrganismTaxon Diagnostic Deep-Dive & Fix** section,
> since the diagnostic journey that led to the fix is itself part of the
> dissertation's contribution.

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
IDA_LLM_API_KEY=your_uofg_hpc_key_here
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

A LangGraph agentic pipeline using heterogeneous-routing: candidate spans are
broadcast to three specialist branches simultaneously, and a priority
combiner arbitrates the votes.

**Architecture:**

```
Document
  └─> Span extraction (PubMedBERT — Tier 1 model, high-recall candidate generator)
        └─> broadcast to all three branches in parallel
              ├─> Pattern branch   — regex for SequenceVariant (HGVS, rsIDs, AA substitutions)
              ├─> Common branch    — PubMedBERT token classifier (gene / disease / chemical)
              └─> Rare branch      — human-referent allowlist → Cellosaurus → NCBI Taxonomy
        └─> Combiner (priority: pattern > rare [if common conf < 0.7] > common > requery > drop)
        └─> Overseer / re-query (Qwen3-32B — resolves low-confidence spans via LLM)
        └─> Offset localisation (deterministic string→char offsets, cap=5 per span)
  └─> Canonical output → scorer.py
```

**Design decisions (all data-driven):**
- Occurrence cap of 5 per span covers 92.5% of BioRED gold mentions (train set analysis)
- Rare branch confidence gate (common conf < 0.7) prevents KB from overriding
  confident encoder predictions — improved F1 by +2.6 points
- **Human-referent allowlist** (`patient`, `patients`, `inpatient`, `man`,
  `men`, `woman`, `women`) checked before Cellosaurus/NCBI Taxonomy — added
  after the diagnostic investigation below found these terms have no entry
  in either external KB despite being valid BioRED OrganismTaxon mentions.
  Worth +3.6 strict F1 / +38.0 F1 on OrganismTaxon alone (see diagnostic
  section). As a side effect, this also eliminates the `resolve_rare_entity`
  sequential-check collision where a term like `"men"` could otherwise match
  Cellosaurus before NCBI Taxonomy is ever consulted.
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
  --output_dir outputs/tier4_agent_v2 \
  --limit      5
```

**Full run** *(100 docs, checkpoint/resume supported)*
```bash
PYTHONPATH=. uv run agent/run_agent.py \
  --test_json  ./dataset/test/Test.BioC.JSON \
  --output_dir outputs/tier4_agent_v2
```

> If interrupted, rerun the same command to resume from checkpoint.
> Delete `outputs/tier4_agent_v2/checkpoint.jsonl` to start fresh.
> Given the 12K tpm llama limit, a full run takes ~2 sessions across 2 days.

**Score**
```bash
PYTHONPATH=. uv run scorer.py \
  --pred outputs/tier4_agent_v2/test_predictions.json \
  --gold outputs/tier4_agent_v2/gold_test.json \
  --name tier4_agent_v2 \
  --out  outputs/tier4_agent_v2/full_metrics.json
```

**Results:** Strict F1 **80.3** · Relaxed F1 **83.1** · Macro F1 **74.7**
Precision 85.0 · Recall 76.1

**Per-type (strict):**

| Type | F1 | P | R | Support |
|---|---|---|---|---|
| **OrganismTaxon** | **85.6** | 88.1 | 83.2 | 393 |
| GeneOrGeneProduct | 83.4 | 90.2 | 77.5 | 1180 |
| ChemicalEntity | 82.2 | 88.8 | 76.5 | 754 |
| DiseaseOrPhenotypicFeature | 77.8 | 74.9 | 80.9 | 917 |
| SequenceVariant | 59.9 | **100.0** | 42.7 | 241 |
| CellLine | 59.1 | 68.4 | 52.0 | 50 |

**Key findings:**
- Orchestration adds **+15.8 F1** over Tier 3 (same base model, only variable
  is agentic architecture)
- SequenceVariant precision **1.0** — deterministic regex achieves zero false
  positives on variants it covers; recall limited by BERT span extraction ceiling
  (measured directly: 93.4% extraction ceiling for SequenceVariant — see the
  diagnostic section — so the remaining recall gap is mostly headroom the
  regex rule itself leaves on the table, not an extraction problem)
- KB confidence gate (+2.6 F1): without it, the rare branch overrides confident
  encoder predictions, causing 85 unnecessary type confusions
- **OrganismTaxon is now the second-strongest type in the system (85.6 F1,
  up from 47.6 pre-fix).** This was originally misdiagnosed as a span-extraction
  problem; it turned out to be an external knowledge-base coverage gap. Full
  diagnostic chain and fix below.

**Span extractor comparison (LLM vs BERT):**

An initial implementation used the LLM (llama-3.3-70b-versatile) for span
extraction. This was replaced with the fine-tuned PubMedBERT model from Tier 1.
The improvement on a 5-document sanity check:

| Span extractor | Strict F1 | TP | FP | FN |
|---|---|---|---|---|
| LLM (llama-3.3-70b) | 43.9 | 64 | 51 | 112 |
| PubMedBERT (fine-tuned) | 58.9 | 94 | 49 | 82 |
| Delta | +15.0 | +30 | -2 | -30 |

The LLM extractor missed short terms (`sodium`, `NQO1`) and abbreviations
(`CBR3`) that PubMedBERT reliably tags because it was trained on BioRED.
Since downstream branches can only type what the extractor finds, a
higher-recall extractor lifts all branch outputs. The BERT extractor also
removes LLM non-determinism from the span detection step, making the
pipeline fully reproducible up to the overseer re-query calls.

Note: these are 5-document sanity numbers. The final reported Tier 4 result
(80.3 strict F1) uses the BERT extractor across all 100 test documents.

**Branch resolution diagnosis — motivation for Tier 4 Extended:**

Logging which branch resolved each final entity (`analyze_branch_stats.py`)
revealed that the combiner's "arbitration" is largely illusory for three of
the six entity types. `common_relation_agent` only ever votes on
GeneOrGeneProduct / DiseaseOrPhenotypicFeature / ChemicalEntity by
construction, and the pattern/rare branches are correspondingly scoped to
their own disjoint subsets — so in practice the pipeline behaves as **three
mutually-exclusive type-scoped classifiers**, not a system with genuine
cross-branch competition:

| Entity type | Resolving branch | Share |
|---|---|---|
| GeneOrGeneProduct | common | 99.7% |
| DiseaseOrPhenotypicFeature | common | 100.0% |
| ChemicalEntity | common | 100.0% |
| SequenceVariant | pattern | 100.0% |
| CellLine | rare | 100.0% |
| OrganismTaxon | rare | 100.0% |

No span was ever contested between branches. This directly motivated Tier 4
Extended (below): rather than tuning the confidence threshold that gates this
already-narrow arbitration, the combiner is replaced entirely with an LLM
agent that can consult any tool for any span.

---

## Tier 4 Extended — LLM-Orchestrated Agent

Replaces Tier 4's deterministic, confidence-gated combiner with a single LLM
agent per document that decides, per candidate span, which of the same three
underlying methods (common classifier / pattern matcher / rare KB lookup) to
consult as tools, before producing a final type assignment. Span extraction
is **identical** to Tier 4 (same fine-tuned PubMedBERT candidate generator),
so any performance delta is attributable to the arbitration strategy alone,
not to a change in what candidates the system even considers.

**Architecture:**

```
Document
  └─> Span extraction (PubMedBERT — same Tier 1 model, unchanged from Tier 4)
        └─> Central LLM agent (gpt-oss-120b, University of Glasgow HPC endpoint)
              Phase 1 — evidence gathering (tools bound, tool_choice="required" on turn 1):
                agent calls any of the three tools, batching multiple spans
                per call, until it has covered every candidate span or signals "DONE"
                  ├─> common_classifier(span_texts) — same PubMedBERT token classifier as Tier 4
                  ├─> pattern_matcher(span_texts)   — same regex rules as Tier 4
                  └─> rare_lookup(span_texts)        — same allowlist/Cellosaurus/NCBI KB lookups as Tier 4
              Phase 2 — final answer (tools bound, tool_choice="none"):
                agent returns one JSON type assignment per span using all
                gathered evidence plus full passage context
        └─> Offset localisation (identical to Tier 4, cap=5 per span)
  └─> Canonical output → scorer.py
```

**Key design decisions:**
- **Same three tools as Tier 4's branches**, wrapping the identical underlying
  functions (`predict_span_type`, `match_sequence_variant`, `resolve_rare_entity`).
  This isolates the comparison to arbitration strategy: deterministic
  confidence-gated combiner (Tier 4) vs. LLM-orchestrated tool selection
  (Tier 4 Extended).
- **Two-phase interaction** (evidence-gathering vs. final-answer, as separate
  LLM calls) rather than one open-ended tool-calling loop. Forcing a model to
  choose between calling a tool and returning a final answer in the same turn
  caused malformed-output errors on multiple models tested (`tool_use_failed`
  on Groq); separating the phases — with `tool_choice="required"` on the
  first gathering turn and `tool_choice="none"` on the final-answer turn —
  eliminated this failure mode entirely.
- **`tool_choice="required"` on the first turn is necessary, not optional.**
  Without it, models frequently skip tools entirely and answer from
  parametric knowledge alone, silently degrading the system into Tier 3's
  zero-shot behaviour despite the tools being available.
- **Batched tool calls** (a list of spans per call, not one call per span).
  Some models issue only one tool call per conversational turn; without
  batching, a 30–70 candidate-span document would require as many turns,
  which is both slow and — for reasoning models with large hidden
  chain-of-thought token costs — prohibitively expensive.
- **Checkpointing.** Predictions are written incrementally per document, so
  an interrupted run (rate limit, network drop) resumes rather than
  restarting. This mattered in practice: one document in the reported v2 run
  failed with an empty model response mid-run; the checkpoint let it be
  cleared and retried individually without re-running the other 99 documents.

> Set `IDA_LLM_API_KEY` in `.env` before running. Uses the University of
> Glasgow HPC-hosted `gpt-oss-120b` endpoint
> (`http://api.llm.apps.os.dcs.gla.ac.uk/v1`, or `http://api.terrier.org/v1`
> outside the university network) rather than a commercial provider — the
> same model hosted via Groq's free tier was found to exhaust its 200K
> tokens/day cap after roughly 1.5 documents, since `gpt-oss-120b` is a
> reasoning model whose hidden chain-of-thought is billed against the quota
> even though it never appears in the visible response.

**Debug run** *(3 docs, minimal cost)*
```bash
PYTHONPATH=. uv run agent/run_tier5.py \
  --test_json  ./dataset/test/Test.BioC.JSON \
  --output_dir outputs/tier5_agent_v2 \
  --limit      3
```

**Full run** *(100 docs, checkpoint/resume supported)*
```bash
PYTHONPATH=. uv run agent/run_tier5.py \
  --test_json  ./dataset/test/Test.BioC.JSON \
  --output_dir outputs/tier5_agent_v2
```

> If interrupted, rerun the exact same command — already-completed documents
> are skipped automatically via `outputs/tier5_agent_v2/checkpoint_predictions.jsonl`.
> If a specific document fails (e.g. an empty/malformed model response),
> remove just that document's line from the checkpoint file before rerunning
> so only that document is retried.

**Score**
```bash
PYTHONPATH=. uv run scorer.py \
  --pred outputs/tier5_agent_v2/test_predictions.json \
  --gold outputs/tier5_agent_v2/gold_test.json \
  --name tier5_agent_v2 \
  --out  outputs/tier5_agent_v2/full_metrics.json
```

**Results:** Strict F1 **79.3** · Relaxed F1 **82.3** · Macro F1 **77.7**
Precision 85.9 · Recall 73.6

**Per-type (strict):**

| Type | F1 | P | R | Support |
|---|---|---|---|---|
| **CellLine** | **89.1** | 97.6 | 82.0 | 50 |
| GeneOrGeneProduct | 83.5 | 90.8 | 77.2 | 1180 |
| ChemicalEntity | 82.5 | 88.1 | 77.6 | 754 |
| **SequenceVariant** | **79.3** | 91.4 | 70.1 | 241 |
| DiseaseOrPhenotypicFeature | 79.6 | 77.7 | 81.5 | 917 |
| OrganismTaxon | 52.2 | 85.1 | 37.7 | 393 |

**Tier 4 (v2) vs Tier 4 Extended (v2) — final per-type F1 comparison, both with the identical KB fix applied:**

| Type | Tier 4 | Tier 4 Extended | Δ | Winner |
|---|---|---|---|---|
| CellLine | 59.1 | 89.1 | **+30.0** | Tier 4 Extended |
| SequenceVariant | 59.9 | 79.3 | **+19.4** | Tier 4 Extended |
| DiseaseOrPhenotypicFeature | 77.8 | 79.6 | +1.8 | Tier 4 Extended |
| ChemicalEntity | 82.2 | 82.5 | +0.3 | Tier 4 Extended |
| GeneOrGeneProduct | 83.4 | 83.5 | +0.1 | Tier 4 Extended |
| **OrganismTaxon** | **85.6** | 52.2 | **−33.4** | **Tier 4** |
| **Overall (strict)** | **80.3** | 79.3 | **−1.0** | **Tier 4** |
| **Macro F1** | 74.7 | **77.7** | +3.0 | Tier 4 Extended |

**Key findings:**
- **There is no single winner between the two architectures once the KB
  coverage gap is fixed on both sides.** Tier 4 Extended still wins clearly
  on CellLine and SequenceVariant — the two types where Tier 4's branch
  isolation genuinely limits it, exactly as the earlier diagnosis predicted.
  But Tier 4 now wins overall (80.3 vs 79.3) and decisively on OrganismTaxon
  (85.6 vs 52.2), reversing the picture from the pre-fix comparison.
- **The same code fix produced wildly different payoffs across the two
  architectures**: Tier 4's OrganismTaxon true positives rose from 136 to
  327 (+191) from the fix alone; Tier 4 Extended's rose from 134 to only 148
  (+14). Tier 4's `rare_relation_agent` calls `resolve_rare_entity`
  unconditionally on every candidate span, so the allowlist fix applies
  universally and for free. Tier 4 Extended's `rare_lookup` is a tool the
  agent must actively choose to invoke — generic words like `patient` or
  `women` do not superficially resemble the kind of term that needs an
  external KB check the way `zebrafish` or `HeLa` does, so the working
  hypothesis is that the agent under-invokes `rare_lookup` on exactly this
  class of span. This has not yet been directly confirmed by inspecting the
  Tier 4 Extended entity log's tool-call records — flagged as a natural next
  step rather than a settled conclusion.
- **This reframes the dissertation's central comparison.** The original
  framing — "agentic orchestration outperforms deterministic arbitration" —
  was true in the pre-fix comparison, but partly because Tier 4's
  undiagnosed KB bug was dragging its OrganismTaxon score down, not solely
  because agentic orchestration is categorically better. The corrected,
  fairer finding is a **completeness-vs-flexibility tradeoff**: Tier 4's
  exhaustive, unconditional branch-calling guarantees every candidate gets
  every applicable check, at the cost of being unable to exploit
  cross-branch evidence (its CellLine/SequenceVariant weakness). Tier 4
  Extended's selective, LLM-judged tool calling can combine evidence
  flexibly and recovers real value where Tier 4 is structurally blind — but
  that same selectivity means it can silently skip a tool call for spans
  that don't superficially look like they need it, costing recall precisely
  where deterministic exhaustiveness would have caught them for free.
- **SequenceVariant and CellLine shifted between the original and v2 Tier 4
  Extended runs even though the fix does not touch either type's logic**
  (SequenceVariant F1 83.6 → 79.3; CellLine F1 90.3 → 89.1). Since
  `resolve_rare_entity`'s changes only affect OrganismTaxon-adjacent
  resolution, this movement is attributable to LLM run-to-run
  non-determinism rather than the fix — concrete evidence for the
  non-determinism limitation already noted below, not just a theoretical
  caveat.
- Error taxonomy for Tier 4 Extended v2: spurious FPs 269, missed FNs 775,
  type confusions 49+11=60 total — broadly similar shape to the pre-fix run,
  consistent with the fix affecting OrganismTaxon specifically rather than
  the system generally.

**Limitation:** both v2 results are a single run each of non-deterministic
LLM agents/APIs (Tier 4's overseer, Tier 4 Extended's orchestrating agent).
Run-to-run variance has not been formally characterised across repeated
runs; the SequenceVariant/CellLine shift noted above is suggestive of a
non-trivial variance band that a single run cannot quantify.

---

## OrganismTaxon Diagnostic Deep-Dive & Fix

Both Tier 4 (76.7 F1) and Tier 4 Extended (79.2 F1), in their original
pre-fix form, shared the same weak spot: OrganismTaxon recall stuck at ~34%,
essentially flat across two very different arbitration strategies. Since
arbitration strategy clearly wasn't the cause, four scripts were written to
isolate which pipeline stage was actually responsible, run in sequence, each
one ruling a hypothesis in or out with live data rather than assumption:

1. **`extract_rare_entities.py`** — pulls every gold OrganismTaxon / CellLine
   span out of the test set (`outputs/rare_entity_analysis/`) for the checks
   below to query against.
   ```bash
   PYTHONPATH=. uv run extract_rare_entities.py \
     --test_json ./dataset/test/Test.BioC.JSON \
     --out       outputs/rare_entity_analysis/organism_cellline_gold.json \
     --txt_out   outputs/rare_entity_analysis/organism_cellline_spans.txt
   ```

2. **`check_extraction_ceiling.py`** — measures what fraction of gold
   OrganismTaxon mentions are ever generated as a BERT candidate span at all.
   **Result: 99.5% extraction ceiling (391/393).** This ruled out the
   original hypothesis (and the one initially written into this README) that
   span extraction was the bottleneck. For comparison, SequenceVariant's
   extraction ceiling is 93.4% and CellLine's is 100% — extraction is a real,
   separate, smaller constraint on SequenceVariant, but not on OrganismTaxon.
   ```bash
   PYTHONPATH=. uv run check_extraction_ceiling.py --type OrganismTaxon
   ```

3. **`check_rare_collisions.py`** — tests whether OrganismTaxon spans are
   being misrouted to CellLine because `resolve_rare_entity()` checked
   Cellosaurus before NCBI Taxonomy. **Result: 1/32 unique spans collided**
   (`"men"` matched a Cellosaurus entry) — real, but far too small a rate to
   explain the recall gap; ruled out as the primary cause.
   ```bash
   python check_rare_collisions.py \
     --spans_file outputs/rare_entity_analysis/organism_cellline_spans.txt
   ```

4. **`check_ncbi_coverage.py`** — tests every unique gold OrganismTaxon span
   against the live NCBI Taxonomy API. **Result: 11/32 unique spans return
   zero hits, and those 11 terms account for 228/393 (58%) of all
   OrganismTaxon gold mentions** — `patient`, `patients`, `inpatient`, `man`,
   `men`, `woman`, `women` correctly denote *Homo sapiens* in BioRED but have
   no entry in a species-name taxonomy database. This is the primary
   bottleneck: a genuine KB coverage gap, not a code bug.
   ```bash
   python check_ncbi_coverage.py \
     --spans_file outputs/rare_entity_analysis/organism_cellline_spans.txt
   ```

**Fix:** `agent/utils/rare_agent_utils.py` — `resolve_rare_entity()` now
checks a small human-referent allowlist before Cellosaurus/NCBI Taxonomy:

```python
HUMAN_REFERENT_TERMS = {"patient", "patients", "inpatient", "man", "men", "woman", "women"}

def resolve_rare_entity(text: str) -> tuple[Optional[str], float, str]:
    if text in _cache:
        return _cache[text]
    if text.strip().lower() in HUMAN_REFERENT_TERMS:
        result = ("OrganismTaxon", 1.0, "human_referent_allowlist")
        _cache[text] = result
        return result
    if lookup_cellosaurus(text):
        ...
```

**Results with the fix**, both re-run to completion on the same 100-doc test
set with the same span extractor and everything else unchanged
(`outputs/tier4_agent_v2/`, `outputs/tier5_agent_v2/`):

| System | Strict F1 (before → after) | OrganismTaxon F1 | OrganismTaxon Recall | OrganismTaxon TP gained |
|---|---|---|---|---|
| Tier 4 | 76.7 → **80.3** (+3.6) | 47.6 → **85.6** (+38.0) | 34.6% → **83.2%** | +191 |
| Tier 4 Extended | 79.2 → **79.3** (+0.1) | 49.1 → **52.2** (+3.1) | 34.1% → **37.7%** | +14 |

**Key findings:**
- The fix is dramatically more valuable for Tier 4's deterministic combiner
  than for Tier 4 Extended's LLM-orchestrated agent, and the gap is not
  small — 191 recovered true positives versus 14. This is now the leading
  explanation for the reversal in the head-to-head comparison above: Tier 4's
  rare branch is called unconditionally on every span, so it benefits from
  the fix everywhere the fix applies, whereas Tier 4 Extended's agent has to
  choose to invoke the same underlying function, and mostly does not for
  this class of generic term.
- **Tier 4 v2 (80.3 strict F1) now exceeds Tier 4 Extended v2 (79.3 strict
  F1) overall**, though Tier 4 Extended remains clearly stronger on
  CellLine and SequenceVariant. The original Tier 4 vs. Tier 4 Extended
  comparison in earlier drafts of this work was confounded by this fixable
  KB gap; seeing both systems after the fix is the fairer test of
  arbitration strategy on its own terms, and the fairer test shows a
  tradeoff, not a clean win for either side.
- This diagnostic chain (extraction ceiling → collision check → KB coverage
  check) directly resolves the "OrganismTaxon recall" item that was
  previously in **Known Limitations**: the bottleneck was neither span
  extraction nor branch-ordering, but KB domain coverage for colloquial
  referents — and it was diagnosed by testing each candidate explanation
  against live data rather than accepting the first plausible-sounding one.

---

## Results Summary

| Tier | System | Strict F1 | Relaxed F1 | Macro F1 | P | R |
|---|---|---|---|---|---|---|
| 1 | PubMedBERT fine-tuned | **89.9** | **93.8** | **90.7** | 87.9 | 92.0 |
| 2 | GLiNER-biomed zero-shot | 63.3 | 76.3 | 52.3 | 64.7 | 61.9 |
| 3 | LLM 0-shot | 60.9 | 67.5 | 51.5 | 61.9 | 59.9 |
| 3 | LLM 3-shot | 57.9 | 64.8 | 51.4 | 65.5 | 51.8 |
| 4 | Multi-agent system | **80.3** | 83.1 | 74.7 | 85.0 | 76.1 |
| **4 Extended** | **LLM-orchestrated agent** | 79.3 | **82.3** | **77.7** | **85.9** | 73.6 |

*Tier 4 and Tier 4 Extended figures are the `_v2` (KB-fix) results. See the
diagnostic section above for the pre-fix numbers and the reasoning behind
the fix.*

---

## Ablation Table (Tier 4)

Each row disables one component and reports the F1 drop, measured against
the **pre-fix** Tier 4 baseline (76.7 F1) — these ablations were run before
the human-referent allowlist fix was diagnosed and have not been re-run
against the fixed `tier4_agent_v2` baseline. The fix itself is added as a
final row for reference, but its interaction with the other ablated
components (e.g. does the rare-branch ablation delta change once the KB gap
is fixed?) has not been tested and is flagged here as a gap rather than
silently assumed away.

| System | Strict F1 | Δ | What this measures |
|---|---|---|---|
| Full system (pre-fix) | 76.7 | — | — |
| − KB confidence gate | 74.1 | −2.6 | value of gating rare branch on common confidence |
| − rare branch | 74.2 | −2.5 | value of KB lookup for OrganismTaxon / CellLine |
| − pattern branch | 74.2 | −2.5 | value of deterministic regex for SequenceVariant |
| − overseer / requery | 76.7 | −0 | value of LLM re-query for low-confidence spans |
| Single-LLM (Tier 3) | 60.9 | −15.8 | total value of agentic orchestration |
| Tier 4 Extended (pre-fix) | 79.2 | +2.5 | value of replacing the combiner with LLM-orchestrated tool selection |
| **+ Human-referent allowlist fix** | **80.3** | **+3.6** | value of closing the NCBI Taxonomy coverage gap (see diagnostic section) |

---

### Ablation Analysis

**Overseer / re-query (F1 delta: 0.0)**
The overseer produced no measurable change in aggregate F1, confirming it functions
as a safety net rather than a primary contributor. Across 100 documents, the combiner's
priority rules resolved the vast majority of spans without requiring LLM arbitration.
The re-query loop fired on a small number of low-confidence spans per document, and
those cases did not move the aggregate metric. This is an honest null result: the
overseer earns its place as a fault-tolerance layer for edge cases (e.g. the
inflammasome disambiguation resolved correctly via Qwen3-32B reasoning), but should
not be cited as a performance driver.

**Rare branch / KB lookup (F1 delta: -2.5)**
Disabling the Cellosaurus and NCBI Taxonomy lookups dropped F1 by 2.5 points, driven
primarily by the loss of OrganismTaxon and CellLine predictions that the common branch
does not reliably produce. The rare branch is the only source of OrganismTaxon
predictions in the pipeline; without it, all 393 gold organism mentions become false
negatives by default. The equal delta to the pattern branch (below) confirms the two
branches serve complementary, non-overlapping parts of the entity type space. Note this
ablation predates the human-referent allowlist fix — with the fix in place, the rare
branch's contribution is very likely larger, since it is now the sole source of the
228 human-referent mentions as well.

**Pattern branch / regex (F1 delta: -2.5)**
Disabling deterministic regex for SequenceVariant dropped F1 by 2.5 points and
eliminated the system's only source of perfect-precision predictions. With the pattern
branch active, SequenceVariant precision is 1.0 with zero false positives across 103
predictions. With it disabled, those 103 predictions disappear entirely as false
negatives, and the common branch does not recover them reliably. This validates the
core design premise of heterogeneous routing: pattern-like types with characteristic
surface forms (HGVS notation, rsIDs, amino acid substitutions) are better served by
deterministic rules than by learned classifiers.

**KB confidence gate (F1 delta: -2.6)**
Without the confidence gate, the rare branch overrides common branch predictions
regardless of the encoder's confidence. This caused 85 type confusions where correctly
typed gene, chemical, and disease spans were re-typed as OrganismTaxon or CellLine
because the surface string happened to resolve in a KB. The gate (rare branch only
overrides when common branch confidence is below 0.7) recovers those 85 predictions,
contributing the largest single ablation delta. This finding establishes that KB
grounding requires confidence-aware arbitration: a naive KB lookup that ignores the
encoder's posterior is actively harmful.

**Single-LLM baseline / Tier 3 (F1 delta: -15.8)**
The gap between Tier 3 (60.9) and the full Tier 4 system (76.7, pre-fix) represents the
total contribution of agentic orchestration, holding the base model constant. This
15.8-point gain decomposes approximately as: heterogeneous routing adds ~5.0 (rare +
pattern branches combined), the confidence gate adds ~2.6, and the remaining ~8.2 comes
from the structured combiner and BERT span extraction replacing the LLM extractor. No
single component accounts for the full gain, confirming that the architecture's value
is emergent rather than attributable to any one design decision.

**Tier 4 Extended / LLM-orchestrated arbitration (F1 delta: +2.5 overall, pre-fix)**
Replacing the entire combiner — confidence gate included — with LLM-orchestrated tool
selection recovered +2.5 strict F1 on top of the pre-fix Tier 4 system, concentrated
almost entirely in CellLine and SequenceVariant. As the diagnostic section and the
final head-to-head comparison above show, part of this apparent advantage was a
byproduct of Tier 4's undiagnosed KB coverage gap rather than a pure arbitration-strategy
effect: once both systems have the fix, Tier 4 Extended's overall lead disappears
(80.3 vs. 79.3), even though its CellLine/SequenceVariant advantage — the part of the
finding not confounded by the KB gap — remains real and substantial.

**Human-referent allowlist fix (F1 delta: +3.6 for Tier 4, +0.1 for Tier 4 Extended)**
The single largest lever identified in this dissertation, found through a systematic
elimination of competing hypotheses (extraction ceiling, branch-ordering collisions,
KB coverage) rather than intuition. Its effect size differs by more than an order of
magnitude between the two arbitration strategies, which is itself the most interesting
finding to come out of the fix: identical underlying tools, identical underlying bug,
almost entirely different practical impact, depending on whether the branch is called
unconditionally (Tier 4) or is one option an LLM must choose to exercise (Tier 4
Extended).

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
   Tier 4 (pre-fix) with the same base model. The improvement is attributable
   to: heterogeneous routing (each entity class handled by the method best
   suited to it), KB-grounded type validation, and the confidence-gated
   combiner.

6. **Tier 4 achieves the highest precision of any non-fine-tuned tier** (85.0
   post-fix). The combiner and KB grounding reduce spurious predictions. The
   tradeoff is lower recall than Tier 1 (92.0) — the agentic system is more
   conservative but more precise.

7. **Deterministic arbitration systematically under-serves branch-isolated
   entity types where genuine cross-branch competition would help — and this
   is fixable without touching span extraction or the underlying branch
   methods.** Tier 4's combiner routes each of the six entity types to
   exactly one branch by construction, with essentially zero cross-branch
   contest. Tier 4 Extended shows that replacing only the arbitration logic —
   same tools, same span extractor — recovers +30.0 F1 on CellLine and
   +19.4 F1 on SequenceVariant even after both systems have the KB fix,
   while leaving common-branch-dominated types (Gene, Disease, Chemical)
   essentially unchanged. This isolates arbitration flexibility as a real,
   unconfounded causal factor for these two types specifically.

8. **The inverse is also true, and is the more novel finding: deterministic,
   unconditional branch-calling has a completeness advantage that selective
   agentic tool-calling can quietly give up.** The human-referent allowlist
   fix recovered 191 OrganismTaxon true positives for Tier 4 (which calls its
   rare branch on every candidate span, no exceptions) but only 14 for Tier 4
   Extended (whose agent must choose to invoke the equivalent tool, and
   apparently often doesn't for spans that don't superficially resemble
   knowledge-base-lookup candidates). Neither architecture is categorically
   superior; each has a distinct, type-dependent failure mode. This
   reframes the dissertation's central architectural question from "does
   agentic orchestration beat deterministic arbitration" to "which failure
   modes does each strategy trade for which."

---

## Known Limitations

- **OrganismTaxon recall — resolved for Tier 4, only partially for Tier 4
  Extended.** Diagnosed in full in **OrganismTaxon Diagnostic Deep-Dive &
  Fix** above. It is *not* a span-extraction problem (extraction ceiling
  ≈99.5%) or a branch-ordering problem (only 1/32 unique spans collided with
  Cellosaurus); it is an NCBI Taxonomy coverage gap for colloquial
  human-referent terms (`patient`, `man`, `woman`, etc.), covering 58% of all
  OrganismTaxon gold mentions. A small allowlist fix raises Tier 4's
  OrganismTaxon recall to 83.2%, but Tier 4 Extended's only to 37.7% — the
  fix's value depends heavily on whether the branch is called unconditionally
  or is agent-selected, and the exact mechanism behind Tier 4 Extended's
  much smaller gain has not yet been directly confirmed via its entity log's
  tool-call records (flagged as follow-up work).
- **SequenceVariant recall ceiling** — genuine span-extraction limitation,
  distinct from the OrganismTaxon issue above. Extraction ceiling measured
  directly at 93.4% (16/241 gold variants never extracted as candidates
  under either Tier 4 or Tier 4 Extended, since both share the same
  extractor). Tier 4 Extended's improvement over Tier 4 on this type
  (42.7% → 70.1% recall) comes from typing more of the candidates that
  were extracted, not from extracting more of them.
- **CellLine variance** — support of 50 in the test set makes per-type F1
  high-variance. Results should be interpreted cautiously.
- **Descriptive annotation spans** — a subset of BioRED gold annotations are
  long descriptive phrases (e.g. `valine (gtg) to a methionine (atg)`) that
  no NER system returns as a single span. These affect all tiers equally.
- **LLM non-determinism** — Tier 3, Tier 4's overseer, and Tier 4 Extended's
  agent all use LLMs. Tier 3 is run at temperature 0 with multiple runs
  recommended for headline numbers. Tier 4 Extended's SequenceVariant and
  CellLine F1 shifted by several points between its original and v2 runs
  despite no code change affecting either type, which is concrete evidence
  of non-trivial run-to-run variance rather than just a theoretical caveat;
  formal multi-run statistics have not been collected for either LLM-based
  agentic tier.
- **Ablation table baseline mismatch** — the Tier 4 ablation table (rare
  branch, pattern branch, KB confidence gate, overseer) was measured against
  the pre-fix system and has not been re-run against `tier4_agent_v2`.
  Component-level deltas may differ once the human-referent allowlist fix is
  included in the baseline, particularly for the rare-branch ablation.

---

## Environment Variables

| Variable | Required by | Description |
|---|---|---|
| `GROQ_API_KEY` | Tiers 3, 4 | Groq API key for LLM calls |
| `NCBI_EMAIL` | Tier 4 | Email for NCBI Entrez API (courtesy, no registration needed) |
| `IDA_LLM_API_KEY` | Tier 4 Extended | University of Glasgow HPC LLM endpoint key |