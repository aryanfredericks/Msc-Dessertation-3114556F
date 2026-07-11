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
│       ├── rare_agent_utils.py   # Cellosaurus + NCBI Taxonomy KB lookups
│       ├── common_agent_utils.py # PubMedBERT span-level type prediction
│       ├── bert_span_extractor.py# BERT candidate span generator
│       ├── overseer_utils.py     # overseer prompt + output schema (Tier 4 only)
│       └── offset_utils.py       # find_occurrences — string to char offsets
└── outputs/
    ├── tier1_pubmedbert/
    ├── tier2_gliner/
    ├── tier3_llm_0shot/
    ├── tier3_llm_3shot/
    ├── tier4_agent/
    └── tier5_agent/            # Tier 4 Extended output directory
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

**Span extractor comparison (LLM vs BERT):**

An initial implementation used the LLM (llama-3.3-70b-versatile) for span
extraction. This was replaced with the fine-tuned PubMedBERT model from Tier 1.
The improvement on a 5-document sanity check:

| Span extractor | Strict F1 | TP | FP | FN |
|---|---|---|---|---|
| LLM (llama-3.3-70b) | 43.9 | 64 | 51 | 112 |
| PubMedBERT (fine-tuned) | 58.9 | 94 | 49 | 82 |
| Delta | +15.0 | +30 | -2 | -30 |

The LLM extractor missed short terms (`sodium`, `NQO1`), abbreviations (`CBR3`),
and organism mentions in colloquial contexts (`patients`, `human`) that PubMedBERT
reliably tags because it was trained on BioRED. Since downstream branches can only
type what the extractor finds, a higher-recall extractor lifts all branch outputs.
The BERT extractor also removes LLM non-determinism from the span detection step,
making the pipeline fully reproducible up to the overseer re-query calls.

Note: these are 5-document sanity numbers. The final reported Tier 4 result (76.7
strict F1) uses the BERT extractor across all 100 test documents.

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
                  └─> rare_lookup(span_texts)        — same Cellosaurus/NCBI KB lookups as Tier 4
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
  an interrupted run (rate limit, network drop) resumes rather than restarting.

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
  --output_dir outputs/tier5_agent \
  --limit      3
```

**Full run** *(100 docs, checkpoint/resume supported)*
```bash
PYTHONPATH=. uv run agent/run_tier5.py \
  --test_json  ./dataset/test/Test.BioC.JSON \
  --output_dir outputs/tier5_agent
```

> If interrupted, rerun the exact same command — already-completed documents
> are skipped automatically via `outputs/tier5_agent/checkpoint_predictions.jsonl`.

**Score**
```bash
PYTHONPATH=. uv run scorer.py \
  --pred outputs/tier5_agent/test_predictions.json \
  --gold outputs/tier5_agent/gold_test.json \
  --name tier4_extended \
  --out  outputs/tier5_agent/full_metrics.json
```

**Results:** Strict F1 **79.2** · Relaxed F1 **82.3** · Macro F1 **78.0**
Precision 86.3 · Recall 73.2

**Per-type (strict):**

| Type | F1 | P | R | Support |
|---|---|---|---|---|
| CellLine | **90.3** | 97.7 | 84.0 | 50 |
| GeneOrGeneProduct | 83.7 | 91.0 | 77.5 | 1180 |
| ChemicalEntity | 82.4 | 88.8 | 76.8 | 754 |
| SequenceVariant | **83.6** | 94.3 | 75.1 | 241 |
| DiseaseOrPhenotypicFeature | 78.8 | 77.3 | 80.5 | 917 |
| OrganismTaxon | 49.1 | 87.6 | 34.1 | 393 |

**Tier 4 vs Tier 4 Extended — per-type F1 delta:**

| Type | Tier 4 | Tier 4 Extended | Δ |
|---|---|---|---|
| CellLine | 57.8 | 90.3 | **+32.5** |
| SequenceVariant | 59.9 | 83.6 | **+23.7** |
| OrganismTaxon | 47.6 | 49.1 | +1.5 |
| DiseaseOrPhenotypicFeature | 77.8 | 78.8 | +1.0 |
| GeneOrGeneProduct | 83.3 | 83.7 | +0.4 |
| ChemicalEntity | 82.2 | 82.4 | +0.2 |
| **Overall (strict)** | **76.7** | **79.2** | **+2.5** |
| **Macro F1** | **68.1** | **78.0** | **+9.9** |

**Key findings:**
- The improvement is **concentrated, not uniform**, and lands exactly where
  the branch-resolution diagnosis (above) predicted it would: CellLine and
  SequenceVariant were the two types Tier 4 resolved via a single isolated
  branch with no possibility of cross-checking. Giving the agent discretion
  to consult multiple tools per span produces large gains specifically there
  (+32.5 and +23.7 F1), while types where Tier 4's common branch already had
  unimpeded access (Gene, Disease, Chemical) move by at most ±1 F1 — there
  was no arbitration bottleneck left to fix on those types.
- **SequenceVariant recall nearly doubles** (42.7% → 75.1%) at a small
  precision cost (100.0% → 94.3%). Tier 4's regex-only pattern branch was
  maximally conservative (perfect precision, but missed any variant mention
  it didn't structurally match); the agent uses the same regex tool as one
  input among several rather than a hard gate, recovering many of the
  variant mentions the rigid rule alone would drop.
- **CellLine improves on both precision and recall simultaneously**
  (65.0%/52.0% → 97.7%/84.0%) — not a tradeoff, a strict win, consistent with
  the agent combining KB lookup with passage context rather than trusting
  the KB result in isolation.
- **OrganismTaxon barely moves** (47.6 → 49.1 F1), and recall in particular
  is essentially flat (34.6% → 34.1%). Since span extraction is unchanged
  from Tier 4, this is expected: OrganismTaxon's bottleneck is at the
  extraction stage (organism mentions like `patients`, `human` are never
  generated as candidate spans in the first place), which is upstream of
  arbitration and therefore not something a smarter combiner can fix. This
  cleanly separates arbitration-fixable errors from extraction-fixable ones.
- **Error taxonomy improves on every axis simultaneously**: spurious false
  positives fall (302 → 259), missed false negatives fall (858 → 793), and
  type-confusion errors nearly halve (73 → 41 total instances).
- This result directly answers the architectural critique that motivated
  this tier: Tier 4's combiner gives disproportionate and largely
  uncontested weight to the common branch for the majority of entity types.
  Replacing fixed-threshold arbitration with agentic tool selection recovers
  a substantial share of the performance lost to that structural rigidity,
  concentrated exactly on the entity types it was diagnosed to affect.

**Limitation:** this is a single run of a non-deterministic LLM agent.
Run-to-run variance has not been characterised (unlike Tier 3, which is
recommended to run 3–5 times); the reported numbers should be read as one
realisation rather than a stable mean.

---

## Results Summary

| Tier | System | Strict F1 | Relaxed F1 | Macro F1 | P | R |
|---|---|---|---|---|---|---|
| 1 | PubMedBERT fine-tuned | **89.9** | **93.8** | **90.7** | 87.9 | 92.0 |
| 2 | GLiNER-biomed zero-shot | 63.3 | 76.3 | 52.3 | 64.7 | 61.9 |
| 3 | LLM 0-shot | 60.9 | 67.5 | 51.5 | 61.9 | 59.9 |
| 3 | LLM 3-shot | 57.9 | 64.8 | 51.4 | 65.5 | 51.8 |
| 4 | Multi-agent system | 76.7 | 79.7 | 68.1 | 83.9 | 70.7 |
| **4 Extended** | **LLM-orchestrated agent** | **79.2** | **82.3** | **78.0** | **86.3** | **73.2** |

---

## Ablation Table (Tier 4)

Each row disables one component and reports the F1 drop.
Shows which parts of the architecture contribute measurably.

| System | Strict F1 | Δ | What this measures |
|---|---|---|---|
| Full system | 76.7 | — | — |
| − KB confidence gate | 74.1 | −2.6 | value of gating rare branch on common confidence |
| − rare branch | 74.2 | −2.5 | value of KB lookup for OrganismTaxon / CellLine |
| − pattern branch | 74.2 | −2.5 | value of deterministic regex for SequenceVariant |
| − overseer / requery | 76.7 | −0 | value of LLM re-query for low-confidence spans |
| Single-LLM (Tier 3) | 60.9 | −15.8 | total value of agentic orchestration |
| Tier 4 Extended | 79.2 | +2.5 | value of replacing the combiner with LLM-orchestrated tool selection |

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
branches serve complementary, non-overlapping parts of the entity type space.

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
The gap between Tier 3 (60.9) and the full Tier 4 system (76.7) represents the total
contribution of agentic orchestration, holding the base model constant. This 15.8-point
gain decomposes approximately as: heterogeneous routing adds ~5.0 (rare + pattern
branches combined), the confidence gate adds ~2.6, and the remaining ~8.2 comes from
the structured combiner and BERT span extraction replacing the LLM extractor. No single
component accounts for the full gain, confirming that the architecture's value is
emergent rather than attributable to any one design decision.

**Tier 4 Extended / LLM-orchestrated arbitration (F1 delta: +2.5 overall, +9.9 macro)**
Replacing the entire combiner — confidence gate included — with LLM-orchestrated tool
selection recovers a further +2.5 strict F1 on top of the full Tier 4 system, holding
span extraction and all three underlying branch methods constant. Critically, this gain
is not uniform: it is concentrated almost entirely in the two entity types (CellLine,
SequenceVariant) that the branch-resolution diagnosis showed were resolved by a single
isolated branch with zero cross-branch competition under Tier 4. This confirms the
KB confidence gate's earlier finding from the opposite direction — where a fixed
threshold prevents harmful KB overrides, it also prevents beneficial ones, and an
agent that can weigh evidence contextually per span recovers value the fixed rule
structurally cannot access.

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

7. **Deterministic arbitration systematically under-serves branch-isolated
   entity types, and this is fixable without touching span extraction or the
   underlying branch methods.** Tier 4's combiner routes each of the six
   entity types to exactly one branch by construction, with essentially zero
   cross-branch contest. Tier 4 Extended shows that replacing only the
   arbitration logic — same tools, same span extractor — recovers +32.5 F1
   on CellLine and +23.7 F1 on SequenceVariant, while leaving common-branch-
   dominated types (Gene, Disease, Chemical) unchanged. This isolates
   arbitration flexibility as the causal factor, distinct from model quality
   or extraction coverage.

---

## Known Limitations

- **OrganismTaxon recall (34.6% Tier 4 / 34.1% Tier 4 Extended)** — BERT span
  extractor misses organism mentions used in descriptive or colloquial
  contexts (`patients`, `human`, `Chinese hamster`). KB lookup can only type
  candidates it receives, and this bottleneck is upstream of arbitration —
  confirmed by Tier 4 Extended leaving this type essentially unchanged
  despite otherwise substantial gains elsewhere.
- **SequenceVariant recall ceiling** — same span extraction limitation.
  138 gold variants were never extracted as candidates under Tier 4; Tier 4
  Extended's improvement (42.7% → 75.1% recall) comes from typing more of
  the candidates that were extracted, not from extracting more of them.
- **CellLine variance** — support of 50 in the test set makes per-type F1
  high-variance. Results should be interpreted cautiously.
- **Descriptive annotation spans** — a subset of BioRED gold annotations are
  long descriptive phrases (e.g. `valine (gtg) to a methionine (atg)`) that
  no NER system returns as a single span. These affect all tiers equally.
- **LLM non-determinism** — Tier 3, Tier 4's overseer, and Tier 4 Extended's
  agent all use LLMs. Tier 3 is run at temperature 0 with multiple runs
  recommended for headline numbers. Tier 4 Extended's reported result is a
  single run; run-to-run variance has not been characterised.

---

## Environment Variables

| Variable | Required by | Description |
|---|---|---|
| `GROQ_API_KEY` | Tiers 3, 4 | Groq API key for LLM calls |
| `NCBI_EMAIL` | Tier 4 | Email for NCBI Entrez API (courtesy, no registration needed) |
| `IDA_LLM_API_KEY` | Tier 4 Extended | University of Glasgow HPC LLM endpoint key |