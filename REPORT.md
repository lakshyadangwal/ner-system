# Project Report — Named Entity Recognition (NER) System

## 1. Introduction

Named Entity Recognition (NER) is a fundamental natural-language-processing
task that automatically identifies and classifies named entities in unstructured
text into pre-defined categories such as persons, organisations, locations,
miscellaneous entities and dates. NER is a building block of many downstream
applications: information extraction, question answering, search indexing,
resume parsing, and content recommendation.

This college project builds an end-to-end NER system that:
- fine-tunes a Transformer (DistilBERT, optionally BERT) on the CoNLL-2003
  dataset for PER / ORG / LOC / MISC detection,
- adds DATE detection with spaCy (CoNLL-2003 has no DATE label),
- merges the two outputs without overlapping spans,
- presents results through a Streamlit graphical user interface with
  colour-coded highlights, an entity table, per-category counts and CSV export.

Everything runs locally and free on an Apple Silicon Mac (M5) using the PyTorch
MPS backend — no CUDA, no fp16/bf16, no paid APIs, no API keys.

## 2. Objective

To build, train, evaluate and deploy a working Named Entity Recognition system
that identifies PER, ORG, LOC, MISC and DATE entities in arbitrary user text,
with:
- a reproducible training pipeline (`train.py`),
- a clean inference engine (`ner_engine.py`),
- a user-friendly GUI (`app.py`),
- full documentation (this report, README.md, VIVA.md).

## 3. Dataset

**CoNLL-2003** (loaded from the Hugging Face Datasets Hub; the `tomaarsen/conll2003`
build was used). It contains English newswire sentences annotated with four
entity types using the BIO scheme:

| Label | Meaning |
|-------|---------|
| PER   | Person  |
| ORG   | Organisation |
| LOC   | Location |
| MISC  | Miscellaneous |
| O     | Outside any entity |

Splits used (tomaarsen/conll2003):
- Train: 14,041 sentences
- Validation: 3,250 sentences
- Test: 3,453 sentences

Label distribution (token level, all splits) is shown in
`label_distribution.png`. The "O" label dominates (~250k tokens), which is
typical for NER; entity tokens are a minority, with PER and LOC being the most
frequent entity types and MISC the least.

**Note on DATE:** CoNLL-2003 has no DATE label. DATE entities are therefore
added at inference time with spaCy's `en_core_web_sm` and merged with the model
output (see Section 5).

## 4. Methodology

### 4.1 Preprocessing
CoNLL-2003 sentences are already tokenised into word lists with integer NER
tags. No further text cleaning is required; the labels are read directly from
the dataset features (`ner_tags`).

### 4.2 Tokenization and subword label alignment
The input words are tokenised with `AutoTokenizer` (`is_split_into_words=True`,
`truncation=True`). Because BERT/DistilBERT use WordPiece subword tokenisation,
one word may map to multiple subwords, and special tokens ([CLS], [SEP]) map to
none. Labels are aligned with `word_ids()` as follows:
- special token (word_id is None)          → -100 (ignored by the loss)
- first subword of a word                  → the word's label
- continuation subwords of the same word   → -100 (ignored by the loss)

`-100` is the ignore index of `torch.nn.CrossEntropyLoss`, so those positions
do not affect training.

### 4.3 Model
`AutoModelForTokenClassification` with:
- `num_labels = 9` (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC),
- `id2label` / `label2id` maps built from the dataset label list.

Default: `distilbert-base-cased` (switchable to `bert-base-cased` via
`--model-name`).

### 4.4 Training setup
Trained with Hugging Face `Trainer` and `DataCollatorForTokenClassification`:

| Hyperparameter | Value |
|----------------|-------|
| Learning rate  | 2e-5 |
| Epochs         | 3 (1 for smoke test) |
| Batch size     | 16 per device |
| Weight decay   | 0.01 |
| Evaluation     | each epoch (`eval_strategy="epoch"`) |
| Saving         | each epoch, `load_best_model_at_end=True` |
| Best model by  | `metric_for_best_model="f1"` |
| Mixed precision| None (no fp16/bf16 on MPS) |
| Telemetry      | `report_to="none"` |
| Seed           | 42 |

The code handles the `evaluation_strategy` vs `eval_strategy` rename across
transformers versions by inspecting `TrainingArguments.__init__`. Likewise for
`tokenizer` vs `processing_class` in `Trainer`.

Training runs on the MPS backend (`torch.backends.mps.is_available() == True`)
with `PYTORCH_ENABLE_MPS_FALLBACK=1`.

### 4.5 Evaluation
`compute_metrics` uses `seqeval` to compute overall precision, recall, F1 and
accuracy at the entity (chunk) level. After training, the full
`seqeval.classification_report` is printed on the TEST split, and overall +
per-entity metrics are saved to `metrics.json`.

## 5. Implementation

### 5.1 Training script (`train.py`)
- Tries CoNLL-2003 dataset IDs in order: `eriktks/conll2003`,
  `tomaarsen/conll2003`, `conll2003`; uses whichever loads.
- `--smoke` flag (CLI) runs 500 train / 200 validation samples for 1 epoch.
- Prints split sizes and label distribution; saves `label_distribution.png`
  (matplotlib Agg backend).
- Tokenizes + aligns labels, fine-tunes, evaluates on TEST, writes
  `metrics.json`, saves model + tokenizer to `./model`.
- Heavily commented for viva explanations.

### 5.2 Inference engine (`ner_engine.py`)
- Loads `./model` with the transformers `token-classification` pipeline,
  `aggregation_strategy="simple"`, device = MPS if available else CPU.
- Pipeline is cached (loaded once).
- Normalises model labels to PER / ORG / LOC / MISC (strips B-/I- prefixes).
- Loads spaCy `en_core_web_sm` and extracts DATE entities.
- Merges model entities + DATE entities with no overlap: model entities win on
  conflict; result sorted by start position.
- `extract_entities(text)` → list of `{text, label, start, end, score}`.
- Raises `ModelNotFoundError` if `./model` is missing; empty text → `[]`.
- Long text is chunked (~400 words) with correct character-offset shifting.
- `__main__` block runs 3 test sentences.

### 5.3 GUI (`app.py`)
Streamlit app (`streamlit run app.py --server.headless true`):
- Wide layout, title, description, sidebar with model info (F1 from
  `metrics.json`) and colour legend (PER blue, ORG green, LOC orange, MISC
  purple, DATE red).
- Text area, "Extract Entities" button, "Load sample text" buttons
  (`st.session_state`), including an Indian-news sample.
- Highlights entities inline in escaped HTML with per-entity label tags.
- pandas table (Entity | Label | Confidence, 3 decimals) and per-category
  counts (`st.metric` + bar chart).
- "Download CSV" button (`st.download_button`).
- Error handling: empty input warning, >5000-character warning/truncation,
  missing-model friendly error; model loading wrapped in `st.cache_resource`.

## 6. Results

Full training on CoNLL-2003 (DistilBERT, 3 epochs). Test-set metrics saved to
`metrics.json`:

Overall (test set, 3,453 sentences):
- Precision: 0.8853
- Recall: 0.9017
- F1: 0.8934
- Accuracy: 0.9787

Per entity:

| Entity | Precision | Recall | F1     | Support |
|--------|-----------|--------|--------|---------|
| PER    | 0.9605    | 0.9462 | 0.9533 | 1,617   |
| ORG    | 0.8521    | 0.8844 | 0.8679 | 1,661   |
| LOC    | 0.9222    | 0.9167 | 0.9194 | 1,668   |
| MISC   | 0.7262    | 0.8048 | 0.7635 | 702     |

Full details are in `metrics.json`. The test F1 of 0.8934 lies within the
expected ≈ 0.88–0.92 range for DistilBERT (BERT would be slightly higher).

## 7. Limitations

- Trained only on English newswire (CoNLL-2003); other domains may degrade.
- DATE detection depends on spaCy's rule-based NER (common date formats only).
- MPS may fall back to CPU for some ops (`PYTORCH_ENABLE_MPS_FALLBACK=1`).
- Single-user Streamlit deployment; no confidence-threshold tuning.

## 8. Conclusion

The project delivers a complete, locally-run NER system: a fine-tuned
DistilBERT model for PER/ORG/LOC/MISC, spaCy-based DATE detection, conflict-free
span merging, and a Streamlit GUI with highlights, a data table, counts and CSV
export. All code runs free on an Apple Silicon Mac with the MPS backend and no
external APIs.

## 9. Future Work

- Fine-tune larger models (BERT, RoBERTa) for higher F1.
- Train on additional data/domains (e.g., OntoNotes, social media).
- Add confidence-threshold slider and batch/file processing.
- Support more entity types and containerise the deployment.
