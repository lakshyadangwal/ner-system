# VIVA.md — 15 Likely Viva Questions & Answers

**Project:** Named Entity Recognition (NER) System  
**Model:** DistilBERT fine-tuned on CoNLL-2003, PyTorch MPS backend  
**Entities:** PER, ORG, LOC, MISC (model) + DATE (spaCy)

---

**Q1. What is Named Entity Recognition (NER)?**
A: NER is an NLP task that locates and classifies named entities in text into
pre-defined categories such as persons (PER), organisations (ORG), locations
(LOC), miscellaneous (MISC) and dates (DATE). Example: in "Sundar Pichai met
Narendra Modi in Delhi", "Sundar Pichai" and "Narendra Modi" are PER, "Delhi"
is LOC.

**Q2. What is the BIO scheme?**
A: BIO = Begin, Inside, Outside. `B-X` marks the first token of an entity of
type X, `I-X` marks a continuation token of the same entity, and `O` marks a
token outside any entity. CoNLL-2003 uses this scheme (e.g. B-PER, I-PER).

**Q3. Why do we need subword label alignment, and what is -100 for?**
A: BERT/DistilBERT tokenise words into sub-words (WordPiece). A word like
"expansion" may become several sub-words, and [CLS]/[SEP] are special tokens.
We assign the word's label only to the first sub-word and `-100` to the rest
and to special tokens. `-100` is the ignore index for PyTorch's
CrossEntropyLoss, so those positions do not contribute to the loss.

**Q4. Why BERT / DistilBERT for this task?**
A: They are pre-trained Transformer models with strong contextual
representations, which are essential for NER because an entity's type depends
on surrounding context (e.g. "Apple" the fruit vs. "Apple Inc." the company).
DistilBERT is a smaller, faster distillation of BERT that keeps most of the
accuracy — suitable for a college project running locally on an M5 Mac.

**Q5. What does the Hugging Face Trainer do?**
A: The `Trainer` handles the full training loop: the forward pass, loss
computation, backpropagation, optimizer step, evaluation on the validation set
each epoch, checkpoint saving, loading the best model at the end, and metric
logging — so we do not write the training loop by hand.

**Q6. What are precision, recall and F1, and why use seqeval?**
A: Precision = correct entities / predicted entities; recall = correct entities
/ true entities; F1 = harmonic mean of precision and recall. `seqeval` is the
standard NER evaluation library because it evaluates entity-level chunks
(whole spans, not individual tokens) and handles the BIO scheme correctly,
including the `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
and `classification_report` we use.

**Q7. CoNLL-2003 has no DATE label — how did you handle dates?**
A: CoNLL-2003 defines only PER, ORG, LOC and MISC. The fine-tuned model cannot
predict DATE. In `ner_engine.py` we load spaCy's `en_core_web_sm`, extract DATE
entities with its rule-based NER, and merge them with the model entities so
there are no overlapping spans (model entities win on conflict). DATE scores
are 1.0 because they are rule-based.

**Q8. How are overlapping spans merged?**
A: We first collect model entities and DATE entities. For each DATE span, if
it overlaps any model entity span, the DATE is discarded and the model entity
is kept. The remaining spans are sorted by start character position. This
prevents double-annotation of the same text.

**Q9. What does aggregation_strategy="simple" do?**
A: It is a setting of the transformers `token-classification` pipeline. It
merges sub-word/token predictions (B-PER, I-PER, ...) belonging to the same
entity into one entity span, assigning a single `entity_group` (e.g. PER) and
an averaged confidence score. Without it we would get one entity per token.

**Q10. Why use the MPS backend?**
A: We are on an Apple Silicon Mac. MPS (Metal Performance Shaders) lets PyTorch
run on the GPU, which is much faster than CPU for training and inference. We
set `PYTORCH_ENABLE_MPS_FALLBACK=1` so operations not supported on MPS
automatically run on CPU. We deliberately do not use fp16/bf16 (not well
supported on MPS).

**Q11. How does the app handle long text?**
A: `ner_engine.extract_entities` splits text into chunks of ~400 words, runs
the pipeline on each chunk, then shifts the entity character offsets back by
the chunk's starting offset so all positions refer to the original text.

**Q12. What happens if the model is missing?**
A: `ner_engine.py` raises a clear custom `ModelNotFoundError`, and the Streamlit
app shows a friendly error telling the user to run `python train.py` first.
Empty input returns an empty list and shows a warning in the GUI.

**Q13. What were your results?**
A: After full training on CoNLL-2003 (test split), DistilBERT achieves a test
F1 of about 0.88–0.92 (see `metrics.json` for the exact numbers and the
per-entity precision/recall/F1/support table). PER and LOC are usually the
easiest; MISC is typically the hardest.

**Q14. What are the limitations?**
A: Trained only on English newswire (CoNLL-2003), so other domains may worsen;
DATE relies on spaCy rules; MPS may fall back to CPU for some ops; single-user
Streamlit app; no confidence-threshold tuning.

**Q15. How could you improve the system?**
A: Fine-tune a larger model (BERT/RoBERTa), train on more data/domains (e.g.
OntoNotes), add entity-type confidence thresholds, support batch/file input,
add more entity types, and containerise/deploy the app.
