#!/usr/bin/env python3
"""
ner_engine.py - Named Entity Recognition inference engine.

Loads the fine-tuned model from ./model (transformers token-classification
pipeline, aggregation_strategy="simple"), normalizes its labels to the canonical
set {PER, ORG, LOC, MISC}, extracts DATE entities with spaCy (CoNLL-2003 has no
DATE label), merges the two without overlapping spans (model entities win on
conflict), and returns a sorted list of entity dicts:

    {"text": str, "label": str, "start": int, "end": int, "score": float}

Run directly:
    python ner_engine.py
"""

import logging
import os
import re

import spacy
import torch
from transformers import pipeline

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_DIR = "./model"
CHUNK_WORDS = 400          # long text is split into chunks of ~400 words
CANONICAL_LABELS = ("PER", "ORG", "LOC", "MISC", "DATE")

logger = logging.getLogger(__name__)


class ModelNotFoundError(RuntimeError):
    """Raised when the fine-tuned model directory is missing."""
    pass


# ---------------------------------------------------------------------------
# Pipeline cache (loaded once, reused across calls)
# ---------------------------------------------------------------------------
_pipeline = None
_nlp = None


def _get_device():
    """Return the device id for the transformers pipeline: mps or cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_pipeline():
    """Lazy-load and cache the token-classification pipeline."""
    global _pipeline
    if _pipeline is None:
        if not os.path.isdir(MODEL_DIR):
            raise ModelNotFoundError(
                f"Model directory '{MODEL_DIR}' not found. "
                "Train the model first: python train.py"
            )
        device = _get_device()
        logger.info("Loading token-classification pipeline from %s (device=%s)", MODEL_DIR, device)
        _pipeline = pipeline(
            "token-classification",
            model=MODEL_DIR,
            tokenizer=MODEL_DIR,
            aggregation_strategy="simple",
            device=device,
        )
    return _pipeline


def _get_nlp():
    """Lazy-load and cache the spaCy English model."""
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model en_core_web_sm")
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _normalize_label(label):
    """
    Map any model entity_group to one of PER, ORG, LOC, MISC.
    Strips B-/I- prefixes, upper-cases, and ignores unknown labels.
    Returns None if the label cannot be mapped.
    """
    if label is None:
        return None
    label = label.upper()
    for prefix in ("B-", "I-"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    if label in CANONICAL_LABELS and label != "DATE":
        return label
    return None


def _spans_overlap(a_start, a_end, b_start, b_end):
    """True if half-open spans [a_start, a_end) and [b_start, b_end) overlap."""
    return a_start < b_end and b_start < a_end


def _safe_chunk_words(tokenizer, target_words=CHUNK_WORDS, max_tokens=None):
    """
    Estimate how many words fit in one chunk without exceeding the model's
    maximum sequence length. Reserves 2 tokens for special tokens ([CLS]/[SEP]).
    """
    if max_tokens is None:
        max_tokens = getattr(tokenizer, "model_max_length", 512) or 512
    max_tokens -= 2  # room for special tokens
    if max_tokens < 16:
        max_tokens = 510
    # Estimate tokens-per-word from a small sample so chunks stay near, but
    # under, the token limit (avoids pipeline truncation losing entities).
    sample_words = min(60, target_words)
    sample_text = " ".join(["word"] * sample_words)  # placeholder, replaced below
    # We measure the ratio lazily in _chunk_spans using real text; here we just
    # return the token budget. The per-call ratio computation happens in
    # _chunk_spans where real text is available.
    return max_tokens


def _chunk_spans(text, tokenizer, target_words=CHUNK_WORDS):
    """
    Split text into chunks of up to ~target_words words, where every chunk
    is guaranteed to fit within the model's maximum sequence length (so the
    pipeline does not truncate and drop entities).

    Returns (chunk_text, start_char_offset) for each chunk.
    """
    # Find every non-whitespace run with its character span.
    word_spans = [m for m in re.finditer(r"\S+", text)]
    if not word_spans:
        return [(text, 0)]

    max_tokens = _safe_chunk_words(tokenizer)
    # Estimate tokens-per-word from a real sample of this text.
    sample_n = min(50, len(word_spans))
    sample_end = word_spans[sample_n - 1].end()
    sample_toks = len(tokenizer(text[:sample_end], truncation=False)["input_ids"])
    ratio = sample_toks / max(1, sample_n)
    chunk_words = max(1, int((max_tokens / ratio) * 0.90))  # 10% safety margin
    chunk_words = min(target_words, chunk_words)

    chunks = []
    i = 0
    n = len(word_spans)
    while i < n:
        j = min(i + chunk_words, n)
        # Safety check: shrink the chunk if it would still exceed the limit.
        while j > i:
            cand = text[word_spans[i].start():word_spans[j - 1].end()]
            if len(tokenizer(cand, truncation=False)["input_ids"]) <= max_tokens:
                break
            j -= 1
        if j == i:
            # A single "word" is longer than the limit; take it anyway.
            j = i + 1
        chunks.append((text[word_spans[i].start():word_spans[j - 1].end()], word_spans[i].start()))
        i = j
    return chunks


def _merge_adjacent_fragments(entities, text):
    """
    Merge pipeline fragments that belong to the same entity.

    The transformers pipeline (even with aggregation_strategy="simple")
    sometimes returns subword fragments as separate entities for uncommon
    words (e.g. "Ad" and "Shibu" for "Aditya Shibu" when the middle subword
    "##itya" is dropped). This function joins consecutive fragments with the
    same label when the gap between them contains no whitespace (i.e. it is a
    pure subword-boundary gap, not a real word separating two entities).

    We do NOT merge when the gap is a separator word (e.g. "Narendra Modi met
    Joe Biden") or contains punctuation (e.g. "Paris, London"), since those
    are distinct entities.
    """
    if not entities:
        return entities

    entities = sorted(entities, key=lambda e: (e["start"], e["end"]))
    merged = [dict(entities[0])]

    for ent in entities[1:]:
        prev = merged[-1]
        same_label = ent["label"] == prev["label"]
        gap = text[prev["end"]:ent["start"]]
        gap_stripped = gap.strip()
        # Common short words that separate two distinct entities (e.g. "Modi met Joe").
        # If the gap is just such a word, do NOT merge.
        separator_words = {
            "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
            "in", "on", "at", "to", "for", "of", "and", "or", "but", "with",
            "by", "from", "as", "into", "via", "per", "vs", "meet", "met",
        }
        is_separator = gap_stripped.lower() in separator_words
        # Punctuation in the gap means two separate entities,
        # e.g. "Paris, London" or "Google; Microsoft".
        has_punct = any(ch in ",;:()[]{}|/&\"\n" for ch in gap)
        # Only bridge gaps that are pure subword artifacts (no separator word)
        # and short enough to be a dropped subword (<= 10 chars).
        gap_is_subword = (not is_separator) and (not has_punct) and len(gap) <= 10
        if same_label and gap_is_subword:
            prev["end"] = max(prev["end"], ent["end"])
            prev["text"] = text[prev["start"]:prev["end"]]
            prev["score"] = max(prev["score"], ent["score"])
        else:
            merged.append(dict(ent))

    return merged


def _model_entities(text):
    """Run the fine-tuned pipeline and return normalized model entities."""
    pipe = _get_pipeline()
    # aggregation_strategy="simple" merges subwords into entity groups.
    raw_entities = pipe(text)
    entities = []
    for ent in raw_entities:
        label = _normalize_label(ent.get("entity_group"))
        if label is None:
            continue
        entities.append({
            "text": text[int(ent["start"]):int(ent["end"])],  # original text (no "##")
            "label": label,
            "start": int(ent["start"]),
            "end": int(ent["end"]),
            "score": float(ent.get("score", 0.0)),
        })
    # Merge subword fragments into whole entities.
    entities = _merge_adjacent_fragments(entities, text)
    return entities


def _date_entities(text):
    """Extract DATE entities with spaCy. Score is 1.0 (rule-based)."""
    nlp = _get_nlp()
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ == "DATE":
            entities.append({
                "text": ent.text,
                "label": "DATE",
                "start": ent.start_char,
                "end": ent.end_char,
                "score": 1.0,
            })
    return entities


def _merge_entities(model_ents, date_ents):
    """
    Merge model entities and DATE entities without overlapping spans.
    Rule: if a DATE span overlaps a model entity, keep the model entity.
    """
    merged = list(model_ents)
    for date in date_ents:
        overlap = False
        for m in model_ents:
            if _spans_overlap(date["start"], date["end"], m["start"], m["end"]):
                overlap = True
                break
        if not overlap:
            merged.append(date)
    merged.sort(key=lambda e: (e["start"], e["end"]))
    return merged


def extract_entities(text):
    """
    Extract named entities from text.

    Returns a list of dicts: {text, label, start, end, score}.
    Labels are always one of PER, ORG, LOC, MISC, DATE.

    Raises ModelNotFoundError if ./model is missing.
    Returns [] for empty input.
    Long text is split into ~400-word chunks with correct character offsets.
    """
    if not text or not text.strip():
        return []

    if not os.path.isdir(MODEL_DIR):
        raise ModelNotFoundError(
            f"Model directory '{MODEL_DIR}' not found. "
            "Train the model first: python train.py"
        )

    # Get the (cached) pipeline once; also reuse its tokenizer for chunking so
    # chunks never exceed the model's max sequence length.
    pipe = _get_pipeline()
    tokenizer = pipe.tokenizer

    all_entities = []
    for chunk_text, offset in _chunk_spans(text, tokenizer):
        model_ents = _model_entities(chunk_text)
        # Shift model entity offsets to absolute character positions.
        for ent in model_ents:
            ent["start"] += offset
            ent["end"] += offset
        date_ents = _date_entities(chunk_text)
        for ent in date_ents:
            ent["start"] += offset
            ent["end"] += offset
        all_entities.extend(_merge_entities(model_ents, date_ents))

    # De-duplicate identical spans (can happen at chunk boundaries) and sort.
    seen = set()
    unique = []
    for ent in all_entities:
        key = (ent["start"], ent["end"], ent["label"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(ent)
    unique.sort(key=lambda e: (e["start"], e["end"]))
    return unique


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_sentences = [
        "Sundar Pichai met Narendra Modi in Delhi on 5 March 2024 to discuss "
        "Google's expansion in India.",
        "Apple Inc. is headquartered in Cupertino, California.",
        "The United Nations signed an agreement in Paris on 12 January 2023.",
    ]

    try:
        for sentence in test_sentences:
            print("-" * 70)
            print("INPUT:", sentence)
            entities = extract_entities(sentence)
            print("ENTITIES:")
            for ent in entities:
                print(
                    "  %-20s | %-6s | start=%-4d end=%-4d | score=%.3f | '%s'"
                    % (
                        ent["label"],
                        ent["label"],
                        ent["start"],
                        ent["end"],
                        ent["score"],
                        ent["text"],
                    )
                )
    except ModelNotFoundError as exc:
        print("ERROR:", exc)
        print("Run 'python train.py' first to create the ./model directory.")
