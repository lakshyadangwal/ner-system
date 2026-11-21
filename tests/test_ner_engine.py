"""
Tests for ner_engine.py.

Run from the project folder:
    python -m unittest discover tests -v

The helper tests run without a trained model. The end-to-end tests need the
fine-tuned model in ./model and are skipped if it is missing.
"""

import os
import unittest

import ner_engine
from ner_engine import (
    _merge_adjacent_fragments,
    _merge_entities,
    _normalize_label,
    _spans_overlap,
    extract_entities,
)


def make_ent(text, word, label, score=0.9):
    """Build an entity dict for `word` at its first position in `text`."""
    start = text.index(word)
    return {"text": word, "label": label, "start": start, "end": start + len(word), "score": score}


class TestHelpers(unittest.TestCase):
    def test_normalize_label(self):
        self.assertEqual(_normalize_label("B-PER"), "PER")
        self.assertEqual(_normalize_label("i-loc"), "LOC")
        self.assertEqual(_normalize_label("ORG"), "ORG")
        self.assertIsNone(_normalize_label("O"))
        self.assertIsNone(_normalize_label("DATE"))  # DATE only comes from spaCy
        self.assertIsNone(_normalize_label(None))

    def test_spans_overlap(self):
        self.assertTrue(_spans_overlap(0, 5, 3, 8))
        self.assertFalse(_spans_overlap(0, 5, 5, 8))  # touching, not overlapping

    def test_subword_fragments_are_merged(self):
        text = "Aditya Shibu joined the team"
        ents = [make_ent(text, "Ad", "PER"), make_ent(text, "Shibu", "PER")]
        merged = _merge_adjacent_fragments(ents, text)
        self.assertEqual([e["text"] for e in merged], ["Aditya Shibu"])

    def test_entities_separated_by_punctuation_stay_apart(self):
        text = "I visited Paris, London and Tokyo"
        ents = [make_ent(text, w, "LOC") for w in ("Paris", "London", "Tokyo")]
        merged = _merge_adjacent_fragments(ents, text)
        self.assertEqual([e["text"] for e in merged], ["Paris", "London", "Tokyo"])

    def test_entities_separated_by_word_stay_apart(self):
        text = "Narendra Modi met Joe Biden"
        ents = [make_ent(text, "Narendra Modi", "PER"), make_ent(text, "Joe Biden", "PER")]
        merged = _merge_adjacent_fragments(ents, text)
        self.assertEqual(len(merged), 2)

    def test_model_entity_wins_over_overlapping_date(self):
        text = "May Smith arrived on 5 May 2024"
        model_ents = [make_ent(text, "May Smith", "PER")]
        date_ents = [make_ent(text, "May", "DATE", 1.0), make_ent(text, "5 May 2024", "DATE", 1.0)]
        merged = _merge_entities(model_ents, date_ents)
        self.assertEqual([(e["text"], e["label"]) for e in merged],
                         [("May Smith", "PER"), ("5 May 2024", "DATE")])


@unittest.skipUnless(os.path.isdir(ner_engine.MODEL_DIR), "trained model not found in ./model")
class TestExtractEntities(unittest.TestCase):
    def labels_for(self, text):
        return {(e["text"], e["label"]) for e in extract_entities(text)}

    def test_empty_text(self):
        self.assertEqual(extract_entities(""), [])
        self.assertEqual(extract_entities("   \n "), [])

    def test_person_org_location_date(self):
        found = self.labels_for(
            "Sundar Pichai met Narendra Modi in Delhi on 5 March 2024 to discuss Google's plans."
        )
        self.assertIn(("Sundar Pichai", "PER"), found)
        self.assertIn(("Delhi", "LOC"), found)
        self.assertIn(("Google", "ORG"), found)
        self.assertIn(("5 March 2024", "DATE"), found)

    def test_offsets_match_text(self):
        text = "Apple Inc. is headquartered in Cupertino, California."
        for ent in extract_entities(text):
            self.assertEqual(text[ent["start"]:ent["end"]], ent["text"])

    def test_special_characters(self):
        text = "António Guterres spoke at the United Nations (UN) in Genève!!! #news @UN"
        ents = extract_entities(text)
        self.assertTrue(len(ents) > 0)
        for ent in ents:
            self.assertEqual(text[ent["start"]:ent["end"]], ent["text"])

    def test_long_text_is_chunked(self):
        sentence = "Tim Cook visited Berlin to meet Siemens engineers. "
        text = sentence * 120  # well over the model's 512-token limit
        ents = extract_entities(text)
        people = [e for e in ents if e["label"] == "PER"]
        self.assertGreater(len(people), 100)
        for ent in ents:
            self.assertEqual(text[ent["start"]:ent["end"]], ent["text"])


if __name__ == "__main__":
    unittest.main()
