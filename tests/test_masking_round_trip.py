"""Mask -> unmask round-trip and span post-processing checks.

These run without the model — they cover the deterministic logic.
"""

import unittest

from pii_mask.identifier_validators import (
    filter_invalid_spans,
    is_structurally_valid,
)
from pii_mask.inference import merge_adjacent_same_label_spans, merge_window_spans
from pii_mask.masking import mask_text, unmask_text


class MaskingRoundTripTest(unittest.TestCase):
    def test_round_trip_restores_exact_text(self):
        text = "Sayın Berkay Gökova, TC: 19283746501. Tel: +905321234567."
        spans = [
            {"label": "KISI_ADI", "start": 6, "end": 19},
            {"label": "TCKN",     "start": 25, "end": 36},
            {"label": "TELEFON",  "start": 43, "end": 56},
        ]
        masked, mapping = mask_text(text, spans)
        self.assertNotIn("Berkay Gökova", masked)
        self.assertIn("«KISI_ADI_1»", masked)
        self.assertEqual(unmask_text(masked, mapping), text)

    def test_repeated_entity_collapses_to_one_placeholder(self):
        text = "Berkay aradı. Daha sonra Berkay tekrar aradı."
        spans = [
            {"label": "KISI_ADI", "start": 0, "end": 6},
            {"label": "KISI_ADI", "start": 25, "end": 31},
        ]
        masked, mapping = mask_text(text, spans)
        self.assertEqual(masked.count("«KISI_ADI_1»"), 2)
        self.assertEqual(len(mapping), 1)

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            mask_text(
                "x",
                [
                    {"label": "A", "start": 0, "end": 5},
                    {"label": "B", "start": 3, "end": 7},
                ],
            )

    def test_empty_input(self):
        self.assertEqual(mask_text("", []), ("", {}))
        self.assertEqual(unmask_text("", {}), "")


class SpanPostProcessingTest(unittest.TestCase):
    def test_glue_eposta_split_at_at(self):
        text = "berkay@example.com"
        spans = [
            {"label": "EPOSTA", "start": 0, "end": 6},
            {"label": "EPOSTA", "start": 7, "end": 18},
        ]
        merged = merge_adjacent_same_label_spans(spans, text)
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0]["start"], merged[0]["end"]), (0, 18))

    def test_space_gap_does_not_merge(self):
        text = "Ali Veli"
        spans = [
            {"label": "KISI_ADI", "start": 0, "end": 3},
            {"label": "KISI_ADI", "start": 4, "end": 8},
        ]
        merged = merge_adjacent_same_label_spans(spans, text)
        self.assertEqual(len(merged), 2)

    def test_window_overlap_unions_same_label(self):
        text = "TR330006100519786457841326"
        spans = [
            {"label": "IBAN", "start": 0, "end": 14},
            {"label": "IBAN", "start": 8, "end": 26},
        ]
        merged = merge_window_spans(spans, text)
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0]["start"], merged[0]["end"]), (0, 26))


class StructuralValidatorsTest(unittest.TestCase):
    def test_tckn_format(self):
        self.assertTrue(is_structurally_valid("TCKN", "12345678901"))
        self.assertFalse(is_structurally_valid("TCKN", "01234567890"))  # leading 0
        self.assertFalse(is_structurally_valid("TCKN", "1234"))

    def test_iban_format(self):
        self.assertTrue(is_structurally_valid("IBAN", "TR330006100519786457841326"))
        self.assertTrue(is_structurally_valid("IBAN", "TR33 0006 1005 1978 6457 8413 26"))
        self.assertFalse(is_structurally_valid("IBAN", "DE89370400440532013000"))

    def test_plaka_format(self):
        self.assertTrue(is_structurally_valid("PLAKA", "34 ABC 123"))
        self.assertTrue(is_structurally_valid("PLAKA", "06-DEF-12"))
        self.assertFalse(is_structurally_valid("PLAKA", "99 ABC 123"))  # province > 81
        self.assertFalse(is_structurally_valid("PLAKA", "34 XYZ 123"))  # X not in TR plate alphabet

    def test_unknown_label_passes_through(self):
        self.assertTrue(is_structurally_valid("KISI_ADI", "Berkay Gökova"))

    def test_filter_drops_only_invalid(self):
        text = "TCKN: 12345678901 ve TCKN: 23"
        spans = [
            {"label": "TCKN", "start": 6, "end": 17},
            {"label": "TCKN", "start": 27, "end": 29},
        ]
        kept = filter_invalid_spans(spans, text)
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
