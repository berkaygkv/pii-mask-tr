"""HTML preview rendering."""

import unittest

from pii_mask.preview import render_preview_html


class PreviewTest(unittest.TestCase):
    def test_renders_chips_and_legend(self):
        text = "Sayın Berkay, TC 12345678901."
        spans = [
            {"label": "KISI_ADI", "start": 6, "end": 12},
            {"label": "TCKN",     "start": 17, "end": 28},
        ]
        html = render_preview_html(text, spans, source="sample.pdf")
        self.assertIn("KISI_ADI", html)
        self.assertIn("TCKN", html)
        self.assertIn("sample.pdf", html)
        self.assertIn("Berkay", html)
        self.assertIn("12345678901", html)
        self.assertIn("<strong>2</strong> entities", html)

    def test_empty_spans(self):
        html = render_preview_html("hello world", [], source="empty.txt")
        self.assertIn("<strong>0</strong> entities", html)
        self.assertIn("hello world", html)

    def test_html_escaped(self):
        html = render_preview_html("<script>alert(1)</script>", [], source="x.txt")
        self.assertNotIn("<script>alert", html)
        self.assertIn("&lt;script&gt;", html)
