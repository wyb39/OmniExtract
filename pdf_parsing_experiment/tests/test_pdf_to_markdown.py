from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from markdown_renderer import render_document
from pdf_to_markdown import (
    Block,
    Document,
    Page,
    _canonical_margin_text,
    _document_from_opendoc_results,
    _heading_level,
    _is_bold,
)


class RendererTests(unittest.TestCase):
    def test_heading_bold_paragraph_and_table(self) -> None:
        document = Document(
            source="synthetic.pdf",
            metadata={},
            pages=[
                Page(
                    number=1,
                    width=600,
                    height=800,
                    blocks=[
                        Block(kind="heading", bbox=(0, 0, 100, 20), text="Paper title", level=1),
                        Block(
                            kind="paragraph",
                            bbox=(0, 30, 300, 50),
                            text="Important result",
                            runs=[
                                {"text": "Important", "bold": True},
                                {"text": " result", "bold": False},
                            ],
                        ),
                        Block(
                            kind="table",
                            bbox=(0, 60, 300, 120),
                            rows=[["A", "B"], ["1", "2"]],
                        ),
                    ],
                )
            ],
        )
        markdown = render_document(document)
        self.assertIn("# Paper title", markdown)
        self.assertIn("**Important** result", markdown)
        self.assertIn("| A | B |", markdown)
        self.assertIn("| --- | --- |", markdown)

    def test_complex_table_uses_html(self) -> None:
        document = Document(
            source="synthetic.pdf",
            metadata={},
            pages=[
                Page(
                    number=1,
                    width=600,
                    height=800,
                    blocks=[
                        Block(
                            kind="table",
                            bbox=(0, 0, 100, 100),
                            rows=[["A", "B"], ["", "2"]],
                            complex_table=True,
                        )
                    ],
                )
            ],
        )
        self.assertIn("<table>", render_document(document))

    def test_style_and_margin_helpers(self) -> None:
        self.assertTrue(_is_bold("ABCDEE+Arial-BoldMT"))
        self.assertEqual(_canonical_margin_text("Page 12"), "page #")

    def test_heading_heuristics_reject_dates_and_affiliations(self) -> None:
        base = {"runs": [], "bbox": (0, 0, 100, 10), "bold_ratio": 0.0}
        date = {**base, "text": "10 October 2018", "font_size": 18.0}
        affiliation = {
            **base,
            "text": "1 Childhood Liver Oncology Group",
            "font_size": 10.0,
        }
        section = {**base, "text": "1. Introduction", "font_size": 11.0}
        self.assertIsNone(_heading_level(date, 10.0))
        self.assertIsNone(_heading_level(affiliation, 10.0))
        self.assertEqual(_heading_level(section, 10.0), 2)

    def test_opendoc_results_keep_markdown_and_heading_structure(self) -> None:
        results = {
            "width": 1200,
            "height": 1600,
            "recognition_results": [
                {
                    "label": "doc_title",
                    "bbox": [100, 100, 1100, 180],
                    "score": 0.98,
                    "text": "OCR Paper",
                },
                {
                    "label": "text",
                    "bbox": [100, 220, 1100, 400],
                    "score": 0.95,
                    "text": "An inline formula: $x^2$.",
                },
                {
                    "label": "paragraph_title",
                    "bbox": [100, 410, 1100, 440],
                    "score": 0.96,
                    "text": "Abstract",
                },
                {
                    "label": "abstract",
                    "bbox": [100, 440, 1100, 448],
                    "score": 0.94,
                    "text": "Abstract: OCR abstract body.",
                },
                {
                    "label": "table",
                    "bbox": [100, 450, 1100, 800],
                    "score": 0.92,
                    "text": "| A | B |\n| --- | --- |\n| 1 | 2 |",
                },
                {
                    "label": "footer",
                    "bbox": [100, 1500, 1100, 1560],
                    "score": 0.99,
                    "text": "Ignored footer",
                },
            ],
        }
        document = _document_from_opendoc_results(Path("ocr.pdf"), results)
        markdown = render_document(document)
        self.assertEqual(document.metadata["backend"], "opendoc")
        self.assertIn("# OCR Paper", markdown)
        self.assertIn("$x^2$", markdown)
        self.assertEqual(markdown.count("## Abstract"), 1)
        self.assertIn("OCR abstract body.", markdown)
        self.assertIn("| A | B |", markdown)
        self.assertNotIn("Ignored footer", markdown)


if __name__ == "__main__":
    unittest.main()
