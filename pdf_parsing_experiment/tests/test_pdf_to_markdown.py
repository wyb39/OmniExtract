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
    _layout_bbox_to_pdf,
    _layout_native_needs_recognition,
    _native_blocks_for_layout_region,
    _native_table_is_plausible,
    _normalize_layout_heading,
    _run_opendoc_layout,
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

    def test_layout_coordinates_and_native_style_mapping(self) -> None:
        self.assertEqual(
            _layout_bbox_to_pdf(
                (100, 200, 600, 800),
                page_width=600,
                page_height=800,
                image_width=1200,
                image_height=1600,
            ),
            (50.0, 100.0, 300.0, 400.0),
        )
        chars = []
        x = 10.0
        for character in "Hybrid Title":
            width = 3.0 if character == " " else 6.0
            chars.append(
                {
                    "text": character,
                    "x0": x,
                    "x1": x + width,
                    "top": 20.0,
                    "bottom": 32.0,
                    "size": 12.0,
                    "fontname": "Arial-Bold",
                }
            )
            x += width
        blocks = _native_blocks_for_layout_region(
            label="doc_title",
            bbox=(10, 20, 200, 40),
            chars=chars,
            page_width=600,
            body_size=10.0,
            confidence=0.95,
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].kind, "heading")
        self.assertEqual(blocks[0].level, 1)
        self.assertEqual(blocks[0].source, "layout-native")
        self.assertGreater(blocks[0].bold_ratio, 0.9)

    def test_layout_heading_and_native_table_filters(self) -> None:
        self.assertEqual(_normalize_layout_heading("a b s t r a c t"), "Abstract")
        self.assertEqual(_normalize_layout_heading("a r t i c l e i n f o"), "ARTICLE INFO")
        self.assertEqual(_normalize_layout_heading("Thermal properties"), "Thermal properties")

        data_table = Block(
            kind="table",
            bbox=(50, 100, 550, 220),
            rows=[
                ["Spot", "01", "02", "03"],
                ["Fe %", "100", "79", "92"],
                ["Y %", "0", "10", "8"],
            ],
            confidence=0.9,
        )
        prose_table = Block(
            kind="table",
            bbox=(20, 100, 580, 650),
            rows=[
                ["Long two-column paragraph " * 3, "Another paragraph " * 4]
                for _ in range(10)
            ],
            confidence=0.95,
        )
        self.assertTrue(
            _native_table_is_plausible(
                data_table,
                page_width=600,
                page_height=800,
                chart_overlap=0.0,
            )
        )
        self.assertFalse(
            _native_table_is_plausible(
                prose_table,
                page_width=600,
                page_height=800,
                chart_overlap=0.0,
            )
        )
        corrupt_title = Block(
            kind="heading",
            bbox=(0, 0, 100, 20),
            text="Oxidation of a CueZr-based alloy",
        )
        self.assertTrue(
            _layout_native_needs_recognition("doc_title", [corrupt_title])
        )
        self.assertFalse(
            _layout_native_needs_recognition("text", [corrupt_title])
        )

    def test_layout_adapter_repairs_799_pixel_tensor(self) -> None:
        import numpy as np

        class FakeSession:
            input_shape = None

            def run(self, output_names, input_dict):
                self.input_shape = input_dict["image"].shape
                return [np.empty((0, 8), dtype=np.float32)]

        class FakeDetector:
            output_names = ["boxes"]
            session = FakeSession()

            def preprocess(self, image):
                return (
                    {
                        "image": np.zeros((1, 3, 800, 799), dtype=np.float32),
                        "im_shape": np.array([[800, 800]], dtype=np.float32),
                        "scale_factor": np.array([[8.0, 6.0]], dtype=np.float32),
                    },
                    (8.0, 6.0),
                    image.shape[0],
                    image.shape[1],
                )

            def postprocess(self, image, outputs, scale, height, width):
                return {"boxes": [], "size": [height, width]}

        detector = FakeDetector()
        result = _run_opendoc_layout(
            detector,
            np.zeros((100, 133, 3), dtype=np.uint8),
        )
        self.assertEqual(detector.session.input_shape, (1, 3, 800, 800))
        self.assertEqual(result["size"], [100, 133])


if __name__ == "__main__":
    unittest.main()
