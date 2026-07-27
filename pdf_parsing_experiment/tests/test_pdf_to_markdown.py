from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


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
    _layout_region_needs_recognition,
    _layout_source_chars,
    _looks_like_colored_chart,
    _native_dense_table_fallback,
    _native_blocks_for_layout_region,
    _native_table_is_plausible,
    _normalize_layout_heading,
    _pdfium_charbox_to_page_bbox,
    parse_pdf,
    _run_opendoc_layout,
    _table_crop_rotation,
    _table_recognition_is_suspicious,
    _token_tail_is_repetitive,
    _unirec_task_max_length,
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

    def test_adjacent_bold_runs_keep_their_separator(self) -> None:
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
                            kind="heading",
                            bbox=(0, 0, 100, 20),
                            text="tumor behavior",
                            runs=[
                                {"text": "tumor", "bold": True},
                                {"text": " ", "bold": False},
                                {"text": "behavior", "bold": True},
                            ],
                            level=1,
                        )
                    ],
                )
            ],
        )

        self.assertEqual(render_document(document), "# **tumor behavior**\n")

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
        self.assertEqual(blocks[0].text, "Hybrid Title")
        self.assertGreater(blocks[0].bold_ratio, 0.9)

    def test_pdfium_charbox_respects_nonzero_page_origin(self) -> None:
        self.assertEqual(
            _pdfium_charbox_to_page_bbox(
                charbox=(70.0, 800.0, 170.0, 820.0),
                page_bbox=(40.0, 50.0, 640.0, 850.0),
            ),
            (30.0, 30.0, 130.0, 50.0),
        )

    def test_layout_source_chars_preserve_explicit_whitespace(self) -> None:
        chars = [
            {"text": "A", "upright": True},
            {"text": " ", "upright": True},
            {"text": "\x00", "upright": True},
            {"text": "B\x00", "upright": True},
            {"text": "ignored", "upright": False},
        ]

        usable = _layout_source_chars(chars)

        self.assertEqual([char["text"] for char in usable], ["A", " ", "B"])

    def test_pdfium_source_spacing_repairs_shifted_spaces_and_wrapped_words(
        self,
    ) -> None:
        chars: list[dict[str, object]] = []
        source_index = 0
        pending_space = False

        for line_text, top, hidden_hyphen_before in (
            ("high energy syn", 10.0, False),
            ("chrotron diffraction and ", 20.0, True),
            ("thermal properties of Zr", 30.0, False),
            ("based plastic manner micro", 40.0, True),
            ("structure ", 50.0, False),
            ("laser-treated", 60.0, False),
            ("spot line", 70.0, True),
        ):
            pending_hyphen = hidden_hyphen_before
            if pending_hyphen:
                # PDFium reports some visible line-end hyphens as U+0002.
                source_index += 1
            x = 10.0
            for value in line_text:
                if value == " ":
                    x += 1.55
                    chars.append(
                        {
                            "text": " ",
                            "x0": x,
                            "x1": x,
                            "top": top + 4.8,
                            "bottom": top + 5.8,
                            "size": 1.0,
                            "fontname": "Times",
                            "upright": True,
                            "source_index": source_index,
                            "source_space_before": False,
                            "source_hyphen_before": False,
                        }
                    )
                    pending_space = True
                else:
                    chars.append(
                        {
                            "text": value,
                            "x0": x,
                            "x1": x + 4.0,
                            "top": top,
                            "bottom": top + 7.0,
                            "size": 8.0,
                            "fontname": "Times",
                            "upright": True,
                            "source_index": source_index,
                            "source_space_before": pending_space,
                            "source_hyphen_before": pending_hyphen,
                        }
                    )
                    x += 4.0
                    pending_space = False
                    pending_hyphen = False
                source_index += 1

        # Simulate a PDF Unicode stream that omits one real space while its
        # visual gap remains identical to confirmed spaces on the same line.
        visible_prefix = ""
        for char in chars:
            if float(char["top"]) != 40.0 or not str(char["text"]).strip():
                continue
            if visible_prefix.endswith("plastic") and char["text"] == "m":
                char["source_space_before"] = False
                break
            visible_prefix += str(char["text"])

        blocks = _native_blocks_for_layout_region(
            label="text",
            bbox=(0.0, 0.0, 300.0, 90.0),
            chars=chars,
            page_width=300.0,
            body_size=8.0,
            confidence=0.99,
        )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(
            blocks[0].text,
            (
                "high energy synchrotron diffraction and thermal properties "
                "of Zr-based plastic manner microstructure "
                "laser-treated-spot line"
            ),
        )

    def test_shifted_small_glyphs_render_as_inline_scripts(self) -> None:
        def char(
            text: str,
            x0: float,
            x1: float,
            top: float,
            bottom: float,
            size: float,
        ) -> dict[str, object]:
            return {
                "text": text,
                "x0": x0,
                "x1": x1,
                "top": top,
                "bottom": bottom,
                "size": size,
                "fontname": "Times",
                "upright": True,
            }

        title_chars = [
            char("Z", 10, 18, 10, 24, 14),
            char("r", 18, 23, 10, 24, 14),
            char("5", 23, 29, 15, 25.5, 10.5),
            char("5", 29, 35, 15, 25.5, 10.5),
            char("C", 35, 44, 10, 24, 14),
            char("u", 44, 52, 10, 24, 14),
            char("3", 52, 58, 15, 25.5, 10.5),
            char("0", 58, 64, 15, 25.5, 10.5),
        ]
        author_chars = []
        x = 10.0
        for value in "B. Chen":
            width = 3.0 if value == " " else 5.0
            author_chars.append(char(value, x, x + width, 40, 51, 11))
            x += width
        author_chars.extend(
            [
                char("a", x + 3, x + 7, 37, 45, 8),
                char(",", x + 7, x + 9, 37, 45, 8),
                char("b", x + 9, x + 13, 37, 45, 8),
            ]
        )

        title = _native_blocks_for_layout_region(
            label="doc_title",
            bbox=(8, 8, 70, 28),
            chars=title_chars,
            page_width=600,
            body_size=11,
            confidence=0.99,
        )[0]
        author = _native_blocks_for_layout_region(
            label="text",
            bbox=(8, 35, 70, 55),
            chars=author_chars,
            page_width=600,
            body_size=11,
            confidence=0.99,
        )[0]
        document = Document(
            source="scripts.pdf",
            metadata={},
            pages=[Page(number=1, width=600, height=800, blocks=[title, author])],
        )

        self.assertEqual(title.text, "Zr55Cu30")
        self.assertEqual(author.text, "B. Chena,b")
        markdown = render_document(document)
        self.assertIn("# Zr<sub>55</sub>Cu<sub>30</sub>", markdown)
        self.assertIn("B. Chen<sup>a,b</sup>", markdown)

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

    def test_hybrid_runs_without_native_prepass(self) -> None:
        expected = Document(source="synthetic.pdf", metadata={}, pages=[])
        with patch(
            "pdf_to_markdown._parse_pdf_hybrid",
            return_value=expected,
        ) as hybrid_parser, patch(
            "pdf_to_markdown._parse_pdf_native"
        ) as native_parser:
            actual = parse_pdf("synthetic.pdf", backend="hybrid")

        self.assertIs(actual, expected)
        hybrid_parser.assert_called_once()
        native_parser.assert_not_called()

    def test_all_layout_tables_are_routed_to_unirec(self) -> None:
        native_block = Block(
            kind="paragraph",
            bbox=(0, 0, 100, 20),
            text="native table-like text",
        )

        self.assertTrue(
            _layout_region_needs_recognition("table", [native_block])
        )
        self.assertFalse(
            _layout_region_needs_recognition("text", [native_block])
        )

    def test_unirec_generation_guards(self) -> None:
        self.assertEqual(_unirec_task_max_length("display_formula", 400, 2048), 512)
        self.assertEqual(_unirec_task_max_length("table", 100, 2048), 534)
        self.assertEqual(_unirec_task_max_length("table", 0, 2048), 2048)
        self.assertEqual(_unirec_task_max_length("text", 100, 2048), 2048)

        prefix = list(range(200))
        repetitive_tail = [4, 8, 15, 16, 23, 42, 7, 9] * 12
        self.assertTrue(_token_tail_is_repetitive(prefix + repetitive_tail))
        self.assertFalse(_token_tail_is_repetitive(list(range(300))))

        explosive_table = "<table>" + "".join(
            "<tr>" + "<td>value</td>" * 5 + "</tr>"
            for _ in range(130)
        ) + "</table>"
        self.assertTrue(
            _table_recognition_is_suspicious(
                explosive_table,
                native_character_count=100,
            )
        )
        self.assertFalse(
            _table_recognition_is_suspicious(
                "<table><tr><td>A</td><td>B</td></tr></table>",
                native_character_count=100,
            )
        )

    def test_table_crop_rotation_and_heatmap_filter(self) -> None:
        import math
        import numpy as np

        rotated_chars = [
            {"text": "A", "upright": False, "angle": -math.pi / 2}
            for _ in range(20)
        ]
        self.assertEqual(_table_crop_rotation(rotated_chars), 90)
        self.assertEqual(
            _table_crop_rotation(
                [{"text": "A", "upright": True, "angle": 0.0}] * 20
            ),
            0,
        )

        heatmap = np.full((100, 100, 3), 255, dtype=np.uint8)
        heatmap[15:85, 15:85] = (180, 210, 250)
        self.assertTrue(
            _looks_like_colored_chart(
                heatmap,
                native_character_count=200,
            )
        )
        plain_table = np.full((100, 100, 3), 255, dtype=np.uint8)
        plain_table[::20, :] = 0
        plain_table[:, ::20] = 0
        self.assertFalse(
            _looks_like_colored_chart(
                plain_table,
                native_character_count=200,
            )
        )

        dense_chars = []
        for row in range(50):
            for column in range(4):
                for offset in range(10):
                    dense_chars.append(
                        {
                            "text": "A",
                            "x0": 10 + column * 100 + offset * 3,
                            "x1": 12 + column * 100 + offset * 3,
                            "top": 10 + row * 10,
                            "bottom": 16 + row * 10,
                            "size": 6,
                            "upright": True,
                        }
                    )
        fallback = _native_dense_table_fallback(
            dense_chars,
            bbox=(0, 0, 500, 600),
            confidence=0.9,
        )
        self.assertIsNotNone(fallback)
        self.assertEqual(len(fallback.rows), 51)
        self.assertEqual(len(fallback.rows[0]), 4)
        self.assertEqual(fallback.rows[0][0], "Column 1")

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
