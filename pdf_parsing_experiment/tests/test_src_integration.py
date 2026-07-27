from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import articleUtil
from gui.yml_generation.yml_document_parsing import generate_document_parsing_yaml
from params import PathSettings, TableExtractionParams
from service import file_to_md
from src.pdf_parser import _ensure_opendoc_models


class ProductionPDFIntegrationTests(unittest.TestCase):
    def test_public_input_models_keep_previous_shape(self) -> None:
        path_settings = PathSettings.model_validate(
            {
                "folder_path": "input",
                "save_path": "output",
                "file_type": "PDF",
                "convert_mode": "byPart",
            }
        )
        table_settings = TableExtractionParams.model_validate(
            {
                "file_folder_path": "input",
                "save_folder_path": "output",
                "non_tabular_file_format": "PDF",
            }
        )
        self.assertEqual(path_settings.file_type, "PDF")
        self.assertEqual(table_settings.non_tabular_file_format, "PDF")
        self.assertFalse(hasattr(path_settings, "pdf_parsing"))
        self.assertFalse(hasattr(table_settings, "pdf_parsing"))

    def test_missing_models_fail_before_open_doc_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "OMNIEXTRACT_OPENDOC_MODEL_DIR"):
                _ensure_opendoc_models(
                    Path(temp_dir),
                    ("PP-DocLayoutV2.onnx",),
                    auto_download=False,
                )

    def test_pdf_batch_uses_hybrid_without_public_parser_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            pdf_path = source / "article.v1.PDF"
            pdf_path.write_bytes(b"synthetic")

            def fake_convert_pdf(pdf, output_path, **kwargs):
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_text("# Article\n", encoding="utf-8")
                self.assertEqual(kwargs["backend"], "hybrid")

            with patch.object(articleUtil, "convert_pdf", side_effect=fake_convert_pdf):
                result = articleUtil.parse_article_to_md(source, output)

            expected = output / "article.v1" / "article.v1.md"
            self.assertEqual(result, [str(expected)])
            self.assertTrue(expected.is_file())

    def test_service_preserves_old_pdf_signature(self) -> None:
        with patch("service.parse_article_to_md", return_value=["article.md"]) as parser:
            result = file_to_md("input", "output", "PDF")
        self.assertEqual(result, ["article.md"])
        parser.assert_called_once_with("input", "output")

    def test_gui_yaml_does_not_expose_parser_options(self) -> None:
        import yaml

        content = generate_document_parsing_yaml(
            "D:/input", "D:/output", "PDF", "byPart"
        )
        parsed = yaml.safe_load(content)
        self.assertNotIn("pdf_parsing", parsed)

    def test_production_batch_and_section_split_smoke(self) -> None:
        pdf_path = PROJECT_ROOT / "pdf_parsing_experiment" / "test_pdfs" / "DAA.pdf"
        if not pdf_path.is_file():
            self.skipTest("corpus PDFs are not available")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "markdown"
            sections = root / "sections"
            source.mkdir()
            shutil.copy2(pdf_path, source / pdf_path.name)
            with patch.object(articleUtil, "convert_pdf") as parser:
                parser.side_effect = lambda pdf, output_path, **kwargs: (
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True),
                    Path(output_path).write_text("# Introduction\ntext\n", encoding="utf-8"),
                )
                result = articleUtil.parse_article_to_md(source, output)
            self.assertIn(str(output / "DAA" / "DAA.md"), result)
            summary = articleUtil.split_md(output, sections)
            self.assertEqual(summary["processed"], 1)
            self.assertTrue((sections / "DAA.json").is_file())


if __name__ == "__main__":
    unittest.main()
