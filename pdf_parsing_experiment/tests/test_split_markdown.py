from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from markdown_renderer import (
    ACADEMIC_SECTION_ORDER,
    split_markdown_text,
    split_md,
)


class AcademicMarkdownSplitterTests(unittest.TestCase):
    def assert_content_preserved(
        self,
        markdown: str,
        sections: dict[str, str],
    ) -> None:
        self.assertEqual(sum(map(len, sections.values())), len(markdown))
        self.assertEqual(tuple(sections), ACADEMIC_SECTION_ORDER)

    def test_main_sections_win_over_structured_abstract_labels(self) -> None:
        markdown = """# Article title

## Abstract
Summary.

## Methods:
Abstract methods.

## Results:
Abstract results.

## Conclusion:
Abstract conclusion.

## **1. Introduction**
Main introduction.

## **2. Materials** **and methods**
Main methods.

## **3. Results**
Main results.

## **4. Discussion**
Main discussion.
"""
        sections = split_markdown_text(markdown)

        self.assertIn("Abstract methods.", sections["Others"])
        self.assertIn("Main introduction.", sections["Introduction"])
        self.assertIn("Main methods.", sections["Method"])
        self.assertNotIn("Abstract methods.", sections["Method"])
        self.assertIn("Main results.", sections["Result"])
        self.assertIn("Main discussion.", sections["Discussion"])
        self.assert_content_preserved(markdown, sections)

    def test_fragmented_and_experimental_headings_are_recognized(self) -> None:
        markdown = """# Article

## **Intro ductio** **n**
Introduction body.

## 2. Alloy design and experimental procedures
Method body.

## **Result s**
Result body.

## **Di scussio** **n**
Discussion body.

## **Refer ences**
[1] Reference body.
"""
        sections = split_markdown_text(markdown)

        self.assertIn("Introduction body.", sections["Introduction"])
        self.assertIn("Method body.", sections["Method"])
        self.assertIn("Result body.", sections["Result"])
        self.assertIn("Discussion body.", sections["Discussion"])
        self.assertIn("[1] Reference body.", sections["Reference"])
        self.assert_content_preserved(markdown, sections)

    def test_multiline_references_are_not_moved_into_result(self) -> None:
        markdown = """# Article

## Introduction
Body.

## Results
Result body.

## References
- [1] First reference
  continuation of first reference
- [2] Second reference
  continuation of second reference

## Conclusion
Conclusion after references.
"""
        sections = split_markdown_text(markdown)

        self.assertIn("continuation of first reference", sections["Reference"])
        self.assertIn("continuation of second reference", sections["Reference"])
        self.assertNotIn("reference", sections["Result"].lower())
        self.assertIn("Conclusion after references.", sections["Conclusion"])
        self.assert_content_preserved(markdown, sections)

    def test_supplementary_methods_do_not_replace_main_methods(self) -> None:
        markdown = """# Article

## Introduction
Introduction.

## Materials/Patients and Methods
Main methods.

## Results
Results.

## References
[1] Main reference.

## Supplementary Methods
Supplementary protocol.

## Supplementary References
[S1] Supplementary reference.
"""
        sections = split_markdown_text(markdown)

        self.assertIn("Main methods.", sections["Method"])
        self.assertNotIn("Supplementary protocol.", sections["Method"])
        self.assertIn(
            "Supplementary protocol.",
            sections["Supporting Information"],
        )
        self.assertIn(
            "[S1] Supplementary reference.",
            sections["Supporting Information"],
        )
        self.assert_content_preserved(markdown, sections)

    def test_combined_results_and_discussion_remains_single_copy(self) -> None:
        markdown = """# Article

## Introduction
Introduction.

## Results and Discussion
Combined content.
"""
        sections = split_markdown_text(markdown)

        self.assertIn("Combined content.", sections["Result"])
        self.assertEqual(sections["Discussion"], "")
        self.assertEqual(
            sum(value.count("Combined content.") for value in sections.values()),
            1,
        )
        self.assert_content_preserved(markdown, sections)

    def test_missing_introduction_is_not_fabricated(self) -> None:
        markdown = """# Article

## Results
Results-first article.

## Methods
Methods follow results.
"""
        sections = split_markdown_text(markdown)

        self.assertEqual(sections["Introduction"], "")
        self.assertIn("Results-first article.", sections["Result"])
        self.assertIn("Methods follow results.", sections["Method"])
        self.assert_content_preserved(markdown, sections)

    def test_common_back_matter_aliases_are_separated(self) -> None:
        markdown = """# Article

## Methods
Main methods.

## Data Availability
Repository link.

## Competing Interests
No competing interests.
"""
        sections = split_markdown_text(markdown)

        self.assertIn("Main methods.", sections["Method"])
        self.assertIn(
            "Repository link.",
            sections["Supporting Information"],
        )
        self.assertIn(
            "No competing interests.",
            sections["Conflict of Interest"],
        )
        self.assert_content_preserved(markdown, sections)

    def test_split_md_processes_flat_and_legacy_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "flat.md").write_text(
                "# Flat\n\n## Introduction\nFlat body.\n",
                encoding="utf-8",
            )
            legacy_directory = source / "legacy"
            legacy_directory.mkdir()
            (legacy_directory / "legacy.md").write_text(
                "# Legacy\n\n## Introduction\nLegacy body.\n",
                encoding="utf-8",
            )

            summary = split_md(source, output)

            self.assertEqual(summary["processed"], 2)
            self.assertEqual(summary["errors"], [])
            flat = json.loads(
                (output / "flat.json").read_text(encoding="utf-8")
            )
            legacy = json.loads(
                (output / "legacy.json").read_text(encoding="utf-8")
            )
            self.assertIn("Flat body.", flat["Introduction"])
            self.assertIn("Legacy body.", legacy["Introduction"])

    def test_split_markdown_file_preserves_crlf_characters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            markdown_path = Path(temporary_directory) / "article.md"
            markdown = "# Article\r\n\r\n## Introduction\r\nBody.\r\n"
            with markdown_path.open("w", encoding="utf-8", newline="") as stream:
                stream.write(markdown)

            source = markdown_path.read_bytes().decode("utf-8")
            output = Path(temporary_directory) / "output"
            summary = split_md(markdown_path.parent, output)
            sections = json.loads(
                (output / "article.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary["errors"], [])
            self.assertEqual(sum(map(len, sections.values())), len(source))
            self.assertIn("\r\n", sections["Introduction"])


if __name__ == "__main__":
    unittest.main()
