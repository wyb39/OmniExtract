"""Markdown rendering for the production PDF parser."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any


ACADEMIC_SECTION_ORDER = (
    "Others",
    "Introduction",
    "Method",
    "Result",
    "Discussion",
    "Conclusion",
    "Funding",
    "Acknowledgement",
    "Reference",
    "Conflict of Interest",
    "Supporting Information",
    "Abbreviations",
)

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_BODY_SECTIONS = {
    "Introduction",
    "Method",
    "Result",
    "Discussion",
    "Conclusion",
}


def _escape_markdown(text: str, *, table_cell: bool = False) -> str:
    value = text.replace("\\", "\\\\")
    value = re.sub(r"(?<!\\)([*_`])", r"\\\1", value)
    if table_cell:
        value = value.replace("|", "\\|").replace("\n", "<br>")
    return value


def _render_runs(runs: list[dict[str, Any]], fallback: str) -> str:
    if not runs:
        return _escape_markdown(fallback)
    rendered: list[str] = []
    for run in runs:
        text = _escape_markdown(str(run.get("text", "")))
        if not text:
            continue
        if text.strip() and (run.get("bold") or run.get("script") in {"sup", "sub"}):
            leading = text[: len(text) - len(text.lstrip())]
            trailing = text[len(text.rstrip()) :]
            core = text.strip()
            script = run.get("script")
            if script in {"sup", "sub"}:
                core = f"<{script}>{core}</{script}>"
            if run.get("bold"):
                core = f"**{core}**"
            rendered.append(f"{leading}{core}{trailing}")
        else:
            rendered.append(text)
    value = "".join(rendered)
    return re.sub(r"\*\*(.*?)\*\*\s+\*\*(.*?)\*\*", r"**\1 \2**", value)


def _render_list(text: str, runs: list[dict[str, Any]]) -> str:
    value = _render_runs(runs, text).strip()
    value = re.sub(r"^[\u2022\u25cf\u25aa\u25e6\u00b7*]\s*", "- ", value)
    value = re.sub(r"^-\s*", "- ", value)
    return value


def _render_gfm_table(rows: list[list[str]]) -> str:
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return ""
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[1:]

    def render_row(row: list[str]) -> str:
        return "| " + " | ".join(_escape_markdown(cell, table_cell=True) for cell in row) + " |"

    output = [render_row(header), "| " + " | ".join("---" for _ in range(width)) + " |"]
    output.extend(render_row(row) for row in body)
    return "\n".join(output)


def _render_html_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    output = ["<table>", "  <thead>", "    <tr>"]
    output.extend(f"      <th>{html.escape(cell).replace(chr(10), '<br>')}</th>" for cell in normalized[0])
    output.extend(["    </tr>", "  </thead>", "  <tbody>"])
    for row in normalized[1:]:
        output.append("    <tr>")
        output.extend(f"      <td>{html.escape(cell).replace(chr(10), '<br>')}</td>" for cell in row)
        output.append("    </tr>")
    output.extend(["  </tbody>", "</table>"])
    return "\n".join(output)


def render_document(document: Any) -> str:
    """Render a lightweight Document object as LLM-friendly Markdown."""

    if document.status == "ocr_required" and not any(page.blocks for page in document.pages):
        return "<!-- ocr_required: no usable text layer -->\n"

    parts: list[str] = []
    for page in document.pages:
        for block in page.blocks:
            if block.kind == "heading":
                level = max(1, min(int(block.level or 2), 6))
                parts.append(f"{'#' * level} {_render_runs(block.runs, block.text).strip()}")
            elif block.kind == "paragraph":
                value = _render_runs(block.runs, block.text).strip()
                if value:
                    parts.append(value)
            elif block.kind == "list":
                value = _render_list(block.text, block.runs)
                if value:
                    parts.append(value)
            elif block.kind == "table":
                value = (
                    _render_html_table(block.rows)
                    if block.complex_table
                    else _render_gfm_table(block.rows)
                )
                if value:
                    parts.append(value)
            elif block.kind == "raw_markdown":
                value = block.text.strip()
                if value:
                    parts.append(value)

    markdown = "\n\n".join(part.rstrip() for part in parts if part.strip())
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return markdown + ("\n" if markdown else "")


def _plain_heading(value: str) -> tuple[str, str]:
    """Return readable and compact forms of a Markdown heading."""

    text = html.unescape(value)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*_`~]", "", text)
    text = re.sub(
        (
            r"^\s*(?:(?i:section)\s+)?"
            r"(?:\d+(?:\.\d+)*(?:[.)]\s*|\s+)|[IVXLCDM]+[.)]\s*)"
            r"[-:–—]?\s*"
        ),
        "",
        text,
    )
    readable = re.sub(r"\s+", " ", text).strip().lower()
    compact = re.sub(r"[^a-z]+", "", readable)
    return readable, compact


def _canonical_academic_heading(value: str) -> str | None:
    """Map a rendered academic heading to the legacy section schema."""

    _, compact = _plain_heading(value)
    if not compact:
        return None

    # Supplementary sections must be checked before their base section names.
    if (
        compact.startswith("supplementary")
        or compact.startswith("supportinginformation")
        or compact.startswith("supportingmaterials")
        or compact.startswith("dataavailability")
        or compact.startswith("codeavailability")
        or compact.startswith("availabilityofdataandmaterials")
        or compact.startswith("dataandmaterialsavailability")
    ):
        return "Supporting Information"
    if (
        compact.startswith("conflictofinterest")
        or compact.startswith("conflictsofinterest")
        or compact.startswith("competinginterest")
        or compact.startswith("declarationofcompetinginterest")
        or compact.startswith("declarationofinterests")
    ):
        return "Conflict of Interest"
    if (
        compact.startswith("funding")
        or compact.startswith("financialsupport")
        or compact.startswith("formattingoffundingsources")
    ):
        return "Funding"
    if compact.startswith("acknowledg"):
        return "Acknowledgement"
    if (
        compact in {"reference", "references", "bibliography", "literaturecited"}
        or compact.startswith("referencesand")
    ):
        return "Reference"
    if compact.startswith("abbreviation"):
        return "Abbreviations"

    if (
        compact.startswith("conclusion")
        or compact.startswith("concluding")
        or compact in {"summary", "summaryandconclusions"}
    ):
        return "Conclusion"
    if compact.startswith("resultsanddiscussion") or compact.startswith(
        "resultsanddiscussions"
    ):
        # Preserve the legacy schema without duplicating combined content.
        return "Result"
    if compact in {"discussion", "discussions", "generaldiscussion"}:
        return "Discussion"
    if compact in {"result", "results", "findings", "mainfindings"}:
        return "Result"

    method_names = {
        "method",
        "methods",
        "methodology",
        "materialandmethod",
        "materialsandmethods",
        "patientsandmethods",
        "materialspatientsandmethods",
        "materialspatientandmethods",
    }
    if compact in method_names:
        return "Method"
    if (
        compact.startswith("experimental")
        or compact.endswith("experimentalprocedure")
        or compact.endswith("experimentalprocedures")
        or compact.endswith("experimentaldetails")
        or compact.endswith("experimentalmethods")
    ):
        return "Method"

    if compact in {
        "introduction",
        "generalintroduction",
        "background",
        "backgroundandobjectives",
    }:
        return "Introduction"

    if compact in {
        "abstract",
        "additionalinformation",
        "articleinfo",
        "authoraffiliations",
        "authorcontribution",
        "authorcontributions",
        "authorstatement",
        "citation",
        "correspondence",
        "figurelegend",
        "figurelegends",
        "graphicalabstract",
        "highlights",
        "keywords",
        "laysummary",
        "table",
        "tables",
    }:
        return "Others"

    return None


def _section_heading(
    line: str,
    *,
    first_heading_index: int | None,
    line_index: int,
) -> str | None:
    match = _MARKDOWN_HEADING_RE.match(line.rstrip("\r\n"))
    if not match:
        return None
    level = len(match.group(1))
    if level > 2:
        return None
    if level == 1 and line_index == first_heading_index:
        # The first H1 is the document title, even when it contains words such
        # as "results" or "methods".
        return None
    return _canonical_academic_heading(match.group(2))


def split_markdown_text(markdown: str) -> dict[str, str]:
    """Split LLM-friendly academic Markdown into the legacy section schema.

    Every source character is assigned exactly once. Structured-abstract labels
    that precede a real Introduction remain in ``Others``; main-body headings
    control the canonical sections. Unknown headings stay with the current
    section so tables, formulas, and nested subsections remain intact.
    """

    lines = markdown.splitlines(keepends=True)
    sections: dict[str, list[str]] = {
        section: [] for section in ACADEMIC_SECTION_ORDER
    }
    first_heading_index = next(
        (
            index
            for index, line in enumerate(lines)
            if _MARKDOWN_HEADING_RE.match(line.rstrip("\r\n"))
        ),
        None,
    )
    heading_sections = [
        _section_heading(
            line,
            first_heading_index=first_heading_index,
            line_index=index,
        )
        for index, line in enumerate(lines)
    ]
    has_introduction = "Introduction" in heading_sections

    current = "Others"
    body_started = False
    for line, candidate in zip(lines, heading_sections):
        if candidate == "Introduction":
            body_started = True
            current = candidate
        elif candidate in _BODY_SECTIONS:
            if body_started or not has_introduction:
                body_started = True
                current = candidate
            else:
                # Methods/Results/Conclusion labels in a structured abstract
                # must not steal the corresponding main-body section.
                current = "Others"
        elif candidate is not None:
            current = candidate
        sections[current].append(line)

    return {
        section: "".join(sections[section])
        for section in ACADEMIC_SECTION_ORDER
    }


def split_markdown_file(file_path: str | Path) -> dict[str, str]:
    """Split one UTF-8 Markdown file into canonical academic sections."""

    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Markdown file does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return split_markdown_text(stream.read())


def _discover_markdown_files(folder_path: Path) -> list[Path]:
    direct = sorted(folder_path.glob("*.md"))
    legacy = sorted(
        path
        for subdirectory in folder_path.iterdir()
        if subdirectory.is_dir()
        for path in [subdirectory / f"{subdirectory.name}.md"]
        if path.is_file()
    )
    return list(dict.fromkeys([*direct, *legacy]))


def split_md(
    folder_path: str | Path,
    save_path: str | Path,
) -> dict[str, Any]:
    """Split parser-produced Markdown files and write compatible JSON files.

    This keeps the production ``split_md(folder_path, save_path)`` call shape,
    while supporting both the new parser's flat output and the legacy
    ``folder/article/article.md`` layout.
    """

    source_directory = Path(folder_path).expanduser().resolve()
    output_directory = Path(save_path).expanduser().resolve()
    if not source_directory.exists():
        raise ValueError(f"Folder path does not exist: {source_directory}")
    if not source_directory.is_dir():
        raise ValueError(f"Path is not a directory: {source_directory}")
    if output_directory.exists() and not output_directory.is_dir():
        raise ValueError(f"Save path is not a directory: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    files = _discover_markdown_files(source_directory)
    processed: list[str] = []
    errors: list[dict[str, str]] = []
    output_names: set[str] = set()
    for markdown_path in files:
        output_name = f"{markdown_path.stem}.json"
        if output_name in output_names:
            errors.append(
                {
                    "file": str(markdown_path),
                    "error": f"duplicate output filename: {output_name}",
                }
            )
            continue
        output_names.add(output_name)
        try:
            sections = split_markdown_file(markdown_path)
            (output_directory / output_name).write_text(
                json.dumps(sections, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            processed.append(markdown_path.name)
        except Exception as exc:
            errors.append(
                {
                    "file": str(markdown_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "processed": len(processed),
        "errors": errors,
        "files": processed,
        "output": str(output_directory),
    }
