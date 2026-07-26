"""Markdown rendering for the compact PDF parser."""

from __future__ import annotations

import html
import re
from typing import Any


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
        if run.get("bold") and text.strip():
            leading = text[: len(text) - len(text.lstrip())]
            trailing = text[len(text.rstrip()) :]
            core = text.strip()
            rendered.append(f"{leading}**{core}**{trailing}")
        else:
            rendered.append(text)
    value = "".join(rendered)
    value = re.sub(r"\*\*\s*\*\*", "", value)
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

    markdown = "\n\n".join(part.rstrip() for part in parts if part.strip())
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return markdown + ("\n" if markdown else "")
