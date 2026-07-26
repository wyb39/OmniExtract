"""Compact, layout-aware PDF to Markdown experiment.

The module intentionally keeps extraction, normalization, and layout heuristics
in one place.  pdfplumber supplies character geometry and tables; pypdf supplies
metadata and a text fallback for pages without a usable character layer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import pdfplumber
from pypdf import PdfReader


BBox = tuple[float, float, float, float]


@dataclass
class Block:
    kind: str
    bbox: BBox
    text: str = ""
    runs: list[dict[str, Any]] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    font_size: float = 0.0
    bold_ratio: float = 0.0
    level: int | None = None
    confidence: float = 1.0
    complex_table: bool = False


@dataclass
class Page:
    number: int
    width: float
    height: float
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Document:
    source: str
    metadata: dict[str, Any]
    pages: list[Page]
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_BOLD_MARKERS = ("bold", "semibold", "demibold", "demi", "black", "heavy", "-bd")
_LIST_RE = re.compile(r"^\s*(?:[\u2022\u25cf\u25aa\u25e6\u00b7\-*]|\(?\d+[.)]|\(?[a-zA-Z][.)])\s+")
_SECTION_RE = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+){0,4})\s+)?"
    r"(abstract|keywords?|introduction|background|related work|materials?(?: and methods?)?|"
    r"methods?|methodology|experimental(?: procedures?)?|results?(?: and discussion)?|discussion|"
    r"conclusions?|acknowledg(?:e)?ments?|references|appendix)\b",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.|\d+\.\d+(?:\.\d+){0,3}\.?)\s+[A-Z][A-Za-z]"
)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def _is_bold(font_name: str) -> bool:
    name = (font_name or "").lower()
    return any(marker in name for marker in _BOLD_MARKERS)


def _inside_bbox(char: dict[str, Any], bbox: BBox, padding: float = 0.5) -> bool:
    x = (float(char.get("x0", 0.0)) + float(char.get("x1", 0.0))) / 2
    y = (float(char.get("top", 0.0)) + float(char.get("bottom", 0.0))) / 2
    return bbox[0] - padding <= x <= bbox[2] + padding and bbox[1] - padding <= y <= bbox[3] + padding


def _bbox_iou(first: BBox, second: BBox) -> float:
    x0, top = max(first[0], second[0]), max(first[1], second[1])
    x1, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, bottom - top)
    if not intersection:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1.0)


def _normalize_rows(rows: Sequence[Sequence[str | None]]) -> list[list[str]]:
    normalized = [[_clean_text(cell or "") for cell in row] for row in rows]
    width = max((len(row) for row in normalized), default=0)
    return [row + [""] * (width - len(row)) for row in normalized]


def _table_score(rows: list[list[str]]) -> float:
    if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
        return 0.0
    cells = [cell for row in rows for cell in row]
    nonempty = sum(bool(cell) for cell in cells)
    return len(rows) * len(rows[0]) * (nonempty / max(len(cells), 1))


def _extract_tables(page: Any, strategy: str = "auto") -> list[Block]:
    settings: list[dict[str, Any] | None] = [None]
    if strategy in {"auto", "strict"}:
        settings.append({"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"})
    if strategy in {"auto", "text"}:
        settings.append(
            {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "min_words_vertical": 3,
                "min_words_horizontal": 1,
                "intersection_tolerance": 4,
            }
        )

    candidates: list[tuple[float, Block]] = []
    for table_settings in settings:
        is_text_strategy = bool(table_settings and table_settings.get("vertical_strategy") == "text")
        try:
            tables = page.find_tables(table_settings=table_settings) if table_settings else page.find_tables()
        except Exception:
            continue
        for table in tables:
            try:
                raw_rows = table.extract()
            except Exception:
                continue
            rows = _normalize_rows(raw_rows or [])
            score = _table_score(rows)
            if score <= 0:
                continue
            bbox = tuple(float(value) for value in table.bbox)
            row_count = len(rows)
            column_count = max((len(row) for row in rows), default=0)
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            nonempty_ratio = sum(bool(cell) for row in rows for cell in row) / max(
                sum(len(row) for row in rows), 1
            )
            if nonempty_ratio < 0.35:
                continue
            nonempty_cells = [cell for row in rows for cell in row if cell]
            numeric_ratio = sum(bool(re.search(r"\d", cell)) for cell in nonempty_cells) / max(
                len(nonempty_cells), 1
            )
            average_cell_length = sum(len(cell) for cell in nonempty_cells) / max(
                len(nonempty_cells), 1
            )
            if is_text_strategy and numeric_ratio < 0.08 and average_cell_length > 12:
                continue
            if is_text_strategy and (
                (bbox_height > float(page.height) * 0.45 and (row_count > 10 or column_count > 5))
                or (bbox_width > float(page.width) * 0.90 and bbox_height > float(page.height) * 0.55)
            ):
                continue
            empty_ratio = sum(not cell for row in rows for cell in row) / max(
                sum(len(row) for row in rows), 1
            )
            complex_table = empty_ratio > 0.32 or any("\n" in (cell or "") for row in raw_rows for cell in row)
            block = Block(
                kind="table",
                bbox=bbox,
                rows=rows,
                confidence=min(1.0, 0.45 + score / 80),
                complex_table=complex_table,
            )
            candidates.append((score, block))

    selected: list[Block] = []
    for _, candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(_bbox_iou(candidate.bbox, existing.bbox) > 0.72 for existing in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda block: (block.bbox[1], block.bbox[0]))


def _line_from_chars(chars: list[dict[str, Any]]) -> dict[str, Any] | None:
    ordered = sorted(chars, key=lambda char: (float(char.get("x0", 0.0)), float(char.get("top", 0.0))))
    if not ordered:
        return None

    parts: list[str] = []
    runs: list[dict[str, Any]] = []
    previous_x1: float | None = None
    previous_size = 10.0
    bold_chars = 0
    visible_chars = 0

    def append_piece(piece: str, bold: bool) -> None:
        if not piece:
            return
        if runs and runs[-1]["bold"] == bold:
            runs[-1]["text"] += piece
        else:
            runs.append({"text": piece, "bold": bold})
        parts.append(piece)

    for char in ordered:
        value = str(char.get("text", ""))
        if not value:
            continue
        x0 = float(char.get("x0", 0.0))
        x1 = float(char.get("x1", x0))
        size = float(char.get("size", previous_size) or previous_size)
        bold = _is_bold(str(char.get("fontname", "")))
        if previous_x1 is not None and not value.isspace():
            gap = x0 - previous_x1
            if gap > max(1.2, min(previous_size, size) * 0.22) and (not parts or not parts[-1].endswith(" ")):
                append_piece(" ", False)
        append_piece(value, bold)
        nonspace = sum(not character.isspace() for character in value)
        visible_chars += nonspace
        bold_chars += nonspace if bold else 0
        previous_x1 = max(previous_x1 or x1, x1)
        previous_size = size

    text = _clean_text("".join(parts))
    if not text:
        return None
    cleaned_runs = []
    for run in runs:
        run_text = re.sub(r" {2,}", " ", run["text"])
        if run_text:
            cleaned_runs.append({"text": run_text, "bold": bool(run["bold"])})
    sizes = [float(char.get("size", 0.0) or 0.0) for char in ordered if str(char.get("text", "")).strip()]
    return {
        "text": text,
        "runs": cleaned_runs,
        "bbox": (
            min(float(char.get("x0", 0.0)) for char in ordered),
            min(float(char.get("top", 0.0)) for char in ordered),
            max(float(char.get("x1", 0.0)) for char in ordered),
            max(float(char.get("bottom", 0.0)) for char in ordered),
        ),
        "font_size": median(sizes) if sizes else 0.0,
        "bold_ratio": bold_chars / max(visible_chars, 1),
        "column": 0,
    }


def _split_visual_line(
    chars: list[dict[str, Any]],
    page_width: float,
    protect_wide_headings: bool = False,
) -> list[list[dict[str, Any]]]:
    ordered = sorted(chars, key=lambda item: float(item.get("x0", 0.0)))
    segments: list[list[dict[str, Any]]] = []
    previous_x1: float | None = None
    previous_size = 10.0
    for char in ordered:
        x0 = float(char.get("x0", 0.0))
        x1 = float(char.get("x1", x0))
        size = float(char.get("size", previous_size) or previous_size)
        if (
            segments
            and previous_x1 is not None
            and x0 - previous_x1 > max(page_width * 0.045, min(previous_size, size) * 4.0)
        ):
            segments.append([])
        if not segments:
            segments.append([])
        segments[-1].append(char)
        previous_x1 = max(previous_x1 or x1, x1)
        previous_size = size

    refined: list[list[dict[str, Any]]] = []
    for segment in segments:
        if len(segment) < 4:
            refined.append(segment)
            continue
        segment_width = float(segment[-1].get("x1", 0.0)) - float(segment[0].get("x0", 0.0))
        segment_font_size = median(
            float(char.get("size", 10.0) or 10.0) for char in segment
        )
        segment_top = min(float(char.get("top", 0.0)) for char in segment)
        central_gaps: list[tuple[float, int, float]] = []
        for index in range(1, len(segment)):
            left = segment[index - 1]
            right = segment[index]
            gap = float(right.get("x0", 0.0)) - float(left.get("x1", 0.0))
            boundary = (float(left.get("x1", 0.0)) + float(right.get("x0", 0.0))) / 2
            size = min(
                float(left.get("size", 10.0) or 10.0),
                float(right.get("size", 10.0) or 10.0),
            )
            if page_width * 0.35 <= boundary <= page_width * 0.65:
                central_gaps.append((gap, index, size))
        best_gap = max(central_gaps, default=None)
        if (
            not (protect_wide_headings and segment_top < 260 and segment_font_size >= 10.5)
            and segment_font_size < 13.0
            and segment_width > page_width * 0.68
            and best_gap
            and best_gap[0] > max(4.0, best_gap[2] * 0.55)
        ):
            refined.extend((segment[: best_gap[1]], segment[best_gap[1] :]))
        else:
            refined.append(segment)
    return refined


def _chars_to_lines(
    chars: Iterable[dict[str, Any]],
    page_width: float,
    protect_wide_headings: bool = False,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for char in sorted(chars, key=lambda item: (float(item.get("top", 0.0)), float(item.get("x0", 0.0)))):
        text = str(char.get("text", ""))
        if not text or text == "\x00":
            continue
        top = float(char.get("top", 0.0))
        size = float(char.get("size", 10.0) or 10.0)
        tolerance = max(1.5, min(3.5, size * 0.22))
        target = None
        for group in reversed(groups[-8:]):
            if abs(top - group["top"]) <= max(tolerance, group["tolerance"]):
                target = group
                break
            if top - group["top"] > 5.0:
                break
        if target is None:
            target = {"top": top, "tolerance": tolerance, "chars": []}
            groups.append(target)
        target["chars"].append(char)
        count = len(target["chars"])
        target["top"] = ((target["top"] * (count - 1)) + top) / count
        target["tolerance"] = max(target["tolerance"], tolerance)

    lines = [
        line
        for group in groups
        for segment in _split_visual_line(
            group["chars"],
            page_width,
            protect_wide_headings=protect_wide_headings,
        )
        if (line := _line_from_chars(segment)) is not None
    ]
    return sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0]))


def _canonical_margin_text(text: str) -> str:
    value = re.sub(r"\d+", "#", text.lower())
    value = re.sub(r"\s+", " ", value)
    return re.sub(r"[^a-z0-9\u4e00-\u9fff# ]", "", value).strip()


def _find_repeated_margins(page_lines: list[tuple[float, list[dict[str, Any]]]]) -> set[str]:
    counter: Counter[str] = Counter()
    for height, lines in page_lines:
        seen = set()
        for line in lines:
            top, bottom = line["bbox"][1], line["bbox"][3]
            if top <= height * 0.12 or bottom >= height * 0.90:
                key = _canonical_margin_text(line["text"])
                if 1 <= len(key) <= 160:
                    seen.add(key)
        counter.update(seen)
    threshold = max(2, math.ceil(len(page_lines) * 0.35))
    return {key for key, count in counter.items() if count >= threshold}


def _assign_columns(lines: list[dict[str, Any]], width: float) -> None:
    mid = width / 2
    gutter = width * 0.035
    left = [line for line in lines if line["bbox"][2] < mid + gutter and line["bbox"][0] < mid]
    right = [line for line in lines if line["bbox"][0] > mid - gutter]
    if len(left) < 4 or len(right) < 4:
        return
    left_span = (min(line["bbox"][1] for line in left), max(line["bbox"][3] for line in left))
    right_span = (min(line["bbox"][1] for line in right), max(line["bbox"][3] for line in right))
    overlap = min(left_span[1], right_span[1]) - max(left_span[0], right_span[0])
    if overlap < min(left_span[1] - left_span[0], right_span[1] - right_span[0]) * 0.2:
        return
    for line in lines:
        x0, _, x1, _ = line["bbox"]
        if x1 < mid + gutter:
            line["column"] = 1
        elif x0 > mid - gutter:
            line["column"] = 2


def _order_items(lines: list[dict[str, Any]], tables: list[Block], width: float) -> list[Any]:
    _assign_columns(lines, width)
    full_width = [
        line
        for line in lines
        if line["column"] == 0 or (line["bbox"][2] - line["bbox"][0]) >= width * 0.62
    ]
    boundaries = sorted({line["bbox"][1] for line in full_width})
    items: list[Any] = [*lines, *tables]

    def band_index(item: Any) -> int:
        bbox = item.bbox if isinstance(item, Block) else item["bbox"]
        return sum(boundary <= bbox[1] + 1.0 for boundary in boundaries)

    def key(item: Any) -> tuple[float, ...]:
        bbox = item.bbox if isinstance(item, Block) else item["bbox"]
        band = band_index(item)
        if isinstance(item, Block):
            column = 0 if (bbox[2] - bbox[0]) > width * 0.58 else (1 if bbox[0] < width / 2 else 2)
        else:
            column = int(item.get("column", 0))
        return (float(band), float(column), bbox[1], bbox[0])

    return sorted(items, key=key)


def _heading_level(line: dict[str, Any], body_size: float, is_title: bool = False) -> int | None:
    text = line["text"].strip()
    if is_title:
        return 1
    if not text or len(text) > 180 or text.endswith((".", ",", ";", ":")):
        return None
    size_ratio = line["font_size"] / max(body_size, 1.0)
    section_prefix = _SECTION_RE.match(text)
    section_tail = text[section_prefix.end() :].strip(" .:\t") if section_prefix else ""
    section_match = bool(section_prefix and len(section_tail) <= 16)
    numbered = bool(_NUMBERED_HEADING_RE.match(text))
    bold = line["bold_ratio"] >= 0.55
    first_letter = next((character for character in text if character.isalpha()), "")
    if not section_match and not numbered and first_letter.islower():
        return None
    if not section_match and not numbered and text.count(",") >= 3:
        return None
    if not section_match and not numbered and not (bold and size_ratio >= 1.12):
        return None
    if numbered:
        prefix = text.split(maxsplit=1)[0]
        depth = prefix.rstrip(".").count(".")
        return min(4, 2 + depth)
    return 2 if size_ratio >= 1.12 or section_match else 3


def _merge_runs(current: list[dict[str, Any]], incoming: list[dict[str, Any]], separator: str) -> None:
    if separator:
        if current and not current[-1]["text"].endswith((" ", "\n")):
            current.append({"text": separator, "bold": False})
    for run in incoming:
        if current and current[-1]["bold"] == run["bold"]:
            current[-1]["text"] += run["text"]
        else:
            current.append(dict(run))


def _build_blocks(
    ordered: list[Any], body_size: float, title_line: dict[str, Any] | None
) -> list[Block]:
    blocks: list[Block] = []
    paragraph_lines: list[dict[str, Any]] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = ""
        runs: list[dict[str, Any]] = []
        for index, line in enumerate(paragraph_lines):
            line_text = line["text"]
            dehyphenate = bool(index and text.endswith("-") and line_text[:1].islower())
            if dehyphenate:
                text = text[:-1]
                if runs:
                    runs[-1]["text"] = runs[-1]["text"].rstrip("-")
                separator = ""
            else:
                separator = " " if index else ""
                text += separator
            text += line_text
            _merge_runs(runs, line["runs"], separator)
        bbox = (
            min(line["bbox"][0] for line in paragraph_lines),
            min(line["bbox"][1] for line in paragraph_lines),
            max(line["bbox"][2] for line in paragraph_lines),
            max(line["bbox"][3] for line in paragraph_lines),
        )
        total_chars = sum(max(len(line["text"]), 1) for line in paragraph_lines)
        bold_ratio = sum(line["bold_ratio"] * max(len(line["text"]), 1) for line in paragraph_lines) / total_chars
        blocks.append(
            Block(
                kind="paragraph",
                text=_clean_text(text),
                runs=runs,
                bbox=bbox,
                font_size=median(line["font_size"] for line in paragraph_lines),
                bold_ratio=bold_ratio,
            )
        )
        paragraph_lines.clear()

    previous_line: dict[str, Any] | None = None
    for item in ordered:
        if isinstance(item, Block):
            flush_paragraph()
            blocks.append(item)
            previous_line = None
            continue
        labeled = re.match(r"^\s*(Abstract|Keywords?)\s*[:.-]\s*(.+)$", item["text"], re.IGNORECASE)
        if labeled:
            flush_paragraph()
            label = labeled.group(1).title()
            content = _clean_text(labeled.group(2))
            blocks.append(
                Block(
                    kind="heading",
                    text=label,
                    bbox=item["bbox"],
                    font_size=item["font_size"],
                    bold_ratio=item["bold_ratio"],
                    level=2,
                    confidence=0.9,
                )
            )
            blocks.append(
                Block(
                    kind="paragraph",
                    text=content,
                    runs=[{"text": content, "bold": False}],
                    bbox=item["bbox"],
                    font_size=item["font_size"],
                )
            )
            previous_line = None
            continue
        level = _heading_level(item, body_size, is_title=item is title_line)
        if level is not None:
            flush_paragraph()
            blocks.append(
                Block(
                    kind="heading",
                    text=item["text"],
                    runs=item["runs"],
                    bbox=item["bbox"],
                    font_size=item["font_size"],
                    bold_ratio=item["bold_ratio"],
                    level=level,
                    confidence=0.75 if level > 1 else 0.9,
                )
            )
            previous_line = None
            continue
        if _LIST_RE.match(item["text"]):
            flush_paragraph()
            blocks.append(
                Block(
                    kind="list",
                    text=item["text"],
                    runs=item["runs"],
                    bbox=item["bbox"],
                    font_size=item["font_size"],
                    bold_ratio=item["bold_ratio"],
                )
            )
            previous_line = None
            continue

        should_break = False
        if previous_line is not None:
            gap = item["bbox"][1] - previous_line["bbox"][3]
            different_column = item.get("column", 0) != previous_line.get("column", 0)
            indent_shift = abs(item["bbox"][0] - previous_line["bbox"][0])
            should_break = (
                different_column
                or gap > max(5.0, body_size * 0.72)
                or (indent_shift > body_size * 2.5 and not previous_line["text"].endswith(("-", ",")))
            )
        if should_break:
            flush_paragraph()
        paragraph_lines.append(item)
        previous_line = item
    flush_paragraph()
    return blocks


def _choose_title(
    page_lines: list[list[dict[str, Any]]],
    body_size: float,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    if not page_lines:
        return None
    boilerplate = ("journal pre-proof", "accepted manuscript", "available online", "downloaded from")
    metadata_title = _clean_text(str(metadata.get("Title", metadata.get("title", ""))))
    normalized_metadata_title = re.sub(r"\W+", "", metadata_title).lower()
    metadata_title_is_usable = (
        len(normalized_metadata_title) >= 20
        and not metadata_title.lower().startswith(("doi", "http", "www."))
        and not any(label in metadata_title.lower() for label in boilerplate)
    )
    if metadata_title_is_usable:
        metadata_matches = []
        for line in page_lines[0]:
            normalized_line = re.sub(r"\W+", "", line["text"]).lower()
            if (
                len(normalized_line) >= 20
                and not any(label in line["text"].lower() for label in boilerplate)
                and (normalized_line in normalized_metadata_title or normalized_metadata_title in normalized_line)
            ):
                metadata_matches.append(line)
        if metadata_matches:
            return max(metadata_matches, key=lambda line: len(line["text"]))
    candidates = [
        line
        for line in page_lines[0]
        if line["bbox"][1] < 300
        and 8 <= len(line["text"]) <= 240
        and line["font_size"] >= body_size * 1.18
        and not _SECTION_RE.match(line["text"])
        and not any(label in line["text"].lower() for label in boilerplate)
    ]
    if not candidates:
        candidates = [
            line
            for line in page_lines[0]
            if line["bbox"][1] < 220
            and 30 <= len(line["text"]) <= 240
            and line["text"].count(",") < 4
            and not _SECTION_RE.match(line["text"])
            and not line["text"].lower().startswith(("please cite", "this is a pdf", "http"))
            and not any(label in line["text"].lower() for label in boilerplate)
        ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda line: (
            line["font_size"] / max(body_size, 1),
            line["bold_ratio"],
            min(len(line["text"]), 120) / 120,
            -line["bbox"][1] / 1000,
        ),
    )


def _coalesce_wrapped_title(
    lines: list[dict[str, Any]],
    body_size: float,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge a title wrapped over adjacent lines before heading classification."""

    boilerplate = ("journal pre-proof", "accepted manuscript", "available online", "downloaded from")
    eligible = [
        line
        for line in sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0]))
        if line["bbox"][1] < 300
        and 8 <= len(line["text"]) <= 240
        and line["text"].count(",") < 4
        and not any(label in line["text"].lower() for label in boilerplate)
    ]
    groups: list[list[dict[str, Any]]] = []
    for line in eligible:
        if not groups:
            groups.append([line])
            continue
        previous = groups[-1][-1]
        vertical_gap = line["bbox"][1] - previous["bbox"][3]
        size_ratio = min(line["font_size"], previous["font_size"]) / max(
            line["font_size"], previous["font_size"], 1.0
        )
        if vertical_gap <= max(line["font_size"], previous["font_size"]) * 1.25 and size_ratio >= 0.84:
            groups[-1].append(line)
        else:
            groups.append([line])

    metadata_title = _clean_text(str(metadata.get("Title", metadata.get("title", ""))))
    normalized_metadata = re.sub(r"\W+", "", metadata_title).lower()
    scored: list[tuple[float, list[dict[str, Any]], str]] = []
    for group in groups:
        if not (2 <= len(group) <= 5):
            continue
        joined = _clean_text(" ".join(line["text"] for line in group))
        normalized_joined = re.sub(r"\W+", "", joined).lower()
        average_size = sum(line["font_size"] for line in group) / len(group)
        similarity = (
            SequenceMatcher(None, normalized_joined, normalized_metadata).ratio()
            if len(normalized_metadata) >= 20
            else 0.0
        )
        early_long_group = group[0]["bbox"][1] < 180 and len(joined) >= 50
        if similarity < 0.72 and average_size < body_size * 1.08 and not early_long_group:
            continue
        score = similarity * 4 + average_size / max(body_size, 1.0) + min(len(joined), 160) / 160
        scored.append((score, group, joined))
    if not scored:
        return lines

    _, group, joined = max(scored, key=lambda item: item[0])
    merged_runs: list[dict[str, Any]] = []
    for index, line in enumerate(group):
        _merge_runs(merged_runs, line["runs"], " " if index else "")
    merged = {
        "text": joined,
        "runs": merged_runs,
        "bbox": (
            min(line["bbox"][0] for line in group),
            min(line["bbox"][1] for line in group),
            max(line["bbox"][2] for line in group),
            max(line["bbox"][3] for line in group),
        ),
        "font_size": median(line["font_size"] for line in group),
        "bold_ratio": sum(line["bold_ratio"] for line in group) / len(group),
        "column": 0,
    }
    group_ids = {id(line) for line in group}
    first_index = min(index for index, line in enumerate(lines) if id(line) in group_ids)
    result = [line for line in lines if id(line) not in group_ids]
    result.insert(first_index, merged)
    return sorted(result, key=lambda line: (line["bbox"][1], line["bbox"][0]))


def _reader_metadata(reader: PdfReader) -> dict[str, Any]:
    metadata: dict[str, Any] = {"page_count": len(reader.pages)}
    raw = reader.metadata or {}
    for key, value in raw.items():
        clean_key = str(key).lstrip("/")
        try:
            metadata[clean_key] = str(value)
        except Exception:
            metadata[clean_key] = repr(value)
    return metadata


def _parse_pdf_native(
    pdf_path: str | Path,
    *,
    password: str | None = None,
    table_strategy: str = "auto",
    max_pages: int | None = None,
) -> Document:
    """Parse a PDF into a lightweight layout model."""

    source = Path(pdf_path)
    if not source.is_file():
        raise FileNotFoundError(source)

    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    reader: PdfReader | None = None

    def get_pypdf_reader() -> PdfReader:
        nonlocal reader
        if reader is None:
            reader = PdfReader(str(source))
            if reader.is_encrypted and not reader.decrypt(password or ""):
                raise ValueError(f"Unable to decrypt PDF: {source}")
            for key, value in _reader_metadata(reader).items():
                metadata.setdefault(key, value)
        return reader

    raw_pages: list[dict[str, Any]] = []
    all_sizes: list[float] = []
    visible_characters = 0
    with pdfplumber.open(str(source), password=password) as plumber_pdf:
        metadata["page_count"] = len(plumber_pdf.pages)
        for key, value in (plumber_pdf.metadata or {}).items():
            metadata[str(key).lstrip("/")] = str(value)
        page_limit = min(len(plumber_pdf.pages), max_pages) if max_pages else len(plumber_pdf.pages)
        for index in range(page_limit):
            plumber_page = plumber_pdf.pages[index]
            tables = _extract_tables(plumber_page, strategy=table_strategy) if table_strategy != "none" else []
            chars = [
                char
                for char in plumber_page.chars
                if char.get("upright", True)
                and not any(_inside_bbox(char, table.bbox) for table in tables)
            ]
            lines = _chars_to_lines(
                chars,
                float(plumber_page.width),
                protect_wide_headings=index == 0,
            )
            if not lines:
                fallback_reader = get_pypdf_reader()
                fallback = _clean_text(fallback_reader.pages[index].extract_text() or "")
                if fallback:
                    warnings.append(f"page {index + 1}: used pypdf text fallback")
                    line_height = max(float(plumber_page.height) / max(fallback.count("\n") + 1, 1), 10.0)
                    lines = []
                    for line_index, text in enumerate(fallback.splitlines()):
                        text = _clean_text(text)
                        if text:
                            lines.append(
                                {
                                    "text": text,
                                    "runs": [{"text": text, "bold": False}],
                                    "bbox": (36.0, line_index * line_height, float(plumber_page.width) - 36.0, (line_index + 1) * line_height),
                                    "font_size": 10.0,
                                    "bold_ratio": 0.0,
                                    "column": 0,
                                }
                            )
            all_sizes.extend(
                line["font_size"]
                for line in lines
                if line["font_size"] > 0 and len(line["text"]) >= 12
            )
            visible_characters += sum(len(line["text"]) for line in lines)
            raw_pages.append(
                {
                    "width": float(plumber_page.width),
                    "height": float(plumber_page.height),
                    "lines": lines,
                    "tables": tables,
                }
            )

    body_size = median(all_sizes) if all_sizes else 10.0
    repeated = _find_repeated_margins([(page["height"], page["lines"]) for page in raw_pages])
    filtered_page_lines: list[list[dict[str, Any]]] = []
    for page in raw_pages:
        filtered_page_lines.append(
            [
                line
                for line in page["lines"]
                if _canonical_margin_text(line["text"]) not in repeated
            ]
        )
    if filtered_page_lines:
        filtered_page_lines[0] = _coalesce_wrapped_title(
            filtered_page_lines[0],
            body_size,
            metadata,
        )
    title_line = _choose_title(filtered_page_lines, body_size, metadata)

    pages: list[Page] = []
    for index, page in enumerate(raw_pages):
        lines = filtered_page_lines[index]
        ordered = _order_items(lines, page["tables"], page["width"])
        blocks = _build_blocks(ordered, body_size, title_line)
        pages.append(Page(number=index + 1, width=page["width"], height=page["height"], blocks=blocks))

    status = "ocr_required" if visible_characters < max(50, page_limit * 10) else "ok"
    if status == "ocr_required":
        warnings.append("The PDF has little or no extractable text; OCR is required.")
    metadata["body_font_size"] = round(body_size, 2)
    metadata["removed_repeated_margins"] = sorted(repeated)
    return Document(source=str(source), metadata=metadata, pages=pages, status=status, warnings=warnings)


def _base_opendoc_label(label: str) -> str:
    value = (label or "text").strip().lower()
    suffix = value.rsplit("_", 1)
    return suffix[0] if len(suffix) == 2 and suffix[1].isdigit() else value


def _opendoc_bbox(value: Any) -> BBox:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(float(item) for item in value)
    return (0.0, 0.0, 0.0, 0.0)


def _document_from_opendoc_results(
    source: Path,
    results: dict[str, Any] | list[dict[str, Any]],
) -> Document:
    """Convert OpenDoc recognition results to the experiment's lightweight model."""

    page_results = results if isinstance(results, list) else [results]
    pages: list[Page] = []
    warnings: list[str] = []
    ignored_labels = {
        "number",
        "footnote",
        "header",
        "header_image",
        "footer",
        "footer_image",
        "aside_text",
        "chart",
    }
    recognized_characters = 0
    timings: list[dict[str, Any]] = []

    for index, result in enumerate(page_results):
        blocks: list[Block] = []
        timings.append(dict(result.get("timing") or {}))

        def append_heading(text: str, bbox: BBox, level: int, confidence: float) -> None:
            clean_text = _clean_text(text)
            if (
                blocks
                and blocks[-1].kind == "heading"
                and _clean_text(blocks[-1].text).casefold() == clean_text.casefold()
            ):
                return
            blocks.append(
                Block(
                    kind="heading",
                    text=clean_text,
                    bbox=bbox,
                    level=level,
                    confidence=confidence,
                )
            )

        for recognition in result.get("recognition_results", []):
            label = _base_opendoc_label(str(recognition.get("label", "text")))
            if label in ignored_labels or recognition.get("is_image"):
                continue
            text = str(recognition.get("text_unirec") or recognition.get("text") or "").strip()
            if not text:
                continue
            bbox = _opendoc_bbox(recognition.get("bbox"))
            confidence = float(recognition.get("score", 1.0) or 0.0)
            recognized_characters += len(text)

            if label == "doc_title":
                append_heading(text, bbox, 1, confidence)
            elif label in {"paragraph_title", "reference"}:
                append_heading(text, bbox, 2, confidence)
            elif label == "abstract":
                abstract_text = re.sub(r"^\s*abstract\s*[:.-]?\s*", "", text, flags=re.IGNORECASE)
                append_heading("Abstract", bbox, 2, confidence)
                if abstract_text:
                    blocks.append(
                        Block(
                            kind="raw_markdown",
                            text=abstract_text,
                            bbox=bbox,
                            confidence=confidence,
                        )
                    )
            else:
                blocks.append(
                    Block(
                        kind="raw_markdown",
                        text=text,
                        bbox=bbox,
                        confidence=confidence,
                    )
                )

        pages.append(
            Page(
                number=int(result.get("pdf_page", index + 1)),
                width=float(result.get("width", 0.0) or 0.0),
                height=float(result.get("height", 0.0) or 0.0),
                blocks=blocks,
            )
        )

    status = "ok" if recognized_characters else "ocr_failed"
    if status == "ocr_failed":
        warnings.append("OpenDoc completed but returned no recognized text.")
    return Document(
        source=str(source),
        metadata={
            "backend": "opendoc",
            "page_count": len(pages),
            "opendoc_timing": timings,
        },
        pages=pages,
        status=status,
        warnings=warnings,
    )


def _parse_pdf_opendoc(
    pdf_path: str | Path,
    *,
    model_dir: str | Path | None = None,
    use_gpu: str = "auto",
    auto_download: bool = True,
    max_length: int = 2048,
    max_parallel_blocks: int = 2,
    max_pages: int | None = None,
) -> Document:
    source = Path(pdf_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        from openocr import OpenOCR
    except ImportError as exc:
        raise RuntimeError(
            "OpenDoc requires openocr-python==0.1.5. Install the project requirements first."
        ) from exc

    base_dir = (
        Path(model_dir)
        if model_dir is not None
        else Path(__file__).resolve().parent / "models" / "opendoc"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    gpu_value: bool | None
    if use_gpu == "auto":
        gpu_value = None
    elif use_gpu == "true":
        gpu_value = True
    elif use_gpu == "false":
        gpu_value = False
    else:
        raise ValueError("use_gpu must be one of: auto, true, false")

    parser = OpenOCR(
        task="doc",
        layout_model_path=str(base_dir / "PP-DocLayoutV2.onnx"),
        unirec_encoder_path=str(base_dir / "unirec_encoder.onnx"),
        unirec_decoder_path=str(base_dir / "unirec_decoder.onnx"),
        tokenizer_mapping_path=str(base_dir / "unirec_tokenizer_mapping.json"),
        use_gpu=gpu_value,
        use_layout_detection=True,
        use_chart_recognition=False,
        auto_download=auto_download,
        max_parallel_blocks=max(1, max_parallel_blocks),
    )
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    input_path = source
    try:
        if source.suffix.lower() == ".pdf" and max_pages is not None:
            import fitz

            source_pdf = fitz.open(str(source))
            if source_pdf.page_count > max_pages:
                temporary_directory = tempfile.TemporaryDirectory(prefix="opendoc_pdf_")
                limited_path = Path(temporary_directory.name) / source.name
                limited_pdf = fitz.open()
                limited_pdf.insert_pdf(
                    source_pdf,
                    from_page=0,
                    to_page=max(0, max_pages - 1),
                )
                limited_pdf.save(str(limited_path))
                limited_pdf.close()
                input_path = limited_path
            source_pdf.close()
        results = parser(
            image_path=str(input_path),
            max_length=max_length,
            merge_layout_blocks=True,
        )
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()
    return _document_from_opendoc_results(source, results)


def parse_pdf(
    pdf_path: str | Path,
    *,
    password: str | None = None,
    table_strategy: str = "auto",
    max_pages: int | None = None,
    backend: str = "auto",
    opendoc_model_dir: str | Path | None = None,
    opendoc_use_gpu: str = "auto",
    opendoc_auto_download: bool = True,
    opendoc_max_length: int = 2048,
    opendoc_max_parallel_blocks: int = 2,
) -> Document:
    """Parse a PDF with native extraction, OpenDoc, or automatic OCR fallback."""

    if backend not in {"native", "auto", "opendoc"}:
        raise ValueError("backend must be one of: native, auto, opendoc")
    opendoc_options = {
        "model_dir": opendoc_model_dir,
        "use_gpu": opendoc_use_gpu,
        "auto_download": opendoc_auto_download,
        "max_length": opendoc_max_length,
        "max_parallel_blocks": opendoc_max_parallel_blocks,
        "max_pages": max_pages,
    }
    if backend == "opendoc":
        return _parse_pdf_opendoc(pdf_path, **opendoc_options)

    native_document = _parse_pdf_native(
        pdf_path,
        password=password,
        table_strategy=table_strategy,
        max_pages=max_pages,
    )
    native_document.metadata.setdefault("backend", "native")
    if backend == "native" or native_document.status != "ocr_required":
        return native_document

    try:
        opendoc_document = _parse_pdf_opendoc(pdf_path, **opendoc_options)
        opendoc_document.warnings.insert(
            0,
            "Native extraction found no usable text; OpenDoc OCR fallback was used.",
        )
        return opendoc_document
    except Exception as exc:
        native_document.warnings.append(
            f"OpenDoc OCR fallback failed: {type(exc).__name__}: {exc}"
        )
        return native_document


def convert_pdf(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    *,
    debug_json_path: str | Path | None = None,
    password: str | None = None,
    table_strategy: str = "auto",
    max_pages: int | None = None,
    backend: str = "auto",
    opendoc_model_dir: str | Path | None = None,
    opendoc_use_gpu: str = "auto",
    opendoc_auto_download: bool = True,
    opendoc_max_length: int = 2048,
    opendoc_max_parallel_blocks: int = 2,
) -> tuple[str, Document]:
    """Parse one PDF, render Markdown, and optionally write output artifacts."""

    document = parse_pdf(
        pdf_path,
        password=password,
        table_strategy=table_strategy,
        max_pages=max_pages,
        backend=backend,
        opendoc_model_dir=opendoc_model_dir,
        opendoc_use_gpu=opendoc_use_gpu,
        opendoc_auto_download=opendoc_auto_download,
        opendoc_max_length=opendoc_max_length,
        opendoc_max_parallel_blocks=opendoc_max_parallel_blocks,
    )
    try:
        from .markdown_renderer import render_document
    except ImportError:
        from markdown_renderer import render_document

    markdown = render_document(document)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown, encoding="utf-8")
    if debug_json_path is not None:
        debug_destination = Path(debug_json_path)
        debug_destination.parent.mkdir(parents=True, exist_ok=True)
        debug_destination.write_text(
            json.dumps(document.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return markdown, document


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Convert PDF files to layout-aware Markdown.")
    parser.add_argument("input", type=Path, help="PDF file or a directory containing PDF files")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--debug-json", action="store_true", help="Write intermediate document JSON")
    parser.add_argument("--table-strategy", choices=("auto", "strict", "text", "none"), default="auto")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument(
        "--backend",
        choices=("auto", "native", "opendoc"),
        default="auto",
        help="auto uses OpenDoc only when native extraction requires OCR",
    )
    parser.add_argument(
        "--opendoc-model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "opendoc",
    )
    parser.add_argument(
        "--opendoc-use-gpu",
        choices=("auto", "true", "false"),
        default="auto",
    )
    parser.add_argument("--opendoc-no-auto-download", action="store_true")
    parser.add_argument("--opendoc-max-length", type=int, default=2048)
    parser.add_argument("--opendoc-max-parallel-blocks", type=int, default=2)
    args = parser.parse_args()

    inputs = sorted(args.input.glob("*.pdf")) if args.input.is_dir() else [args.input]
    if not inputs:
        parser.error(f"No PDF files found under {args.input}")
    failed = 0
    for pdf_path in inputs:
        try:
            output_path = args.output / f"{pdf_path.stem}.md"
            debug_path = args.output / f"{pdf_path.stem}.debug.json" if args.debug_json else None
            _, document = convert_pdf(
                pdf_path,
                output_path,
                debug_json_path=debug_path,
                table_strategy=args.table_strategy,
                max_pages=args.max_pages,
                backend=args.backend,
                opendoc_model_dir=args.opendoc_model_dir,
                opendoc_use_gpu=args.opendoc_use_gpu,
                opendoc_auto_download=not args.opendoc_no_auto_download,
                opendoc_max_length=args.opendoc_max_length,
                opendoc_max_parallel_blocks=args.opendoc_max_parallel_blocks,
            )
            print(f"{pdf_path.name}: {document.status} -> {output_path}")
        except Exception as exc:
            failed += 1
            print(f"{pdf_path.name}: ERROR: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
