"""Compact, layout-aware PDF to Markdown experiment.

The module intentionally keeps extraction, normalization, and layout heuristics
in one place.  The hybrid backend uses PDFium for rendering and native character
geometry, OpenDoc for layout and selective recognition, and pypdf for metadata.
The explicit native backend retains pdfplumber extraction for comparison.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
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
    source: str = ""
    label: str = ""


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
_LAYOUT_IGNORED_LABELS = {
    "aside_text",
    "chart",
    "footer",
    "footer_image",
    "footnote",
    "header",
    "header_image",
    "image",
    "number",
    "seal",
    "vision_footnote",
}
_LAYOUT_FORMULA_LABELS = {"display_formula", "inline_formula", "formula_number"}
_LAYOUT_HEADING_LABELS = {"doc_title", "paragraph_title", "reference"}
_LAYOUT_DETECTOR_CACHE: dict[tuple[str, str], Any] = {}
_SELECTIVE_RECOGNIZER_CACHE: dict[tuple[str, str, int], Any] = {}


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
                source="native-table",
                label="table",
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

    def append_piece(piece: str, bold: bool, script: str | None = None) -> None:
        if not piece:
            return
        if (
            runs
            and runs[-1].get("bold") == bold
            and runs[-1].get("script") == script
        ):
            runs[-1]["text"] += piece
        else:
            runs.append({"text": piece, "bold": bold, "script": script})
        parts.append(piece)

    def trim_space_before_script() -> None:
        while parts and parts[-1].isspace():
            parts.pop()
        if parts:
            parts[-1] = parts[-1].rstrip()
            if not parts[-1]:
                parts.pop()
        while runs and not str(runs[-1].get("text", "")).rstrip():
            runs.pop()
        if runs:
            runs[-1]["text"] = str(runs[-1].get("text", "")).rstrip()
            if not runs[-1]["text"]:
                runs.pop()

    for char in ordered:
        value = str(char.get("text", ""))
        if not value:
            continue
        x0 = float(char.get("x0", 0.0))
        x1 = float(char.get("x1", x0))
        size = float(char.get("size", previous_size) or previous_size)
        bold = _is_bold(str(char.get("fontname", "")))
        script = str(char.get("script", "") or "") or None
        if script is not None:
            trim_space_before_script()
        if previous_x1 is not None and not value.isspace() and script is None:
            gap = x0 - previous_x1
            if gap > max(1.2, min(previous_size, size) * 0.22) and (not parts or not parts[-1].endswith(" ")):
                append_piece(" ", False, None)
        append_piece(value, bold, script)
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
            cleaned_runs.append(
                {
                    "text": run_text,
                    "bold": bool(run["bold"]),
                    "script": run.get("script"),
                }
            )
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


def _attach_script_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach small vertically shifted glyph groups to their dominant text line."""

    def metrics(group: dict[str, Any]) -> dict[str, float] | None:
        visible = [
            char
            for char in group["chars"]
            if str(char.get("text", "")).strip()
        ]
        if not visible:
            return None
        sizes = [float(char.get("size", 0.0) or 0.0) for char in visible]
        tops = [float(char.get("top", 0.0)) for char in visible]
        bottoms = [float(char.get("bottom", 0.0)) for char in visible]
        return {
            "size": median(sizes),
            "top": median(tops),
            "bottom": median(bottoms),
            "center": median(
                (float(char.get("top", 0.0)) + float(char.get("bottom", 0.0)))
                / 2
                for char in visible
            ),
            "x0": min(float(char.get("x0", 0.0)) for char in visible),
            "x1": max(float(char.get("x1", 0.0)) for char in visible),
        }

    group_metrics = [metrics(group) for group in groups]
    assignments: dict[int, list[tuple[int, str | None]]] = defaultdict(list)
    attached: set[int] = set()
    for index, source in enumerate(group_metrics):
        if source is None or source["size"] <= 0:
            continue
        candidates: list[tuple[float, int, dict[str, float]]] = []
        source_height = max(source["bottom"] - source["top"], 0.1)
        for target_index, target in enumerate(group_metrics):
            if target_index == index or target is None:
                continue
            size_ratio = source["size"] / max(target["size"], 0.1)
            if not 0.45 <= size_ratio <= 0.86:
                continue
            target_height = max(target["bottom"] - target["top"], 0.1)
            vertical_overlap = max(
                0.0,
                min(source["bottom"], target["bottom"])
                - max(source["top"], target["top"]),
            )
            if vertical_overlap / min(source_height, target_height) < 0.35:
                continue
            center_delta = abs(source["center"] - target["center"])
            if center_delta > target["size"] * 0.58:
                continue
            horizontal_gap = max(
                target["x0"] - source["x1"],
                source["x0"] - target["x1"],
                0.0,
            )
            if horizontal_gap > target["size"] * 1.5:
                continue
            candidates.append((center_delta, target_index, target))
        if not candidates:
            continue
        _, target_index, target = min(candidates, key=lambda item: item[0])
        offset = source["center"] - target["center"]
        if (
            offset < -target["size"] * 0.10
            and source["bottom"] < target["bottom"] - target["size"] * 0.08
        ):
            script: str | None = "sup"
        elif (
            offset > target["size"] * 0.10
            and source["bottom"] > target["bottom"] + target["size"] * 0.04
        ):
            script = "sub"
        else:
            script = None
        assignments[target_index].append((index, script))
        attached.add(index)

    merged: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if index in attached:
            continue
        chars = list(group["chars"])
        for source_index, script in assignments.get(index, []):
            chars.extend(
                {
                    **char,
                    "script": script
                    if str(char.get("text", "")).strip()
                    else None,
                }
                for char in groups[source_index]["chars"]
            )
        merged.append({**group, "chars": chars})
    return merged


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

    groups = _attach_script_groups(groups)
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
            current.append({"text": separator, "bold": False, "script": None})
    for run in incoming:
        if (
            current
            and current[-1].get("bold") == run.get("bold")
            and current[-1].get("script") == run.get("script")
        ):
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
                source="native",
                label="text",
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
                    source="native",
                    label="paragraph_title",
                )
            )
            blocks.append(
                Block(
                    kind="paragraph",
                    text=content,
                    runs=[{"text": content, "bold": False}],
                    bbox=item["bbox"],
                    font_size=item["font_size"],
                    source="native",
                    label="abstract",
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
                    source="native",
                    label="doc_title" if level == 1 else "paragraph_title",
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
                    source="native",
                    label="text",
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


def _opendoc_gpu_value(use_gpu: str) -> bool | None:
    if use_gpu == "auto":
        return None
    if use_gpu == "true":
        return True
    if use_gpu == "false":
        return False
    raise ValueError("use_gpu must be one of: auto, true, false")


def _opendoc_model_base(model_dir: str | Path | None) -> Path:
    base_dir = (
        Path(model_dir)
        if model_dir is not None
        else Path(__file__).resolve().parent / "models" / "opendoc"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _ensure_utf8_console_output() -> None:
    """Prevent OpenDoc's Unicode progress messages from failing on Windows GBK."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        encoding = str(getattr(stream, "encoding", "") or "").lower()
        if reconfigure is not None and encoding not in {"utf-8", "utf8"}:
            reconfigure(encoding="utf-8", errors="replace")


def _prepare_opendoc_gpu_runtime(use_gpu: str) -> None:
    """Preload Torch's CUDA DLLs before ONNX Runtime creates GPU sessions."""

    if use_gpu == "false":
        return
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        if use_gpu == "true":
            raise RuntimeError(
                "GPU OpenDoc requires a CUDA-enabled torch installation so its "
                "CUDA and cuDNN DLLs can be loaded on Windows."
            ) from exc


def _require_cuda_session(session: Any, component: str, use_gpu: str) -> None:
    """Reject ONNX Runtime's otherwise silent CPU fallback for explicit GPU runs."""

    if use_gpu != "true":
        return
    providers = list(session.get_providers())
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            f"{component} requested GPU but created providers {providers}. "
            "Check onnxruntime-gpu and CUDA/cuDNN DLL availability."
        )


def _get_opendoc_layout_detector(
    model_dir: str | Path | None,
    use_gpu: str,
    auto_download: bool,
) -> Any:
    """Load and cache PP-DocLayoutV2 without loading UniRec."""

    _ensure_utf8_console_output()
    _prepare_opendoc_gpu_runtime(use_gpu)
    try:
        from openocr.tools.infer_doc_onnx import LayoutDetectorONNX
    except ImportError as exc:
        raise RuntimeError(
            "Hybrid parsing requires openocr-python==0.1.5."
        ) from exc

    base_dir = _opendoc_model_base(model_dir)
    cache_key = (str(base_dir.resolve()), use_gpu)
    detector = _LAYOUT_DETECTOR_CACHE.get(cache_key)
    if detector is None:
        detector = LayoutDetectorONNX(
            str(base_dir / "PP-DocLayoutV2.onnx"),
            use_gpu=_opendoc_gpu_value(use_gpu),
            threshold=0.5,
            auto_download=auto_download,
        )
        _require_cuda_session(detector.session, "OpenDoc layout", use_gpu)
        _LAYOUT_DETECTOR_CACHE[cache_key] = detector
    return detector


def _get_selective_opendoc_recognizer(
    model_dir: str | Path | None,
    use_gpu: str,
    auto_download: bool,
    max_parallel_blocks: int,
) -> Any:
    """Load and cache UniRec without creating a second layout detector."""

    _ensure_utf8_console_output()
    _prepare_opendoc_gpu_runtime(use_gpu)
    try:
        from openocr.tools.infer_doc_onnx import OpenDocONNX
    except ImportError as exc:
        raise RuntimeError(
            "Selective recognition requires openocr-python==0.1.5."
        ) from exc

    base_dir = _opendoc_model_base(model_dir)
    workers = max(1, max_parallel_blocks)
    cache_key = (str(base_dir.resolve()), use_gpu, workers)
    recognizer = _SELECTIVE_RECOGNIZER_CACHE.get(cache_key)
    if recognizer is None:
        recognizer = OpenDocONNX(
            unirec_encoder_path=str(base_dir / "unirec_encoder.onnx"),
            unirec_decoder_path=str(base_dir / "unirec_decoder.onnx"),
            tokenizer_mapping_path=str(base_dir / "unirec_tokenizer_mapping.json"),
            use_gpu=_opendoc_gpu_value(use_gpu),
            use_layout_detection=False,
            use_chart_recognition=False,
            auto_download=auto_download,
            max_parallel_blocks=workers,
        )
        _require_cuda_session(
            recognizer.vlm_recognizer.encoder_session,
            "UniRec encoder",
            use_gpu,
        )
        _require_cuda_session(
            recognizer.vlm_recognizer.decoder_session,
            "UniRec decoder",
            use_gpu,
        )
        _SELECTIVE_RECOGNIZER_CACHE[cache_key] = recognizer
    return recognizer


def _render_pdfium_page(page: Any) -> Any:
    """Render one PDFium page as the BGR image expected by OpenDoc."""

    import cv2
    import numpy as np

    width, height = page.get_size()
    scale = 2.0
    if width * scale > 2000 or height * scale > 2000:
        scale = 1.0
    bitmap = page.render(scale=scale)
    try:
        image = bitmap.to_pil().convert("RGB")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    finally:
        bitmap.close()


def _pdfium_font_name(text_page: Any, index: int) -> str:
    import pypdfium2.raw as pdfium_c

    flags = ctypes.c_int()
    size = pdfium_c.FPDFText_GetFontInfo(
        text_page,
        index,
        None,
        0,
        ctypes.byref(flags),
    )
    if size <= 1:
        return ""
    buffer = ctypes.create_string_buffer(size)
    pdfium_c.FPDFText_GetFontInfo(
        text_page,
        index,
        buffer,
        size,
        ctypes.byref(flags),
    )
    return buffer.value.decode("utf-8", errors="replace")


def _pdfium_page_chars(
    page: Any,
    *,
    include_rotated: bool = False,
) -> list[dict[str, Any]]:
    """Extract only native text geometry, without parsing vector drawings."""

    import pypdfium2.raw as pdfium_c

    page_height = float(page.get_height())
    text_page = page.get_textpage()
    chars: list[dict[str, Any]] = []
    try:
        for index in range(text_page.count_chars()):
            codepoint = int(pdfium_c.FPDFText_GetUnicode(text_page, index))
            if not codepoint:
                continue
            try:
                text = chr(codepoint)
            except ValueError:
                continue
            if text in {"\r", "\n", "\f", "\v"}:
                continue
            if text.isspace():
                text = " "
            elif not text.isprintable():
                continue
            try:
                left, bottom, right, top = text_page.get_charbox(index)
                (
                    loose_left,
                    loose_bottom,
                    loose_right,
                    loose_top,
                ) = text_page.get_charbox(
                    index,
                    loose=True,
                )
            except Exception:
                continue
            font_name = _pdfium_font_name(text_page, index)
            font_weight = int(pdfium_c.FPDFText_GetFontWeight(text_page, index))
            if font_weight >= 600 and not _is_bold(font_name):
                font_name = f"{font_name}-Bold"
            angle = float(pdfium_c.FPDFText_GetCharAngle(text_page, index))
            signed_angle = (angle + math.pi) % (2 * math.pi) - math.pi
            normalized_angle = abs(signed_angle)
            raw_font_size = float(
                pdfium_c.FPDFText_GetFontSize(text_page, index)
            )
            chars.append(
                {
                    "text": text,
                    "x0": float(min(loose_left, loose_right)),
                    "x1": float(max(loose_left, loose_right)),
                    "top": page_height - float(max(loose_bottom, loose_top)),
                    "bottom": page_height
                    - float(min(loose_bottom, loose_top)),
                    "size": max(
                        raw_font_size,
                        abs(float(loose_top) - float(loose_bottom)),
                    ),
                    "fontname": font_name,
                    "upright": min(normalized_angle, abs(math.pi - normalized_angle))
                    < 0.10,
                    "angle": signed_angle,
                }
            )
    finally:
        text_page.close()
    if include_rotated:
        return [
            char
            for char in chars
            if str(char.get("text", "")).replace("\x00", "")
        ]
    return _layout_source_chars(chars)


def _run_opendoc_layout(detector: Any, image: Any) -> dict[str, Any]:
    """Run layout with an exact 800x800 tensor.

    openocr-python 0.1.5 calculates the resize dimensions through floating-point
    multiplication and can truncate one side to 799 for some aspect ratios.
    PP-DocLayoutV2 has a fixed 800x800 input, so normalize the tensor here.
    """

    import cv2
    import numpy as np

    input_dict, scale, original_height, original_width = detector.preprocess(image)
    if tuple(input_dict["image"].shape[-2:]) != (800, 800):
        resized = cv2.resize(image, (800, 800), interpolation=cv2.INTER_LINEAR)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_dict = {
            "im_shape": np.array([[800, 800]], dtype=np.float32),
            "image": (
                resized_rgb.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
                / 255.0
            ),
            "scale_factor": np.array(
                [[800 / original_height, 800 / original_width]],
                dtype=np.float32,
            ),
        }
        scale = (800 / original_height, 800 / original_width)
    outputs = detector.session.run(detector.output_names, input_dict)
    return detector.postprocess(
        image,
        outputs,
        scale,
        original_height,
        original_width,
    )


def _layout_bbox_to_pdf(
    bbox: BBox,
    *,
    page_width: float,
    page_height: float,
    image_width: float,
    image_height: float,
) -> BBox:
    """Convert OpenDoc raster coordinates to pdfplumber page coordinates."""

    scale_x = page_width / max(image_width, 1.0)
    scale_y = page_height / max(image_height, 1.0)
    return (
        max(0.0, min(page_width, bbox[0] * scale_x)),
        max(0.0, min(page_height, bbox[1] * scale_y)),
        max(0.0, min(page_width, bbox[2] * scale_x)),
        max(0.0, min(page_height, bbox[3] * scale_y)),
    )


def _bbox_intersection_ratio(first: BBox, second: BBox) -> float:
    """Intersection divided by the smaller box area."""

    x0, top = max(first[0], second[0]), max(first[1], second[1])
    x1, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(min(first_area, second_area), 1.0)


def _visible_character_count(chars: Iterable[dict[str, Any]]) -> int:
    return sum(
        sum(not character.isspace() for character in str(char.get("text", "")))
        for char in chars
    )


def _layout_source_chars(chars: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep explicit PDF whitespace for layout-region text reconstruction."""

    usable: list[dict[str, Any]] = []
    for char in chars:
        if not char.get("upright", True):
            continue
        raw_text = str(char.get("text", ""))
        text = raw_text.replace("\x00", "")
        if not text:
            continue
        if text != raw_text:
            char = {**char, "text": text}
        usable.append(char)
    return usable


def _merge_layout_lines(lines: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Merge native lines inside one OpenDoc region while preserving bold runs."""

    if not lines:
        return None
    text = ""
    runs: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
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
    total_chars = sum(max(len(line["text"]), 1) for line in lines)
    return {
        "text": _clean_text(text),
        "runs": runs,
        "font_size": median(line["font_size"] for line in lines),
        "bold_ratio": sum(
            line["bold_ratio"] * max(len(line["text"]), 1) for line in lines
        )
        / max(total_chars, 1),
    }


def _layout_heading_level(text: str, label: str, body_size: float, merged: dict[str, Any]) -> int:
    if label == "doc_title":
        return 1
    probe = {
        "text": text,
        "runs": merged.get("runs", []),
        "bbox": (0.0, 0.0, 0.0, 0.0),
        "font_size": float(merged.get("font_size", body_size) or body_size),
        "bold_ratio": float(merged.get("bold_ratio", 0.0) or 0.0),
    }
    return _heading_level(probe, body_size) or 2


def _normalize_layout_heading(text: str) -> str:
    """Collapse letter-spaced journal headings without altering normal titles."""

    value = _clean_text(text)
    compact = re.sub(r"[^a-z]", "", value.casefold())
    known = {
        "abstract": "Abstract",
        "articleinfo": "ARTICLE INFO",
        "acknowledgments": "Acknowledgments",
        "acknowledgements": "Acknowledgements",
        "references": "References",
    }
    single_letter_tokens = [
        token for token in re.split(r"\s+", value) if token and token.isalpha()
    ]
    if compact in known and len(single_letter_tokens) >= 3 and all(
        len(token) == 1 for token in single_letter_tokens
    ):
        return known[compact]
    return value


def _native_blocks_for_layout_region(
    *,
    label: str,
    bbox: BBox,
    chars: list[dict[str, Any]],
    page_width: float,
    body_size: float,
    confidence: float,
    source: str = "layout-native",
) -> list[Block]:
    lines = _chars_to_lines(chars, page_width)
    merged = _merge_layout_lines(lines)
    if not merged or not merged["text"]:
        return []

    text = (
        _normalize_layout_heading(merged["text"])
        if label in _LAYOUT_HEADING_LABELS
        else merged["text"]
    )
    heading_runs = merged["runs"] if text == merged["text"] else []
    common = {
        "bbox": bbox,
        "font_size": merged["font_size"],
        "bold_ratio": merged["bold_ratio"],
        "confidence": confidence,
        "source": source,
        "label": label,
    }
    if label in _LAYOUT_HEADING_LABELS:
        return [
            Block(
                kind="heading",
                text=text,
                runs=heading_runs,
                level=_layout_heading_level(text, label, body_size, merged),
                **common,
            )
        ]
    if label == "abstract":
        abstract_text = re.sub(
            r"^\s*abstract\s*[:.-]?\s*", "", text, flags=re.IGNORECASE
        )
        blocks = [
            Block(
                kind="heading",
                text="Abstract",
                level=2,
                **common,
            )
        ]
        if abstract_text:
            blocks.append(
                Block(
                    kind="paragraph",
                    text=abstract_text,
                    runs=merged["runs"] if abstract_text == text else [],
                    **common,
                )
            )
        return blocks
    kind = "list" if _LIST_RE.match(text) else "paragraph"
    return [
        Block(
            kind=kind,
            text=text,
            runs=merged["runs"],
            **common,
        )
    ]


def _recognized_blocks_for_layout_region(
    *,
    label: str,
    bbox: BBox,
    text: str,
    confidence: float,
) -> list[Block]:
    value = text.strip()
    if not value:
        return []
    common = {
        "bbox": bbox,
        "confidence": confidence,
        "source": "layout-unirec",
        "label": label,
    }
    if label == "doc_title":
        return [Block(kind="heading", text=_clean_text(value), level=1, **common)]
    if label in {"paragraph_title", "reference"}:
        return [Block(kind="heading", text=_clean_text(value), level=2, **common)]
    if label == "abstract":
        abstract_text = re.sub(
            r"^\s*abstract\s*[:.-]?\s*", "", value, flags=re.IGNORECASE
        )
        blocks = [Block(kind="heading", text="Abstract", level=2, **common)]
        if abstract_text:
            blocks.append(Block(kind="raw_markdown", text=abstract_text, **common))
        return blocks
    return [Block(kind="raw_markdown", text=value, **common)]


def _layout_native_needs_recognition(label: str, blocks: list[Block]) -> bool:
    """OCR critical headings when their native font mapping is visibly corrupt."""

    if label not in _LAYOUT_HEADING_LABELS or not blocks:
        return False
    text = " ".join(block.text for block in blocks)
    chemical_dash_as_e = re.search(r"\b[A-Z][a-z]?e[A-Z][a-z]?\b", text)
    return "(cid:" in text or chemical_dash_as_e is not None


def _layout_region_needs_recognition(
    label: str,
    native_fallback: list[Block],
) -> bool:
    """Route every table and formula through UniRec, regardless of native text."""

    return (
        label == "table"
        or label in _LAYOUT_FORMULA_LABELS
        or not native_fallback
        or _layout_native_needs_recognition(label, native_fallback)
    )


def _native_table_is_plausible(
    table: Block,
    *,
    page_width: float,
    page_height: float,
    chart_overlap: float,
) -> bool:
    rows = table.rows
    row_count = len(rows)
    column_count = max((len(row) for row in rows), default=0)
    nonempty = [cell for row in rows for cell in row if cell.strip()]
    if row_count < 2 or column_count < 2 or len(nonempty) < 4:
        return False
    average_length = sum(len(cell) for cell in nonempty) / len(nonempty)
    numeric_ratio = sum(bool(re.search(r"\d", cell)) for cell in nonempty) / len(nonempty)
    height_ratio = (table.bbox[3] - table.bbox[1]) / max(page_height, 1.0)
    width_ratio = (table.bbox[2] - table.bbox[0]) / max(page_width, 1.0)
    strong_grid = row_count >= 3 and column_count >= 3 and numeric_ratio >= 0.20
    if height_ratio > 0.30 and average_length > 24:
        return False
    if row_count >= 8 and average_length > 30:
        return False
    if width_ratio > 0.88 and height_ratio > 0.22 and average_length > 28:
        return False
    if chart_overlap > 0.25 and not strong_grid:
        return False
    return strong_grid or (
        table.confidence >= 0.70
        and average_length <= 24
        and (numeric_ratio >= 0.15 or row_count <= 6)
    )


def _region_insertion_order(regions: list[dict[str, Any]], bbox: BBox) -> float:
    for region in regions:
        region_bbox = region["bbox"]
        if region_bbox[1] >= bbox[1] - 1.0:
            return float(region["order"]) - 0.25
    return float(len(regions)) + 0.25


def _crop_layout_region(image: Any, bbox: BBox, label: str) -> Any | None:
    x0, top, x1, bottom = (int(round(value)) for value in bbox)
    height, width = image.shape[:2]
    x0, x1 = max(0, x0), min(width, x1)
    top, bottom = max(0, top), min(height, bottom)
    if x1 <= x0 or bottom <= top:
        return None
    cropped = image[top:bottom, x0:x1]
    if cropped.size == 0:
        return None
    if label in _LAYOUT_FORMULA_LABELS and label != "formula_number":
        try:
            from openocr.tools.infer_doc_onnx import crop_margin

            cropped = crop_margin(cropped)
        except Exception:
            pass
    return cropped


def _table_crop_rotation(chars: Sequence[dict[str, Any]]) -> int:
    """Return the quarter-turn needed to make predominantly rotated text upright."""

    visible = [
        char
        for char in chars
        if str(char.get("text", "")).strip()
    ]
    rotated = [char for char in visible if not char.get("upright", True)]
    if len(rotated) < 10 or len(rotated) / max(len(visible), 1) < 0.60:
        return 0
    angle = median(float(char.get("angle", 0.0)) for char in rotated)
    if abs(abs(angle) - math.pi / 2) > 0.20:
        return 0
    # PDFium reports the long2009 page-7 table as -90 degrees. Rotating its
    # raster crop clockwise restores the reading direction.
    return 90 if angle < 0 else -90


def _rotate_crop(image: Any, rotation: int) -> Any:
    if not rotation:
        return image
    import cv2

    code = (
        cv2.ROTATE_90_CLOCKWISE
        if rotation == 90
        else cv2.ROTATE_90_COUNTERCLOCKWISE
    )
    return cv2.rotate(image, code)


def _looks_like_colored_chart(
    image: Any,
    *,
    native_character_count: int,
) -> bool:
    """Detect heatmaps that PP-DocLayoutV2 occasionally labels as tables."""

    if native_character_count < 100:
        return False
    import cv2
    import numpy as np

    if image is None or not getattr(image, "size", 0):
        return False
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colored_ratio = float(
        np.mean((saturation >= 20) & (value <= 253))
    )
    saturation_upper_quartile = float(np.percentile(saturation, 75))
    return colored_ratio >= 0.05 and saturation_upper_quartile >= 20


def _native_dense_table_fallback(
    chars: Sequence[dict[str, Any]],
    *,
    bbox: BBox,
    confidence: float,
) -> Block | None:
    """Recover oversized native tables that exceed UniRec's token capacity."""

    upright_chars = _layout_source_chars(chars)
    if _visible_character_count(upright_chars) < 1800:
        return None
    cell_lines = _chars_to_lines(
        upright_chars,
        max(bbox[2] - bbox[0], 1.0),
    )
    grouped_rows: list[dict[str, Any]] = []
    for line in sorted(
        cell_lines,
        key=lambda item: (item["bbox"][1], item["bbox"][0]),
    ):
        center = (line["bbox"][1] + line["bbox"][3]) / 2
        target = next(
            (
                row
                for row in reversed(grouped_rows[-4:])
                if abs(center - row["center"]) <= 3.5
            ),
            None,
        )
        if target is None:
            target = {"center": center, "lines": []}
            grouped_rows.append(target)
        target["lines"].append(line)
        target["center"] = sum(
            (item["bbox"][1] + item["bbox"][3]) / 2
            for item in target["lines"]
        ) / len(target["lines"])

    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in grouped_rows:
        if len(row["lines"]) >= 3:
            current.append(row)
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)
    grid_rows = max(runs, key=len, default=[])
    if len(grid_rows) < 30:
        return None

    column_count = Counter(
        len(row["lines"]) for row in grid_rows
    ).most_common(1)[0][0]
    if column_count < 3:
        return None
    anchor_rows = [
        sorted(row["lines"], key=lambda item: item["bbox"][0])
        for row in grid_rows
        if len(row["lines"]) == column_count
    ]
    if len(anchor_rows) < max(5, len(grid_rows) // 2):
        return None
    anchors = [
        median(row[index]["bbox"][0] for row in anchor_rows)
        for index in range(column_count)
    ]
    rows: list[list[str]] = []
    for row in grid_rows:
        cells = [""] * column_count
        for line in sorted(row["lines"], key=lambda item: item["bbox"][0]):
            index = min(
                range(column_count),
                key=lambda candidate: abs(
                    line["bbox"][0] - anchors[candidate]
                ),
            )
            cells[index] = re.sub(
                r"([×x]10)([+-]?\d+)$",
                r"\1^\2",
                _clean_text(
                    f"{cells[index]} {line['text']}"
                ),
            )
        rows.append(cells)
    if not rows:
        return None
    grid_bbox = (
        min(line["bbox"][0] for row in grid_rows for line in row["lines"]),
        min(line["bbox"][1] for row in grid_rows for line in row["lines"]),
        max(line["bbox"][2] for row in grid_rows for line in row["lines"]),
        max(line["bbox"][3] for row in grid_rows for line in row["lines"]),
    )
    return Block(
        kind="table",
        bbox=grid_bbox,
        rows=[
            [f"Column {index + 1}" for index in range(column_count)],
            *rows,
        ],
        confidence=confidence,
        complex_table=True,
        source="layout-native-capacity",
        label="table",
    )


def _unirec_task_max_length(
    label: str,
    native_character_count: int,
    configured_max_length: int,
) -> int:
    """Use native content as a conservative upper bound for autoregressive OCR."""

    configured = max(32, int(configured_max_length))
    if label in _LAYOUT_FORMULA_LABELS:
        return min(configured, 512)
    if label == "table" and native_character_count >= 20:
        estimated = 384 + round(native_character_count * 1.5)
        return min(configured, max(512, estimated))
    return configured


def _token_tail_is_repetitive(token_ids: Sequence[int]) -> bool:
    """Stop exact token cycles that otherwise run until the hard token limit."""

    if len(token_ids) < 256:
        return False
    tail = token_ids[-192:]
    if tail and Counter(tail).most_common(1)[0][1] / len(tail) >= 0.90:
        return True
    for unit_length, repeats in ((8, 12), (16, 8), (32, 6)):
        span = unit_length * repeats
        if len(token_ids) < span:
            continue
        unit = list(token_ids[-unit_length:])
        if len(set(unit)) > 1 and list(token_ids[-span:]) == unit * repeats:
            return True
    return False


def _generate_unirec_text(
    vlm_recognizer: Any,
    image: Any,
    *,
    max_length: int,
) -> tuple[str, int, str]:
    """Run UniRec generation with EOS, repetition, and length stop reporting."""

    import cv2
    import numpy as np
    from PIL import Image
    from openocr.tools.infer_unirec_onnx import clean_special_tokens

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    encoder_hidden_states, cross_k, cross_v = vlm_recognizer.encode_image(
        pil_image
    )
    tokenizer = vlm_recognizer.tokenizer
    generated_ids = [tokenizer.bos_token_id]
    batch_size = encoder_hidden_states.shape[0]
    past_key_values = [
        (
            np.zeros(
                (
                    batch_size,
                    vlm_recognizer.num_heads,
                    0,
                    vlm_recognizer.head_dim,
                ),
                dtype=np.float32,
            ),
            np.zeros(
                (
                    batch_size,
                    vlm_recognizer.num_heads,
                    0,
                    vlm_recognizer.head_dim,
                ),
                dtype=np.float32,
            ),
        )
        for _ in range(vlm_recognizer.num_decoder_layers)
    ]
    stop_reason = "max_length"
    for step in range(max_length - 1):
        logits, past_key_values = vlm_recognizer.decode_step(
            generated_ids[-1],
            step,
            cross_k,
            cross_v,
            past_key_values,
            padding_idx=tokenizer.pad_token_id,
        )
        next_token_id = int(np.argmax(logits[0, -1, :]))
        generated_ids.append(next_token_id)
        if next_token_id == tokenizer.eos_token_id:
            stop_reason = "eos"
            break
        if _token_tail_is_repetitive(generated_ids):
            stop_reason = "repetition"
            break
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return clean_special_tokens(decoded), len(generated_ids), stop_reason


def _postprocess_unirec_text(raw_label: str, text: str) -> str:
    from openocr.tools import infer_doc_onnx

    if "table" in raw_label:
        return infer_doc_onnx.markdown_converter._handle_table(text)
    if "formula" in raw_label and raw_label != "formula_number":
        return infer_doc_onnx.markdown_converter._handle_formula(text)
    return infer_doc_onnx.markdown_converter._handle_text(text)


def _table_recognition_is_suspicious(
    text: str,
    *,
    native_character_count: int,
) -> bool:
    """Reject structurally explosive table output when native text is available."""

    if native_character_count < 80 or not text:
        return False
    row_count = len(re.findall(r"<tr(?:\s|>)", text, flags=re.IGNORECASE))
    cell_count = len(re.findall(r"<td(?:\s|>)", text, flags=re.IGNORECASE))
    excessive_text = len(text) > native_character_count * 12 + 1500
    excessive_structure = (
        row_count > max(120, native_character_count // 3)
        or cell_count > max(500, native_character_count)
    )
    return excessive_text and excessive_structure


def _recognize_layout_tasks(
    tasks: list[dict[str, Any]],
    *,
    model_dir: str | Path | None,
    use_gpu: str,
    auto_download: bool,
    max_length: int,
    max_parallel_blocks: int,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    recognizer = _get_selective_opendoc_recognizer(
        model_dir,
        use_gpu,
        auto_download,
        max_parallel_blocks,
    )
    results: list[dict[str, Any] | None] = [None] * len(tasks)

    def recognize(index: int, task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        task_max_length = _unirec_task_max_length(
            task["label"],
            int(task.get("native_character_count", 0)),
            max_length,
        )
        started = time.perf_counter()
        status = "ok"
        error = ""
        text = ""
        token_count = 0
        stop_reason = "error"
        try:
            raw_text, token_count, stop_reason = _generate_unirec_text(
                recognizer.vlm_recognizer,
                task["image"],
                max_length=task_max_length,
            )
            text = _postprocess_unirec_text(task["raw_label"], raw_text)
            if task["label"] == "table" and _table_recognition_is_suspicious(
                text,
                native_character_count=int(
                    task.get("native_character_count", 0)
                ),
            ):
                status = "rejected_suspicious"
                text = ""
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        return index, {
            "text": text,
            "seconds": round(time.perf_counter() - started, 6),
            "token_count": token_count,
            "max_length": task_max_length,
            "stop_reason": stop_reason,
            "status": status,
            "error": error,
        }

    workers = min(max(1, max_parallel_blocks), len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(recognize, index, task): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                result_index, result = future.result()
                results[result_index] = result
            except Exception as exc:
                results[index] = {
                    "text": "",
                    "seconds": 0.0,
                    "token_count": 0,
                    "max_length": max_length,
                    "stop_reason": "error",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    try:
        from openocr.tools.infer_doc_onnx import convert_otsl_to_html
    except ImportError:
        convert_otsl_to_html = None
    normalized: list[dict[str, Any]] = []
    for task, result in zip(tasks, results):
        normalized_result = dict(result or {})
        value = str(normalized_result.get("text") or "").strip()
        if (
            task["label"] == "table"
            and value
            and "<table" not in value.lower()
            and convert_otsl_to_html is not None
        ):
            html_value = convert_otsl_to_html(value)
            if html_value:
                value = html_value
        normalized_result["text"] = value
        normalized.append(normalized_result)
    return normalized


def _parse_pdf_hybrid(
    pdf_path: str | Path,
    *,
    password: str | None = None,
    model_dir: str | Path | None = None,
    use_gpu: str = "auto",
    auto_download: bool = True,
    max_length: int = 2048,
    max_parallel_blocks: int = 4,
    max_pages: int | None = None,
) -> Document:
    """Run document-wide layout first, then fill regions with native text/UniRec."""

    import pypdfium2

    source = Path(pdf_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    detector = _get_opendoc_layout_detector(model_dir, use_gpu, auto_download)
    warnings: list[str] = []
    pages: list[Page] = []
    layout_timings: list[float] = []
    render_timings: list[float] = []
    recognition_seconds = 0.0
    source_counts: Counter[str] = Counter()
    total_native_chars = 0
    mapped_native_chars = 0
    total_layout_regions = 0
    total_ocr_regions = 0
    total_table_regions = 0
    recognition_diagnostics: list[dict[str, Any]] = []
    pending_pages: list[dict[str, Any]] = []

    pdfium_document = pypdfium2.PdfDocument(str(source), password=password)
    page_limit = len(pdfium_document)
    if max_pages is not None:
        page_limit = min(page_limit, max_pages)

    layout_pages: list[dict[str, Any]] = []
    try:
        # Pass 1: finish layout for every page before native text extraction starts.
        for page_index in range(page_limit):
            pdfium_page = pdfium_document[page_index]
            try:
                page_width = float(pdfium_page.get_width())
                page_height = float(pdfium_page.get_height())
                started = time.perf_counter()
                page_image = _render_pdfium_page(pdfium_page)
                render_timings.append(time.perf_counter() - started)
            finally:
                pdfium_page.close()

            started = time.perf_counter()
            layout_result = _run_opendoc_layout(detector, page_image)
            layout_timings.append(time.perf_counter() - started)

            image_height, image_width = page_image.shape[:2]
            regions: list[dict[str, Any]] = []
            for order, box in enumerate(layout_result.get("boxes", [])):
                raw_label = str(box.get("label", "text"))
                label = _base_opendoc_label(raw_label)
                image_bbox = _opendoc_bbox(box.get("coordinate"))
                regions.append(
                    {
                        "order": order,
                        "raw_label": raw_label,
                        "label": label,
                        "image_bbox": image_bbox,
                        "bbox": _layout_bbox_to_pdf(
                            image_bbox,
                            page_width=page_width,
                            page_height=page_height,
                            image_width=float(image_width),
                            image_height=float(image_height),
                        ),
                        "score": float(box.get("score", 1.0) or 0.0),
                    }
                )
            total_layout_regions += len(regions)
            total_table_regions += sum(
                region["label"] == "table" for region in regions
            )
            layout_pages.append(
                {
                    "number": page_index + 1,
                    "width": page_width,
                    "height": page_height,
                    "image": page_image,
                    "regions": regions,
                }
            )

        # Pass 2: PDFium's text API ignores vector drawings and returns char geometry.
        native_font_sizes: list[float] = []
        for page_index, layout_page in enumerate(layout_pages):
            pdfium_page = pdfium_document[page_index]
            try:
                all_chars = _pdfium_page_chars(
                    pdfium_page,
                    include_rotated=True,
                )
            finally:
                pdfium_page.close()
            chars = _layout_source_chars(all_chars)
            layout_page["chars"] = chars
            layout_page["all_chars"] = all_chars
            native_font_sizes.extend(
                float(char.get("size", 0.0))
                for char in chars
                if float(char.get("size", 0.0)) > 0
                and not str(char.get("text", "")).isspace()
            )
    finally:
        pdfium_document.close()

    body_size = median(native_font_sizes) if native_font_sizes else 10.0
    repeated = _find_repeated_margins(
        [
            (
                page["height"],
                _chars_to_lines(page["chars"], page["width"]),
            )
            for page in layout_pages
        ]
    )

    if layout_pages:
        page_limit = min(page_limit, len(layout_pages))
        if max_pages is not None:
            page_limit = min(page_limit, max_pages)

        for page_index in range(page_limit):
            layout_page = layout_pages[page_index]
            page_image = layout_page["image"]
            page_width = layout_page["width"]
            page_height = layout_page["height"]
            regions = layout_page["regions"]

            table_regions = [
                region for region in regions if region["label"] == "table"
            ]
            chars = layout_page["chars"]
            all_chars = layout_page["all_chars"]
            total_native_chars += _visible_character_count(chars)
            assignments: list[list[dict[str, Any]]] = [[] for _ in regions]
            unassigned_chars: list[dict[str, Any]] = []
            for char in chars:
                # Tables are wholly owned by UniRec; do not leak their native text
                # into layout-native or unmatched-text fallback blocks.
                if any(
                    _inside_bbox(char, region["bbox"], padding=1.0)
                    for region in table_regions
                ):
                    continue
                candidate_indices = [
                    index
                    for index, region in enumerate(regions)
                    if region["label"] != "table"
                    and _inside_bbox(char, region["bbox"], padding=1.0)
                ]
                if not candidate_indices:
                    unassigned_chars.append(char)
                    continue
                selected_index = min(
                    candidate_indices,
                    key=lambda index: (
                        (regions[index]["bbox"][2] - regions[index]["bbox"][0])
                        * (regions[index]["bbox"][3] - regions[index]["bbox"][1]),
                        index,
                    ),
                )
                assignments[selected_index].append(char)

            entries: list[tuple[float, Block]] = []
            recognition_tasks: list[dict[str, Any]] = []
            for region_index, region in enumerate(regions):
                label = region["label"]
                region_chars = assignments[region_index]
                region_all_chars = [
                    char
                    for char in all_chars
                    if _inside_bbox(char, region["bbox"], padding=1.0)
                ]
                if label in _LAYOUT_IGNORED_LABELS:
                    continue
                native_fallback = (
                    []
                    if label == "table"
                    else _native_blocks_for_layout_region(
                        label="text" if label in _LAYOUT_FORMULA_LABELS else label,
                        bbox=region["bbox"],
                        chars=region_chars,
                        page_width=page_width,
                        body_size=body_size,
                        confidence=region["score"],
                    )
                )
                needs_recognition = _layout_region_needs_recognition(
                    label,
                    native_fallback,
                )
                if needs_recognition:
                    cropped = _crop_layout_region(
                        page_image, region["image_bbox"], label
                    )
                    if cropped is not None:
                        native_character_count = _visible_character_count(
                            region_all_chars
                        )
                        rotation = (
                            _table_crop_rotation(region_all_chars)
                            if label == "table"
                            else 0
                        )
                        native_capacity_fallback = (
                            _native_dense_table_fallback(
                                [
                                    char
                                    for char in all_chars
                                    if _inside_bbox(
                                        char,
                                        region["bbox"],
                                        padding=45.0,
                                    )
                                ],
                                bbox=region["bbox"],
                                confidence=region["score"],
                            )
                            if label == "table" and not rotation
                            else None
                        )
                        if native_capacity_fallback is not None:
                            entries.append(
                                (
                                    float(region["order"]),
                                    native_capacity_fallback,
                                )
                            )
                            source_counts[
                                native_capacity_fallback.source
                            ] += 1
                            recognition_diagnostics.append(
                                {
                                    "page": page_index + 1,
                                    "label": label,
                                    "bbox": list(
                                        native_capacity_fallback.bbox
                                    ),
                                    "crop_width": int(cropped.shape[1]),
                                    "crop_height": int(cropped.shape[0]),
                                    "native_character_count": (
                                        native_character_count
                                    ),
                                    "rotation": 0,
                                    "seconds": 0.0,
                                    "token_count": 0,
                                    "max_length": 0,
                                    "stop_reason": (
                                        "native_capacity_fallback"
                                    ),
                                    "status": "native_capacity_fallback",
                                    "output_characters": sum(
                                        len(cell)
                                        for row in native_capacity_fallback.rows
                                        for cell in row
                                    ),
                                    "rows": len(
                                        native_capacity_fallback.rows
                                    ),
                                    "columns": max(
                                        (
                                            len(row)
                                            for row
                                            in native_capacity_fallback.rows
                                        ),
                                        default=0,
                                    ),
                                }
                            )
                            warnings.append(
                                f"page {page_index + 1}: oversized table "
                                "used native grid fallback to avoid UniRec "
                                "token overflow"
                            )
                            continue
                        if label == "table" and _looks_like_colored_chart(
                            cropped,
                            native_character_count=native_character_count,
                        ):
                            recognition_diagnostics.append(
                                {
                                    "page": page_index + 1,
                                    "label": label,
                                    "bbox": list(region["bbox"]),
                                    "crop_width": int(cropped.shape[1]),
                                    "crop_height": int(cropped.shape[0]),
                                    "native_character_count": (
                                        native_character_count
                                    ),
                                    "rotation": 0,
                                    "seconds": 0.0,
                                    "token_count": 0,
                                    "max_length": 0,
                                    "stop_reason": "skipped_chart",
                                    "status": "skipped_chart",
                                    "output_characters": 0,
                                }
                            )
                            warnings.append(
                                f"page {page_index + 1}: table-like region "
                                "skipped as probable chart or heatmap"
                            )
                            continue
                        cropped = _rotate_crop(cropped, rotation)
                        recognition_tasks.append(
                            {
                                "order": float(region["order"]),
                                "label": label,
                                "raw_label": region["raw_label"],
                                "bbox": region["bbox"],
                                "score": region["score"],
                                "image": cropped.copy(),
                                "fallback": native_fallback,
                                "page_number": page_index + 1,
                                "native_character_count": (
                                    native_character_count
                                ),
                                "rotation": rotation,
                                "crop_width": int(cropped.shape[1]),
                                "crop_height": int(cropped.shape[0]),
                            }
                        )
                        continue
                for offset, block in enumerate(native_fallback):
                    entries.append((float(region["order"]) + offset / 100, block))
                    source_counts[block.source] += 1
                mapped_native_chars += _visible_character_count(region_chars)

            if unassigned_chars:
                fallback_lines = _chars_to_lines(unassigned_chars, page_width)
                fallback_lines = [
                    line
                    for line in fallback_lines
                    if _canonical_margin_text(line["text"]) not in repeated
                ]
                fallback_blocks = _build_blocks(fallback_lines, body_size, None)
                for block in fallback_blocks:
                    block.source = "native-fallback"
                    block.label = block.label or "text"
                    entries.append(
                        (_region_insertion_order(regions, block.bbox), block)
                    )
                    source_counts["native-fallback"] += 1

            pending_pages.append(
                {
                    "number": page_index + 1,
                    "width": page_width,
                    "height": page_height,
                    "entries": entries,
                    "recognition_tasks": recognition_tasks,
                }
            )
            layout_page["image"] = None

    all_recognition_tasks = [
        task
        for pending_page in pending_pages
        for task in pending_page["recognition_tasks"]
    ]
    recognition_results: list[dict[str, Any]] = []
    if all_recognition_tasks:
        started = time.perf_counter()
        try:
            recognition_results = _recognize_layout_tasks(
                all_recognition_tasks,
                model_dir=model_dir,
                use_gpu=use_gpu,
                auto_download=auto_download,
                max_length=max_length,
                max_parallel_blocks=max_parallel_blocks,
            )
        except Exception as exc:
            for page_number in sorted(
                {task["page_number"] for task in all_recognition_tasks}
            ):
                warnings.append(
                    f"page {page_number}: selective UniRec failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            recognition_results = [
                {
                    "text": "",
                    "seconds": 0.0,
                    "token_count": 0,
                    "max_length": max_length,
                    "stop_reason": "error",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                for _ in all_recognition_tasks
            ]
        recognition_seconds += time.perf_counter() - started

    recognition_offset = 0
    for pending_page in pending_pages:
        entries = pending_page["entries"]
        page_tasks = pending_page["recognition_tasks"]
        page_results = recognition_results[
            recognition_offset : recognition_offset + len(page_tasks)
        ]
        recognition_offset += len(page_tasks)
        for task, recognition_result in zip(page_tasks, page_results):
            recognized_text = str(recognition_result.get("text") or "")
            diagnostic = {
                "page": task["page_number"],
                "label": task["label"],
                "bbox": list(task["bbox"]),
                "crop_width": task["crop_width"],
                "crop_height": task["crop_height"],
                "native_character_count": task["native_character_count"],
                "rotation": task["rotation"],
                "segment_index": task.get("segment_index", 1),
                "segment_count": task.get("segment_count", 1),
                "seconds": recognition_result.get("seconds", 0.0),
                "token_count": recognition_result.get("token_count", 0),
                "max_length": recognition_result.get("max_length", max_length),
                "stop_reason": recognition_result.get(
                    "stop_reason",
                    "unknown",
                ),
                "status": recognition_result.get("status", "unknown"),
                "output_characters": len(recognized_text),
            }
            if recognition_result.get("error"):
                diagnostic["error"] = recognition_result["error"]
            recognition_diagnostics.append(diagnostic)
            if diagnostic["status"] == "error":
                warnings.append(
                    f"page {task['page_number']}: UniRec {task['label']} "
                    f"failed: {diagnostic.get('error', 'unknown error')}"
                )
            elif diagnostic["status"] == "rejected_suspicious":
                warnings.append(
                    f"page {task['page_number']}: suspicious UniRec table "
                    "output was rejected"
                )
            elif diagnostic["stop_reason"] == "max_length":
                warnings.append(
                    f"page {task['page_number']}: UniRec {task['label']} "
                    f"reached token limit {diagnostic['max_length']}"
                )
            recognized_blocks = _recognized_blocks_for_layout_region(
                label=task["label"],
                bbox=task["bbox"],
                text=recognized_text,
                confidence=task["score"],
            )
            selected_blocks = recognized_blocks or task["fallback"]
            for offset, block in enumerate(selected_blocks):
                entries.append((task["order"] + offset / 100, block))
                source_counts[block.source] += 1
            if recognized_blocks:
                total_ocr_regions += 1
            else:
                if task["label"] == "table":
                    warnings.append(
                        f"page {task['page_number']}: UniRec returned no table content"
                    )
                mapped_native_chars += sum(
                    len(block.text) for block in task["fallback"]
                )

        ordered_blocks: list[Block] = []
        for _, block in sorted(entries, key=lambda item: item[0]):
            if (
                block.kind == "heading"
                and ordered_blocks
                and ordered_blocks[-1].kind == "heading"
                and _clean_text(ordered_blocks[-1].text).casefold()
                == _clean_text(block.text).casefold()
            ):
                continue
            ordered_blocks.append(block)
        pages.append(
            Page(
                number=pending_page["number"],
                width=pending_page["width"],
                height=pending_page["height"],
                blocks=ordered_blocks,
            )
        )

    output_characters = sum(
        len(block.text) + sum(len(cell) for row in block.rows for cell in row)
        for page in pages
        for block in page.blocks
    )
    status = "ok" if output_characters >= max(50, len(pages) * 10) else "ocr_required"
    if status == "ocr_required":
        warnings.append("Hybrid parsing produced little usable text; OCR is still required.")
    metadata: dict[str, Any] = {}
    try:
        reader = PdfReader(str(source))
        if reader.is_encrypted and not reader.decrypt(password or ""):
            raise ValueError(f"Unable to decrypt PDF metadata: {source}")
        metadata.update(_reader_metadata(reader))
    except Exception as exc:
        warnings.append(
            f"PDF metadata could not be read: {type(exc).__name__}: {exc}"
        )
    metadata.update(
        {
            "backend": "hybrid",
            "page_count": len(pages),
            "renderer": "pypdfium2",
            "native_text_provider": "pypdfium2",
            "table_provider": "unirec",
            "body_font_size": round(body_size, 2),
            "removed_repeated_margins": sorted(repeated),
            "layout_provider": list(detector.session.get_providers()),
            "layout_seconds": round(sum(layout_timings), 6),
            "layout_page_seconds": [round(value, 6) for value in layout_timings],
            "render_seconds": round(sum(render_timings), 6),
            "selective_unirec_seconds": round(recognition_seconds, 6),
            "layout_regions": total_layout_regions,
            "layout_table_regions": total_table_regions,
            "selective_unirec_regions": total_ocr_regions,
            "unirec_region_diagnostics": recognition_diagnostics,
            "unirec_stop_reasons": dict(
                sorted(
                    Counter(
                        item["stop_reason"]
                        for item in recognition_diagnostics
                    ).items()
                )
            ),
            "native_character_coverage": round(
                mapped_native_chars / max(total_native_chars, 1), 4
            ),
            "block_sources": dict(sorted(source_counts.items())),
        }
    )
    return Document(
        source=str(source),
        metadata=metadata,
        pages=pages,
        status=status,
        warnings=warnings,
    )


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
                    source="opendoc",
                    label="doc_title" if level == 1 else "paragraph_title",
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
                            source="opendoc",
                            label="abstract",
                        )
                    )
            else:
                blocks.append(
                    Block(
                        kind="raw_markdown",
                        text=text,
                        bbox=bbox,
                        confidence=confidence,
                        source="opendoc",
                        label=label,
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
    password: str | None = None,
    model_dir: str | Path | None = None,
    use_gpu: str = "auto",
    auto_download: bool = True,
    max_length: int = 2048,
    max_parallel_blocks: int = 4,
    max_pages: int | None = None,
) -> Document:
    import pypdfium2

    _ensure_utf8_console_output()
    _prepare_opendoc_gpu_runtime(use_gpu)
    source = Path(pdf_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        from openocr import OpenOCR
    except ImportError as exc:
        raise RuntimeError(
            "OpenDoc requires openocr-python==0.1.5. Install the project requirements first."
        ) from exc

    base_dir = _opendoc_model_base(model_dir)
    gpu_value = _opendoc_gpu_value(use_gpu)

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
    pdfium_document = pypdfium2.PdfDocument(str(source), password=password)
    page_limit = len(pdfium_document)
    if max_pages is not None:
        page_limit = min(page_limit, max_pages)
    results: list[dict[str, Any]] = []
    try:
        for page_index in range(page_limit):
            pdfium_page = pdfium_document[page_index]
            try:
                image = _render_pdfium_page(pdfium_page)
            finally:
                pdfium_page.close()
            result = parser(
                img_numpy=image,
                max_length=max_length,
                merge_layout_blocks=True,
            )
            result["pdf_page"] = page_index + 1
            results.append(result)
    finally:
        pdfium_document.close()
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
    opendoc_max_parallel_blocks: int = 4,
) -> Document:
    """Parse with native, full OpenDoc, or layout-first hybrid extraction."""

    if backend not in {"native", "auto", "hybrid", "opendoc"}:
        raise ValueError("backend must be one of: native, auto, hybrid, opendoc")
    opendoc_options = {
        "password": password,
        "model_dir": opendoc_model_dir,
        "use_gpu": opendoc_use_gpu,
        "auto_download": opendoc_auto_download,
        "max_length": opendoc_max_length,
        "max_parallel_blocks": opendoc_max_parallel_blocks,
        "max_pages": max_pages,
    }
    if backend == "opendoc":
        return _parse_pdf_opendoc(pdf_path, **opendoc_options)

    if backend == "native":
        native_document = _parse_pdf_native(
            pdf_path,
            password=password,
            table_strategy=table_strategy,
            max_pages=max_pages,
        )
        native_document.metadata.setdefault("backend", "native")
        return native_document

    try:
        return _parse_pdf_hybrid(pdf_path, **opendoc_options)
    except Exception as exc:
        if backend == "hybrid":
            raise
        fallback = _parse_pdf_opendoc(pdf_path, **opendoc_options)
        fallback.warnings.append(
            f"Hybrid parsing failed; full OpenDoc fallback was used: "
            f"{type(exc).__name__}: {exc}"
        )
        fallback.metadata["backend"] = "opendoc-fallback"
        return fallback


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
    opendoc_max_parallel_blocks: int = 4,
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
        choices=("auto", "hybrid", "native", "opendoc"),
        default="auto",
        help="auto/hybrid use layout-first OpenDoc, PDFium text, and UniRec tables",
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
    parser.add_argument("--opendoc-max-parallel-blocks", type=int, default=4)
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
