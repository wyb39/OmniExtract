"""Document-shaped adapters for the production batch parsers.

Legacy parsers accept a whole folder and often log-and-continue, which loses
the failing filename and exception.  These adapters expose one document per
worker, use private temporary files, and return a ``ProcessingReport``.  They
contain no HTTP/CLI/UI policy; callers decide where to persist the report.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from error_handling import DocumentStageError, ProcessingReport, run_isolated


_SUFFIXES = {
    "pdf": {".pdf"},
    "sciencedirect": {".xml"},
    "pmc": {".xml"},
    "arxiv": {".tex"},
}


def _normalise_format(file_type: str) -> str:
    value = str(file_type).strip().lower().replace("_", "")
    aliases = {
        "pdf": "pdf",
        "sciencedirect": "sciencedirect",
        "sciencedirectxml": "sciencedirect",
        "pmc": "pmc",
        "pmcxml": "pmc",
        "arxiv": "arxiv",
        "tex": "arxiv",
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported file type: {file_type}. "
            "Supported types: PDF, scienceDirect, PMC, Arxiv"
        )
    return aliases[value]


def _discover(folder_path: str | Path, file_type: str) -> tuple[Path, list[Path], str]:
    root = Path(folder_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Input path is not a directory: {root}")
    normalised = _normalise_format(file_type)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUFFIXES[normalised]
    )
    if not files:
        raise FileNotFoundError(f"No {normalised} documents found in {root}")
    return root, files, normalised


def _document_id(root: Path, source: Path) -> str:
    return source.relative_to(root).as_posix()


def _output_stem(root: Path, source: Path) -> str:
    """Return a readable Unicode-safe candidate output stem.

    ``\\w`` is Unicode-aware in Python, so Chinese and other non-ASCII names
    are preserved instead of all collapsing to ``document``.
    """

    relative = source.relative_to(root).with_suffix("").as_posix()
    return re.sub(r"[^\w.-]+", "__", relative).strip("._") or "document"


def _output_stems(root: Path, files: list[Path]) -> dict[Path, str]:
    """Resolve normalization collisions with a stable relative-path hash."""

    candidates = {source: _output_stem(root, source) for source in files}
    counts: dict[str, int] = {}
    for candidate in candidates.values():
        key = candidate.casefold()  # Windows output paths are case-insensitive.
        counts[key] = counts.get(key, 0) + 1

    resolved: dict[Path, str] = {}
    for source, candidate in candidates.items():
        if counts[candidate.casefold()] > 1:
            relative = source.relative_to(root).as_posix()
            digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
            candidate = f"{candidate}__{digest}"
        resolved[source] = candidate
    return resolved


def _converter(file_type: str) -> Callable[[Path, Path], Any]:
    # Imports stay lazy so non-parsing CLI commands do not initialize parser
    # dependencies or model libraries.
    from articleUtil import (
        PDF_PARSER_BACKEND,
        PubMedCentralXmlParser,
        ScienceDirectXmlParser,
        TeXProcessor,
    )
    from pdf_parser import convert_pdf

    if file_type == "pdf":
        return lambda source, destination: convert_pdf(
            source,
            destination,
            backend=PDF_PARSER_BACKEND,
        )
    if file_type == "pmc":
        return lambda source, destination: PubMedCentralXmlParser(
            str(source)
        ).to_markdown(str(destination))
    if file_type == "sciencedirect":
        return lambda source, destination: ScienceDirectXmlParser(
            str(source)
        ).to_markdown(str(destination))

    def convert_tex(source: Path, destination: Path) -> None:
        destination.write_text(
            TeXProcessor(str(source)).process(),
            encoding="utf-8",
        )

    return convert_tex


def documents_to_markdown(
    folder_path: str | Path,
    save_path: str | Path,
    file_type: str,
    *,
    max_workers: int = 1,
) -> tuple[list[str], ProcessingReport]:
    """Convert every source independently and retain successful Markdown."""

    root, files, normalised = _discover(folder_path, file_type)
    output = Path(save_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    convert = _converter(normalised)
    stems = _output_stems(root, files)

    def worker(source: Path) -> str:
        stem = stems[source]
        destination = output / f"{stem}.md"
        temporary = output / f".{stem}.md.tmp"
        try:
            convert(source, temporary)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise OSError("The parser did not generate Markdown output")
            os.replace(temporary, destination)
            return str(destination)
        finally:
            temporary.unlink(missing_ok=True)

    outcomes, report = run_isolated(
        files,
        worker,
        lambda source: _document_id(root, source),
        stage="markdown_convert",
        workflow_id=output.name or "file_to_md",
        max_workers=max_workers,
    )
    return [
        str(outcome.value)
        for outcome in outcomes
        if outcome.issue is None and outcome.value is not None
    ], report


def documents_to_json(
    folder_path: str | Path,
    save_path: str | Path,
    file_type: str,
    convert_mode: str,
    *,
    max_workers: int = 1,
) -> tuple[list[str], ProcessingReport]:
    """Convert source -> Markdown -> JSON within one isolated document worker."""

    root, files, normalised = _discover(folder_path, file_type)
    output = Path(save_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    mode = str(convert_mode).strip().lower()
    if mode not in {"bypart", "wholedoc"}:
        raise ValueError("convert_mode must be byPart or wholeDoc")
    convert = _converter(normalised)
    stems = _output_stems(root, files)

    def worker(source: Path) -> str:
        stem = stems[source]
        destination = output / f"{stem}.json"
        temporary_json = output / f".{stem}.json.tmp"
        with tempfile.TemporaryDirectory(prefix=f"omniextract-{stem}-") as temp:
            markdown = Path(temp) / f"{stem}.md"
            try:
                convert(source, markdown)
                if not markdown.is_file() or markdown.stat().st_size == 0:
                    raise OSError("The parser did not generate Markdown output")
            except Exception as error:
                raise DocumentStageError("markdown_convert", error) from error
            try:
                if mode == "wholedoc":
                    payload = {"Document": markdown.read_text(encoding="utf-8")}
                else:
                    from pdf_markdown_renderer import split_markdown_file

                    payload = split_markdown_file(markdown)
                    if not isinstance(payload, dict):
                        raise TypeError("Markdown splitter must return a mapping")
                temporary_json.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(temporary_json, destination)
            except Exception as error:
                raise DocumentStageError("json_convert", error) from error
        temporary_json.unlink(missing_ok=True)
        return str(destination)

    outcomes, report = run_isolated(
        files,
        worker,
        lambda source: _document_id(root, source),
        stage="document_parse",
        workflow_id=output.name or "file_to_json",
        max_workers=max_workers,
    )
    return [
        str(outcome.value)
        for outcome in outcomes
        if outcome.issue is None and outcome.value is not None
    ], report


def process_table_documents(
    parsed_path: str | Path,
    output_path: str | Path,
    processor: Callable[[Path, Path], Any],
) -> tuple[list[Any], ProcessingReport]:
    """Run the model-based table pipeline once per source document directory.

    The production table parser creates one child directory per source
    document.  Keeping those directories as the isolation unit prevents one
    document's classification/extraction error from cancelling its siblings.
    Processing is sequential at this outer level because DSPy model
    configuration is global; each document may still use its configured
    internal thread count.
    """

    parsed = Path(parsed_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not parsed.is_dir():
        raise ValueError(f"Parsed table path is not a directory: {parsed}")
    output.mkdir(parents=True, exist_ok=True)

    document_dirs = [
        child
        for child in sorted(parsed.iterdir())
        if child.is_dir()
        and child.name not in {"json_classify", "json_example"}
        and any(path.suffix.lower() == ".tsv" for path in child.rglob("*.tsv"))
    ]
    if not document_dirs and any(
        path.suffix.lower() == ".tsv" for path in parsed.iterdir() if path.is_file()
    ):
        document_dirs = [parsed]
    if not document_dirs:
        raise FileNotFoundError("No parsed table documents were found")

    def worker(source: Path) -> Any:
        document_id = source.name if source != parsed else "tables"
        destination = output / document_id if source != parsed else output
        return processor(source, destination)

    outcomes, report = run_isolated(
        document_dirs,
        worker,
        lambda source: source.name if source != parsed else "tables",
        stage="table_extract",
        workflow_id=output.name or "table_extraction",
        max_workers=1,
    )
    return [
        outcome.value
        for outcome in outcomes
        if outcome.issue is None and outcome.value is not None
    ], report


__all__ = [
    "documents_to_json",
    "documents_to_markdown",
    "process_table_documents",
]
