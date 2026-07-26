"""Run the parser over test_pdfs and write compact quality metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from pdf_to_markdown import convert_pdf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=EXPERIMENT_DIR / "test_pdfs")
    parser.add_argument("--output", type=Path, default=EXPERIMENT_DIR / "tests" / "output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--debug-json", action="store_true")
    parser.add_argument("--table-strategy", choices=("auto", "strict", "text", "none"), default="auto")
    parser.add_argument("--backend", choices=("native", "auto", "opendoc"), default="native")
    args = parser.parse_args()

    pdfs = sorted(args.input.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for pdf_path in pdfs:
        started = time.perf_counter()
        entry = {"file": pdf_path.name}
        try:
            markdown_path = args.output / f"{pdf_path.stem}.md"
            debug_path = args.output / f"{pdf_path.stem}.debug.json" if args.debug_json else None
            markdown, document = convert_pdf(
                pdf_path,
                markdown_path,
                debug_json_path=debug_path,
                table_strategy=args.table_strategy,
                max_pages=args.max_pages,
                backend=args.backend,
            )
            blocks = [block for page in document.pages for block in page.blocks]
            entry.update(
                {
                    "status": document.status,
                    "pages": len(document.pages),
                    "markdown_chars": len(markdown),
                    "headings": sum(block.kind == "heading" for block in blocks),
                    "tables": sum(block.kind == "table" for block in blocks),
                    "warnings": document.warnings,
                }
            )
        except Exception as exc:
            entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        entry["seconds"] = round(time.perf_counter() - started, 3)
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = sum(entry["status"] == "error" for entry in results)
    print(f"processed={len(results)} failures={failures} summary={summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
