# PDF Parsing Experiment

This directory is a temporary workspace for developing and comparing the new PDF parsing pipeline before it is moved into `src`.

See [PLAN.md](PLAN.md) for the detailed implementation plan, output contract, architecture, and acceptance criteria.

## Intended pipeline

1. Accept a PDF input and collect document metadata.
2. Extract page text and layout-aware text blocks with `pypdf` and `pdfplumber`.
3. Detect tables, images, headers, footers, and page boundaries.
4. Normalize the result into a lightweight `Document -> Page -> Block` model.
5. Add fixtures and comparison checks for representative PDFs.
6. Move the validated implementation into the appropriate `src` module.

## Development boundary

- Keep the core implementation in no more than two files:
  `pdf_to_markdown.py` and `markdown_renderer.py`.
- Keep corpus evaluation, snapshots, and generated outputs under `tests/`; they do
  not count toward the core-file limit.
- Keep experimental code, fixtures, and comparison output in this directory.
- Do not import this experiment from production code until the pipeline is validated.
- Avoid committing generated or sensitive PDF files; use small synthetic fixtures where possible.

## Prototype usage

Python API:

```python
from pdf_to_markdown import convert_pdf

markdown, document = convert_pdf(
    "input.pdf",
    "output.md",
    debug_json_path="output.debug.json",
)
```

Command line:

```powershell
.\.venv\Scripts\python.exe pdf_parsing_experiment\pdf_to_markdown.py `
  pdf_parsing_experiment\test_pdfs `
  --output pdf_parsing_experiment\tests\output `
  --debug-json
```

Regression checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s pdf_parsing_experiment\tests -p "test_*.py" -v

.\.venv\Scripts\python.exe pdf_parsing_experiment\tests\evaluate_corpus.py `
  --max-pages 5
```

The evaluator writes Markdown and `summary.json` under `tests/output/`, which is
ignored by Git.

## Current limits

- Scanned PDFs return `ocr_required`; OCR is intentionally outside this prototype.
- Vector chart labels can still enter the text stream because PDF figures do not
  always expose a reliable image boundary.
- Formula reconstruction and cross-page table merging are not implemented yet.
- Borderless table detection is heuristic and deliberately rejects paragraph-like
  candidates to avoid turning prose into false tables.

## Migration checklist

- [x] Define the lightweight intermediate document model in `pdf_to_markdown.py`.
- [x] Implement `pypdf` fallback and `pdfplumber` layout/table extraction together.
- [x] Implement Markdown and table rendering in `markdown_renderer.py`.
- [ ] Compare results against the current parser.
- [x] Add regression tests and corpus evaluation.
- [ ] Move the validated components into `src` and update callers.
