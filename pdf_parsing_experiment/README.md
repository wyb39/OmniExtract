# PDF Parsing Experiment

This directory is a temporary workspace for developing and comparing the new PDF parsing pipeline before it is moved into `src`.

See [PLAN.md](PLAN.md) for the detailed implementation plan, output contract, architecture, and acceptance criteria.

## Intended pipeline

1. Render every PDF page with `pypdfium2`.
2. Finish OpenDoc PP-DocLayoutV2 for the whole document before native extraction.
3. Read native character geometry, fonts, and bold styles with PDFium and map
   them into the detected regions.
4. Send detected tables to UniRec. If a machine-generated table has a stable
   native grid that would exceed UniRec's token capacity, recover that grid
   with PDFium characters instead of invoking `pdfplumber`.
5. Invoke UniRec for formulas, scanned pages, and low-coverage regions.
6. Normalize the result into a lightweight `Document -> Page -> Block` model.
7. Add fixtures and comparison checks for representative PDFs.
8. Move the validated implementation into the appropriate `src` module.

The hybrid and full OpenDoc PDF paths do not use PyMuPDF. `pdfplumber` remains
available only in the explicit `native` comparison backend.

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
    backend="auto",
)
```

Command line:

```powershell
.\.venv\Scripts\python.exe pdf_parsing_experiment\pdf_to_markdown.py `
  pdf_parsing_experiment\test_pdfs `
  --output pdf_parsing_experiment\tests\output `
  --debug-json
```

Parser backends:

```powershell
# Layout-first hybrid (default): OpenDoc layout + native text + selective UniRec
.\.venv\Scripts\python.exe pdf_parsing_experiment\pdf_to_markdown.py `
  input.pdf --backend auto --output output

# Explicit layout-first hybrid
.\.venv\Scripts\python.exe pdf_parsing_experiment\pdf_to_markdown.py `
  input.pdf --backend hybrid --output output

# Native-only comparison baseline
.\.venv\Scripts\python.exe pdf_parsing_experiment\pdf_to_markdown.py `
  input.pdf --backend native --output output

# Force OpenDoc for every page
.\.venv\Scripts\python.exe pdf_parsing_experiment\pdf_to_markdown.py `
  input.pdf --backend opendoc --output output

# Do not download models during execution
.\.venv\Scripts\python.exe pdf_parsing_experiment\pdf_to_markdown.py `
  input.pdf --backend opendoc --opendoc-no-auto-download
```

OpenDoc uses `openocr-python==0.1.5`. Its four required model files are stored
under `models/opendoc/` and ignored by Git:

- `PP-DocLayoutV2.onnx`
- `unirec_encoder.onnx`
- `unirec_decoder.onnx`
- `unirec_tokenizer_mapping.json`

The models total approximately 943 MB. The installed ONNX Runtime exposes CUDA
and CPU providers. `--opendoc-use-gpu true` requires CUDA for Layout and UniRec;
`--opendoc-use-gpu false` forces CPU.

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

- `auto` and `hybrid` complete OpenDoc Layout for every page before extracting
  native text. Ordinary machine-generated body text is not OCRed. Tables and
  formulas use UniRec unless an oversized native table is recovered by the
  token-capacity fallback.
- Color-dense heatmaps mislabeled as tables are skipped as charts, and
  predominantly rotated table crops are normalized before UniRec.
- Debug JSON records per-region UniRec time, generated token count, configured
  limit, stop reason, rotation, and fallback status.
- Full OpenDoc OCR remains much slower than Layout plus native character mapping.
- Vector chart labels can still enter the text stream because PDF figures do not
  always expose a reliable image boundary.
- Cross-page table merging is not implemented yet.

## Migration checklist

- [x] Define the lightweight intermediate document model in `pdf_to_markdown.py`.
- [x] Keep `pdfplumber` as an explicit native-only comparison backend.
- [x] Implement Markdown and table rendering in `markdown_renderer.py`.
- [x] Add full OpenDoc parsing and OCR fallback.
- [x] Make OpenDoc Layout the unified `auto` entry and map native characters by region.
- [x] Remove PyMuPDF from the PDF flow and use PDFium for rendering/native text.
- [x] Route detected tables through UniRec with an oversized native-grid
  capacity fallback.
- [ ] Compare results against the current parser.
- [x] Add regression tests and corpus evaluation.
- [ ] Move the validated components into `src` and update callers.
