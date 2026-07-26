# PDF Parsing Experiment

This directory is a temporary workspace for developing and comparing the new PDF parsing pipeline before it is moved into `src`.

## Intended pipeline

1. Accept a PDF input and collect document metadata.
2. Extract page text and layout-aware text blocks with `pypdf` and `pdfplumber`.
3. Detect tables, images, headers, footers, and page boundaries.
4. Normalize the result into a stable intermediate document model.
5. Add fixtures and comparison checks for representative PDFs.
6. Move the validated implementation into the appropriate `src` module.

## Development boundary

- Keep experimental code, fixtures, and comparison output in this directory.
- Do not import this experiment from production code until the pipeline is validated.
- Avoid committing generated or sensitive PDF files; use small synthetic fixtures where possible.

## Migration checklist

- [ ] Define the intermediate document model.
- [ ] Implement text extraction with `pypdf`.
- [ ] Implement layout and table extraction with `pdfplumber`.
- [ ] Compare results against the current parser.
- [ ] Add regression tests and representative fixtures.
- [ ] Move the validated components into `src` and update callers.
