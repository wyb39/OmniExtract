# Dependency Installation Note

## PDF packages

- Installed in the project `.venv`:
  - `pypdf==6.14.2`
  - `pdfplumber==0.11.10`
- Both `pypdf` and `pdfplumber` import successfully.

## Existing dependency conflicts

Installing `pdfplumber` upgraded these packages:

- `Pillow` to `12.3.0`
- `pypdfium2` to `5.12.1`

`pip check` reports:

- `marker-pdf 1.6.1` requires `Pillow<11.0.0,>=10.1.0`.
- `surya-ocr 0.13.0` requires `Pillow<11.0.0,>=10.2.0`.
- `pdftext 0.6.2` requires `pypdfium2==4.30.0`.
- `surya-ocr 0.13.0` requires `pypdfium2==4.30.0`.

Be aware that the existing marker/surya/pdftext workflow may be affected. Do not downgrade these dependencies without checking the PDF package requirements and the project's intended workflow.
