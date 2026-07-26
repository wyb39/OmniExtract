# Dependency Installation Note

## PDF packages

- Installed in the project `.venv`:
  - `pypdf==6.14.2`
  - `pdfplumber==0.11.7`
- Both `pypdf` and `pdfplumber` import successfully.

## Resolved dependency conflicts

Installing `pdfplumber==0.11.10` initially upgraded these packages:

- `Pillow` to `12.3.0`
- `pypdfium2` to `5.12.1`

That conflicted with:

- `marker-pdf 1.6.1` requires `Pillow<11.0.0,>=10.1.0`.
- `surya-ocr 0.13.0` requires `Pillow<11.0.0,>=10.2.0`.
- `pdftext 0.6.2` requires `pypdfium2==4.30.0`.
- `surya-ocr 0.13.0` requires `pypdfium2==4.30.0`.

The environment was corrected by pinning:

- `pdfplumber==0.11.7`
- `Pillow==10.4.0`
- `pypdfium2==4.30.0`
- `pdfminer.six==20250506`

`pdfplumber==0.11.7` requires `Pillow>=9.1` and `pypdfium2>=4.18.0`, so these versions are compatible with the existing Marker/Surya/pdftext stack.
