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

## OpenDoc OCR fallback

OpenDoc was added through:

- `openocr-python==0.1.5`
- `onnxruntime==1.23.2`
- `PyMuPDF==1.28.0`
- `opencv-python==4.11.0.86`

Compatibility constraints:

- Keep `numpy==1.24.3`; an unconstrained install selected OpenCV 5 and attempted
  to upgrade NumPy to 2.x.
- Keep both OpenCV distributions at `4.11.0.86` because the existing application
  uses `opencv-python-headless`, while OpenOCR declares `opencv-python`.
- `pip check` reports no broken requirements.
- The installed ONNX Runtime currently exposes `CPUExecutionProvider`, not CUDA.

OpenDoc model files are stored under
`pdf_parsing_experiment/models/opendoc/` and ignored by Git. The required layout,
encoder, decoder, and tokenizer files total approximately 943 MB.
