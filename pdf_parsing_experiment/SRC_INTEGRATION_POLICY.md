# Production PDF integration policy

This policy supersedes the earlier migration draft in `SRC_INTEGRATION_PLAN.md`.

- The production PDF path is the integrated parser in `src/pdf_parser.py`.
- `src/articleUtil.py` uses `PDF_PARSER_BACKEND = "hybrid"` by default.
- To run full OpenDoc, change that internal constant to `"opendoc"` and
  provision models under `models/opendoc` or set
  `OMNIEXTRACT_OPENDOC_MODEL_DIR`.
- Marker is not a production dependency, backend, or fallback.
- The `native` backend remains available only for experiment/comparison code;
  it is not exposed through production article, service, CLI, API, GUI, or YAML
  inputs.
- Existing public input models and function signatures remain unchanged. New
  parser tuning parameters are intentionally internal implementation details.
