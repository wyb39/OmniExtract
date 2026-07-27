# PDF parser production integration plan

## Scope

Move the validated parser and Markdown renderer from `pdf_parsing_experiment`
into `src`, while preserving the existing document, table, CLI, API, GUI, YAML,
Markdown, and JSON contracts.

## Production decisions

1. The production PDF path uses the migrated parser in `src/pdf_parser.py` and
   renderer in `src/pdf_markdown_renderer.py`.
2. The production default is Hybrid (`OpenDoc layout + PDFium native text +
   selective UniRec`).
3. Marker is not part of the production flow and is not a fallback.
4. The `native` backend is retained only for experiment/comparison code and is
   not exposed through production interfaces.
5. Existing public inputs remain unchanged. No parser tuning fields are added
   to Pydantic models, API payloads, CLI data, GUI forms, or YAML templates.

## Implementation layout

```text
src/
├── pdf_parser.py
├── pdf_markdown_renderer.py
├── articleUtil.py       # fixed internal PDF_PARSER_BACKEND switch
├── tableUtil.py         # PDF tables use the same production parser
├── service.py
├── params.py
├── cli/
└── router/
```

`pdf_parsing_experiment/pdf_to_markdown.py` and
`pdf_parsing_experiment/markdown_renderer.py` are thin experiment wrappers;
they are not production compatibility or fallback layers.

## Internal OpenDoc switch

The only production backend switch is deliberately internal:

```python
# src/articleUtil.py
PDF_PARSER_BACKEND = "hybrid"  # change to "opendoc" for full OpenDoc
```

OpenDoc models are resolved from `OMNIEXTRACT_OPENDOC_MODEL_DIR`, the repository
`models/opendoc` directory, or the transitional experiment model directory.

## Verification and rollout

- Compile `src`, `gui`, and the experiment package.
- Run the full experiment/integration unittest suite.
- Verify old `PathSettings` and `TableExtractionParams` payloads still parse.
- Verify `file_to_md`, `file_to_json`, and `parse_table_to_tsv` retain their
  previous call signatures.
- Run representative corpus PDFs in a GPU/model-provisioned environment.
- Compare Hybrid and full OpenDoc output quality before changing the internal
  switch for a deployment.

The last corpus-quality step requires the host GPU/model environment and is not
performed automatically by the unit-test suite.
