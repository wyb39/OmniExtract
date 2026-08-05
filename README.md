# OmniExtract
OmniExtract is an LLM-based automatic extraction tool based on [DSPy](https://github.com/stanfordnlp/dspy), specifically designed for information extraction tasks from literature and documents. It utilizes prompt optimization engineering to enhance extraction performance based on curated data, and provides various file format parsing tools, supporting batch extraction of multi-property entities from original documents (in formats such as PDF or XML) and tabular files.

A video tutorial about OmniExtract is available at https://www.bilibili.com/video/BV12QywBhE1p.

## How to Extract Multi-Property Entities from Multiple Files

You can extract multi-property entities from multiple files by following these steps:

1. **Configure the Large Language Model**
   Set up the connection and parameters for the LLM you wish to use for extraction. This includes specifying model endpoints, API keys, and model-specific configurations.

2. **Parse the Files to be Extracted**
   Use the built-in file parsing utilities to process your source documents. OmniExtract supports various formats including PDF, XML, and tabular files.

3. **Optimize Prompts Based on Existing Data (Optional)**
   Leverage OmniExtract's prompt optimization engineering capabilities to refine your extraction prompts using curated data, which can significantly improve extraction accuracy and consistency.

4. **Extract Information from Documents or Tables**
   Execute the extraction process to retrieve multi-property entities from your parsed documents. The extracted data will be structured according to your specified output format.

## Key Features
- Batch processing of multiple documents
- Support for various file formats
- Prompt optimization for improved extraction performance
- Structured output of multi-property entities

## Quick Start

### Prerequisites
- Python 3.10
- Git

### Download the code
```bash
cd OmniExtract
```

### Create a Virtual Environment
Create a virtual environment to isolate the project dependencies:

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
Install the required Python packages using pip. It is recommended to install
torch separately first so the version matches your system:

1. Install the torch version that matches your system. Choose the command
   suitable for your platform from https://pytorch.org/get-started/locally/,
   for example:
   ```bash
   pip install torch
   ```
2. Remove the `torch` dependency from `requirements.txt` (it was already
   installed in step 1; reinstalling it would overwrite the version you chose).
3. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Please note that you need to install the correct torch version for your
system, and you must install torch first before running
`pip install -r requirements.txt`.**

### Start the Web UI Service
After installing dependencies, start the web UI service:
```bash
python src/main.py
```
Make sure your virtual environment is activated.
Then open your browser and navigate to http://127.0.0.1:9000/omniextract/ to
use the tool. It uses the same API handlers as the existing CLI service and
serves the migrated templates from `ui_jinja/templates`.

The Jinja pages submit four background workflows under the same prefix:
`run_workflow_doc_extraction`, `run_workflow_doc_extraction_optimized`,
`run_workflow_prompt_optimization`, and `run_workflow_table_extraction`.
Each request returns a `workflow_id` and a tokenized `workflow_url`. Open that
URL to view the workflow status and download the result when it is ready. The
page refreshes itself every five minutes, so no browser polling code is needed.
The JSON status endpoint is available at `/omniextract/api/workflow/{workflow_id}/status`
when called with the returned token. Result files are served only through the
tokenized artifact links shown on the status page; local filesystem paths are
never exposed.

Contact email is optional. If it is omitted, the workflow remains fully usable
through the returned status URL and does not attempt an email notification.

Email notifications are optional. Configure `OMNI_EXTRACT_SMTP_SENDER` and
`OMNI_EXTRACT_SMTP_PASSWORD` (plus the optional SMTP server variables) before
starting the service to enable them.

The older Dash-based GUI is still available. Start it with `python -m gui.app`
and open http://127.0.0.1:8050/ in your browser to use the tool.

For Linux server deployment, set a password to encrypt the model `api_key` and expose the service externally.
The GUI already listens on `0.0.0.0` by default, so just set the encryption key and start it:
```bash
OMNI_EXTRACT_ENCRYPTION_KEY=YOUR_PASSWORD python -m gui.app
```

### Model runtime settings

Two independent DSPy response-cache switches are available as service startup
flags of `python src/main.py`:

- `--cache-for-optimization` / `--no-cache-for-optimization` (default: enabled)
  applies to `optim`, `optim_custom`, image prompt optimization, and their
  model-based metrics.
- `--cache-for-other` / `--no-cache-for-other` (default: disabled) applies to
  prediction, judging, parsing, table extraction, and model connection tests.

These switches control DSPy's local exact-response cache. They do not enable
or disable a provider's prompt cache. Calls served by DSPy's response cache are
excluded from `model_calls`, `input_tokens`, and `output_tokens` in the
processing report.

### Processing error reports

CLI commands and the four Jinja background workflows create
`processing_report.json` beside their result files. A failed document is
isolated from the other documents in the same batch, so a workflow can finish
with `processing_status: "partial"` and still return all usable results.

The report contains the workflow result, affected documents, and aggregated
provider-reported model token usage:

```json
{
  "workflow_id": "20260730_182514_567020_43c4ac02",
  "processing_status": "partial",
  "failed_documents": [
    {
      "document_id": "broken.pdf",
      "issues": [
        {
          "stage": "markdown_convert",
          "code": "MARKDOWN_GENERATION_FAILED",
          "message": "The parser did not generate Markdown output",
          "action": "Check the source document or try another supported parser.",
          "retryable": false
        }
      ]
    }
  ],
  "token_usage": {
    "model_calls": 3,
    "input_tokens": 12480,
    "output_tokens": 936,
    "cached_input_tokens": 8192,
    "cache_creation_input_tokens": null
  }
}
```

The `token_usage` fields aggregate provider-reported token usage; either cache
field is `null` when the provider does not return enough information.

The supported error codes, grouped by category, are:

**Input / source / parsing issues**

| Code | Meaning / Suggested action |
|---|---|
| `SOURCE_NOT_FOUND` | Input file does not exist. Check and re-upload it. |
| `SOURCE_INVALID` | Input file is invalid. Check the file format and encoding. |
| `FILE_ACCESS_DENIED` | File or directory access denied. Check permissions and retry. |
| `DOCUMENT_PARSE_FAILED` | Document could not be parsed. Check the format and re-upload. |
| `MARKDOWN_GENERATION_FAILED` | Markdown conversion failed. Check the document or try another parser. |
| `JSON_CONVERSION_FAILED` | JSON conversion failed. Check the generated Markdown and section structure. |
| `TABLE_PARSE_FAILED` | Table parsing failed. Check the document/table format and retry. |

**Model / provider issues**

| Code | Meaning / Suggested action |
|---|---|
| `MODEL_TIMEOUT` (retryable) | Model request timed out. Check the model service and reduce threads. |
| `MODEL_RATE_LIMITED` (retryable) | Rate limited. Wait and retry, or reduce the number of threads. |
| `MODEL_UNAVAILABLE` (retryable) | Model service unavailable. Check the API base and provider status, then retry. |
| `MODEL_AUTH_FAILED` | Authentication failed. Check the configured model credentials. |

**Prediction / extraction issues**

| Code | Meaning / Suggested action |
|---|---|
| `PREDICTION_FAILED` | Prediction failed. Check the input fields and model service, then retry. |
| `JUDGEMENT_FAILED` | Judging failed. Check the judge model settings; the result may still be usable. |
| `TABLE_EXTRACTION_FAILED` | Table extraction failed. Check the table prompts and model service, then retry. |

**Prompt optimization issues**

| Code | Meaning / Suggested action |
|---|---|
| `OPTIMIZATION_FAILED` | Prompt optimization failed. Check the dataset and `optim.log`, then retry. |
| `OPTIM_DATASET_NOT_FOUND` | Optimization dataset missing. Set `dataset` to an existing JSON, CSV, TSV or XLSX file. |
| `OPTIM_DATASET_EMPTY` | Optimization dataset is empty. Add at least two complete records. |
| `OPTIM_DATASET_TOO_SMALL` | Optimization dataset is too small. Provide at least two complete, valid records. |

**Output issues**

| Code | Meaning / Suggested action |
|---|---|
| `OUTPUT_WRITE_FAILED` | Could not write the output. Check permissions, free space and file locks. |

**Fallback**

| Code | Meaning / Suggested action |
|---|---|
| `TASK_FAILED` | Unclassified task failure. Check the task input and application log, then retry. |

`retryable: true` issues can typically be resolved by retrying. Detailed
parser/model tracebacks remain in the application logs. For CLI runs, the JSON
printed at completion includes `processing_report`; for Jinja workflows, the
status page lists affected documents and exposes the report as a downloadable
artifact, and the report is also included inside the returned result ZIP.

Runtime output under `process/`, the local `error_handling_experiment/`
workspace, and `settings/model_settings_*.json` are ignored by Git. Model
settings may contain credentials and must be configured locally on each
deployment.

### Use the Command Line and Configuration Files
You can start using OmniExtract through the command-line interface. Please refer to the README file in the src/yml directory for detailed configuration instructions.


## Important Notice

> **PDF input uses the Marker-based parser.**
> Marker is the production PDF backend on main. The backend name is fixed in
> `src/parsing/articleUtil.py` via `PDF_PARSER_BACKEND` (default `"marker"`) and
> is intentionally internal: it does not change the existing CLI/API/YAML input
> parameters. Please ensure compliance with Marker's usage requirements and
> licensing terms.
> Refer to Marker's official documentation for details:
> https://github.com/datalab-to/marker
> https://github.com/datalab-to/marker/blob/master/README.md
