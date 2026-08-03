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
- Python 3.10 or higher
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
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```
**Please note that you may need to install right torch version depending on your system.**

### Start the GUI Service
After installing dependencies, start the local GUI service:
```bash
cd gui
python app.py
```
Make sure your virtual environment is activated.
Then open your browser and navigate to http://127.0.0.1:8050/ to use the tool.

The Jinja2 web UI is also available from the FastAPI service. Start it with
`python src/main.py`, then open http://127.0.0.1:9000/omniextract/. It uses the
same API handlers as the existing CLI service and serves the migrated templates
from `ui_jinja/templates`.

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

### Model runtime settings

The supported model providers are `openai`, `vllm`, `ollama`, `qwen`,
`deepseek`, `gemini`, `anthropic`, `sglang`, `openrouter`, and `custom`.
Provider prefixes are normalized, so both `gpt-4.1` and `openai/gpt-4.1`
produce the same OpenAI model identifier. Leave Gemini `api_base` empty to use
LiteLLM's provider URL generation. Qwen accepts either `DASHSCOPE_API_KEY` or
`QWEN_API_KEY` when no key is stored in the model settings.

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

Thinking/reasoning settings consist of:

- `thinking_enabled`: enables explicit provider controls when supported.
- `reasoning_effort`: `low`, `medium`, or `high`.
- `thinking_budget_tokens`: an optional provider-specific reasoning budget.

OpenAI, OpenRouter, Anthropic, Qwen, vLLM, SGLang, and compatible custom
endpoints receive provider-specific thinking parameters. With the pinned
LiteLLM adapter, DeepSeek, Gemini, and Ollama use the selected model's native
thinking behavior without additional request fields. Anthropic requires a
thinking budget of at least 1024 tokens, and `max_tokens` must be greater than
that budget.

Sampling settings are validated before an LM is created. `temperature` must be
between 0 and 2, `top_p` and `min_p` between 0 and 1, and `top_k` and token
limits must be positive. `top_k` is rejected for OpenAI and DeepSeek. `min_p`
is accepted only for vLLM, Ollama, SGLang, OpenRouter, and compatible custom
endpoints.

### Processing error reports

CLI commands and the four Jinja background workflows now create
`processing_report.json` beside their result files. A failed document is
isolated from the other documents in the same batch. Therefore, a workflow can
finish with `processing_status: "partial"` and still return all usable results.

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

`cached_input_tokens` records input tokens read from the provider's prompt
cache. `cache_creation_input_tokens` records newly cached input tokens when the
provider exposes that metric. Either cache field is `null` when the provider
does not return enough information to calculate a complete total. DSPy local
response-cache hits are not provider calls and are not included in these
totals.

Detailed parser/model tracebacks remain in the application logs. Prompt
optimization also keeps DSPy's original `optim.log`; its processing report
only explains the main failure and the recommended correction. An optimization
dataset must contain at least two complete, valid records.

For CLI runs, the JSON printed at completion includes `processing_report`.
When a task-level failure occurs, the CLI prints the report path and exits with
a non-zero status. For Jinja workflows, the status page lists affected
documents and exposes `processing_report.json` as a downloadable artifact.
The report is also included inside the returned result ZIP.

The production mechanism is concentrated in three commented modules:
`src/error_handling.py` defines the report contract, exception mapping and
isolated batch executor; `src/processing_adapters.py` provides the
single-document parsing boundary; and `src/token_usage.py` normalizes and
aggregates provider usage fields. The existing service, CLI and workflow files
only contain integration calls.

Runtime output under `process/`, the local `error_handling_experiment/`
workspace, and `settings/model_settings_*.json` are ignored by Git. Model
settings may contain credentials and must be configured locally on each
deployment.

For Linux server deployment, set a password to encrypt the model `api_key` and expose the service externally:
First, modify `gui/app.py` to listen on all interfaces by changing the default host to `0.0.0.0` (replace `os.environ.get("HOST", "127.0.0.1")` with `os.environ.get("HOST", "0.0.0.0")`).
```bash
cd gui
OMNI_EXTRACT_ENCRYPTION_KEY=YOUR_PASSWORD python app.py
```

### Use the Command Line and Configuration Files
You can start using OmniExtract through the command-line interface. Please refer to the README file in the src/yml directory for detailed configuration instructions.


## Important Notice

> **PDF input uses the integrated Hybrid/OpenDoc parser.**
> Hybrid is the production default. To switch the production flow to full
> OpenDoc, edit `PDF_PARSER_BACKEND` in `src/articleUtil.py` to `"opendoc"`.
> This switch is intentionally internal and does not change the existing
> CLI/API/YAML input parameters.
<!--
> Marker compatibility notice removed from the production documentation.
>
> Marker remains available as an explicit backend and fallback; please ensure
> compliance with Marker’s usage requirements and licensing terms.
> Refer to marker’s official documentation for details:
> https://github.com/datalab-to/marker
> https://github.com/datalab-to/marker/blob/master/README.md
-->
