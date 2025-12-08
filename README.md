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
- Python 3.8 or higher
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

For Linux server deployment, set a password to encrypt the model `api_key` and expose the service externally:
First, modify `gui/app.py` to listen on all interfaces by changing the default host to `0.0.0.0` (replace `os.environ.get("HOST", "127.0.0.1")` with `os.environ.get("HOST", "0.0.0.0")`).
```bash
cd gui
OMNI_EXTRACT_ENCRYPTION_KEY=YOUR_PASSWORD python app.py
```

### Use the Command Line and Configuration Files
You can start using OmniExtract through the command-line interface. Please refer to the README file in the src/yml directory for detailed configuration instructions.


## Important Notice

> **This project uses the marker tool to parse PDF files.**
> Please ensure compliance with marker’s usage requirements and licensing terms.
> Refer to marker’s official documentation for details:
> https://github.com/datalab-to/marker
> https://github.com/datalab-to/marker/blob/master/README.md
