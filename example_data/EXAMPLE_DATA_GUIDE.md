# OmniExtract Example Data Guide

## Overview

This guide explains how to use the example data provided with OmniExtract for demonstrating document information extraction capabilities. The example data consists of **AKC (American Kennel Club) breed standard files** that have been carefully prepared for testing and demonstration purposes.

## Dataset Information

- **Source**: American Kennel Club (AKC) breed standard documents
- **Training Data**: 62 breed standard files with manually extracted information for curation
- **Test Data**: 100 breed standard files for testing the extraction pipeline
- **Extracted Properties**: 39 physical and behavioral traits including size, color, height, weight, coat characteristics, and personality traits

## Prerequisites

Before using the example data, ensure you have:

1. **Followed the main project README** to set up your environment
2. **Created a virtual environment** as specified in the installation instructions
3. **Installed all dependencies** using the project's requirements

## Setup Instructions

### 1. Download Breed Standard Files

Run the download script to fetch the AKC breed standard PDF files:

```bash
cd /data/OmniExtract_code/example_data
python download_breed_pdfs.py
```

This script will download the breed standard PDF files from AKC sources and save them in the appropriate directory.

### 2. Generate Predefined Configuration Files

To simplify testing, generate the predefined form and YML configuration files:

```bash
cd /data/OmniExtract_code/example_data
python generate_example_tasks.py
```

This script creates:
- Predefined extraction forms for breed standard files
- YML configuration files for the extraction tasks
- Directory structure for organizing extraction results

### 3. Start the Service

Follow the main project instructions to start the OmniExtract service.

### 4. (Optional) Watch the Tutorial Guide

If available, watch the operation guide or tutorial videos to understand the workflow before proceeding.

### 5. Configure Model Settings

In the **Model Config** section of the application:
- Configure your preferred LLM model (OpenAI, Anthropic, etc.)
- Set appropriate API keys and parameters
- Ensure the model is properly connected and tested

## Document Information Extraction Workflow

### Option A: Original Extraction (Without Prompt Optimization)

#### 8.1 Document Parsing
1. Go to **Document Parsing** section
2. Load the "document parsing for breed files (for curation)" task form
3. Run the parsing task on the downloaded PDF files
4. The system will parse and prepare the documents for extraction

#### 8.2 Document Extraction
1. Navigate to **Document Extraction** → **Original** section
2. Load the "breed files curation (original)" task
3. Run the extraction task
4. Review the extracted information from breed standard files

### Option B: Optimized Extraction (With Prompt Optimization)

#### 9.1 Document Parsing for Optimized Workflow
1. In **Document Parsing**, load "document parsing for breed files (curated)" task
2. Run the parsing task (if you haven't done step 8.1)

#### 9.2 Build Optimization Dataset
1. Run the **Build Optimization Dataset** module
2. Execute "build optimization dataset for breed" task
3. This creates a dataset for prompt optimization based on curated data

#### 9.3 Prompt Optimization
1. Go to **Prompt Optimization** section
2. Run "prompt optimization for breed" task
3. The system will optimize extraction prompts using the curated dataset

#### 9.4 Document Parsing (If Needed)
1. If you skipped step 8.1, load "document parsing for breed files (for curation)" task
2. Run parsing on the PDF files

#### 9.5 Optimized Document Extraction
1. Navigate to **Document Extraction** → **Optimized** section
2. Load "breed files curation (optimized)" task
3. Run the extraction with optimized prompts

## EWAS Table Extraction

- Test data source: Tables S2 and S3 from `https://pmc.ncbi.nlm.nih.gov/articles/PMC6022498/`
- Use `download_ewas_tables.py` to download data into `example_data/ewas_sup_tables`
- Use `generate_example_tasks.py` to generate example forms (if not already run)
- In the `Table Extraction` module, run `EWAS table parsing` under `Table Files Parsing`
- In `Extraction From Tables`, run the `EWAS table extraction` task
- Results are available in `example_data/ewas_sup_tables_extract_result/format_tables`; irrelevant Table S2 will not produce results, while Table S3 will generate the corresponding `tsv` file

Example commands:

```bash
cd /data/OmniExtract_code/example_data
python download_ewas_tables.py
python generate_example_tasks.py
```
