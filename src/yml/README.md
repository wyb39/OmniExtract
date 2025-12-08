# OmniExtract YAML Configuration Guide

## Introduction

This directory contains YAML configuration file templates for OmniExtract, a tool designed for batch extraction of multi-attribute entity information from raw scientific research literature and supplementary materials.

## Usage Overview

OmniExtract provides YAML configuration file templates for various functionalities. Users can modify these templates and run the program via command line.

## Command Line Usage

The general command format is:

```
python run_cli.py <command> <config_file.yml>
```

### Available Commands and Configuration Files

| Function                                         | Command               | Configuration File               | Description                                                                                                                                                                                                 |
| ------------------------------------------------ | --------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model Configuration                              | modify_model          | model_config.yml                 | Configure online or locally deployed models for information extraction, prompt generation, result evaluation, and table extraction                                                                          |
| Document Parsing                                 | file_to_json          | file_to_json_config.yml          | Parse multiple raw documents (PDF, XML, Tex, etc.) into markdown files and further parse them into JSON files according to sections like Introduction, Method, Result, etc.                                 |
| Information Extraction                           | pred_original            | extraction_original_config.yml      | Batch extract information from parsed documents, outputting JSON or table files of extraction results                                                                                                       |
| Prompt Optimization - Dataset Building           | build_optm_set        | build_optm_set_config.yml        | Build prompt optimization dataset using curated original literature and curated table files (containing article_field column with original literature names), outputting JSON files for prompt optimization |
| Prompt Optimization - Generate Optimized Prompts | optim_custom          | optim_custom_config.yml          | Generate optimized instructions and examples after building the dataset                                                                                                                                     |
| Information Extraction with Optimized Prompts    | pred_optimized        | pred_optimized_config.yml        | Perform information extraction based on optimized settings and prompt results                                                                                                                               |
| Table Parsing                                    | parse_table_to_tsv    | parse_table_to_tsv_config.yml    | Parse tables in Excel, Tsv, Csv and other formats into TSV format                                                                                                                                           |
| Table Information Extraction                     | extract_table_service | extract_table_service_config.yml | Activate ReAct agent to extract table information and output results to TSV files after table parsing                                                                                                       |

## Configuration File Structure

Each configuration file follows a specific structure tailored to its functionality, generally including:

- File/directory path configurations
- Model settings
- Extraction parameters
- Output configurations

Refer to each individual configuration file template for detailed parameter explanations and examples.

## Directory Organization

- `extraction/`: Configuration files for information extraction tasks
- `optm/`: Configuration files for prompt optimization tasks
- `table/`: Configuration files for table processing tasks
- `model/`: Configuration files for model settings
- `file_parse/`: Configuration files for file parsing tasks

## Example Workflow

1. Configure model settings using `model_config.yml`
2. Parse raw documents using `file_to_json_config.yml`
3. Extract information using `extraction_original_config.yml`
4. (Optional) Build optimization dataset using `build_optm_set_config.yml`
5. (Optional) Generate optimized prompts using `optim_custom_config.yml`
6. (Optional) Perform optimized extraction using `pred_optimized_config.yml`
7. For table data: parse tables using `parse_table_to_tsv_config.yml` and extract information using `extract_table_service_config.yml`

## Notes

- Replace placeholder values in configuration files with your actual paths and settings
- For best results, follow the recommended workflow when using multiple functionalities
- Ensure all required dependencies are installed before running the commands
