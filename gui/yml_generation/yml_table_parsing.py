import os
import yaml
from typing import Dict, Any


def generate_table_parsing_yaml(
    file_folder_path: str,
    save_folder_path: str,
    non_tabular_file_format: str = "PDF",
    output_path: str = "assets/yml/parse_table_to_tsv_config.yml",
) -> str:
    """
    Generate YAML configuration file for table parsing

    Args:
        file_folder_path: Source file folder path
        save_folder_path: Result save folder path
        non_tabular_file_format: Non-tabular file format (PDF, scienceDirect, PMC, Arxiv)
        output_path: Output YAML file path

    Returns:
        Generated YAML string
    """

    # Build configuration dictionary
    config = {
        "file_folder_path": file_folder_path,
        "save_folder_path": save_folder_path,
        "non_tabular_file_format": non_tabular_file_format,
    }

    # Generate YAML string
    yaml_str = yaml.dump(
        config, default_flow_style=False, allow_unicode=True, sort_keys=False
    )

    # Add file header and comments
    header = """# parse_table_to_tsv Configuration File
# This file is used to configure parameters for table parsing functionality

"""

    usage_instructions = """
# Usage Instructions:
# 1. Set file_folder_path to the folder path containing source files with tables to parse
# 2. Set save_folder_path to the folder path where parsed TSV files will be saved
# 3. Set non_tabular_file_format parameter based on source file type, optional values are 'PDF', 'scienceDirect', 'PMC', 'Arxiv'
# 4. If file encoding is not utf-8, you can modify the encoding parameter
# 5. If you need to see detailed processing information, you can set verbose to true
"""

    # Combine complete YAML content
    full_yaml = header + yaml_str + usage_instructions

    # Save to file if output path is specified
    if output_path:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_yaml)

    return full_yaml


def extract_table_parsing_config_from_callback(
    table_folder_path: str, table_save_path: str, table_file_type: str
) -> Dict[str, Any]:
    """
    Extract data from table parsing form callback function

    Args:
        table_folder_path: Source folder path
        table_save_path: Save path
        table_file_type: File type

    Returns:
        Dictionary containing all table parsing configuration data
    """

    # Build configuration dictionary
    config = {
        "file_folder_path": table_folder_path,
        "save_folder_path": table_save_path,
        "non_tabular_file_format": table_file_type,
    }

    return config


def save_table_parsing_config_to_yaml(
    table_folder_path: str,
    table_save_path: str,
    table_file_type: str,
    output_dir: str = "assets/yml",
    filename: str | None = None,
) -> str:
    """
    Save table parsing configuration to YAML file

    Args:
        table_folder_path: Source folder path
        table_save_path: Save path
        table_file_type: File type
        output_dir: Output directory
        filename: File name, uses default file name if None

    Returns:
        Saved file path
    """

    # Use default file name if not specified
    if filename is None:
        filename = "parse_table_to_tsv_config.yml"

    # Build output path
    output_path = os.path.join(output_dir, filename)

    # Generate YAML configuration
    generate_table_parsing_yaml(
        file_folder_path=table_folder_path,
        save_folder_path=table_save_path,
        non_tabular_file_format=table_file_type,
        output_path=output_path,
    )

    return output_path


def generate_extract_table_yaml(
    parsed_file_path: str,
    save_folder_path: str,
    output_fields: list,
    classify_prompt: str,
    extract_prompt: str,
    num_threads: int = 6,
    encoding: str = "utf-8",
    output_path: str = "assets/yml/extract_table_service_config.yml",
) -> str:
    """
    Generate YAML configuration file for table extraction service

    Args:
        parsed_file_path: Parsed file path
        save_folder_path: Result save folder path
        output_fields: Output field configuration list
        classify_prompt: Classification prompt
        extract_prompt: Extraction prompt
        num_threads: Number of threads, default 6
        encoding: File encoding, default utf-8
        output_path: Output YAML file path

    Returns:
        Generated YAML string
    """

    # Build configuration dictionary
    config = {
        "parsed_file_path": parsed_file_path,
        "save_folder_path": save_folder_path,
        "outputFields": output_fields,
        "classify_prompt": classify_prompt,
        "extract_prompt": extract_prompt,
        "num_threads": num_threads,
        "encoding": encoding,
    }

    # Generate YAML string
    yaml_str = yaml.dump(
        config, default_flow_style=False, allow_unicode=True, sort_keys=False
    )

    # Add file header and comments
    header = """# extract_table_service Configuration File
# This file is used to configure parameters for table extraction service functionality

"""

    usage_instructions = """
# Usage Instructions:
# 1. Set parsed_file_path to the folder path containing parsed files
# 2. Set save_folder_path to the folder path where extracted tables will be saved
# 3. Configure outputFields with the list of fields you want to extract
# 4. Set appropriate prompts for classification and extraction
# 5. If you want to extract directly without classification, set extract_directly to true
# 6. Adjust num_threads according to your system capabilities
# 7. If file encoding is not utf-8, you can modify the encoding parameter
"""

    # Combine complete YAML content
    full_yaml = header + yaml_str + usage_instructions

    # Save to file if output path is specified
    if output_path:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_yaml)

    return full_yaml


def extract_table_extraction_config_from_callback(
    extract_parsed_file_path: str,
    extract_save_folder_path: str,
    extract_output_fields: list,
    extract_classify_prompt: str,
    extract_extract_prompt: str,
    extract_num_threads: int,
) -> Dict[str, Any]:
    """
    Extract data from table extraction form callback function

    Args:
        extract_parsed_file_path: Parsed file path
        extract_save_folder_path: Save folder path
        extract_output_fields: Output field configuration
        extract_classify_prompt: Classification prompt
        extract_extract_prompt: Extraction prompt
        extract_num_threads: Number of threads

    Returns:
        Dictionary containing all table extraction configuration data
    """

    # Build configuration dictionary
    config = {
        "parsed_file_path": extract_parsed_file_path,
        "save_folder_path": extract_save_folder_path,
        "outputFields": extract_output_fields,
        "classify_prompt": extract_classify_prompt,
        "extract_prompt": extract_extract_prompt,
        "num_threads": extract_num_threads,
        "encoding": "utf-8",  # Default encoding
    }

    return config


def save_extract_table_config_to_yaml(
    extract_parsed_file_path: str,
    extract_save_folder_path: str,
    extract_output_fields: list,
    extract_classify_prompt: str,
    extract_extract_prompt: str,
    extract_num_threads: int,
    output_dir: str = "assets/yml",
    filename: str | None = None,
) -> str:
    """
    Save table extraction configuration to YAML file

    Args:
        extract_parsed_file_path: Parsed file path
        extract_save_folder_path: Save folder path
        extract_output_fields: Output field configuration
        extract_classify_prompt: Classification prompt
        extract_extract_prompt: Extraction prompt
        extract_num_threads: Number of threads
        output_dir: Output directory
        filename: File name, uses default file name if None

    Returns:
        Saved file path
    """

    # Use default file name if not specified
    if filename is None:
        filename = "extract_table_service_config.yml"

    # Build output path
    output_path = os.path.join(output_dir, filename)

    # Generate YAML configuration
    generate_extract_table_yaml(
        parsed_file_path=extract_parsed_file_path,
        save_folder_path=extract_save_folder_path,
        output_fields=extract_output_fields,
        classify_prompt=extract_classify_prompt,
        extract_prompt=extract_extract_prompt,
        num_threads=extract_num_threads,
        output_path=output_path,
    )

    return output_path
