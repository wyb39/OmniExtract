import yaml
import os
from typing import Dict, Any, Optional


def generate_document_parsing_yaml(
    folder_path: str,
    save_path: str,
    file_type: str,
    convert_mode: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate YAML configuration file based on document parsing form data

    Args:
        folder_path: Source folder path
        save_path: Save path
        file_type: File type (PDF, scienceDirect, PMC, Arxiv)
        convert_mode: Conversion mode (byPart, wholeDoc)
        output_path: Output file path, returns YAML string if None

    Returns:
        YAML string or saved file path
    """

    # Directly build YAML content that completely matches the template
    yaml_str = f"""# YAML Configuration for File to JSON Conversion
# Copy this file and modify according to your needs

# === FILE SOURCE CONFIGURATION ===
# Define the source folder containing files to be converted

folder_path: {folder_path}  # Path to the folder containing source files

# === OUTPUT CONFIGURATION ===
# Define where the converted JSON files will be saved
save_path: {save_path}  # Path to save JSON results

# === FILE TYPE CONFIGURATION ===
# Specify the type of files to be processed
# Optional values: 'PDF', 'scienceDirect', 'PMC', 'Arxiv'
file_type: "{file_type}"

# === CONVERSION MODE ===
# Choose the conversion strategy
# Optional values: 'byPart' (divide the document by article parts), 'wholeDoc' (convert entire document as one)
convert_mode: "{convert_mode}"

# === USAGE INSTRUCTIONS ===
# 1. Replace folder_path with your actual source folder path
# 2. Modify save_path to your desired output directory
# 3. Update file_type according to your source file format
# 4. Choose appropriate convert_mode based on your requirements
# 5. The converted JSON files can be used for building datasets for prompt optimization or extraction tasks"""

    # Save to file if output path is specified
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_str)

        return output_path

    return yaml_str


def extract_document_parsing_config_from_callback(
    folder_path: str, save_path: str, file_type: str, convert_mode: str
) -> Dict[str, Any]:
    """
    Extract data from document parsing form callback function

    Args:
        folder_path: Source folder path
        save_path: Save path
        file_type: File type
        convert_mode: Conversion mode

    Returns:
        Dictionary containing all document parsing configuration data
    """

    # Build configuration dictionary
    config = {
        "folder_path": folder_path,
        "save_path": save_path,
        "file_type": file_type,
        "convert_mode": convert_mode,
    }

    return config


def save_document_parsing_config_to_yaml(
    folder_path: str,
    save_path: str,
    file_type: str,
    convert_mode: str,
    output_dir: str = "assets/yml",
    filename: str | None = None,
) -> str:
    """
    Save document parsing configuration to YAML file

    Args:
        folder_path: Source folder path
        save_path: Save path
        file_type: File type
        convert_mode: Conversion mode
        output_dir: Output directory
        filename: File name, uses default file name if None

    Returns:
        Saved file path
    """

    # Use default file name if not specified
    if filename is None:
        filename = "file_to_json_config.yml"

    # Build output path
    output_path = os.path.join(output_dir, filename)

    # Generate YAML configuration
    generate_document_parsing_yaml(
        folder_path=folder_path,
        save_path=save_path,
        file_type=file_type,
        convert_mode=convert_mode,
        output_path=output_path,
    )

    return output_path
