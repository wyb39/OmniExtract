import yaml
import os
from typing import Dict, List, Any


## ===============Document Extraction===================
def generate_novel_config_yaml(
    dataset_path: str,
    save_dir: str,
    input_fields: List[Dict[str, str]],
    output_fields: List[Dict[str, str]],
    initial_prompt: str,
    judging: str = "",
    task: str = "Extraction",
    threads: int = 6,
    multiple: bool = False,
    output_path: str | None = None,
) -> str:
    """
    Generate YAML configuration file based on Novel section form data

    Args:
        dataset_path: Dataset path
        save_dir: Save directory
        input_fields: List of input fields, each field contains name, field_type, description
        output_fields: List of output fields, each field contains name, field_type, description
        initial_prompt: Initial prompt
        judging: Evaluation mode
        task: Task type
        threads: Number of threads
        multiple: Whether to extract multiple entities
        output_path: Output file path, returns YAML string if None

    Returns:
        YAML string or saved file path
    """

    # Build configuration dictionary
    config = {
        "# PredictionSettings YAML Configuration": "",
        "# Configuration for dataset extraction from Nature Communications journal articles": "",
        "# Copy this file and modify according to your needs": "",
        "# === INPUT FIELDS ===": "",
        "# Define what data your model will receive as input": "",
        "inputFields": input_fields,
        "# === OUTPUT FIELDS ===": "",
        "# Define what your model should extract or generate from the input": "",
        "outputFields": output_fields,
        "# === PROMPT CONFIGURATION ===": "",
        "initial_prompt": initial_prompt,
        "# === DATA AND SAVE CONFIGURATION ===": "",
        "dataset": dataset_path,
        "save_dir": save_dir,
        "# === EVALUATION SETTINGS ===": "",
        "judging": judging,
        "# === PROCESSING SETTINGS ===": "",
        "task": task,
        "threads": threads,
        "multiple": multiple,
        "# === USAGE INSTRUCTIONS ===": "",
        "# 1. Replace dataset path with your actual dataset file path": "",
        "# 2. Modify save_dir to your desired output directory": "",
        "# 3. Update inputFields and outputFields to match your data structure": "",
        "# 4. Ensure your dataset has columns matching all field names": "",
        "# 5. Adjust threads based on your system's capabilities": "",
    }

    # Generate YAML string
    yaml_str = ""
    for key, value in config.items():
        if key.startswith("#"):
            # Comment line
            yaml_str += f"{key}\n"
        elif key == "initial_prompt":
            # Handle multi-line prompt
            yaml_str += f"{key}: |\n"
            for line in value.split("\n"):
                yaml_str += f"  {line}\n"
        elif isinstance(value, list):
            # List fields (inputFields and outputFields)
            yaml_str += f"{key}:\n"
            for item in value:
                yaml_str += f'  - name: "{item.get("name", "")}"\n'
                yaml_str += f'    field_type: "{item.get("field_type", "str")}"\n'
                if "description" in item:
                    yaml_str += f'    description: "{item["description"]}"\n'
                else:
                    yaml_str += "\n"

                # Add range_min property (if exists)
                if "range_min" in item and item["range_min"] is not None:
                    yaml_str += f"    range_min: {item['range_min']}\n"

                # Add range_max property (if exists)
                if "range_max" in item and item["range_max"] is not None:
                    yaml_str += f"    range_max: {item['range_max']}\n"

                # Add literal_list property (if exists)
                if "literal_list" in item and item["literal_list"] is not None:
                    if isinstance(item["literal_list"], list):
                        literal_str = ", ".join(
                            f'"{str(val)}"' for val in item["literal_list"]
                        )
                        yaml_str += f"    literal_list: [{literal_str}]\n"
                    elif isinstance(item["literal_list"], str):
                        # Handle comma-separated string
                        literal_items = [
                            val.strip()
                            for val in item["literal_list"].split(",")
                            if val.strip()
                        ]
                        if literal_items:
                            literal_str = ", ".join(f'"{val}"' for val in literal_items)
                            yaml_str += f"    literal_list: [{literal_str}]\n"
        else:
            # Regular field
            if isinstance(value, bool):
                yaml_str += f"{key}: {str(value).lower()}\n"
            elif isinstance(value, int):
                yaml_str += f"{key}: {value}\n"
            elif key in ["dataset", "save_dir"]:
                # Special handling for dataset and save_dir - don't wrap in quotes
                yaml_str += f"{key}: {value}\n"
            else:
                yaml_str += f'{key}: "{value}"\n'

    # Save to file if output path is specified
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_str)

        return output_path

    return yaml_str


def extract_form_data_from_callback(
    dataset_path: str,
    save_dir: str,
    judging: str,
    threads: int,
    # Other form fields need to be obtained from the callback function
    input_fields_data: List[Dict[str, str]] | None = None,
    output_fields_data: List[Dict[str, str]] | None = None,
    initial_prompt: str | None = None,
    task: str = "Extraction",
    multiple: bool = False,
) -> Dict[str, Any]:
    """
    Extract form data from callback function

    Args:
        dataset_path: Dataset path
        save_dir: Save directory
        judging: Evaluation mode
        threads: Number of threads
        input_fields_data: Input field data
        output_fields_data: Output field data
        initial_prompt: Initial prompt
        task: Task type
        multiple: Whether to extract multiple entities

    Returns:
        Dictionary containing all configuration data
    """

    # Use default values if no input field data is provided
    if input_fields_data is None:
        input_fields_data = [
            {
                "name": "Method",
                "field_type": "str",
                "description": "The method section of a Nature Communications journal article",
            }
        ]

    # Use default values if no output field data is provided
    if output_fields_data is None:
        output_fields_data = [
            {
                "name": "extracted_data",
                "field_type": "list",
                "description": "Extracted data from documents",
            }
        ]

    # Process range and literal properties of input fields
    for field in input_fields_data:
        # Clean up empty values, ensuring only actually set properties are included
        if "range_min" in field and (
            field["range_min"] == "" or field["range_min"] is None
        ):
            del field["range_min"]
        if "range_max" in field and (
            field["range_max"] == "" or field["range_max"] is None
        ):
            del field["range_max"]
        if "literal_list" in field and (
            not field["literal_list"]
            or field["literal_list"] == ""
            or field["literal_list"] == []
        ):
            del field["literal_list"]

    # Process range and literal properties of output fields
    for field in output_fields_data:
        # Clean up empty values, ensuring only actually set properties are included
        if "range_min" in field and (
            field["range_min"] == "" or field["range_min"] is None
        ):
            del field["range_min"]
        if "range_max" in field and (
            field["range_max"] == "" or field["range_max"] is None
        ):
            del field["range_max"]
        if "literal_list" in field and (
            not field["literal_list"]
            or field["literal_list"] == ""
            or field["literal_list"] == []
        ):
            del field["literal_list"]

    # Use default value if no initial prompt is provided
    if initial_prompt is None:
        initial_prompt = """You are a data analyst organizing the data usage in literature published in the journal Nature Communications. Your goals are: 1. Extract the identifiers of all public datasets mentioned in the literature 2. Extract the identifiers of all non-public datasets (i.e., self-created datasets) provided in the literature 3. Extract the names of all databases used in the literature. Please carefully read the methods section of the literature to completely and accurately extract the above information."""

    return {
        "dataset_path": dataset_path,
        "save_dir": save_dir,
        "input_fields": input_fields_data,
        "output_fields": output_fields_data,
        "initial_prompt": initial_prompt,
        "judging": judging,
        "task": task,
        "threads": threads,
        "multiple": multiple,
    }


def generate_novel_config_from_callback(
    dataset_path: str,
    save_dir: str,
    judging: str,
    threads: int,
    input_fields_data: List[Dict[str, str]] | None = None,
    output_fields_data: List[Dict[str, str]] | None = None,
    initial_prompt: str | None = None,
    task: str = "Extraction",
    multiple: bool = False,
    output_path: str | None = None,
) -> str:
    """
    Generate Novel configuration YAML file directly from callback function data

    Args:
        dataset_path: Dataset path
        save_dir: Save directory
        judging: Evaluation mode
        threads: Number of threads
        input_fields_data: Input field data
        output_fields_data: Output field data
        initial_prompt: Initial prompt
        task: Task type
        multiple: Whether to extract multiple entities
        output_path: Output file path

    Returns:
        YAML string or saved file path
    """

    # Extract form data
    form_data = extract_form_data_from_callback(
        dataset_path=dataset_path,
        save_dir=save_dir,
        judging=judging,
        threads=threads,
        input_fields_data=input_fields_data,
        output_fields_data=output_fields_data,
        initial_prompt=initial_prompt,
        task=task,
        multiple=multiple,
    )

    # Generate YAML configuration
    return generate_novel_config_yaml(
        dataset_path=form_data["dataset_path"],
        save_dir=form_data["save_dir"],
        input_fields=form_data["input_fields"],
        output_fields=form_data["output_fields"],
        initial_prompt=form_data["initial_prompt"],
        judging=form_data["judging"],
        task=form_data["task"],
        threads=form_data["threads"],
        multiple=form_data["multiple"],
        output_path=output_path,
    )


def generate_optimized_config_yaml(
    load_dir: str,
    dataset: str,
    save_dir: str,
    judging: str = "",
    output_file: str = "result.json",
    threads: int = 6,
    output_path: str | None = None,
) -> str:
    """
    Generate YAML configuration file based on Optimized section form data

    Args:
        load_dir: Optimization result loading directory
        dataset: Dataset path
        save_dir: Save directory
        judging: Evaluation mode
        output_file: Output file name
        threads: Number of threads
        output_path: Output file path, returns YAML string if None

    Returns:
        YAML string or saved file path
    """

    # Build configuration dictionary
    config = {
        "# YAML Configuration for Extraction tasks with optimization": "",
        "# Copy this file and modify according to your needs": "",
        "# === REQUIRED PATHS ===": "",
        "# Directory containing the optimized settings and prompts from previous optimization": "",
        "load_dir": load_dir,
        "# Dataset to run predictions on": "",
        "dataset": dataset,
        "# Directory where prediction results will be saved": "",
        "save_dir": save_dir,
        "# === EVALUATION SETTINGS ===": "",
        '# Evaluation mode: "confidence" (evaluate prediction confidence), "score" (evaluate prediction quality), or "" for no judgement': "",
        "judging": judging,
        "# === OUTPUT CONFIGURATION ===": "",
        "# Output file name for the prediction results": "",
        "output_file": output_file,
        "# === USAGE INSTRUCTIONS ===": "",
        "# 1. Ensure load_dir contains the optimization results:": "",
        "#    - optim_settings.json (optimized PredictionSettings)": "",
        "#    - optim_prompt.json (optimized prompts)": "",
        "# 2. Replace dataset path with your actual dataset file path": "",
        "# 3. Modify save_dir to your desired output directory": "",
        "# 4. The endpoint will load optimized settings and prompts from load_dir": "",
        "# 5. Predictions will be run on your dataset with the optimized configuration": "",
    }

    # Generate YAML string
    yaml_str = ""
    for key, value in config.items():
        if key.startswith("#"):
            # Comment line
            yaml_str += f"{key}\n"
        elif key in ["load_dir", "dataset", "save_dir"]:
            # Path fields, add comments - don't wrap in quotes
            yaml_str += f'{key}: {value}'
            if key == "load_dir":
                yaml_str += "  # Path to optimization results directory"
            elif key == "dataset":
                yaml_str += (
                    "  # Path to the dataset file (JSON, CSV, TSV, or Excel format)"
                )
            elif key == "save_dir":
                yaml_str += "  # Path to save results"
            yaml_str += "\n"
        elif key == "judging":
            # Evaluation mode field
            yaml_str += f'{key}: "{value}"'
            if value == "":
                yaml_str += "  # No evaluation"
            else:
                yaml_str += f"  # Evaluate prediction {value}"
            yaml_str += "\n"
        elif key == "output_file":
            # Output file field
            yaml_str += f'{key}: "{value}"'
            yaml_str += "  # Output file name for the prediction results"
            yaml_str += "\n"
        elif key == "threads":
            # Thread count field
            yaml_str += f"{key}: {value}\n"
        else:
            # Regular field
            if isinstance(value, int):
                yaml_str += f"{key}: {value}\n"
            else:
                yaml_str += f'{key}: "{value}"\n'

    # Save to file if output path is specified
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_str)

        return output_path

    return yaml_str


def generate_optimized_config_from_callback(
    load_dir: str,
    dataset: str,
    save_dir: str,
    judging: str = "",
    threads: int = 6,
    output_file: str = "result.json",
    output_path: str | None = None,
) -> str:
    """
    Generate Optimized configuration YAML file directly from callback function data

    Args:
        load_dir: Optimization result loading directory
        dataset: Dataset path
        save_dir: Save directory
        judging: Evaluation mode
        threads: Number of threads
        output_file: Output file name
        output_path: Output file path

    Returns:
        YAML string or saved file path
    """

    return generate_optimized_config_yaml(
        load_dir=load_dir,
        dataset=dataset,
        save_dir=save_dir,
        judging=judging,
        output_file=output_file,
        threads=threads,
        output_path=output_path,
    )