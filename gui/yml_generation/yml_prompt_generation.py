import os
from typing import Dict, Any, List, Optional


def generate_prompt_optimization_yaml(
    dataset_path: str,
    save_dir: str,
    experiment_name: str,
    input_fields: List[Dict[str, Any]],
    output_fields: List[Dict[str, Any]],
    initial_prompt: str,
    task: str,
    optim_burden: str,
    threads: int,
    demos: int,
    multiple: bool,
    ai_evaluation: bool,
    recall_prior: bool = False,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate YAML configuration file based on prompt optimization form data

    Args:
        dataset_path: Dataset path
        save_dir: Save directory
        experiment_name: Experiment name
        input_fields: List of input fields
        output_fields: List of output fields
        initial_prompt: Initial prompt template
        task: Task type (QA, Extraction)
        optim_burden: Optimization burden (light, medium, heavy)
        threads: Number of threads
        demos: Number of examples
        multiple: Whether to extract multiple entities
        ai_evaluation: Whether to use AI evaluation
        recall_prior: Whether to prioritize recall
        output_path: Output file path, returns YAML string if None

    Returns:
        YAML string or saved file path
    """

    # Build YAML content
    yaml_content = {
        "# Template Configuration for Optim Custom": None,
        "# Copy this file and modify according to your needs": None,
        "": None,
        "# === INPUT FIELDS ===": None,
        "# Define what data your model will receive": None,
        "inputFields": input_fields,
        "": None,
        "# === OUTPUT FIELDS ===": None,
        "# Define what your model should extract or generate": None,
        "outputFields": output_fields,
        "": None,
        "# === PROMPT CONFIGURATION ===": None,
        "initial_prompt": initial_prompt,
        "": None,
        "# === DATA AND SAVE CONFIGURATION ===": None,
        "dataset": dataset_path,
        "save_dir": save_dir,
        "": None,
        "# === OPTIMIZATION SETTINGS ===": None,
        "task": task,
        "optim_burden": optim_burden,
        "threads": threads,
        "demos": demos,
        "": None,
        "# === EXTRACTION MODE ===": None,
        "multiple": multiple,
        "recall_prior": recall_prior,
        "ai_evaluation": ai_evaluation,
        "": None,
        "# === USAGE INSTRUCTIONS ===": None,
        "# 1. Replace all placeholder paths with your actual file paths": None,
        "# 2. Modify inputFields and outputFields to match your data structure": None,
        "# 3. Update initial_prompt with your specific task instructions": None,
        "# 4. Ensure your dataset CSV has columns matching all field names": None,
        "# 5. Run with: python cli_handler.py optim_custom path/to/this/config.yml": None,
    }

    # Generate YAML string
    yaml_str = ""
    for key, value in yaml_content.items():
        if key.startswith("#"):
            yaml_str += f"{key}\n"
        elif key == "":
            yaml_str += "\n"
        elif key == "initial_prompt":
            yaml_str += f"{key}: |\n"
            for line in value.split("\n"):
                yaml_str += f"  {line}\n"
        elif key == "inputFields":
            yaml_str += f"{key}:\n"
            for field in value:
                yaml_str += '  - name: "{}"\n'.format(field.get("name", ""))
                yaml_str += '    field_type: "{}"\n'.format(
                    field.get("field_type", "str")
                )
                yaml_str += '    description: "{}"\n'.format(
                    field.get("description", "")
                )

                # Add optional fields
                if "range" in field:
                    yaml_str += "    range: {}\n".format(field["range"])
                if "literal" in field:
                    yaml_str += "    literal: {}\n".format(field["literal"])
        elif key == "outputFields":
            yaml_str += f"{key}:\n"
            for field in value:
                yaml_str += '  - name: "{}"\n'.format(field.get("name", ""))
                yaml_str += '    field_type: "{}"\n'.format(
                    field.get("field_type", "str")
                )
                yaml_str += '    description: "{}"\n'.format(
                    field.get("description", "")
                )

                # Add optional fields
                if "range" in field:
                    yaml_str += "    range: {}\n".format(field["range"])
                if "literal" in field:
                    yaml_str += "    literal: {}\n".format(field["literal"])
        else:
            yaml_str += f"{key}: {value}\n"

    # Save to file if output path is specified
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_str)

        return output_path

    return yaml_str


def extract_prompt_optimization_config_from_callback(
    dataset_path: str,
    save_dir: str,
    experiment_name: str,
    input_fields_data: List[Dict[str, Any]],
    output_fields_data: List[Dict[str, Any]],
    initial_prompt: str,
    task: str,
    optim_burden: str,
    threads: int,
    demos: int,
    multiple: bool,
    ai_evaluation: bool,
    recall_prior: bool = False,
) -> Dict[str, Any]:
    """
    Extract data from prompt optimization form callback function

    Args:
        dataset_path: Dataset path
        save_dir: Save directory
        experiment_name: Experiment name
        input_fields_data: Input field data
        output_fields_data: Output field data
        initial_prompt: Initial prompt template
        task: Task type
        optim_burden: Optimization burden
        threads: Number of threads
        demos: Number of examples
        multiple: Whether to extract multiple entities
        ai_evaluation: Whether to use AI evaluation
        recall_prior: Whether to prioritize recall

    Returns:
        Dictionary containing all prompt optimization configuration data
    """

    # Process input fields
    input_fields = []
    for field in input_fields_data:
        field_config = {
            "name": field.get("name", ""),
            "field_type": field.get("field_type", "str"),
            "description": field.get("description", ""),
        }

        # Add optional fields
        if "range_min" in field and "range_max" in field:
            field_config["range"] = [field["range_min"], field["range_max"]]

        if "literal_list" in field:
            field_config["literal"] = field["literal_list"]

        input_fields.append(field_config)

    # Process output fields
    output_fields = []
    for field in output_fields_data:
        field_config = {
            "name": field.get("name", ""),
            "field_type": field.get("field_type", "str"),
            "description": field.get("description", ""),
        }

        # Add optional fields
        if "range_min" in field and "range_max" in field:
            field_config["range"] = [field["range_min"], field["range_max"]]

        if "literal_list" in field:
            field_config["literal"] = field["literal_list"]

        output_fields.append(field_config)

    # Build configuration dictionary
    config = {
        "dataset_path": dataset_path,
        "save_dir": save_dir,
        "experiment_name": experiment_name,
        "input_fields": input_fields,
        "output_fields": output_fields,
        "initial_prompt": initial_prompt,
        "task": task,
        "optim_burden": optim_burden,
        "threads": threads,
        "demos": demos,
        "multiple": multiple,
        "ai_evaluation": ai_evaluation,
        "recall_prior": recall_prior,
    }

    return config


def save_prompt_optimization_config_to_yaml(
    dataset_path: str,
    save_dir: str,
    experiment_name: str,
    input_fields_data: List[Dict[str, Any]],
    output_fields_data: List[Dict[str, Any]],
    initial_prompt: str,
    task: str,
    optim_burden: str,
    threads: int,
    demos: int,
    multiple: bool,
    ai_evaluation: bool,
    recall_prior: bool = False,
    output_dir: str = "assets/yml",
    filename: str | None = None,
) -> str:
    """
    Save prompt optimization configuration to YAML file

    Args:
        dataset_path: Dataset path
        save_dir: Save directory
        experiment_name: Experiment name
        input_fields_data: Input field data
        output_fields_data: Output field data
        initial_prompt: Initial prompt template
        task: Task type
        optim_burden: Optimization burden
        threads: Number of threads
        demos: Number of examples
        multiple: Whether to extract multiple entities
        ai_evaluation: Whether to use AI evaluation
        recall_prior: Whether to prioritize recall
        output_dir: Output directory
        filename: File name, uses default file name if None

    Returns:
        Saved file path
    """

    # Generate filename based on experiment name if not specified
    if filename is None:
        safe_name = "".join(
            c for c in experiment_name if c.isalnum() or c in ("-", "_")
        ).rstrip()
        filename = (
            f"optim_{safe_name}_config.yml" if safe_name else "optim_custom_config.yml"
        )

    # Build output path
    output_path = os.path.join(output_dir, filename)

    # Generate YAML configuration
    generate_prompt_optimization_yaml(
        dataset_path=dataset_path,
        save_dir=save_dir,
        experiment_name=experiment_name,
        input_fields=input_fields_data,
        output_fields=output_fields_data,
        initial_prompt=initial_prompt,
        task=task,
        optim_burden=optim_burden,
        threads=threads,
        demos=demos,
        multiple=multiple,
        ai_evaluation=ai_evaluation,
        recall_prior=recall_prior,
        output_path=output_path,
    )

    return output_path


def create_input_field(
    name: str,
    field_type: str,
    description: str,
    range_min: Any = None,
    range_max: Any = None,
    literal_list: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Create input field configuration

    Args:
        name: Field name
        field_type: Field type
        description: Field description
        range_min: Minimum range value
        range_max: Maximum range value
        literal_list: Literal list

    Returns:
        Input field configuration dictionary
    """
    field: Dict[str, Any] = {
        "name": name,
        "field_type": field_type,
        "description": description,
    }

    if range_min is not None and range_max is not None:
        field["range_min"] = range_min
        field["range_max"] = range_max

    if literal_list:
        field["literal_list"] = literal_list

    return field


def create_output_field(
    name: str,
    field_type: str,
    description: str,
    range_min: Any = None,
    range_max: Any = None,
    literal_list: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Create output field configuration

    Args:
        name: Field name
        field_type: Field type
        description: Field description
        range_min: Minimum range value
        range_max: Maximum range value
        literal_list: Literal list

    Returns:
        Output field configuration dictionary
    """
    field: Dict[str, Any] = {
        "name": name,
        "field_type": field_type,
        "description": description,
    }

    if range_min is not None and range_max is not None:
        field["range_min"] = range_min
        field["range_max"] = range_max

    if literal_list is not None:
        field["literal_list"] = literal_list

    return field
