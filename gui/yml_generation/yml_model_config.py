import yaml
import os
from typing import Dict, Any, Optional


def generate_model_config_yaml(
    model_name: str,
    model_type: str,
    api_base: str,
    api_key: str | None,
    model_usage: str,
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate YAML configuration file based on model configuration form data

    Args:
        model_name: Model name
        model_type: Model type (openai, vllm, ollama, qwen, deepseek, gemini, anthropic, sglang, custom)
        api_base: API base URL
        api_key: API key
        model_usage: Model usage (main, prompt_generation, judge, coder)
        temperature: Temperature parameter (0.0-2.0)
        max_tokens: Maximum number of tokens
        top_p: Top-p sampling parameter (optional)
        top_k: Top-k sampling parameter (optional)
        min_p: Min-p sampling parameter (optional)
        output_path: Output file path, returns YAML string if None

    Returns:
        YAML string or saved file path
    """

    # Build configuration dictionary
    config = {
        "# Model Configuration Template": "",
        "# Copy this file and modify according to your needs": "",
        "# This file contains the configuration for ModelSettings class as defined in src/model.py": "",
        "# === MODEL IDENTIFICATION ===": "",
        "# Define the model name and provider type": "",
        "model_name": model_name,
        "model_type": model_type,
        "# Valid values: openai, vllm, ollama, qwen, deepseek, gemini, anthropic, sglang, custom": "",
        "# === API CONFIGURATION ===": "",
        "# Configure API endpoint and authentication": "",
        "api_base": api_base,
        "# Default URLs for different providers:": "",
        "# - openai: https://api.openai.com/v1": "",
        "# - vllm: http://localhost:8000": "",
        "# - ollama: http://localhost:11434": "",
        "# - qwen: https://dashscope.aliyuncs.com/compatible-mode/v1": "",
        "# - deepseek: https://api.deepseek.com": "",
        "# - gemini: https://generativelanguage.googleapis.com/v1beta": "",
        "# - anthropic: https://api.anthropic.com": "",
        "# - sglang: http://localhost:30000": "",
        "# - custom: [User provided URL]": "",
        "api_key": api_key if api_key and api_key != "YOUR-API-KEY" else "YOUR-API-KEY",
        "# API key for authentication (string or null)": "",
        "# === MODEL USAGE CONFIGURATION ===": "",
        "# Define how the model will be used in the system": "",
        "model_usage": model_usage,
        "# Valid values: main, prompt_generation, judge, coder": "",
        "# - main: Primary model for general tasks, if the model for prompt generation/judge/coding is not specified, this model will be used instead": "",
        "# - prompt_generation: Model for generating prompts": "",
        "# - judge: Model for evaluation and scoring": "",
        "# - coder: Model for code generation and programming tasks": "",
        "# === SAMPLING PARAMETERS ===": "",
        "# Control model output behavior and randomness": "",
        "temperature": temperature,
        "# Temperature for model output randomness (float)": "",
        "# Range: 0.0 to 2.0, lower values make output more deterministic": "",
        "max_tokens": max_tokens if max_tokens else 2500,
        "# Maximum tokens for model response (integer or null)": "",
        "# Set to null for no limit, or specify a positive integer": "",
        "top_p": top_p if top_p is not None else "null",
        "# Top-p sampling parameter (float between 0.0 and 1.0, or null)": "",
        "# Alternative to temperature, controls diversity via nucleus sampling": "",
        "top_k": top_k if top_k is not None else "null",
        "# Top-k sampling parameter (positive integer or null)": "",
        "# Controls diversity by considering only the top k tokens": "",
        "min_p": min_p if min_p is not None else "null",
        "# Minimum probability threshold (float between 0.0 and 1.0, or null)": "",
        "# Minimum probability for a token to be considered": "",
        "# === USAGE INSTRUCTIONS ===": "",
        "# 1. Update model_name to your desired model": "",
        "# 2. Set model_type according to your provider": "",
        "# 3. Configure api_base with your endpoint URL": "",
        "# 4. Set api_key (use null for local models without authentication)": "",
        "# 5. Choose appropriate model_usage based on your use case": "",
        "# 6. Adjust sampling parameters as needed for your task": "",
    }

    # Generate YAML string
    yaml_str = ""
    for key, value in config.items():
        if key.startswith("#"):
            # Comment line
            yaml_str += f"{key}\n"
        else:
            # Configuration field
            if isinstance(value, bool):
                yaml_str += f"{key}: {str(value).lower()}\n"
            elif isinstance(value, (int, float)):
                yaml_str += f"{key}: {value}\n"
            elif value == "null":
                yaml_str += f"{key}: null\n"
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


def extract_model_config_from_callback(
    model_name: str,
    model_type: str,
    api_base: str,
    api_key: str,
    model_usage: str,
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Extract data from model configuration form callback function

    Args:
        model_name: Model name
        model_type: Model type
        api_base: API base URL
        api_key: API key
        model_usage: Model usage
        temperature: Temperature parameter
        max_tokens: Maximum number of tokens
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        min_p: Min-p sampling parameter

    Returns:
        Dictionary containing all model configuration data
    """

    # Build configuration dictionary
    config = {
        "model_name": model_name,
        "model_type": model_type,
        "api_base": api_base,
        "api_key": api_key if api_key and api_key != "YOUR-API-KEY" else "YOUR-API-KEY",
        "model_usage": model_usage,
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens else 2500,
        "top_p": top_p if top_p is not None else None,
        "top_k": top_k if top_k is not None else None,
        "min_p": min_p if min_p is not None else None,
    }

    return config


def save_model_config_to_yaml(
    model_name: str,
    model_type: str,
    api_base: str,
    api_key: str | None,
    model_usage: str,
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    output_dir: str = "assets/yml",
    filename: str | None = None,
) -> str:
    """
    Save model configuration to YAML file

    Args:
        model_name: Model name
        model_type: Model type
        api_base: API base URL
        api_key: API key
        model_usage: Model usage
        temperature: Temperature parameter
        max_tokens: Maximum number of tokens
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        min_p: Min-p sampling parameter
        output_dir: Output directory
        filename: File name, generated based on model_usage if None

    Returns:
        Saved file path
    """

    # Generate filename based on model_usage if not specified
    if filename is None:
        filename = f"{model_usage}_config.yml"

    # Build output path
    output_path = os.path.join(output_dir, filename)

    # Generate YAML configuration
    generate_model_config_yaml(
        model_name=model_name,
        model_type=model_type,
        api_base=api_base,
        api_key=api_key,
        model_usage=model_usage,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        output_path=output_path,
    )

    return output_path
