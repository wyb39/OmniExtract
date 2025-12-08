import yaml
import os
from typing import Type, TypeVar, Any

T = TypeVar('T')


def yaml_to_class(yaml_path: str, cls: Type[T]) -> T:
    """
    Parse YAML file into an instance of the specified class

    Args:
        yaml_path: Path to the YAML file
        cls: Target class type

    Returns:
        Instance of the specified class

    Raises:
        FileNotFoundError: When the YAML file does not exist
        yaml.YAMLError: When YAML parsing fails
        TypeError: When parsed data cannot be converted to the target class
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML parsing error: {str(e)}")

    if not isinstance(data, dict):
        raise TypeError(f"YAML content must be a dictionary, current type: {type(data)}")

    try:
        return cls(**data)
    except Exception as e:
        raise TypeError(f"Failed to convert data to {cls.__name__} type: {str(e)}")
