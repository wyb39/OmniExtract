import yaml
import os
from typing import Dict, List, Any

def generate_build_optm_dataset_yml(form_data: Dict[str, Any], output_path: str | None = None) -> str:
    """
    Generate yml configuration file for build_optm_dataset based on form data
    
    Args:
        form_data: Dictionary containing form data
        output_path: Output file path, returns yml content string if None
        
    Returns:
        Generated yml content string
    """
    
    # Build yml configuration data
    yml_config = {
        'json_path': form_data.get('json_path', 'path/to/your/json/files'),
        'dataset': form_data.get('curated_dataset_path', 'path/to/your/curated/dataset.json'),
        'fields': [],
        'multiple': form_data.get('multiple_entities', False),
        'article_field': form_data.get('article_field', 'article_field')
    }
    
    # Add save directory if provided
    save_directory = form_data.get('save_directory')
    if save_directory:
        yml_config['save_dir'] = save_directory
    
    # Process field configuration
    fields = form_data.get('fields', [])
    for field in fields:
        field_config = {
            'name': field.get('name', 'field_name'),
            'field_type': field.get('field_type', 'str'),
            'description': field.get('description', 'Description of this field')
        }
        
        # Add range limits (if any)
        if field.get('add_range') and field.get('range_min') is not None:
            field_config['range_min'] = float(field['range_min'])
        if field.get('add_range') and field.get('range_max') is not None:
            field_config['range_max'] = float(field['range_max'])
            
        # Add literal list (if any)
        if field.get('add_literal') and field.get('literal_list'):
            literal_list = [item.strip() for item in field['literal_list'].split(',')]
            field_config['literal_list'] = literal_list
            
        yml_config['fields'].append(field_config)
    
    # Process article parts (if any)
    article_parts = form_data.get('article_parts', [])
    if article_parts:
        yml_config['article_parts'] = article_parts
    
    # Generate yml content
    yml_content = yaml.dump(yml_config, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Add comment header
    header = """# Template Configuration for Build Optm Set
# This config is used for the /api/build_optm_set endpoint

"""
    
    full_content = header + yml_content
    
    # Write to file if needed
    if output_path:
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if there is a directory path
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
    
    return full_content

def generate_build_optm_dataset_yml_from_dash_callback(
    json_path: str,
    curated_dataset_path: str,
    fields_data: List[Dict[str, Any]],
    multiple_entities: bool,
    article_field: str,
    article_parts: List[str] | None = None,
    save_directory: str | None = None
) -> str:
    """
    Generate yml configuration directly from Dash callback function parameters
    
    Args:
        json_path: JSON file path
        curated_dataset_path: Curated dataset path
        fields_data: List of field configuration data
        multiple_entities: Whether to allow multi-entity extraction
        article_field: Article field name
        article_parts: List of article parts
        save_directory: Custom save directory path
        
    Returns:
        Generated yml content string
    """
    
    form_data = {
        'json_path': json_path,
        'curated_dataset_path': curated_dataset_path,
        'fields': fields_data,
        'multiple_entities': multiple_entities,
        'article_field': article_field,
        'article_parts': article_parts or [],
        'save_directory': save_directory
    }
    
    return generate_build_optm_dataset_yml(form_data)
