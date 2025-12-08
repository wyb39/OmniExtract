"""
Data Writeback Module
Used to populate JSON configuration information back into corresponding page forms
"""

import json
from typing import Dict, Any, List, Union
import dash
from dash import dcc, html


def writeback_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate model configuration information back into the model configuration form
    
    Args:
        config: Dictionary containing model configuration
        
    Returns:
        Dictionary containing form component IDs and corresponding values
    """
    result = {}
    
    result['model-name'] = config.get('model_name', '')
    result['model-type'] = config.get('model_type', 'OpenAI')
    result['api-base'] = config.get('api_base', '')
    result['api-key'] = config.get('api_key', '')
    result['temperature'] = config.get('temperature', 0.7)
    result['max-tokens'] = config.get('max_tokens', 2048)
    result['top-p'] = config.get('top_p', 0.9)
    result['frequency-penalty'] = config.get('frequency_penalty', 0.0)
    result['presence-penalty'] = config.get('presence_penalty', 0.0)
    result['timeout'] = config.get('timeout', 60)
    result['max-retries'] = config.get('max_retries', 3)
    
    return result


def writeback_run_optimization(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate run optimization configuration information back into the run optimization form
    
    Args:
        config: Dictionary containing run optimization configuration
        
    Returns:
        Dictionary containing form component IDs and corresponding values
    """
    result = {}
    
    result['dataset-path'] = config.get('dataset', config.get('dataset_path', ''))
    result['save-dir'] = config.get('save_dir', '')
    result['experiment-name'] = config.get('experiment_name', '')
    result['initial-prompt'] = config.get('initial_prompt', '')
    result['task'] = config.get('task', 'Extraction')
    result['optim-burden'] = config.get('optim_burden', 'medium')
    result['threads'] = config.get('threads', 6)
    result['demos'] = config.get('demos', 1)
    result['multiple'] = config.get('multiple', False)
    result['ai-evaluation'] = config.get('ai_evaluation', False)
    
    if 'inputFields' in config and isinstance(config['inputFields'], list):
        input_fields = config['inputFields']
        result['input-fields-data'] = input_fields
    else:
        result['input-fields-data'] = []
    
    if 'outputFields' in config and isinstance(config['outputFields'], list):
        output_fields = config['outputFields']
        result['output-fields-data'] = output_fields
    else:
        result['output-fields-data'] = []
    
    return result


def writeback_table_extraction(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate table extraction configuration information back into the table extraction form
    
    Args:
        config: Dictionary containing table extraction configuration
        
    Returns:
        Dictionary containing form component IDs and corresponding values
    """
    result = {}
    
    result['table-dataset-path'] = config.get('parsed_file_path', '')
    result['table-save-dir'] = config.get('save_folder_path', '')
    result['table-threads'] = config.get('num_threads', 6) 
    
    if 'outputFields' in config and isinstance(config['outputFields'], list):
        output_fields = config['outputFields']
        result['table-output-fields-data'] = output_fields
    else:
        result['table-output-fields-data'] = []
    
    result['table-classify-prompt'] = config.get('classify_prompt', '')
    
    result['table-extract-prompt'] = config.get('extract_prompt', '')
    
    result['table-extract-directly'] = config.get('extract_directly', False)
    
    return result


def writeback_build_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate build dataset configuration information back into the build dataset form
    
    Args:
        config: Dictionary containing build dataset configuration
        
    Returns:
        Dictionary containing form component IDs and corresponding values
    """
    result = {}
    
    result['json-path'] = config.get('json_path', '')
    result['curated-dataset-path'] = config.get('dataset', '')
    result['save-dir'] = config.get('save_dir', '')
    
    result['multiple-entities'] = config.get('multiple', False)
    
    result['article-field'] = config.get('article_field', '')
    result['article-parts'] = config.get('article_parts', [])
    
    if 'fields' in config and isinstance(config['fields'], list):
        fields = config['fields']
        result['fields-data'] = fields
    else:
        result['fields-data'] = []
    
    return result


def writeback_novel_extraction(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate original (previously named novel) extraction configuration back into the form
    """
    result = {}
    result['novel-dataset-path'] = config.get('dataset', '')
    result['novel-save-dir'] = config.get('save_dir', '')
    result['novel-initial-prompt'] = config.get('initial_prompt', '')
    result['novel-judging'] = config.get('judging', '')
    result['novel-threads'] = config.get('threads', 6)
    result['novel-multiple'] = config.get('multiple', False)
    if 'inputFields' in config and isinstance(config['inputFields'], list):
        input_fields = config['inputFields']
        result['novel-input-fields-data'] = input_fields
    else:
        result['novel-input-fields-data'] = []
    if 'outputFields' in config and isinstance(config['outputFields'], list):
        output_fields = config['outputFields']
        result['novel-output-fields-data'] = output_fields
    else:
        result['novel-output-fields-data'] = []
    return result

def writeback_original_extraction(config: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    result['original-dataset-path'] = config.get('dataset', '')
    result['original-save-dir'] = config.get('save_dir', '')
    result['original-initial-prompt'] = config.get('initial_prompt', '')
    result['original-judging'] = config.get('judging', '')
    result['original-threads'] = config.get('threads', 6)
    result['original-multiple'] = config.get('multiple', False)
    if 'inputFields' in config and isinstance(config['inputFields'], list):
        input_fields = config['inputFields']
        result['original-input-fields-data'] = input_fields
    else:
        result['original-input-fields-data'] = []
    if 'outputFields' in config and isinstance(config['outputFields'], list):
        output_fields = config['outputFields']
        result['original-output-fields-data'] = output_fields
    else:
        result['original-output-fields-data'] = []
    return result


def writeback_optimized_extraction(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate optimized extraction configuration information back into the optimized extraction form
    
    Args:
        config: Dictionary containing optimized extraction configuration
        
    Returns:
        Dictionary containing form component IDs and corresponding values
    """
    result = {}
    
    if 'load_dir' in config:
        result['optim-load-dir'] = config['load_dir']
    
    if 'dataset' in config:
        result['optim-dataset-path'] = config['dataset']
    
    if 'save_dir' in config:
        result['optim-save-dir'] = config['save_dir']
    
    if 'judging' in config:
        result['optim-judging'] = config['judging']
    
    if 'threads' in config:
        result['optim-threads'] = config['threads']
    
    return result


def writeback_document_parsing(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate document parsing configuration information back into the document parsing form
    
    Args:
        config: Dictionary containing document parsing configuration
        
    Returns:
        Dictionary containing form component IDs and corresponding values
    """
    result = {}
    
    if 'folder_path' in config:
        result['folder-path'] = config['folder_path']
    
    if 'save_path' in config:
        result['save-path'] = config['save_path']
    
    if 'file_type' in config:
        result['file-type'] = config['file_type']
    
    if 'convert_mode' in config:
        result['convert-mode'] = config['convert_mode']
    
    return result


def writeback_table_parsing(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate table parsing configuration information back into the table parsing form
    
    Args:
        config: Dictionary containing table parsing configuration
        
    Returns:
        Dictionary containing form component IDs and corresponding values
    """
    result = {}
    
    if 'file_folder_path' in config:
        result['table-folder-path'] = config['file_folder_path']
    
    if 'save_folder_path' in config:
        result['table-save-path'] = config['save_folder_path']
    
    if 'non_tabular_file_format' in config:
        result['table-file-type'] = config['non_tabular_file_format']
    
    return result


def writeback_config_to_form(config_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the corresponding writeback function based on configuration type
    
    Args:
        config_type: Configuration type, corresponding to the prefix of handle_ functions
        config: Dictionary containing configuration information
        
    Returns:
        Dictionary containing form component IDs and corresponding values
    """
    writeback_functions = {
        'model_config': writeback_model_config,
        'run_optimization': writeback_run_optimization,
        'table_extraction': writeback_table_extraction,
        'build_dataset': writeback_build_dataset,
        'novel_extraction': writeback_novel_extraction,
        'original_extraction': writeback_original_extraction,
        'optimized_extraction': writeback_optimized_extraction,
        'document_parsing': writeback_document_parsing,
        'table_parsing': writeback_table_parsing
    }
    
    writeback_func = writeback_functions.get(config_type)
    
    if writeback_func:
        return writeback_func(config)
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def generate_dynamic_fields(fields_data: List[Dict[str, Any]], field_type: str, prefix: str = "") -> List[html.Div]:
    """
    Generate dynamic form fields based on field data
    
    Args:
        fields_data: List of field data dictionaries
        field_type: Field type ('input' or 'output')
        
    Returns:
        List of generated form field components
    """
    fields = []
    
    for i, field in enumerate(fields_data):
        if field_type == 'output':
            field_id = f"{prefix}output-field-{i}"
        else:
            field_id = f"{prefix}{field_type}-field-{i}"
        
        field_name = field.get('name', '')
        field_type_value = field.get('field_type', 'str')
        field_description = field.get('description', '')
        
        field_type_text = "Output" if field_type == "output" else "Input"
        
        field_div = html.Div([
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}, children=[
                html.H4(f"{field_type_text} Field", style={"margin": "0", "fontSize": "16px"}),
                html.Button(
                    "×",
                    id={'type': f'delete-{prefix}{field_type}-field', 'index': field_id},
                    className="delete-button",
                    style={
                        "background": "none",
                        "border": "none",
                        "color": "#6b7280",
                        "fontSize": "20px",
                        "cursor": "pointer",
                        "padding": "0 5px",
                        "lineHeight": "1",
                        "borderRadius": "4px",
                        "hover": {"background": "#f3f4f6", "color": "#ef4444"}
                    }
                )
            ]),
            html.Div([
                html.Div(style={"display": "flex", "gap": "20px"}, children=[
                    html.Div(style={"flex": 1}, children=[
                        html.Label(f"{field_type_text} Field Name", className="form-label"),
                        dcc.Input(
                            id={'type': f'{prefix}{field_type}-name', 'index': field_id},
                            placeholder="extracted_value",
                            value=field_name if field_name else f'extracted_value_{i}',
                            className="form-input",
                            style={"height": "38px"}
                        )
                    ]),
                    html.Div(style={"flex": 1}, children=[
                        html.Label(f"{field_type_text} Field Type", className="form-label"),
                        dcc.Dropdown(
                            id={'type': f'{prefix}{field_type}-type', 'index': field_id},
                            options=[
                                {"label": "String", "value": "str"},
                                {"label": "Integer", "value": "int"},
                                {"label": "Float", "value": "float"},
                                {"label": "Boolean", "value": "bool"},
                                {"label": "List", "value": "list"},
                                {"label": "Literal", "value": "literal"},
                                {"label": "List Literal", "value": "list_literal"}
                            ],
                            value=field_type_value,
                            style={"height": "38px","border":"1px solid #e1e5e9","border-radius":"6px"}
                        )
                    ])
                ])
            ], className="form-group"),
            html.Div([
                html.Label(f"{field_type_text} Field Description", className="form-label"),
                dcc.Input(
                    id={'type': f'{prefix}{field_type}-description', 'index': field_id},
                    placeholder="Description of what to extract",
                    value=field_description,
                    className="form-input"
                )
            ], className="form-group"),
            html.Div(style={"display": "flex", "gap": "20px"}, children=[
                html.Div(style={"flex": 1}, children=[
                    dcc.Checklist(
                        id={'type': f'{prefix}{field_type}-add-range', 'index': field_id},
                        options=[{"label": "Add Range Limits", "value": "true"}],
                        className="form-checkbox"
                    )
                ]),
                html.Div(style={"flex": 1}, children=[
                    dcc.Checklist(
                        id={'type': f'{prefix}{field_type}-add-literal', 'index': field_id},
                        options=[{"label": "Add Literal List", "value": "true"}],
                        className="form-checkbox"
                    )
                ])
            ]),
            html.Div(id={'type': f'{prefix}{field_type}-range-container', 'index': field_id}, style={"display": "none"}, children=[
                html.Div([
                    html.Label("Range Minimum", className="form-label"),
                    dcc.Input(
                        id={'type': f'{prefix}{field_type}-range-min', 'index': field_id},
                        type="number",
                        placeholder="Min value",
                        className="form-input"
                    )
                ], className="form-group"),
                html.Div([
                    html.Label("Range Maximum", className="form-label"),
                    dcc.Input(
                        id={'type': f'{prefix}{field_type}-range-max', 'index': field_id},
                        type="number",
                        placeholder="Max value",
                        className="form-input"
                    )
                ], className="form-group")
            ]),
            html.Div(id={'type': f'{prefix}{field_type}-literal-container', 'index': field_id}, style={"display": "none"}, children=[
                html.Div([
                    html.Label("Literal List (comma separated)", className="form-label"),
                    dcc.Input(
                        id={'type': f'{prefix}{field_type}-literal-list', 'index': field_id},
                        placeholder="option1,option2,option3",
                        className="form-input"
                    )
                ], className="form-group")
            ])
        ], id=field_id, className="section-container", style={"border": "1px solid #e1e5e9", "borderRadius": "8px", "padding": "16px", "marginBottom": "20px"})
        
        fields.append(field_div)
    return fields

def generate_extract_output_fields(fields_data: List[Dict[str, Any]]) -> List[html.Div]:
    return generate_dynamic_fields(fields_data, 'output', prefix='extract-')


def generate_dataset_fields(fields_data: List[Dict[str, Any]]) -> List[html.Div]:
    """
    Generate dataset fields specifically for build_optm_dataset module
    """
    components = generate_dynamic_fields(fields_data, 'field')
    for i, field_div in enumerate(components):
        field_id = getattr(field_div, "id", None)
        try:
            header_div = field_div.children[0]
            if hasattr(header_div, "children") and isinstance(header_div.children, list) and header_div.children:
                h4 = header_div.children[0]
                if hasattr(h4, "children"):
                    h4.children = "Field"
                if len(header_div.children) > 1:
                    btn = header_div.children[1]
                    if hasattr(btn, "id") and isinstance(btn.id, dict):
                        btn.id["type"] = "delete-field"
        except Exception:
            pass

        data = fields_data[i] if i < len(fields_data) else {}
        range_min = data.get("range_min")
        range_max = data.get("range_max")
        literal_list = data.get("literal_list")
        literal_str = ",".join(str(x) for x in literal_list) if isinstance(literal_list, list) else (literal_list or "")

        def _update(target_type, attr, val):
            stack = [field_div]
            while stack:
                node = stack.pop()
                children = getattr(node, "children", None)
                if isinstance(children, list):
                    stack.extend(children)
                elif children is not None:
                    stack.append(children)
                node_id = getattr(node, "id", None)
                if isinstance(node_id, dict):
                    if node_id.get("type") == target_type and node_id.get("index") == field_id:
                        if attr == "style":
                            style = dict(getattr(node, "style", {}) or {})
                            style.update(val)
                            setattr(node, "style", style)
                        else:
                            setattr(node, attr, val)

        if range_min is not None and range_max is not None:
            _update("field-add-range", "value", ["true"])
            _update("field-range-container", "style", {"display": "block"})
            _update("field-range-min", "value", range_min)
            _update("field-range-max", "value", range_max)
        if literal_str:
            _update("field-add-literal", "value", ["true"])
            _update("field-literal-container", "style", {"display": "block"})
            _update("field-literal-list", "value", literal_str)
    print(components)
    return components


def generate_dataset_output_fields(fields_data: List[Dict[str, Any]]) -> List[html.Div]:
    components = generate_dynamic_fields(fields_data, 'output', prefix='dataset-')
    for i, field_div in enumerate(components):
        field_id = getattr(field_div, "id", None)
        try:
            header_div = field_div.children[0]
            if hasattr(header_div, "children") and isinstance(header_div.children, list) and header_div.children:
                h4 = header_div.children[0]
                if hasattr(h4, "children"):
                    h4.children = "Field"
        except Exception:
            pass

        data = fields_data[i] if i < len(fields_data) else {}
        range_min = data.get("range_min")
        range_max = data.get("range_max")
        literal_list = data.get("literal_list")
        literal_str = ",".join(str(x) for x in literal_list) if isinstance(literal_list, list) else (literal_list or "")

        def _update(target_type, attr, val):
            stack = [field_div]
            while stack:
                node = stack.pop()
                children = getattr(node, "children", None)
                if isinstance(children, list):
                    stack.extend(children)
                elif children is not None:
                    stack.append(children)
                node_id = getattr(node, "id", None)
                if isinstance(node_id, dict):
                    if node_id.get("type") == target_type and node_id.get("index") == field_id:
                        if attr == "style":
                            style = dict(getattr(node, "style", {}) or {})
                            style.update(val)
                            setattr(node, "style", style)
                        else:
                            setattr(node, attr, val)

        if range_min is not None and range_max is not None:
            _update("dataset-output-add-range", "value", ["true"])
            _update("dataset-output-range-container", "style", {"display": "block"})
            _update("dataset-output-range-min", "value", range_min)
            _update("dataset-output-range-max", "value", range_max)
        if literal_str:
            _update("dataset-output-add-literal", "value", ["true"])
            _update("dataset-output-literal-container", "style", {"display": "block"})
            _update("dataset-output-literal-list", "value", literal_str)
    return components
