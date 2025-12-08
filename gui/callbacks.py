import dash
from dash import Input, Output, State, callback, html, dcc, MATCH, ALL
import dash_bootstrap_components as dbc
import os
import json
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any
from yml_generation.yml_model_config import save_model_config_to_yaml
from yml_generation.yml_document_parsing import save_document_parsing_config_to_yaml
from yml_generation.yml_prompt_generation import save_prompt_optimization_config_to_yaml
from yml_generation.yml_table_parsing import save_table_parsing_config_to_yaml
from writeback import generate_dataset_fields, generate_extract_output_fields, writeback_config_to_form, generate_dynamic_fields, generate_dataset_output_fields
from call_cli import create_initial_log, cancel_task
from process_manager import process_manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from cli.cli_handler import modify_model, run_model_test_call


def parse_field_element_to_dict(field_element, field_type="input"):
    """
    Parse field element to dictionary format

    Args:
        field_element: Dash HTML component element
        field_type: Field type ("input" or "output")

    Returns:
        dict: Parsed field configuration dictionary, returns None if parsing fails
    """
    try:
        # Get field ID
        field_id = None
        if hasattr(field_element, "id"):
            field_id = field_element.id
        elif hasattr(field_element, "props") and "id" in field_element.props:
            field_id = field_element.props["id"]
        elif "props" in field_element and "id" in field_element["props"]:
            field_id = field_element["props"]["id"]

        if not field_id:
            return None

        # Initialize field configuration
        field_config = {"name": "", "field_type": "str", "description": ""}

        # Get field name
        name_input_id = {"type": f"{field_type}-name", "index": field_id}
        if hasattr(field_element, "children") and field_element.children:
            for child in field_element.children:
                if hasattr(child, "children") and child.children:
                    for sub_child in child.children:
                        if (
                            hasattr(sub_child, "id")
                            and isinstance(sub_child.id, dict)
                            and sub_child.id.get("type") == f"{field_type}-name"
                        ):
                            field_config["name"] = (
                                sub_child.value if hasattr(sub_child, "value") else ""
                            )
                            break

        # Get field type
        type_dropdown_id = {"type": f"{field_type}-type", "index": field_id}
        if hasattr(field_element, "children") and field_element.children:
            for child in field_element.children:
                if hasattr(child, "children") and child.children:
                    for sub_child in child.children:
                        if (
                            hasattr(sub_child, "id")
                            and isinstance(sub_child.id, dict)
                            and sub_child.id.get("type") == f"{field_type}-type"
                        ):
                            field_config["field_type"] = (
                                sub_child.value
                                if hasattr(sub_child, "value")
                                else "str"
                            )
                            break

        # Get field description
        desc_input_id = {"type": f"{field_type}-description", "index": field_id}
        if hasattr(field_element, "children") and field_element.children:
            for child in field_element.children:
                if hasattr(child, "children") and child.children:
                    for sub_child in child.children:
                        if (
                            hasattr(sub_child, "id")
                            and isinstance(sub_child.id, dict)
                            and sub_child.id.get("type") == f"{field_type}-description"
                        ):
                            field_config["description"] = (
                                sub_child.value if hasattr(sub_child, "value") else ""
                            )
                            break

        # Check if there are range constraints
        range_min_id = {"type": f"{field_type}-range-min", "index": field_id}
        range_max_id = {"type": f"{field_type}-range-max", "index": field_id}
        range_min_val = None
        range_max_val = None

        if hasattr(field_element, "children") and field_element.children:
            for child in field_element.children:
                if hasattr(child, "children") and child.children:
                    for sub_child in child.children:
                        if hasattr(sub_child, "id") and isinstance(sub_child.id, dict):
                            if sub_child.id.get("type") == f"{field_type}-range-min":
                                range_min_val = (
                                    sub_child.value
                                    if hasattr(sub_child, "value")
                                    else None
                                )
                            elif sub_child.id.get("type") == f"{field_type}-range-max":
                                range_max_val = (
                                    sub_child.value
                                    if hasattr(sub_child, "value")
                                    else None
                                )

        if range_min_val is not None and range_max_val is not None:
            field_config["range_min"] = range_min_val
            field_config["range_max"] = range_max_val

        # Check if there is a literal list
        literal_list_id = {"type": f"{field_type}-literal-list", "index": field_id}
        literal_list_val = None

        if hasattr(field_element, "children") and field_element.children:
            for child in field_element.children:
                if hasattr(child, "children") and child.children:
                    for sub_child in child.children:
                        if (
                            hasattr(sub_child, "id")
                            and isinstance(sub_child.id, dict)
                            and sub_child.id.get("type") == f"{field_type}-literal-list"
                        ):
                            literal_list_val = (
                                sub_child.value if hasattr(sub_child, "value") else None
                            )
                            break

        if literal_list_val:
            field_config["literal_list"] = ",".join(
                [item.strip() for item in literal_list_val.split(",")]
            )

        # Validate required fields
        if not field_config["name"]:
            return None

        return field_config

    except Exception as e:
        print(f"Error parsing field element: {e}")
        return None


def get_form_values_for_module(module_key, taskcard_config_path_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    actual_config_path = get_actual_config_path_from_ctx(ctx, taskcard_config_path_list)
    if not actual_config_path:
        raise ValueError("NO_PATH")
    config_data = generic_writeback_input(actual_config_path)
    return writeback_config_to_form(module_key, config_data)


def parse_field_data_from_states(
    field_names,
    field_types,
    field_descriptions,
    field_range_mins=None,
    field_range_maxs=None,
    field_literal_lists=None,
    field_type="input",
):
    """
    Parse field configurations from callback State data

    Args:
        field_names: List of field names
        field_types: List of field types
        field_descriptions: List of field descriptions
        field_range_mins: List of field minimum values (optional)
        field_range_maxs: List of field maximum values (optional)
        field_literal_lists: List of field literal lists (optional)
        field_type: Field type ("input" or "output")

    Returns:
        list: List of field configuration dictionaries
    """
    fields = []

    for i in range(len(field_names)):
        if field_names[i]:  # Only add when field name is not empty
            field_config = {
                "name": field_names[i],
                "field_type": field_types[i] if i < len(field_types) else "str",
                "description": field_descriptions[i]
                if i < len(field_descriptions)
                else "",
            }

            # Add optional range constraints
            if (
                field_range_mins
                and field_range_maxs
                and i < len(field_range_mins)
                and i < len(field_range_maxs)
                and field_range_mins[i] is not None
                and field_range_maxs[i] is not None
            ):
                field_config["range_min"] = field_range_mins[i]
                field_config["range_max"] = field_range_maxs[i]

            # Add optional literal list
            if (
                field_literal_lists
                and i < len(field_literal_lists)
                and field_literal_lists[i]
            ):
                field_config["literal_list"] = [
                    item.strip() for item in field_literal_lists[i].split(",")
                ]

            fields.append(field_config)

    return fields


@callback(
    Output("model-config-modal", "is_open"),
    Output("model-config-modal-body", "children"),
    [
        Input("test-connection", "n_clicks"),
        Input("save-config", "n_clicks"),
        Input("close-model-config-modal", "n_clicks"),
    ],
    [
        State("model-name", "value"),
        State("model-type", "value"),
        State("api-base", "value"),
        State("api-key", "value"),
        State("model-usage", "value"),
        State("temperature", "value"),
        State("max-tokens", "value"),
        State("top-p", "value"),
        State("top-k", "value"),
        State("min-p", "value"),
    ],
    prevent_initial_call=True,
)
def handle_model_config_actions(
    test_clicks,
    save_clicks,
    close_clicks,
    model_name,
    model_type,
    api_base,
    api_key,
    model_usage,
    temperature,
    max_tokens,
    top_p,
    top_k,
    min_p,
):
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, None

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "close-model-config-modal":
        return False, None

    try:
        if button_id == "test-connection":
            complete_config = {
                "model_name": model_name,
                "model_type": model_type,
                "api_base": api_base,
                "api_key": api_key,
                "model_usage": model_usage,
                "temperature": temperature,
                "max_tokens": max_tokens if max_tokens else None,
                "top_p": float(top_p) if top_p else None,
                "top_k": int(top_k) if top_k else None,
                "min_p": float(min_p) if min_p else None,
                "prompt": "Hello"
            }

            try:
                result = run_model_test_call(complete_config)
                inner = result.get("result", {})
                def config_summary():
                    return html.Div([
                        html.H4("Configuration Summary", className="section-title"),
                        html.Div([
                            html.Div([
                                html.Span("Model Name: ", style={"fontWeight": "bold"}),
                                html.Span(model_name)
                            ], className="config-item"),
                            html.Div([
                                html.Span("Model Type: ", style={"fontWeight": "bold"}),
                                html.Span(model_type)
                            ], className="config-item"),
                            html.Div([
                                html.Span("Model Usage: ", style={"fontWeight": "bold"}),
                                html.Span(model_usage)
                            ], className="config-item"),
                            html.Div([
                                html.Span("API Base URL: ", style={"fontWeight": "bold"}),
                                html.Span(api_base)
                            ], className="config-item"),
                            html.Div([
                                html.Span("Temperature: ", style={"fontWeight": "bold"}),
                                html.Span(temperature)
                            ], className="config-item"),
                            html.Div([
                                html.Span("Max Tokens: ", style={"fontWeight": "bold"}),
                                html.Span(max_tokens if max_tokens else "None")
                            ], className="config-item"),
                            html.Div([
                                html.Span("Top-p: ", style={"fontWeight": "bold"}),
                                html.Span(float(top_p) if top_p else "None")
                            ], className="config-item"),
                            html.Div([
                                html.Span("Top-k: ", style={"fontWeight": "bold"}),
                                html.Span(int(top_k) if top_k else "None")
                            ], className="config-item"),
                            html.Div([
                                html.Span("Min-p: ", style={"fontWeight": "bold"}),
                                html.Span(float(min_p) if min_p else "None")
                            ], className="config-item"),
                        ], className="config-summary")
                    ])

                if inner.get("success"):
                    body = html.Div([
                        html.Div("Connection test successful!", className="status-success"),
                        config_summary()
                    ])
                    return True, body
                err = inner.get("error") or "Connection test failed"
                body = html.Div([
                    html.Div(f"Connection test failed: {err}", className="status-error"),
                    config_summary()
                ])
                return True, body
            except Exception as e:
                body = html.Div([
                    html.Div(f"Connection test error: {str(e)}", className="status-error")
                ])
                return True, body
        elif button_id == "save-config":
            # Configuration saving logic
            # Create complete config with API key for internal use
            complete_config = {
                "model_name": model_name,
                "model_type": model_type,
                "api_base": api_base,
                "api_key": api_key,
                "model_usage": model_usage,
                "temperature": temperature,
                "max_tokens": max_tokens if max_tokens else None,
                "top_p": float(top_p) if top_p else None,
                "top_k": int(top_k) if top_k else None,
                "min_p": float(min_p) if min_p else None,
            }

            # Create safe config without API key for external storage
            safe_config = {
                "model_name": model_name,
                "model_type": model_type,
                "api_base": api_base,
                "model_usage": model_usage,
                "temperature": temperature,
                "max_tokens": max_tokens if max_tokens else None,
                "top_p": float(top_p) if top_p else None,
                "top_k": int(top_k) if top_k else None,
                "min_p": float(min_p) if min_p else None,
            }

            # Generate timestamp for creating subdirectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create runs/model_config subdirectory
            runs_base_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "runs", "model_config"
            )
            sub_dir = os.path.join(runs_base_dir, f"run_{timestamp}")
            os.makedirs(sub_dir, exist_ok=True)

            # Generate YAML configuration file under runs/model_config subdirectory
            yaml_filename = f"{model_usage}_config_{timestamp}.yml"

            output_message = None
            try:
                yaml_output_path = save_model_config_to_yaml(
                    model_name=model_name,
                    model_type=model_type,
                    api_base=api_base,
                    api_key=None,  # Don't save API key in YAML
                    model_usage=model_usage,
                    temperature=temperature,
                    max_tokens=max_tokens if max_tokens else 2500,
                    top_p=float(top_p) if top_p else None,
                    top_k=int(top_k) if top_k else None,
                    min_p=float(min_p) if min_p else None,
                    output_dir=sub_dir,
                    filename=yaml_filename,
                )
                print(f"YAML configuration saved to: {yaml_output_path}")

                log_path = os.path.join(sub_dir, f"modify_model_{timestamp}.json")
                
                # Create initial log in main process (using safe config without API key)
                create_initial_log(name=f"{timestamp}", data=safe_config, log_path=log_path)

                from call_cli import run_task_and_update_log

                result = run_task_and_update_log(
                    callable_obj=modify_model,
                    data=complete_config,
                    log_path=log_path
                )

                def format_output_text(text):
                        text_str = str(text)
                        start_key = "OMNI_EXTRACT_ENCRYPTION_KEY"
                        end_key = "app.py"
                        start_idx = text_str.find(start_key)
                        end_idx = text_str.find(end_key, start_idx)
                        if start_idx != -1 and end_idx != -1:
                            end_idx += len(end_key)
                            return [
                                html.Span(text_str[:start_idx]),
                                html.Span(text_str[start_idx:end_idx], style={"color": "red", "fontWeight": "bold"}),
                                html.Span(text_str[end_idx:])
                            ]
                        return [html.Span(text_str)]

                def config_summary():
                    return html.Div([
                        html.H4("Configuration Summary", className="section-title"),
                        html.Div([
                            html.Div([
                                html.Span("Model Name: ", style={"fontWeight": "bold"}),
                                html.Span(model_name)
                            ], className="config-item"),
                            html.Div([
                                html.Span("Model Type: ", style={"fontWeight": "bold"}),
                                html.Span(model_type)
                            ], className="config-item"),
                            html.Div([
                                html.Span("Model Usage: ", style={"fontWeight": "bold"}),
                                html.Span(model_usage)
                            ], className="config-item"),
                            html.Div([
                                html.Span("API Base URL: ", style={"fontWeight": "bold"}),
                                html.Span(api_base)
                            ], className="config-item"),
                            html.Div([
                                html.Span("Temperature: ", style={"fontWeight": "bold"}),
                                html.Span(temperature)
                            ], className="config-item"),
                            html.Div([
                                html.Span("Max Tokens: ", style={"fontWeight": "bold"}),
                                html.Span(max_tokens if max_tokens else "None")
                            ], className="config-item"),
                            html.Div([
                                html.Span("Top-p: ", style={"fontWeight": "bold"}),
                                html.Span(float(top_p) if top_p else "None")
                            ], className="config-item"),
                            html.Div([
                                html.Span("Top-k: ", style={"fontWeight": "bold"}),
                                html.Span(int(top_k) if top_k else "None")
                            ], className="config-item"),
                            html.Div([
                                html.Span("Min-p: ", style={"fontWeight": "bold"}),
                                html.Span(float(min_p) if min_p else "None")
                            ], className="config-item"),
                        ], className="config-summary")
                    ])

                if result.get("status") == "succeed":
                    body = html.Div([
                        html.Div("Configuration saved successfully!", className="status-success"),
                        html.Div(f"YAML path: {yaml_output_path}", className="status-info", style={"marginTop": "10px", "fontSize": "12px"}),
                        html.Div("Model modification task completed", className="status-info", style={"marginTop": "5px", "fontSize": "12px"}),
                        config_summary(),
                        html.Div(f"Result: {result.get('result')}", className="status-info", style={"marginTop": "5px", "fontSize": "12px"}),
                    ])
                    return True, body
                else:
                    body = html.Div([
                        html.Div("Configuration saved successfully!", className="status-success"),
                        html.Div(f"YAML path: {yaml_output_path}", className="status-info", style={"marginTop": "10px", "fontSize": "12px"}),
                        html.Div("Model modification task failed", className="status-error", style={"marginTop": "5px", "fontSize": "12px"}),
                        config_summary(),
                        html.Div(f"Error: {result.get('error')}", className="status-error", style={"marginTop": "5px", "fontSize": "12px"}),
                    ])
                    return True, body
            except Exception as e:
                print(f"Error generating YAML configuration: {e}")
                body = html.Div([
                    html.Div(f"Error generating YAML configuration: {str(e)}", className="status-error")
                ])
                return True, body

        return False, None
    except Exception as e:
        body = html.Div([
            html.Div(f"Error: {str(e)}", className="status-error")
        ])
        return True, body

 


@callback(
    Output("input-range-container", "style"),
    Input("input-add-range", "value"),
    prevent_initial_call=True,
)
def toggle_input_range_fields(add_range):
    if add_range and "true" in add_range:
        return {"display": "block"}
    else:
        return {"display": "none"}


@callback(
    Output("input-literal-container", "style"),
    Input("input-add-literal", "value"),
    prevent_initial_call=True,
)
def toggle_input_literal_fields(add_literal):
    if add_literal and "true" in add_literal:
        return {"display": "block"}
    else:
        return {"display": "none"}


@callback(
    Output("output-range-container", "style"),
    Input("output-add-range", "value"),
    prevent_initial_call=True,
)
def toggle_output_range_fields(add_range):
    if add_range and "true" in add_range:
        return {"display": "block"}
    else:
        return {"display": "none"}


@callback(
    Output("output-literal-container", "style"),
    Input("output-add-literal", "value"),
    prevent_initial_call=True,
)
def toggle_output_literal_fields(add_literal):
    if add_literal and "true" in add_literal:
        return {"display": "block"}
    else:
        return {"display": "none"}


# Callbacks for adding and deleting InputField
@callback(
    Output("input-fields-container", "children"),
    Input("add-input-field", "n_clicks"),
    Input({"type": "delete-input-field", "index": ALL}, "n_clicks"),
    State("input-fields-container", "children"),
    prevent_initial_call=True,
)
def manage_input_fields(add_clicks, dynamic_clicks, children):
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return children

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    button_value = ctx.triggered[0]["value"]

    # 防止初始化时的误触发：检查n_clicks是否为有效点击值
    if button_value is None or button_value == 0:
        return children

    # If the add InputField button was clicked
    if button_id == "add-input-field":
        # Generate unique field ID that doesn't conflict with existing ones
        existing_ids = set()
        for child in children:
            # Check different ID storage methods
            child_id = None
            if hasattr(child, "id"):
                child_id = child.id
            elif hasattr(child, "props") and "id" in child.props:
                child_id = child.props["id"]
            elif "props" in child and "id" in child["props"]:
                child_id = child["props"]["id"]

            if child_id and child_id.startswith("input-field-"):
                existing_ids.add(child_id)

        # Find the next available ID
        counter = 0
        new_field_id = f"input-field-{counter}"
        while new_field_id in existing_ids:
            counter += 1
            new_field_id = f"input-field-{counter}"
        new_field = html.Div(
            [
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "flex-start",
                    },
                    children=[
                        html.H4(
                            "Input Field", style={"margin": "0", "fontSize": "16px"}
                        ),
                        html.Button(
                            "×",
                            id={"type": "delete-input-field", "index": new_field_id},
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
                                "hover": {"background": "#f3f4f6", "color": "#ef4444"},
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "gap": "20px"},
                    children=[
                        html.Div(
                            style={"flex": 1},
                            children=[
                                html.Label("Input Field Name", className="form-label"),
                                dcc.Input(
                                    id={"type": "input-name", "index": new_field_id},
                                    placeholder="input_value",
                                    value=f"input_value_{add_clicks}",
                                    className="form-input",
                                    style={"height": "38px"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={"flex": 1},
                            children=[
                                html.Label("Input Field Type", className="form-label"),
                                dcc.Dropdown(
                                    id={"type": "input-type", "index": new_field_id},
                                    options=[
                                        {"label": "String", "value": "str"},
                                        {"label": "Integer", "value": "int"},
                                        {"label": "Float", "value": "float"},
                                        {"label": "Boolean", "value": "bool"},
                                        {"label": "List", "value": "list"},
                                        {"label": "Literal", "value": "literal"},
                                        {
                                            "label": "List Literal",
                                            "value": "list_literal",
                                        },
                                    ],
                                    value="str",
                                    style={
                                        "height": "38px",
                                        "border": "1px solid #e1e5e9",
                                        "border-radius": "6px",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    [
                        html.Label("Input Field Description", className="form-label"),
                        dcc.Input(
                            id={"type": "input-description", "index": new_field_id},
                            placeholder="Description of this input field",
                            className="form-input",
                        ),
                    ],
                    className="form-group",
                ),
                html.Div(
                    style={"display": "flex", "gap": "20px"},
                    children=[
                        html.Div(
                            style={"flex": 1},
                            children=[
                                dcc.Checklist(
                                    id={
                                        "type": "input-add-range",
                                        "index": new_field_id,
                                    },
                                    options=[
                                        {"label": "Add Range Limits", "value": "true"}
                                    ],
                                    className="form-checkbox",
                                )
                            ],
                        ),
                        html.Div(
                            style={"flex": 1},
                            children=[
                                dcc.Checklist(
                                    id={
                                        "type": "input-add-literal",
                                        "index": new_field_id,
                                    },
                                    options=[
                                        {"label": "Add Literal List", "value": "true"}
                                    ],
                                    className="form-checkbox",
                                )
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id={"type": "input-range-container", "index": new_field_id},
                    style={"display": "none"},
                    children=[
                        html.Div(
                            [
                                html.Label("Range Minimum", className="form-label"),
                                dcc.Input(
                                    id={
                                        "type": "input-range-min",
                                        "index": new_field_id,
                                    },
                                    type="number",
                                    placeholder="Min value",
                                    className="form-input",
                                ),
                            ],
                            className="form-group",
                        ),
                        html.Div(
                            [
                                html.Label("Range Maximum", className="form-label"),
                                dcc.Input(
                                    id={
                                        "type": "input-range-max",
                                        "index": new_field_id,
                                    },
                                    type="number",
                                    placeholder="Max value",
                                    className="form-input",
                                ),
                            ],
                            className="form-group",
                        ),
                    ],
                ),
                html.Div(
                    id={"type": "input-literal-container", "index": new_field_id},
                    style={"display": "none"},
                    children=[
                        html.Div(
                            [
                                html.Label(
                                    "Literal List (comma separated)",
                                    className="form-label",
                                ),
                                dcc.Input(
                                    id={
                                        "type": "input-literal-list",
                                        "index": new_field_id,
                                    },
                                    placeholder="option1,option2,option3",
                                    className="form-input",
                                ),
                            ],
                            className="form-group",
                        )
                    ],
                ),
            ],
            id=new_field_id,
            className="section-container",
            style={
                "border": "1px solid #e1e5e9",
                "borderRadius": "8px",
                "padding": "16px",
                "marginBottom": "20px",
            },
        )

        # Add new InputField to existing list
        children.append(new_field)

    # If the delete button was clicked
    else:
        # Ensure at least one InputField is retained
        if len(children) > 1:
            # Determine the field ID to remove
            field_id_to_remove = None

            # Parse dynamic delete button ID
            try:
                button_data = json.loads(button_id)
                field_id_to_remove = button_data["index"]
            except Exception as e:
                print(f"Error parsing button ID: {e}")

            # If the field ID to remove is found
            if field_id_to_remove:
                # Use helper method to find and remove the corresponding InputField
                for i, child in enumerate(children):
                    child_id = None
                    if hasattr(child, "id"):
                        child_id = child.id
                    elif hasattr(child, "props") and "id" in child.props:
                        child_id = child.props["id"]
                    elif "props" in child and "id" in child["props"]:
                        child_id = child["props"]["id"]

                    # If a matching ID is found, remove the child element
                    if child_id == field_id_to_remove:
                        children.pop(i)
                        break
    print(children)
    return children


# ===== PROMPT OPTIMIZATION CALLBACKS =====


# Handle Run Optimization button click event
@callback(
    Output("optimization-results", "children"),
    Input("run-optimization", "n_clicks"),
    [
        State("dataset-path", "value"),
        State("save-dir", "value"),
        State("task-name-prompt-optimization", "value"),
        State("initial-prompt", "value"),
        State("task", "value"),
        State("optim-burden", "value"),
        State("threads", "value"),
        State("demos", "value"),
        State("multiple", "value"),
        State("ai-evaluation", "value"),
    ]
    + [
        State({"type": "input-name", "index": ALL}, "value"),
        State({"type": "input-type", "index": ALL}, "value"),
        State({"type": "input-description", "index": ALL}, "value"),
        State({"type": "input-range-min", "index": ALL}, "value"),
        State({"type": "input-range-max", "index": ALL}, "value"),
        State({"type": "input-literal-list", "index": ALL}, "value"),
        State({"type": "output-name", "index": ALL}, "value"),
        State({"type": "output-type", "index": ALL}, "value"),
        State({"type": "output-description", "index": ALL}, "value"),
        State({"type": "output-range-min", "index": ALL}, "value"),
        State({"type": "output-range-max", "index": ALL}, "value"),
        State({"type": "output-literal-list", "index": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def handle_run_optimization(
    n_clicks,
    dataset_path,
    save_dir,
    task_name_prompt_optimization,
    initial_prompt,
    task,
    optim_burden,
    threads,
    demos,
    multiple,
    ai_evaluation,
    input_names,
    input_types,
    input_descriptions,
    input_range_mins,
    input_range_maxs,
    input_literal_lists,
    output_names,
    output_types,
    output_descriptions,
    output_range_mins,
    output_range_maxs,
    output_literal_lists,
):
    try:
        # Validate required fields
        if not dataset_path:
            return html.Div(
                [html.Div("Error: Dataset Path is required!", className="status-error")]
            )

        if not save_dir:
            return html.Div(
                [
                    html.Div(
                        "Error: Save Directory is required!", className="status-error"
                    )
                ]
            )

        

        # Use helper method to build input field configurations
        input_fields = parse_field_data_from_states(
            input_names,
            input_types,
            input_descriptions,
            input_range_mins,
            input_range_maxs,
            input_literal_lists,
            field_type="input",
        )

        # Use helper method to build output field configurations
        output_fields = parse_field_data_from_states(
            output_names,
            output_types,
            output_descriptions,
            output_range_mins,
            output_range_maxs,
            output_literal_lists,
            field_type="output",
        )

        # Process checkbox values
        multiple_bool = "true" in multiple if multiple else False
        ai_evaluation_bool = "true" in ai_evaluation if ai_evaluation else False

        # Generate timestamp for creating subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create runs/prompt_optimization subdirectory
        runs_base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "runs", "prompt_optimization"
        )
        sub_dir = os.path.join(runs_base_dir, f"run_{timestamp}")
        os.makedirs(sub_dir, exist_ok=True)

        config = {
            "inputFields": input_fields,
            "outputFields": output_fields,
            "initial_prompt": initial_prompt,
            "dataset": dataset_path,
            "save_dir": save_dir,
            "task": task or "Extraction",
            "optim_burden": optim_burden or "medium",
            "threads": threads or 6,
            "demos": demos or 1,
            "multiple": multiple_bool,
            "ai_evaluation": ai_evaluation_bool,
        }

        # Generate YAML configuration file under runs/prompt_optimization subdirectory
        yaml_filename = f"optim_{timestamp}.yml"
        yaml_output_path = os.path.join(sub_dir, yaml_filename)

        try:
            # Use yml_generation module to save configuration
            saved_yaml_path = save_prompt_optimization_config_to_yaml(
                dataset_path=dataset_path,
                save_dir=save_dir,
                experiment_name=task_name_prompt_optimization,
                input_fields_data=input_fields,
                output_fields_data=output_fields,
                initial_prompt=initial_prompt or "Please optimize this prompt template",
                task=task or "Extraction",
                optim_burden=optim_burden or "medium",
                threads=threads or 6,
                demos=demos or 1,
                multiple=multiple_bool,
                ai_evaluation=ai_evaluation_bool,
                recall_prior=False,
                output_dir=sub_dir,
                filename=yaml_filename,
            )
            print(f"YAML configuration saved to: {saved_yaml_path}")
        except Exception as e:
            print(f"Error generating YAML configuration: {e}")
            return html.Div(
                [
                    html.Div(
                        f"Error generating YAML configuration: {str(e)}",
                        className="status-error",
                    )
                ]
            )

        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                return html.Div(
                    [
                        html.Div(
                            f"Error creating save directory: {str(e)}",
                            className="status-error",
                        )
                    ]
                )

        # Create initial log in main process
        try:
            log_path = os.path.join(sub_dir, f"optim_{timestamp}.json")
            # Create initial log
            create_initial_log(name=task_name_prompt_optimization, data=config, log_path=log_path)
            
            # Build Python code string for subprocess
            config_str = repr(config).replace("None", "None")
            log_path_fixed = log_path.replace("\\", "\\\\")
            python_code = f"""
import sys
sys.path.extend(['{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}', '{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")}'])
from gui.call_cli import run_task_and_update_log
from src.cli.cli_handler import run_optim_custom

config = {config_str}
# Run task and update log
result = run_task_and_update_log(callable_obj=run_optim_custom, data=config, log_path=r'{log_path_fixed}')
print(result)
"""

            proc_log_filename = os.path.basename(log_path).replace(".json", ".log")
            process = process_manager.start_python_code(
                python_code=python_code,
                key=log_path,
                capture_output=False,
                text=True,
                log_to_key_dir=True,
                log_filename=proc_log_filename,
            )

            # stdout, stderr = process.communicate()

            # if process.returncode == 0:
            #     print(f"Optimization task started successfully")
            #     print(f"Task output: {stdout}")
            # else:
            #     print(f"Error in optimization task: {stderr}")

        except Exception as e:
            print(f"Error running optimization task: {e}")

        # Return success message and configuration summary
        return html.Div(
            [
                html.Div(
                    "Prompt optimization started successfully!",
                    className="status-success",
                ),
                html.H4("Configuration Summary", className="section-title"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "Dataset Path: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(dataset_path),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Save Directory: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(save_dir),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Input Fields: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(f"{len(input_fields)} fields configured"),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Output Fields: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(f"{len(output_fields)} fields configured"),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span("Task: ", style={"fontWeight": "bold"}),
                                html.Span(task or "Extraction"),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Optimization Burden: ",
                                    style={"fontWeight": "bold"},
                                ),
                                html.Span(optim_burden or "medium"),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span("Threads: ", style={"fontWeight": "bold"}),
                                html.Span(str(threads or 6)),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span("Demos: ", style={"fontWeight": "bold"}),
                                html.Span(str(demos or 1)),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Multiple Entities: ", style={"fontWeight": "bold"}
                                ),
                                html.Span("Yes" if multiple_bool else "No"),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "AI Evaluation: ", style={"fontWeight": "bold"}
                                ),
                                html.Span("Yes" if ai_evaluation_bool else "No"),
                            ],
                            className="config-item",
                        ),
                    ],
                    className="config-summary",
                ),
                html.Div(
                    [
                        html.Div(
                            f"YAML config saved to: runs/prompt_optimization/run_{timestamp}/optim_{timestamp}.yml",
                            className="status-info",
                            style={"marginTop": "10px", "fontSize": "12px"},
                        )
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            f"Optimization task started with timestamp: {timestamp}",
                            className="status-info",
                            style={"marginTop": "5px", "fontSize": "12px"},
                        )
                    ]
                ),
            ]
        )

    except Exception as e:
        return html.Div(
            [
                html.Div(
                    f"Error during prompt optimization: {str(e)}",
                    className="status-error",
                )
            ]
        )


# Handle table parsing output field range limit checkbox show/hide
@callback(
    Output({"type": "extract-output-range-container", "index": MATCH}, "style"),
    Input({"type": "extract-output-add-range", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def toggle_extract_output_range_fields(add_range):
    if add_range and "true" in add_range:
        return {"display": "block"}
    else:
        return {"display": "none"}


# Handle table parsing output field literal list checkbox show/hide
@callback(
    Output({"type": "extract-output-literal-container", "index": MATCH}, "style"),
    Input({"type": "extract-output-add-literal", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def toggle_extract_output_literal_fields(add_literal):
    if add_literal and "true" in add_literal:
        return {"display": "block"}
    else:
        return {"display": "none"}


# Handle Table Extraction start button click event
@callback(
    Output("table-extraction-results", "children"),
    Input("start-table-extraction", "n_clicks"),
    [
        State("task-name-table-extraction", "value"),
        State("extract-parsed-file-path", "value"),
        State("extract-save-folder-path", "value"),
        State("extract-classify-prompt", "value"),
        State("extract-extract-prompt", "value"),
        State("extract-num-threads", "value"),
    ]
    + [
        State({"type": "extract-output-name", "index": ALL}, "value"),
        State({"type": "extract-output-type", "index": ALL}, "value"),
        State({"type": "extract-output-description", "index": ALL}, "value"),
        State({"type": "extract-output-range-min", "index": ALL}, "value"),
        State({"type": "extract-output-range-max", "index": ALL}, "value"),
        State({"type": "extract-output-literal-list", "index": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def handle_table_extraction(
    n_clicks,
    task_name,
    parse_file_path,
    save_dir,
    classification_prompt,
    extraction_prompt,
    threads,
    output_names,
    output_types,
    output_descriptions,
    output_range_mins,
    output_range_maxs,
    output_literal_lists,
):
    try:
        # Validate required fields
        if not parse_file_path:
            return html.Div(
                [
                    html.Div(
                        "Error: Parse File Path is required!", className="status-error"
                    )
                ]
            )

        if not save_dir:
            return html.Div(
                [
                    html.Div(
                        "Error: Save Directory is required!", className="status-error"
                    )
                ]
            )

        # Build output field configurations
        output_fields = []
        for i in range(len(output_names)):
            if output_names[i]:  # Only add when field name is not empty
                field_config = {
                    "name": output_names[i],
                    "field_type": output_types[i] if i < len(output_types) else "str",
                    "description": output_descriptions[i]
                    if i < len(output_descriptions)
                    else "",
                }

                # Add optional range constraints
                if (
                    i < len(output_range_mins)
                    and i < len(output_range_maxs)
                    and output_range_mins[i] is not None
                    and output_range_maxs[i] is not None
                ):
                    field_config["range_min"] = output_range_mins[i]
                    field_config["range_max"] = output_range_maxs[i]

                # Add optional literal list
                if i < len(output_literal_lists) and output_literal_lists[i]:
                    field_config["literal_list"] = [
                        item.strip() for item in output_literal_lists[i].split(",")
                    ]

                output_fields.append(field_config)

        # Create timestamp for creating subdirectory
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create runs/table_extraction subdirectory
        runs_base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "runs", "table_extraction"
        )
        sub_dir = os.path.join(runs_base_dir, f"run_{timestamp}")
        os.makedirs(sub_dir, exist_ok=True)

        # Generate YAML configuration file path
        yaml_filename = f"extract_table_service_config_{timestamp}.yml"
        yaml_output_path = os.path.join(sub_dir, yaml_filename)

        # use yml_generation module to save configuration
        try:
            from yml_generation.yml_table_parsing import (
                save_extract_table_config_to_yaml,
            )

            # Generate YAML configuration
            saved_yaml_path = save_extract_table_config_to_yaml(
                extract_parsed_file_path=parse_file_path,
                extract_save_folder_path=save_dir,
                extract_output_fields=output_fields,
                extract_classify_prompt=classification_prompt
                or "Please classify the table content",
                extract_extract_prompt=extraction_prompt
                or "Please extract information from the table",
                extract_num_threads=threads or 6,
                output_dir=sub_dir,
                filename=yaml_filename,
            )
            print(f"YAML configuration saved to: {saved_yaml_path}")
        except Exception as e:
            print(f"Error generating YAML configuration: {e}")
            return html.Div(
                [
                    html.Div(
                        f"Error generating YAML configuration: {str(e)}",
                        className="status-error",
                    )
                ]
            )

        # Create configuration object for subprocess call
        config = {
            "parsed_file_path": parse_file_path,
            "save_folder_path": save_dir,
            "outputFields": output_fields,
            "classify_prompt": classification_prompt
            or "Please classify the table content",
            "extract_prompt": extraction_prompt
            or "Please extract information from the table",
            "extract_directly": False,
            "num_threads": threads or 6,
        }

        # Set log file path
        log_path = os.path.join(sub_dir, f"extract_table_service_{timestamp}.json")
        
        # Create initial log in main process
        log_name = task_name if task_name else f"extract_table_{timestamp}"
        create_initial_log(name=log_name, data=config, log_path=log_path)

        # Use subprocess to call run_task_with_logging
        try:
            config_str = repr(config).replace("None", "None")
            log_path_fixed = log_path.replace(
                "\\", "\\\\"
            )  # Properly handle backslashes in Windows paths

            python_code = f"""
import sys
sys.path.extend(['{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}', '{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")}'])
from gui.call_cli import run_task_and_update_log
from src.cli.cli_handler import run_extract_table_service

config = {config_str}
result = run_task_and_update_log(
    callable_obj=run_extract_table_service,
    data=config,
    log_path=r'{log_path_fixed}'
)
print(result)
"""

            proc_log_filename = os.path.basename(log_path).replace(".json", ".log")
            process = process_manager.start_python_code(
                python_code=python_code,
                key=log_path,
                capture_output=False,
                text=True,
                log_to_key_dir=True,
                log_filename=proc_log_filename,
            )

            # stdout, stderr = process.communicate()

            # if process.returncode == 0:
            #     print(f"Table extraction task completed successfully")
            #     print(f"Task output: {stdout}")
            # else:
            #     print(f"Error in table extraction task: {stderr}")

        except Exception as e:
            print(f"Error running table extraction task: {e}")

        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                return html.Div(
                    [
                        html.Div(
                            f"Error creating save directory: {str(e)}",
                            className="status-error",
                        )
                    ]
                )

        # Return success message and configuration summary
        return html.Div(
            [
                html.Div(
                    "Table extraction started successfully!", className="status-success"
                ),
                html.H4("Configuration Summary", className="section-title"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "Parse File Path: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(parse_file_path),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Save Directory: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(save_dir),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Output Fields: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(
                                    str(len(output_fields)) + " fields configured"
                                ),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span("Threads: ", style={"fontWeight": "bold"}),
                                html.Span(str(threads or 6)),
                            ],
                            className="config-item",
                        ),
                    ],
                    className="config-summary",
                ),
                html.Div(
                    [
                        html.Div(
                            f"YAML config saved to: runs/table_extraction/run_{timestamp}/extract_table_service_config_{timestamp}.yml",
                            className="status-info",
                            style={"marginTop": "10px", "fontSize": "12px"},
                        )
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            f"Table extraction task started with timestamp: {timestamp}",
                            className="status-info",
                            style={"marginTop": "5px", "fontSize": "12px"},
                        )
                    ]
                ),
            ]
        )

    except Exception as e:
        return html.Div(
            [
                html.Div(
                    f"Error during table extraction: {str(e)}", className="status-error"
                )
            ]
        )


# Callback function extracted from app.py
@callback(
    Output("page-content", "children"),
    Output("nav-model-config", "className"),
    Output("nav-document-parsing", "className"),
    Output("nav-doc-extraction", "className"),
    Output("nav-build-optm-dataset", "className"),
    Output("nav-prompt-optimization", "className"),
    Output("nav-table-parsing", "className"),
    Input("url", "pathname"),
)
def display_page(pathname):
    from model_config import model_config_layout
    from prompt_optimization import prompt_optimization_layout
    from doc_extraction import doc_extraction_layout
    from build_optm_dataset import build_optm_dataset_layout
    from document_parsing import document_parsing_layout
    from table_parsing import table_parsing_layout
    
    # Check if main model configuration exists
    def check_main_config():
        """Check if main model configuration file exists"""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        config_dir = os.path.join(project_root, "settings")
        config_path = os.path.join(config_dir, "model_settings_main.json")
        return os.path.exists(config_path)
    
    has_main_config = check_main_config()

    # Default style class names
    model_config_class = "nav-link"
    document_parsing_class = "nav-link"
    doc_extraction_class = "nav-link"
    build_optm_dataset_class = "nav-link"
    prompt_optimization_class = "nav-link"
    table_parsing_class = "nav-link"

    # Set the corresponding navigation link to active state based on current path
    if pathname == "/model-config":
        model_config_class = "nav-link active"
        content = model_config_layout()
    elif pathname == "/doc-extraction":
        doc_extraction_class = "nav-link active"
        content = doc_extraction_layout()
    elif pathname == "/document-parsing":
        document_parsing_class = "nav-link active"
        content = document_parsing_layout()
    elif pathname == "/table-parsing":
        table_parsing_class = "nav-link active"
        content = table_parsing_layout()
    elif pathname == "/prompt-optimization":
        prompt_optimization_class = "nav-link active"
        content = prompt_optimization_layout()
    elif pathname == "/build-optm-dataset":
        build_optm_dataset_class = "nav-link active"
        content = build_optm_dataset_layout()
    else:
        # Add a dismissible path input reminder card for first-time page visit
        # This card reminds users to input full absolute paths using English characters only
        # Persist close state using localStorage via dcc.Store
        path_tip_state = dcc.Store(id="path-tip-state", storage_type="local")
        path_tip_card = html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            "Path Input Reminder",
                            style={
                                "fontWeight": "700",
                                "fontSize": "16px",
                                "color": "#92400e",
                            },
                        ),
                        html.Button(
                            "×",
                            id="close-path-tip",
                            title="Close",
                            style={
                                "background": "none",
                                "border": "none",
                                "color": "#92400e",
                                "fontSize": "18px",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "marginBottom": "8px",
                    },
                ),
                html.Div(
                    [
                        html.P(
                            "Please ensure you enter a full absolute path.",
                            style={"margin": "0 0 6px 0", "color": "#78350f"},
                        ),
                        html.P(
                            [
                                "e.g.: ",
                                html.Code(
                                    r"C:\\Users\\Alice\\Projects\\OmniExtract\\data\\input\\file.pdf",
                                    style={
                                        "backgroundColor": "#fff8e1",
                                        "padding": "2px 6px",
                                        "borderRadius": "4px",
                                        "border": "1px solid #f59e0b",
                                        "color": "#92400e",
                                    },
                                ),
                            ],
                            style={"margin": "0 0 6px 0", "color": "#78350f"},
                        ),
                        html.P(
                            "Path must use English characters only (ASCII).",
                            style={
                                "margin": "0",
                                "fontWeight": "600",
                                "color": "#92400e",
                            },
                        ),
                    ]
                ),
            ],
            id="path-tip-card",
            style={
                "backgroundColor": "#fffbeb",
                "border": "1px solid #f59e0b",
                "borderLeft": "6px solid #fbbf24",
                "color": "#92400e",
                "padding": "12px 14px",
                "borderRadius": "10px",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.06)",
                "marginBottom": "16px",
            },
        )
        # Create model_config module introduction
        model_config_intro = html.Div([
            html.H3("Model Configuration", className="module-title", 
                    style={"color": "#1e40af", "marginBottom": "15px", "paddingBottom": "8px", "borderBottom": "3px solid #3b82f6"}),
            html.P(
                [
                    "The ",
                    html.A(
                        "Model Configuration",
                        href="/model-config",
                        style={
                            "color": "#2563eb",
                            "textDecoration": "none",
                            "fontWeight": "bold",
                            "borderBottom": "2px solid #2563eb",
                            "paddingBottom": "2px",
                            "transition": "all 0.3s ease"
                        }
                    ),
                    " module allows you to set up and manage different AI models for various tasks. "
                    "You can configure models for general usage, prompt generation, evaluation, and coding tasks. "
                    "Each model can have different parameters such as temperature, max tokens, and other sampling parameters "
                    "to optimize performance for specific use cases."
                ],
                className="intro-paragraph",
                style={"marginBottom": "20px", "lineHeight": "1.6", "fontSize": "16px", 
                        "color": "#374151"}
            )
        ], style={"marginBottom": "30px", "padding": "20px", 
                  "borderRadius": "12px", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", 
                  "borderLeft": "4px solid #3b82f6"})
        
        # Create document extraction workflow introduction
        doc_extraction_intro = html.Div([
            html.H3("Extract information from PDF/XML documents", className="module-title", 
                    style={"color": "#166534", "marginBottom": "15px", "paddingBottom": "8px", "borderBottom": "3px solid #22c55e"}),
            html.Ul([
                html.Li([
                    "Prerequisites: PDF files, or XML files from ",
                    html.A("PubMed Central", href="https://www.ncbi.nlm.nih.gov/pmc/", target="_blank", 
                           style={"color": "#2563eb", "textDecoration": "none", "transition": "color 0.2s ease"}),
                    "/",
                    html.A("ScienceDirect", href="https://www.sciencedirect.com/", target="_blank", 
                           style={"color": "#2563eb", "textDecoration": "none", "transition": "color 0.2s ease"})
                ], style={"marginBottom": "10px"}),
                html.Li([
                    "Information extraction workflow:",
                    html.Ol([
                        html.Li([
                            "Use the ",
                            html.A(
                                "Document Parsing",
                                href="/document-parsing",
                                style={
                                    "color": "#2563eb",
                                    "textDecoration": "none",
                                    "fontWeight": "bold",
                                    "borderBottom": "2px solid #2563eb",
                                    "paddingBottom": "2px",
                                    "transition": "all 0.3s ease"
                                }
                            ),
                            " module to parse files. You can split document content into different sections based on requirements, or keep the entire document content."
                        ], style={"marginBottom": "8px"}),
                        html.Li([
                            "Fill out the forms in the Novel section of the ",
                            html.A(
                                "Document Extraction",
                                href="/document-extraction",
                                style={
                                    "color": "#2563eb",
                                    "textDecoration": "none",
                                    "fontWeight": "bold",
                                    "borderBottom": "2px solid #2563eb",
                                    "paddingBottom": "2px",
                                    "transition": "all 0.3s ease"
                                }
                            ),
                            " module, standardize the output format and fill in extraction instructions to extract relevant information."
                        ], style={"marginBottom": "8px"})
                    ], style={"marginTop": "10px", "marginLeft": "20px"})
                ])
            ], style={"lineHeight": "1.6", "fontSize": "16px", "color": "#374151"})
        ], style={"marginBottom": "30px", "padding": "20px", 
                  "borderRadius": "12px", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", 
                  "borderLeft": "4px solid #22c55e"})
        
        # Create optimized prompt extraction workflow introduction
        optimized_prompt_intro = html.Div([
            html.H3("Optimize extraction information prompt", className="module-title", 
                    style={"color": "#7e22ce", "marginBottom": "15px", "paddingBottom": "8px", "borderBottom": "3px solid #a855f7"}),
            html.Ul([
                html.Li("Prerequisites: Extracted information and source original files", 
                        style={"marginBottom": "10px"}),
                html.Li([
                    "Information extraction workflow:",
                    html.Ol([
                        html.Li([
                            "Use the ",
                            html.A(
                                "Document Parsing",
                                href="/document-parsing",
                                style={
                                    "color": "#2563eb",
                                    "textDecoration": "none",
                                    "fontWeight": "bold",
                                    "borderBottom": "2px solid #2563eb",
                                    "paddingBottom": "2px",
                                    "transition": "all 0.3s ease"
                                }
                            ),
                            " module to parse files. You can split document content into different sections based on requirements, or keep the entire document content."
                        ], style={"marginBottom": "8px"}),
                        html.Li([
                            "Use the ",
                            html.A(
                                "Build Optimization Dataset",
                                href="/build-optm-dataset",
                                style={
                                    "color": "#2563eb",
                                    "textDecoration": "none",
                                    "fontWeight": "bold",
                                    "borderBottom": "2px solid #2563eb",
                                    "paddingBottom": "2px",
                                    "transition": "all 0.3s ease"
                                }
                            ),
                            " module to build a dataset for prompt optimization."
                        ], style={"marginBottom": "8px"}),
                        html.Li([
                            "Fill out the forms in the ",
                            html.A(
                                "Prompt Optimization",
                                href="/prompt-optimization",
                                style={
                                    "color": "#2563eb",
                                    "textDecoration": "none",
                                    "fontWeight": "bold",
                                    "borderBottom": "2px solid #2563eb",
                                    "paddingBottom": "2px",
                                    "transition": "all 0.3s ease"
                                }
                            ),
                            " module, standardize the output format and fill in initial extraction instruction."
                        ], style={"marginBottom": "8px"}),
                        html.Li([
                            "Use the Optimized section of the ",
                            html.A(
                                "Document Extraction",
                                href="/doc-extraction",
                                style={
                                    "color": "#2563eb",
                                    "textDecoration": "none",
                                    "fontWeight": "bold",
                                    "borderBottom": "2px solid #2563eb",
                                    "paddingBottom": "2px",
                                    "transition": "all 0.3s ease"
                                }
                            ),
                            " module to extract information using the optimized prompt."
                        ], style={"marginBottom": "8px"})
                    ], style={"marginTop": "10px", "marginLeft": "20px"})
                ])
            ], style={"lineHeight": "1.6", "fontSize": "16px", "color": "#374151"})
        ], style={"marginBottom": "30px", "padding": "20px", 
                  "borderRadius": "12px", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", 
                  "borderLeft": "4px solid #a855f7"})
        
        # Create table extraction workflow introduction
        table_extraction_intro = html.Div([
            html.H3("Extract information from table files", className="module-title", 
                    style={"color": "#dc2626", "marginBottom": "15px", "paddingBottom": "8px", "borderBottom": "3px solid #f87171"}),
            html.Ul([
                html.Li("Prerequisites: Various table format files", 
                        style={"marginBottom": "10px"}),
                html.Li([
                    "Information extraction workflow:",
                    html.Ol([
                        html.Li([
                            "Use the ",
                            html.A(
                                "Table Files Parsing",
                                href="/table-parsing",
                                style={
                                    "color": "#2563eb",
                                    "textDecoration": "none",
                                    "fontWeight": "bold",
                                    "borderBottom": "2px solid #2563eb",
                                    "paddingBottom": "2px",
                                    "transition": "all 0.3s ease"
                                }
                            ),
                            " section of the Table Extraction module to parse table files uniformly into TSV format files."
                        ], style={"marginBottom": "8px"}),
                        html.Li([
                            "Use the ",
                            html.A(
                                "Extraction From Tables",
                                href="/table-parsing",
                                style={
                                    "color": "#2563eb",
                                    "textDecoration": "none",
                                    "fontWeight": "bold",
                                    "borderBottom": "2px solid #2563eb",
                                    "paddingBottom": "2px",
                                    "transition": "all 0.3s ease"
                                }
                            ),
                            " section of the Table Extraction module to standardize the output format and fill in extraction instructions, calling the predefined ReAct Agent to extract table information."
                        ], style={"marginBottom": "8px"})
                    ], style={"marginTop": "10px", "marginLeft": "20px"})
                ])
            ], style={"lineHeight": "1.6", "fontSize": "16px", "color": "#374151"})
        ], style={"marginBottom": "30px", "padding": "20px", 
                  "borderRadius": "12px", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", 
                  "borderLeft": "4px solid #f87171"})
        
        # Base content with tool introduction and path tip card
        base_content = [
            path_tip_state,
            path_tip_card,
            html.H1("Welcome to OmniExtract", className="module-title", 
                    style={"color": "#0f172a", "marginBottom": "25px", "paddingBottom": "15px", 
                           "borderBottom": "3px solid #3b82f6", "textAlign": "center", 
                           "fontSize": "2.5rem", "fontWeight": "700", "letterSpacing": "-0.5px"}),
            html.Div(
                [
                    html.P(
                        [
                            "OmniExtract is an LLM-based automatic extraction tool based on ",
                            html.A("DSPy", href="https://github.com/stanfordnlp/dspy", target="_blank", 
                                   style={"color": "#2563eb", "textDecoration": "none", "fontWeight": "600", 
                                          "transition": "all 0.3s ease"}),
                            ", specifically designed for information extraction tasks from literature and documents. It utilizes prompt optimization engineering to enhance extraction performance based on curated data, and provides various file format parsing tools, supporting batch extraction of multi-property entities from original documents (in formats such as PDF or XML) and tabular files. A video tutorial about OmniExtract is available at ",
                            html.A("OmniExtract - Tutorial", href="https://www.bilibili.com/video/BV12QywBhE1p", target="_blank", 
                                   style={"color": "#2563eb", "textDecoration": "none", "fontWeight": "600", 
                                          "transition": "all 0.3s ease"}),
                        ],
                        className="intro-paragraph",
                        style={"marginBottom": "20px", "lineHeight": "1.7", "fontSize": "17px", 
                                "color": "#334155", 
                                "padding": "20px", "borderRadius": "12px", 
                                "boxShadow": "0 4px 6px rgba(0,0,0,0.05)"}
                    ),
                ],
                style={"marginBottom": "40px"}
            )
        ]
        
        # Add model_config introduction based on configuration status
        if has_main_config:
            # If config exists, add model_config intro at the end
            content = html.Div(base_content + [doc_extraction_intro, optimized_prompt_intro, table_extraction_intro, model_config_intro])
        else:
            # If no config, add model_config intro after tool intro with red warning
            warning = html.Div(
                "⚠️ Warning: No model configuration found. Please complete the model configuration first before using other modules.",
                style={"color": "#b91c1c", "fontWeight": "bold", "marginBottom": "20px", "padding": "15px", 
                        "borderRadius": "8px", 
                        "border": "1px solid #fca5a5", "boxShadow": "0 2px 4px rgba(239,68,68,0.1)",
                        "fontSize": "16px"}
            )
            content = html.Div(base_content + [warning, doc_extraction_intro, optimized_prompt_intro, table_extraction_intro, model_config_intro])

    return (
        content,
        model_config_class,
        document_parsing_class,
        doc_extraction_class,
        build_optm_dataset_class,
        prompt_optimization_class,
        table_parsing_class,
    )

# Persist path tip close state to local storage when close button is clicked
@callback(
    Output("path-tip-state", "data"),
    Input("close-path-tip", "n_clicks"),
    State("path-tip-state", "data"),
    prevent_initial_call=True,
)
def persist_path_tip_close(n_clicks, data):
    if n_clicks:
        return {"dismissed": True}
    return dash.no_update

# Sync path tip card visibility with persisted state
@callback(
    Output("path-tip-card", "style"),
    Input("path-tip-state", "data"),
    State("path-tip-card", "style"),
)
def sync_path_tip_visibility(data, current_style):
    dismissed = isinstance(data, dict) and data.get("dismissed") is True
    new_style = dict(current_style or {})
    if dismissed:
        new_style["display"] = "none"
        return new_style
    # Ensure visible if not dismissed
    if "display" in new_style:
        new_style.pop("display")
    return new_style


# ===== BUILD OPTIMIZATION DATASET CALLBACKS =====

# Note: Range and Literal container display/hide is handled by dynamic callbacks below


# Callback functions for adding and removing Fields
@callback(
    Output("fields-container", "children"),
    Input("add-field", "n_clicks"),
    Input({"type": "delete-field", "index": ALL}, "n_clicks"),
    State("fields-container", "children"),
    prevent_initial_call=True,
)
def manage_fields(add_clicks, dynamic_clicks, children):
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return children

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # If Add Field button was clicked
    if button_id == "add-field":
        # Create new Field component (match initial field styles and IDs)
        new_field_id = f"field-{add_clicks}"
        new_field = html.Div([
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}, children=[
                html.H4("Field", style={"margin": "0", "fontSize": "16px"}),
                html.Button(
                    "×",
                    id={"type": "delete-field", "index": new_field_id},
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
            html.Div(style={"display": "flex", "gap": "20px"}, children=[
                html.Div(style={"flex": 1}, children=[
                    html.Label("Field Name", className="form-label"),
                    dcc.Input(
                        id={"type": "field-name", "index": new_field_id},
                        placeholder="field_name",
                        value=f"field_name_{add_clicks}",
                        className="form-input",
                        style={"height": "38px"}
                    )
                ]),
                html.Div(style={"flex": 1}, children=[
                    html.Label("Field Type", className="form-label"),
                    dcc.Dropdown(
                        id={"type": "field-type", "index": new_field_id},
                        options=[
                            {"label": "String", "value": "str"},
                            {"label": "Integer", "value": "int"},
                            {"label": "Float", "value": "float"},
                            {"label": "Boolean", "value": "bool"},
                            {"label": "List", "value": "list"},
                            {"label": "Literal", "value": "literal"},
                            {"label": "List Literal", "value": "list_literal"},
                            {"label": "Image", "value": "image"}
                        ],
                        value="str",
                        clearable=False,
                        style={"height": "38px","border":"1px solid #e1e5e9","border-radius":"6px"}
                    )
                ])
            ]),
            html.Div([
                html.Label("Field Description", className="form-label"),
                dcc.Input(
                    id={"type": "field-description", "index": new_field_id},
                    placeholder="Description of this field",
                    value="Description of this field",
                    className="form-input"
                )
            ], className="form-group"),
            html.Div(style={"display": "flex", "gap": "20px"}, children=[
                html.Div(style={"flex": 1}, children=[
                    dcc.Checklist(
                        id={"type": "field-add-range", "index": new_field_id},
                        options=[{"label": "Add Range Limits", "value": "true"}],
                        className="form-checkbox"
                    )
                ]),
                html.Div(style={"flex": 1}, children=[
                    dcc.Checklist(
                        id={"type": "field-add-literal", "index": new_field_id},
                        options=[{"label": "Add Literal List", "value": "true"}],
                        className="form-checkbox"
                    )
                ])
            ]),
            html.Div(id={"type": "field-range-container", "index": new_field_id}, style={"display": "none"}, children=[
                html.Div([
                    html.Label("Range Minimum", className="form-label"),
                    dcc.Input(
                        id={"type": "field-range-min", "index": new_field_id},
                        type="number",
                        placeholder="Min value",
                        className="form-input"
                    )
                ], className="form-group"),
                html.Div([
                    html.Label("Range Maximum", className="form-label"),
                    dcc.Input(
                        id={"type": "field-range-max", "index": new_field_id},
                        type="number",
                        placeholder="Max value",
                        className="form-input"
                    )
                ], className="form-group")
            ]),
            html.Div(id={"type": "field-literal-container", "index": new_field_id}, style={"display": "none"}, children=[
                html.Div([
                    html.Label("Literal List (comma separated)", className="form-label"),
                    dcc.Input(
                        id={"type": "field-literal-list", "index": new_field_id},
                        placeholder="option1,option2,option3",
                        className="form-input"
                    )
                ], className="form-group")
            ])
        ], id=new_field_id, className="section-container", style={"border": "1px solid #e1e5e9", "border-radius": "8px", "padding": "16px", "margin-bottom": "20px"})

        # Add new Field to existing list
        children.append(new_field)

    # If Delete button was clicked
    else:
        # Ensure at least one Field is kept
        if len(children) > 1:
            # Determine the field ID to remove
            field_id_to_remove = None

            # Parse dynamic delete button ID
            try:
                button_data = json.loads(button_id)
                field_id_to_remove = button_data["index"]
            except Exception as e:
                print(f"Error parsing button ID: {e}")

            # If the field ID to remove is found
            if field_id_to_remove:
                # Find and remove the corresponding Field
                for i, child in enumerate(children):
                    # Check different ID storage methods
                    child_id = None
                    if hasattr(child, "id"):
                        child_id = child.id
                    elif hasattr(child, "props") and "id" in child.props:
                        child_id = child.props["id"]
                    elif "props" in child and "id" in child["props"]:
                        child_id = child["props"]["id"]

                    # If a matching ID is found, remove the child element
                    if child_id == field_id_to_remove:
                        children.pop(i)
                        break

    return children


# Handle Range container display/hide for dynamically created Fields
@callback(
    Output({"type": "field-range-container", "index": MATCH}, "style"),
    Input({"type": "field-add-range", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def toggle_dynamic_field_range_fields(add_range):
    if add_range and "true" in add_range:
        return {"display": "block"}
    else:
        return {"display": "none"}


# Handle Literal container display/hide for dynamically created Fields
@callback(
    Output({"type": "field-literal-container", "index": MATCH}, "style"),
    Input({"type": "field-add-literal", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def toggle_dynamic_field_literal_fields(add_literal):
    if add_literal and "true" in add_literal:
        return {"display": "block"}
    else:
        return {"display": "none"}

# Handle Range container display/hide for dynamically created Dataset Output Fields
@callback(
    Output({"type": "dataset-output-range-container", "index": MATCH}, "style"),
    Input({"type": "dataset-output-add-range", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def toggle_dynamic_dataset_output_range_fields(add_range):
    if add_range and "true" in add_range:
        return {"display": "block"}
    else:
        return {"display": "none"}

# Handle Literal container display/hide for dynamically created Dataset Output Fields
@callback(
    Output({"type": "dataset-output-literal-container", "index": MATCH}, "style"),
    Input({"type": "dataset-output-add-literal", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def toggle_dynamic_dataset_output_literal_fields(add_literal):
    if add_literal and "true" in add_literal:
        return {"display": "block"}
    else:
        return {"display": "none"}


# Handle Build Dataset button click event
@callback(
    Output("build-results", "children"),
    Input("start-building-dataset", "n_clicks"),
    [
        State("json-path", "value"),
        State("curated-dataset-path", "value"),
        State("save-directory", "value"),
        State("article-field", "value"),
        State("article-parts", "value"),
        State("multiple-entities", "value"),
        State("task-name-dataset", "value"),
    ]
    + [
        State({"type": "dataset-output-name", "index": ALL}, "value"),
        State({"type": "dataset-output-type", "index": ALL}, "value"),
        State({"type": "dataset-output-description", "index": ALL}, "value"),
        State({"type": "dataset-output-range-min", "index": ALL}, "value"),
        State({"type": "dataset-output-range-max", "index": ALL}, "value"),
        State({"type": "dataset-output-literal-list", "index": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def handle_build_dataset(
    n_clicks,
    json_path,
    curated_dataset_path,
    save_directory,
    article_field,
    article_parts,
    multiple_entities,
    task_name_dataset,
    field_names,
    field_types,
    field_descriptions,
    field_range_mins,
    field_range_maxs,
    field_literal_lists,
):
    try:
        # Validate required fields
        if not json_path:
            return html.Div(
                [
                    html.Div(
                        "Error: JSON Files Path is required!", className="status-error"
                    )
                ]
            )

        if not curated_dataset_path:
            return html.Div(
                [
                    html.Div(
                        "Error: Curated Dataset Path is required!",
                        className="status-error",
                    )
                ]
            )
        
        if not save_directory:
            return html.Div(
                [
                    html.Div(
                        "Error: Save Directory is required!",
                        className="status-error",
                    )
                ]
            )
        # Build field data using dynamic field parsing
        fields_data = parse_field_data_from_states(
            field_names,
            field_types,
            field_descriptions,
            field_range_mins,
            field_range_maxs,
            field_literal_lists,
            field_type="output",
        )

        # Import yml generation function
        from yml_generation.yml_build_optm_dataset import (
            generate_build_optm_dataset_yml_from_dash_callback,
        )

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory
        output_dir = os.path.join("runs", "build_optim_dataset", f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        config = {
            "json_path": json_path,
            "dataset": curated_dataset_path,
            "save_dir": save_directory,
            "fields": fields_data,
            "multiple": bool(multiple_entities and "true" in multiple_entities),
            "article_field": article_field,
            "article_parts": article_parts,
        }

        # Generate yml file path
        yml_filename = f"build_optm_set_config_{timestamp}.yml"
        yml_output_path = os.path.join(output_dir, yml_filename)

        # Generate YAML configuration
        yml_content = generate_build_optm_dataset_yml_from_dash_callback(
            json_path=json_path,
            curated_dataset_path=curated_dataset_path,
            fields_data=fields_data,
            multiple_entities=bool(multiple_entities and "true" in multiple_entities),
            article_field=article_field,
            article_parts=article_parts or [],
            save_directory=save_directory,
        )

        # Save yml file
        with open(yml_output_path, "w", encoding="utf-8") as f:
            f.write(yml_content)

        # Create log file path
        log_path = os.path.join(output_dir, f"build_dataset_{timestamp}.json")
        
        # Create initial log in main process
        create_initial_log(name=task_name_dataset or f"{timestamp}", data=config, log_path=log_path)

        # Use subprocess to call run_task_with_logging
        config_str = repr(config).replace("None", "None")

        # Properly handle backslashes in Windows paths
        log_path_fixed = log_path.replace("\\", "\\\\")

        python_code = f"""
import sys
sys.path.extend(['{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}', '{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")}'])
from gui.call_cli import run_task_and_update_log
from src.cli.cli_handler import run_build_optm_set

config = {config_str}
result = run_task_and_update_log(
    callable_obj=run_build_optm_set,
    data=config,
    log_path=r'{log_path_fixed}'
)
print(result)
"""

        proc_log_filename = os.path.basename(log_path).replace(".json", ".log")
        process = process_manager.start_python_code(
            python_code=python_code,
            key=log_path,
            capture_output=True,
            text=True,
            log_to_key_dir=True,
            log_filename=proc_log_filename,
        )

        # stdout, stderr = process.communicate()

        # if process.returncode == 0:
        #     print(f"Dataset build task completed successfully")
        #     print(f"Task output: {stdout}")
        # else:
        #     print(f"Error in dataset build task: {stderr}")

        # Return success message and configuration summary
        return html.Div(
            [
                html.Div(
                    "Optimization dataset build started successfully!",
                    className="status-success",
                ),
                html.H4("Configuration Summary", className="section-title"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "JSON Files Path: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(json_path),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Curated Dataset Path: ",
                                    style={"fontWeight": "bold"},
                                ),
                                html.Span(curated_dataset_path),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Save Directory: ",
                                    style={"fontWeight": "bold"},
                                ),
                                html.Span(save_directory if save_directory else "Default location"),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Article Field Name: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(article_field),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Article Parts: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(
                                    ", ".join(article_parts)
                                    if article_parts
                                    else "Entire article"
                                ),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Multiple Entities: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(
                                    "Yes"
                                    if multiple_entities and "true" in multiple_entities
                                    else "No"
                                ),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Fields Configured: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(str(len(fields_data))),
                            ],
                            className="config-item",
                        ),
                    ],
                    className="config-summary",
                ),
                html.Div(
                    [
                        html.Div(
                            f"YAML config saved to: runs/build_optim_dataset/run_{timestamp}/build_optm_set_config_{timestamp}.yml",
                            className="status-info",
                            style={"marginTop": "10px", "fontSize": "12px"},
                        )
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            f"Build dataset task started with timestamp: {timestamp}",
                            className="status-info",
                            style={"marginTop": "5px", "fontSize": "12px"},
                        )
                    ]
                ),
            ]
        )

    except Exception as e:
        return html.Div(
            [
                html.Div(
                    f"Error during dataset build: {str(e)}", className="status-error"
                )
            ]
        )


# Handle Range container display/hide for dynamically created InputFields
@callback(
    Output({"type": "input-range-container", "index": MATCH}, "style"),
    Input({"type": "input-add-range", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def toggle_dynamic_input_range_fields(add_range):
    if add_range and "true" in add_range:
        return {"display": "block"}
    else:
        return {"display": "none"}


# Handle Literal container display/hide for dynamically created InputFields
@callback(
    Output({"type": "input-literal-container", "index": MATCH}, "style"),
    Input({"type": "input-add-literal", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def toggle_dynamic_input_literal_fields(add_literal):
    if add_literal and "true" in add_literal:
        return {"display": "block"}
    else:
        return {"display": "none"}


# Callback functions for adding and removing OutputFields
@callback(
    Output("output-fields-container", "children"),
    Input("add-output-field", "n_clicks"),
    Input({"type": "delete-output-field", "index": ALL}, "n_clicks"),
    State("output-fields-container", "children"),
    prevent_initial_call=True,
)
def manage_output_fields(add_clicks, dynamic_clicks, children):
    return _manage_output_fields_common(add_clicks, dynamic_clicks, children, "add-output-field", "output")


@callback(
    Output("extract-output-fields-container", "children"),
    [
        Input("add-extract-output-field", "n_clicks"),
        Input({"type": "delete-extract-output-field", "index": ALL}, "n_clicks"),
    ],
    State("extract-output-fields-container", "children"),
    prevent_initial_call=True,
)
def manage_extract_output_fields(add_clicks, dynamic_clicks, children):
    return _manage_output_fields_common(add_clicks, dynamic_clicks, children, "add-extract-output-field", "extract-output")


def _manage_output_fields_common(add_clicks, dynamic_clicks, children, add_button_id, id_prefix):
    ctx = dash.callback_context
    if not ctx.triggered:
        return children
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    button_value = ctx.triggered[0]["value"]
    if button_value is None or button_value == 0:
        return children
    if button_id == add_button_id:
        existing_ids = set()
        for child in children:
            child_id = None
            if hasattr(child, "id"):
                child_id = child.id
            elif hasattr(child, "props") and "id" in child.props:
                child_id = child.props["id"]
            elif "props" in child and "id" in child["props"]:
                child_id = child["props"]["id"]
            if child_id and child_id.startswith(f"{id_prefix}-field-"):
                existing_ids.add(child_id)
        counter = 0
        new_field_id = f"{id_prefix}-field-{counter}"
        while new_field_id in existing_ids:
            counter += 1
            new_field_id = f"{id_prefix}-field-{counter}"
        header_text = "Output Field" if id_prefix in ("output", "extract-output") else "Field"
        new_field = html.Div([
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}, children=[
                html.H4(header_text, style={"margin": "0", "fontSize": "16px"}),
                html.Button("×", id={"type": f"delete-{id_prefix}-field", "index": new_field_id}, className="delete-button", style={"background": "none", "border": "none", "color": "#6b7280", "fontSize": "20px", "cursor": "pointer", "padding": "0 5px", "lineHeight": "1", "borderRadius": "4px", "hover": {"background": "#f3f4f6", "color": "#ef4444"}})
            ]),
            html.Div(style={"display": "flex", "gap": "20px"}, children=[
                html.Div(style={"flex": 1}, children=[
                    html.Label("Field Name", className="form-label"),
                    dcc.Input(id={"type": f"{id_prefix}-name", "index": new_field_id}, placeholder="field_name", value=f"field_{add_clicks}", className="form-input", style={"height": "38px"})
                ]),
                html.Div(style={"flex": 1}, children=[
                    html.Label("Field Type", className="form-label"),
                    dcc.Dropdown(id={"type": f"{id_prefix}-type", "index": new_field_id}, options=[
                        {"label": "String", "value": "str"},
                        {"label": "Integer", "value": "int"},
                        {"label": "Float", "value": "float"},
                        {"label": "Boolean", "value": "bool"},
                        {"label": "List", "value": "list"},
                        {"label": "Literal", "value": "literal"},
                        {"label": "List Literal", "value": "list_literal"}
                    ], value="str", clearable=False, style={"height": "38px", "border": "1px solid #e1e5e9", "border-radius": "6px"})
                ])
            ]),
            html.Div([
                html.Label("Field Description", className="form-label"),
                dcc.Input(id={"type": f"{id_prefix}-description", "index": new_field_id}, placeholder="Description of what to extract", className="form-input")
            ], className="form-group"),
            html.Div(style={"display": "flex", "gap": "20px"}, children=[
                html.Div(style={"flex": 1}, children=[
                    dcc.Checklist(id={"type": f"{id_prefix}-add-range", "index": new_field_id}, options=[{"label": "Add Range Limits", "value": "true"}], className="form-checkbox")
                ]),
                html.Div(style={"flex": 1}, children=[
                    dcc.Checklist(id={"type": f"{id_prefix}-add-literal", "index": new_field_id}, options=[{"label": "Add Literal List", "value": "true"}], className="form-checkbox")
                ])
            ]),
            html.Div(id={"type": f"{id_prefix}-range-container", "index": new_field_id}, style={"display": "none"}, children=[
                html.Div([
                    html.Label("Range Minimum", className="form-label"),
                    dcc.Input(id={"type": f"{id_prefix}-range-min", "index": new_field_id}, type="number", placeholder="Min value", className="form-input")
                ], className="form-group"),
                html.Div([
                    html.Label("Range Maximum", className="form-label"),
                    dcc.Input(id={"type": f"{id_prefix}-range-max", "index": new_field_id}, type="number", placeholder="Max value", className="form-input")
                ], className="form-group")
            ]),
            html.Div(id={"type": f"{id_prefix}-literal-container", "index": new_field_id}, style={"display": "none"}, children=[
                html.Div([
                    html.Label("Literal List (comma separated)", className="form-label"),
                    dcc.Input(id={"type": f"{id_prefix}-literal-list", "index": new_field_id}, placeholder="option1,option2,option3", className="form-input")
                ], className="form-group")
            ])
        ], id=new_field_id, className="section-container", style={"border": "1px solid #e1e5e9", "borderRadius": "8px", "padding": "16px", "marginBottom": "20px"})
        children.append(new_field)
    else:
        if len(children) > 1:
            field_id_to_remove = None
            try:
                button_data = json.loads(button_id)
                field_id_to_remove = button_data["index"]
            except Exception:
                pass
            if field_id_to_remove:
                for i, child in enumerate(children):
                    child_id = None
                    if hasattr(child, "id"):
                        child_id = child.id
                    elif hasattr(child, "props") and "id" in child.props:
                        child_id = child.props["id"]
                    elif "props" in child and "id" in child["props"]:
                        child_id = child["props"]["id"]
                    if child_id == field_id_to_remove:
                        children.pop(i)
                        break
    return children


# Handle Range container display/hide for dynamically created OutputFields
@callback(
    Output({"type": "output-range-container", "index": MATCH}, "style"),
    Input({"type": "output-add-range", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def toggle_dynamic_output_range_fields(add_range):
    if add_range and "true" in add_range:
        return {"display": "block"}
    else:
        return {"display": "none"}


# Handle Literal container display/hide for dynamically created OutputFields
@callback(
    Output({"type": "output-literal-container", "index": MATCH}, "style"),
    Input({"type": "output-add-literal", "index": MATCH}, "value"),
    prevent_initial_call=True,
)
def toggle_dynamic_output_literal_fields(add_literal):
    if add_literal and "true" in add_literal:
        return {"display": "block"}
    else:
        return {"display": "none"}

# Callback functions for adding and removing Dataset Output Fields
@callback(
    Output("dataset-output-fields-container", "children"),
    Input("add-dataset-output-field", "n_clicks"),
    Input({"type": "delete-dataset-output-field", "index": ALL}, "n_clicks"),
    State("dataset-output-fields-container", "children"),
    prevent_initial_call=True,
)
def manage_dataset_output_fields(add_clicks, dynamic_clicks, children):
    return _manage_output_fields_common(add_clicks, dynamic_clicks, children, "add-dataset-output-field", "dataset-output")


# Handle Original Extraction Start button click event
@callback(
    Output("original-extraction-results", "children"),
    Input("start-extraction", "n_clicks"),
    [
        State("dataset-path", "value"),
        State("save-dir", "value"),
        State("judging", "value"),
        State("threads", "value"),
        State("initial-prompt", "value"),
        State("task", "value"),
        State("multiple", "value"),
        State("task-name-original", "value"),
    ]
    + [
        State({"type": "input-name", "index": ALL}, "value"),
        State({"type": "input-type", "index": ALL}, "value"),
        State({"type": "input-description", "index": ALL}, "value"),
        State({"type": "input-range-min", "index": ALL}, "value"),
        State({"type": "input-range-max", "index": ALL}, "value"),
        State({"type": "input-literal-list", "index": ALL}, "value"),
    ]
    + [
        State({"type": "output-name", "index": ALL}, "value"),
        State({"type": "output-type", "index": ALL}, "value"),
        State({"type": "output-description", "index": ALL}, "value"),
        State({"type": "output-range-min", "index": ALL}, "value"),
        State({"type": "output-range-max", "index": ALL}, "value"),
        State({"type": "output-literal-list", "index": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def handle_original_extraction(
    n_clicks,
    dataset_path,
    save_dir,
    judging,
    threads,
    initial_prompt,
    task,
    multiple,
    task_name_original,
    input_names,
    input_types,
    input_descriptions,
    input_range_mins,
    input_range_maxs,
    input_literal_lists,
    output_names,
    output_types,
    output_descriptions,
    output_range_mins,
    output_range_maxs,
    output_literal_lists,
):
    try:
        # Validate required fields
        if not dataset_path:
            return html.Div(
                [html.Div("Error: Dataset Path is required!", className="status-error")]
            )

        if not save_dir:
            return html.Div(
                [
                    html.Div(
                        "Error: Save Directory is required!", className="status-error"
                    )
                ]
            )

        # Use parse_field_data_from_states to build input field configurations
        input_fields = parse_field_data_from_states(
            field_names=input_names,
            field_types=input_types,
            field_descriptions=input_descriptions,
            field_range_mins=input_range_mins,
            field_range_maxs=input_range_maxs,
            field_literal_lists=input_literal_lists,
            field_type="input",
        )

        # If no input fields, add default field
        if not input_fields:
            input_fields = [
                {
                    "name": "Method",
                    "field_type": "str",
                    "description": "Input field description",
                }
            ]

        # Use parse_field_data_from_states to build output field configurations
        output_fields = parse_field_data_from_states(
            field_names=output_names,
            field_types=output_types,
            field_descriptions=output_descriptions,
            field_range_mins=output_range_mins,
            field_range_maxs=output_range_maxs,
            field_literal_lists=output_literal_lists,
            field_type="output",
        )

        # If no output fields, add default field
        if not output_fields:
            output_fields = [
                {
                    "name": "extracted_value",
                    "field_type": "str",
                    "description": "Extracted data from documents",
                }
            ]

        # Create runs/doc_extraction subpath
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_subdir = os.path.join("runs", "doc_extraction", "original", f"run_{timestamp}")
        # full_save_dir = os.path.join(save_dir, run_subdir)

        # Ensure directory exists
        os.makedirs(run_subdir, exist_ok=True)

        # Generate YAML configuration file path
        yaml_filename = f"config_{timestamp}.yml"
        yaml_path = os.path.join(run_subdir, yaml_filename)

        config = {
            "dataset_path": dataset_path,
            "save_dir": save_dir,
            "input_fields": input_fields,
            "output_fields": output_fields,
            "initial_prompt": initial_prompt,
            "judging": judging,
            "threads": threads,
            "multiple": multiple,
        }

        # Call yml_generation module to generate YAML file
        try:
            from yml_generation.yml_doc_extraction import generate_novel_config_yaml

            # Generate YAML configuration
            generate_novel_config_yaml(
                dataset_path=dataset_path,
                save_dir=save_dir,
                input_fields=input_fields,
                output_fields=output_fields,
                initial_prompt=initial_prompt
                or "You are a data analyst organizing the data usage in literature published in the journal Nature Communications. Your goals are: 1. Extract the identifiers of all public datasets mentioned in the literature 2. Extract the identifiers of all non-public datasets (i.e., self-created datasets) provided in the literature 3. Extract the names of all databases used in the literature. Please carefully read the methods section of the literature to completely and accurately extract the above information.",
                judging=judging or "",
                task=task or "Extraction",
                threads=threads or 6,
                multiple=bool(multiple and "true" in multiple),
                output_path=yaml_path,
            )

            # Prepare config data for calling run_pred_original
            config = {
                "dataset": dataset_path,
                "save_dir": save_dir,
                "inputFields": input_fields,
                "outputFields": output_fields,
                "initial_prompt": initial_prompt
                or "You are a data analyst organizing the data usage in literature published in the journal Nature Communications. Your goals are: 1. Extract the identifiers of all public datasets mentioned in the literature 2. Extract the identifiers of all non-public datasets (i.e., self-created datasets) provided in the literature 3. Extract the names of all databases used in the literature. Please carefully read the methods section of the literature to completely and accurately extract the above information.",
                "judging": judging or "",
                "task": task or "Extraction",
                "threads": threads or 6,
                "multiple": bool(multiple and "true" in multiple),
            }

            # Use subprocess to call run_task_with_logging
            log_path = os.path.join(run_subdir, f"original_extraction_{timestamp}.json")
            
            # Create initial log in main process
            create_initial_log(name=task_name_original or f"{timestamp}", data=config, log_path=log_path)
            
            log_path_fixed = log_path.replace("\\", "\\\\")
            config_str = repr(config).replace("None", "None")
            python_code = f"""
import sys
sys.path.extend(['{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}', '{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")}'])
from gui.call_cli import run_task_and_update_log
from src.cli.cli_handler import run_pred_original

config = {config_str}
result = run_task_and_update_log(
    callable_obj=run_pred_original,
    data=config,
    log_path=r'{log_path_fixed}')
print(result)
"""

            proc_log_filename = os.path.basename(log_path).replace(".json", ".log")
            process = process_manager.start_python_code(
                python_code=python_code,
                key=log_path,
                capture_output=False,
                text=True,
                log_to_key_dir=True,
                log_filename=proc_log_filename,
            )

            # Can choose to wait for process completion or handle asynchronously
            # stdout, stderr = process.communicate()

            # Create success message
            success_message = html.Div(
                [
                    html.Div(
                        "Original extraction started successfully!",
                        className="status-success",
                    ),
                    html.H4("Configuration Summary", className="section-title"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "Dataset Path: ", style={"fontWeight": "bold"}
                                    ),
                                    html.Span(dataset_path),
                                ],
                                className="config-item",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        "Save Directory: ", style={"fontWeight": "bold"}
                                    ),
                                    html.Span(save_dir),
                                ],
                                className="config-item",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        "Judging Mode: ", style={"fontWeight": "bold"}
                                    ),
                                    html.Span(judging or "None"),
                                ],
                                className="config-item",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        "Threads: ", style={"fontWeight": "bold"}
                                    ),
                                    html.Span(threads or 6),
                                ],
                                className="config-item",
                            ),
                        ],
                        className="config-summary",
                    ),
                    html.Div(
                        [
                            html.Div(
                                f"YAML config saved to: runs/doc_extraction/novel/run_{timestamp}/config_{timestamp}.yml",
                                className="status-info",
                                style={"marginTop": "10px", "fontSize": "12px"},
                            )
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                f"Original extraction task started with timestamp: {timestamp}",
                                className="status-info",
                                style={"marginTop": "5px", "fontSize": "12px"},
                            )
                        ]
                    ),
                ]
            )

            return success_message

        except ImportError as e:
            return html.Div(
                [
                    html.Div(
                        f"Error importing YAML generation module: {str(e)}",
                        className="status-error",
                    )
                ]
            )
        except Exception as e:
            return html.Div(
                [
                    html.Div(
                        f"Error generating YAML config: {str(e)}",
                        className="status-error",
                    )
                ]
            )

    except Exception as e:
        return html.Div(
            [
                html.Div(
                    f"Error during original extraction: {str(e)}", className="status-error"
                )
            ]
        )


# Handle Optimized Extraction Start button click event
@callback(
    Output("optim-extraction-results", "children"),
    Input("optim-start-extraction", "n_clicks"),
    [
        State("optim-load-dir", "value"),
        State("optim-dataset-path", "value"),
        State("optim-save-dir", "value"),
        State("optim-judging", "value"),
        State("optim-threads", "value"),
        State("task-name-optimized", "value"),
    ],
    prevent_initial_call=True,
)
def handle_optimized_extraction(
    n_clicks, load_dir, dataset_path, save_dir, judging, threads, task_name_optimized
):
    try:
        # Validate required fields
        if not load_dir:
            return html.Div(
                [
                    html.Div(
                        "Error: Load Directory is required!", className="status-error"
                    )
                ]
            )

        if not dataset_path:
            return html.Div(
                [html.Div("Error: Dataset Path is required!", className="status-error")]
            )

        if not save_dir:
            return html.Div(
                [
                    html.Div(
                        "Error: Save Directory is required!", className="status-error"
                    )
                ]
            )

        # Import methods from yml_generation module
        from yml_generation.yml_doc_extraction import (
            generate_optimized_config_from_callback,
        )

        # Generate timestamp for creating subdirectory
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create runs\doc_extraction\optm\{timestamp} subpath
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "runs", "doc_extraction", "optm"
        )
        run_subdir = os.path.join(base_dir, timestamp)

        # Ensure subdirectory exists
        os.makedirs(run_subdir, exist_ok=True)

        config = {
            "load_dir": load_dir,
            "dataset": dataset_path,
            "save_dir": save_dir,
            "judging": judging,
            "threads": threads or 6,
            "output_file": "result.json",
        }

        # Generate YAML configuration file path
        yaml_filename = f"optimized_extraction_config_{timestamp}.yml"
        yaml_output_path = os.path.join(run_subdir, yaml_filename)

        # Use yml_generation module to generate YAML configuration file
        try:
            yaml_file_path = generate_optimized_config_from_callback(
                load_dir=load_dir,
                dataset=dataset_path,
                save_dir=save_dir,
                judging=judging,
                threads=threads or 6,
                output_file="result.json",
                output_path=yaml_output_path,
            )
            print(f"YAML configuration generated successfully: {yaml_file_path}")

            # Create log file path
            log_path = os.path.join(run_subdir, f"pred_optimized_{timestamp}.json")
            
            # Create initial log in main process
            create_initial_log(name=task_name_optimized or f"{timestamp}", data=config, log_path=log_path)

            # Correctly handle backslashes in Windows paths
            log_path_fixed = log_path.replace("\\", "\\\\")

            # Build project root directory path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            src_path = os.path.join(project_root, "src")
            config_str = repr(config).replace("None", "None")

            # Build Python code to call run_task_with_logging
            python_code = f"""
import sys
sys.path.extend(['{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}', '{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")}'])
from gui.call_cli import run_task_and_update_log
from src.cli.cli_handler import run_pred_optimized

config = {config_str}
result = run_task_and_update_log(
    callable_obj=run_pred_optimized,
    data=config,
    log_path=r'{log_path_fixed}'
)
print(result)
"""

            proc_log_filename = os.path.basename(log_path).replace(".json", ".log")
            process = process_manager.start_python_code(
                python_code=python_code,
                key=log_path,
                capture_output=True,
                text=True,
                log_to_key_dir=True,
                log_filename=proc_log_filename,
            )

            # stdout, stderr = process.communicate()

            # if process.returncode == 0:
            #     print(f"Optimized extraction task completed successfully")
            #     print(f"Task output: {stdout}")
            # else:
            #     print(f"Error in optimized extraction task: {stderr}")

        except Exception as e:
            print(f"Error generating YAML configuration: {e}")
            yaml_file_path = None

        # Create configuration object
        config = {
            "load_dir": load_dir,
            "dataset": dataset_path,
            "save_dir": save_dir,
            "judging": judging,
            "threads": threads or 6,
        }

        # Print configuration information (may call other functions for processing in actual applications)
        print(f"Starting optimized extraction with configuration: {config}")

        # Create result directory (if it doesn't exist)
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                return html.Div(
                    [
                        html.Div(
                            f"Error creating save directory: {str(e)}",
                            className="status-error",
                        )
                    ]
                )

        # Simulate extraction process
        # In actual applications, the actual extraction function would be called here

        # Return success message and configuration summary
        return html.Div(
            [
                html.Div(
                    "Optimized extraction started successfully!",
                    className="status-success",
                ),
                html.H4("Configuration Summary", className="section-title"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "Load Directory: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(load_dir),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Dataset Path: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(dataset_path),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Save Directory: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(save_dir),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Judging Mode: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(judging or "None"),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span("Threads: ", style={"fontWeight": "bold"}),
                                html.Span(threads or 6),
                            ],
                            className="config-item",
                        ),
                    ],
                    className="config-summary",
                ),
                html.Div(
                    [
                        html.Div(
                            f"YAML config saved to: runs/doc_extraction/optm/{timestamp}/{yaml_filename}",
                            className="status-info",
                            style={"marginTop": "10px", "fontSize": "12px"},
                        )
                    ]
                )
                if yaml_file_path
                else html.Div(
                    [
                        html.Div(
                            "Warning: YAML configuration generation failed",
                            className="status-warning",
                            style={"marginTop": "10px", "fontSize": "12px"},
                        )
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            f"Optimized extraction task started with timestamp: {timestamp}",
                            className="status-info",
                            style={"marginTop": "5px", "fontSize": "12px"},
                        )
                    ]
                )
                if yaml_file_path
                else None,
            ]
        )

    except Exception as e:
        return html.Div(
            [
                html.Div(
                    f"Error during optimized extraction: {str(e)}",
                    className="status-error",
                )
            ]
        )


# ===== DOCUMENT PARSING CALLBACKS =====


# Handle Document Parsing Start button click event
@callback(
    Output("parsing-results", "children"),
    Input("start-document-parsing", "n_clicks"),
    [
        State("task-name", "value"),
        State("folder-path", "value"),
        State("save-path", "value"),
        State("file-type", "value"),
        State("convert-mode", "value"),
    ],
    prevent_initial_call=True,
)
def handle_document_parsing(
    n_clicks, task_name, folder_path, save_path, file_type, convert_mode
):
    try:
        # Validate required fields
        if not folder_path:
            return html.Div(
                [
                    html.Div(
                        "Error: Source Folder Path is required!",
                        className="status-error",
                    )
                ]
            )

        if not save_path:
            return html.Div(
                [html.Div("Error: Save Path is required!", className="status-error")]
            )

        # Create save directory (if it doesn't exist)
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except Exception as e:
                return html.Div(
                    [
                        html.Div(
                            f"Error creating save directory: {str(e)}",
                            className="status-error",
                        )
                    ]
                )

        # Generate timestamp for creating subdirectory and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create runs/doc_parsing subpath
        runs_base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "runs", "doc_parsing"
        )
        sub_dir = os.path.join(runs_base_dir, f"run_{timestamp}")
        os.makedirs(sub_dir, exist_ok=True)

        # Generate YAML configuration file to runs/doc_parsing subpath
        yaml_filename = f"file_to_json_config_{timestamp}.yml"
        yaml_output_path = os.path.join(sub_dir, yaml_filename)

        # Create configuration object for display
        config = {
            "folder_path": folder_path,
            "save_path": save_path,
            "file_type": file_type,
            "convert_mode": convert_mode,
        }

        try:
            # Use yml_generation module to save configuration
            saved_yaml_path = save_document_parsing_config_to_yaml(
                folder_path=folder_path,
                save_path=save_path,
                file_type=file_type,
                convert_mode=convert_mode,
                output_dir=sub_dir,
                filename=yaml_filename,
            )
            print(f"YAML configuration saved to: {saved_yaml_path}")

            # Create log file path
            log_path = os.path.join(sub_dir, f"file_to_json_{timestamp}.json")
            
            # Create initial log in main process
            create_initial_log(name=f"{timestamp}", data=config, log_path=log_path)

            # Correctly handle backslashes in Windows paths
            log_path_fixed = log_path.replace("\\", "\\\\")
            config_str = repr(config).replace("None", "None")

            # Build Python code to call run_task_and_update_log
            python_code = f"""
import sys
sys.path.extend(['{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}', '{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")}'])
from gui.call_cli import run_task_and_update_log
from src.cli.cli_handler import run_file_to_json

config = {config_str}
result = run_task_and_update_log(
    callable_obj=run_file_to_json,
    data=config,
    log_path=r'{log_path_fixed}'
)
print(result)
"""

            proc_log_filename = os.path.basename(log_path).replace(".json", ".log")
            process = process_manager.start_python_code(
                python_code=python_code,
                key=log_path,
                capture_output=False,
                text=True,
                log_to_key_dir=True,
                log_filename=proc_log_filename,
            )

            # stdout, stderr = process.communicate()

            # if process.returncode == 0:
            #     print(f"File to JSON conversion task completed successfully")
            #     print(f"Task output: {stdout}")
            # else:
            #     print(f"Error in file to JSON conversion task: {stderr}")

        except Exception as e:
            print(f"Error generating YAML configuration: {e}")
            return html.Div(
                [
                    html.Div(
                        f"Error generating YAML configuration: {str(e)}",
                        className="status-error",
                    )
                ]
            )

        # Print configuration information
        print(f"Starting document parsing with configuration: {config}")

        # Return success message and configuration summary
        return html.Div(
            [
                html.Div(
                    "Document parsing started successfully!", className="status-success"
                ),
                html.H4("Configuration Summary", className="section-title"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("Task Name: ", style={"fontWeight": "bold"}),
                                html.Span(
                                    task_name if task_name else f"Default ({timestamp})"
                                ),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Source Folder Path: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(folder_path),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span("Save Path: ", style={"fontWeight": "bold"}),
                                html.Span(save_path),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span("File Type: ", style={"fontWeight": "bold"}),
                                html.Span(file_type),
                            ],
                            className="config-item",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Conversion Mode: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(convert_mode),
                            ],
                            className="config-item",
                        ),
                    ],
                    className="config-summary",
                ),
                html.Div(
                    [
                        html.Div(
                            f"YAML config saved to: runs/doc_parsing/run_{timestamp}/file_to_json_config_{timestamp}.yml",
                            className="status-info",
                            style={"marginTop": "10px", "fontSize": "12px"},
                        )
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            f"Document parsing task started with timestamp: {timestamp}",
                            className="status-info",
                            style={"marginTop": "5px", "fontSize": "12px"},
                        )
                    ]
                ),
            ]
        )

    except Exception as e:
        return html.Div(
            [
                html.Div(
                    f"Error during document parsing: {str(e)}", className="status-error"
                )
            ]
        )


# ===== TABLE PARSING CALLBACKS =====


# Handle Table Files Parsing Start button click event
@callback(
    Output("table-files-parsing-results", "children"),
    Input("start-table-parsing", "n_clicks"),
    [
        State("task-name-table-parsing", "value"),
        State("table-folder-path", "value"),
        State("table-save-path", "value"),
        State("table-file-type", "value"),
    ],
    prevent_initial_call=True,
)
def handle_table_parsing(n_clicks, task_name, folder_path, save_path, file_type):
    try:
        print(
            f"Table parsing callback triggered - n_clicks: {n_clicks}, task_name: {task_name}, folder_path: {folder_path}, save_path: {save_path}, file_type: {file_type}"
        )

        # Validate required fields
        if not folder_path:
            print("Error: Source Folder Path is required!")
            return html.Div(
                [
                    html.Div(
                        "Error: Source Folder Path is required!",
                        className="status-error",
                    )
                ]
            )

        if not save_path:
            print("Error: Save Path is required!")
            return html.Div(
                [html.Div("Error: Save Path is required!", className="status-error")]
            )

        # Create configuration object
        config = {
            "file_folder_path": folder_path,
            "save_folder_path": save_path,
            "non_tabular_file_format": file_type,
        }

        # Print configuration information
        print(f"Starting table parsing with configuration: {config}")

        # Create save directory (if it doesn't exist)
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except Exception as e:
                return html.Div(
                    [
                        html.Div(
                            f"Error creating save directory: {str(e)}",
                            className="status-error",
                        )
                    ]
                )

        # Generate timestamp for creating subpath
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create runs/table_parsing subpath
        runs_base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "runs", "table_parsing"
        )
        sub_dir = os.path.join(runs_base_dir, f"run_{timestamp}")
        os.makedirs(sub_dir, exist_ok=True)

        # Generate YAML configuration file to runs/table_parsing subpath
        yaml_filename = f"parse_table_to_tsv_config_{timestamp}.yml"

        try:
            # Use yml_generation module to save configuration
            saved_yaml_path = save_table_parsing_config_to_yaml(
                table_folder_path=folder_path,
                table_save_path=save_path,
                table_file_type=file_type,
                output_dir=sub_dir,
                filename=yaml_filename,
            )
            print(f"YAML configuration saved to: {saved_yaml_path}")

            # Set log file path
            log_path = os.path.join(sub_dir, f"parse_table_to_tsv_{timestamp}.json")
            
            # Create initial log in main process
            log_name = task_name if task_name else f"{timestamp}"
            create_initial_log(name=log_name, data=config, log_path=log_path)
            
            log_path_fixed = log_path.replace("\\", "\\\\")
            config_str = repr(config).replace("None", "None")

            # Use subprocess to call run_task_and_update_log
            python_code = f"""
import sys
sys.path.extend(['{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}', '{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")}'])
from gui.call_cli import run_task_and_update_log
from src.cli.cli_handler import run_parse_table_to_tsv

config = {config_str}
result = run_task_and_update_log(
    callable_obj=run_parse_table_to_tsv,
    data=config,
    log_path=r'{log_path_fixed}')
print(result)
"""

            proc_log_filename = os.path.basename(log_path).replace(".json", ".log")
            process = process_manager.start_python_code(
                python_code=python_code,
                key=log_path,
                capture_output=True,
                text=True,
                log_to_key_dir=True,
                log_filename=proc_log_filename,
            )

            # stdout, stderr = process.communicate()

            # if process.returncode == 0:
            #     print(f"Table parsing task completed successfully")
            #     print(f"Task output: {stdout}")
            # else:
            #     print(f"Error in table parsing task: {stderr}")

        except Exception as e:
            print(f"Error generating YAML configuration or running task: {e}")
            return html.Div(
                [
                    html.Div(
                        f"Error generating YAML configuration or running task: {str(e)}",
                        className="status-error",
                    )
                ]
            )

        # Return success message and configuration summary
        return html.Div(
            [
                html.Div(
                    "Table parsing started successfully!", className="status-success"
                ),
                html.H4("Configuration Summary", className="section-title"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Source Folder Path: ",
                                            style={"fontWeight": "bold"},
                                        ),
                                        html.Span(folder_path),
                                    ],
                                    className="config-item",
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Save Path: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(save_path),
                                    ],
                                    className="config-item",
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "File Type: ", style={"fontWeight": "bold"}
                                        ),
                                        html.Span(file_type),
                                    ],
                                    className="config-item",
                                ),
                            ],
                            className="config-summary",
                        )
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            f"YAML config saved to: runs/table_parsing/run_{timestamp}/parse_table_to_tsv_config_{timestamp}.yml",
                            className="status-info",
                            style={"marginTop": "10px", "fontSize": "12px"},
                        )
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            f"Table parsing task started with timestamp: {timestamp}",
                            className="status-info",
                            style={"marginTop": "5px", "fontSize": "12px"},
                        )
                    ]
                ),
            ]
        )

    except Exception as e:
        print(f"Error in table parsing callback: {str(e)}")
        import traceback

        traceback.print_exc()
        return html.Div(
            [
                html.Div(
                    f"Error during table parsing: {str(e)}", className="status-error"
                )
            ]
        )


 


# Task card toggle callback
@callback(
    Output({"type": "collapse", "index": MATCH}, "is_open"),
    Input({"type": "toggle-btn", "index": MATCH}, "n_clicks"),
    State({"type": "collapse", "index": MATCH}, "is_open"),
    prevent_initial_call=True,
)
def toggle_task_card(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output({"type": "cancel-task-btn", "index": MATCH}, "disabled"),
    Output({"type": "cancel-task-btn", "index": MATCH}, "children"),
    Output({"type": "status-badge", "index": MATCH}, "children"),
    Output({"type": "status-badge", "index": MATCH}, "color"),
    Output({"type": "cancel-error", "index": MATCH}, "children"),
    Input({"type": "cancel-task-btn", "index": MATCH}, "n_clicks"),
    State({"type": "cancel-log-path", "index": MATCH}, "data"),
    prevent_initial_call=True
)
def handle_cancel_task(n_clicks, log_path):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    btn_text = "Cancelling..."
    success = cancel_task(log_path)
    if success:
        return True, "Cancelled", "cancelled", "secondary", ""
    alert = dbc.Alert("failed to cancel task", color="danger")
    return False, "Cancel task", "running", "primary", alert


# ===== CONFIGURATION WRITEBACK CALLBACKS =====


# Handle model configuration load and writeback
@callback(
    [
        Output("config-load-status", "children", allow_duplicate=True),
        Output("model-name", "value", allow_duplicate=True),
        Output("model-type", "value", allow_duplicate=True),
        Output("api-base", "value", allow_duplicate=True),
        Output("api-key", "value", allow_duplicate=True),
        Output("api-version", "value", allow_duplicate=True),
        Output("temperature", "value", allow_duplicate=True),
        Output("max-tokens", "value", allow_duplicate=True),
        Output("top-p", "value", allow_duplicate=True),
        Output("frequency-penalty", "value", allow_duplicate=True),
        Output("presence-penalty", "value", allow_duplicate=True),
        Output("timeout", "value", allow_duplicate=True),
        Output("max-retries", "value", allow_duplicate=True),
    ],
    Input({"type": "load-model-config", "index": ALL}, "n_clicks"),
    [State({"type": "load-model-config-file-path", "index": ALL}, "data")],
    prevent_initial_call=True,
)
def handle_model_config_writeback(n_clicks_list, taskcard_config_path_list):
    """
    Handle model configuration loading and writeback
    """
    # Get the triggered button context
    ctx = dash.callback_context
    if not ctx.triggered:
        return (
            "",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    try:
        form_values = get_form_values_for_module("model_config", taskcard_config_path_list)
    except ValueError as e:
        if str(e) == "NO_PATH":
            return (
                "",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        return (
            html.Div(
                f"Failed to load configuration: {str(e)}", className="status-error"
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    status_msg = html.Div(
        "Configuration loaded successfully!", className="status-success"
    )
    return (
        status_msg,
        form_values.get("model-name"),
        form_values.get("model-type"),
        form_values.get("api-base"),
        form_values.get("api-key"),
        form_values.get("api-version"),
        form_values.get("temperature"),
        form_values.get("max-tokens"),
        form_values.get("top-p"),
        form_values.get("frequency-penalty"),
        form_values.get("presence-penalty"),
        form_values.get("timeout"),
        form_values.get("max-retries"),
    )


# Handle run optimization configuration writeback
@callback(
    [
        Output("optim-config-load-status", "children", allow_duplicate=True),
        Output("optim-config-modal-body", "children", allow_duplicate=True),
        Output("optim-config-modal", "is_open", allow_duplicate=True),
        Output("dataset-path", "value", allow_duplicate=True),
        Output("save-dir", "value", allow_duplicate=True),
        Output("task-name-prompt-optimization", "value", allow_duplicate=True),
        Output("initial-prompt", "value", allow_duplicate=True),
        Output("task", "value", allow_duplicate=True),
        Output("optim-burden", "value", allow_duplicate=True),
        Output("threads", "value", allow_duplicate=True),
        Output("demos", "value", allow_duplicate=True),
        Output("multiple", "value", allow_duplicate=True),
        Output("ai-evaluation", "value", allow_duplicate=True),
        Output("input-fields-container", "children", allow_duplicate=True),
        Output("output-fields-container", "children", allow_duplicate=True),
    ],
    Input({"type": "load-optim-config", "index": ALL}, "n_clicks"),
    [State({"type": "load-optim-config-file-path", "index": ALL}, "data")],
    prevent_initial_call=True,
)
def handle_run_optimization_writeback(n_clicks_list, taskcard_config_path_list):
    """
    Handle run optimization configuration loading and writeback
    """
    # Get the triggered button context
    ctx = dash.callback_context
    if not ctx.triggered:
        return (
            "",
            "",
            False,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    try:
        form_values = get_form_values_for_module('run_optimization', taskcard_config_path_list)
    except ValueError as e:
        if str(e) == 'NO_PATH':
            error_content = html.Div([
                html.H5("No Configuration File Path", className="text-warning mb-3"),
                html.P("No configuration file path was provided.", className="mb-2"),
            ])
            return (
                "",
                error_content,
                True,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        error_content = html.Div([
            html.H5("Failed to Load Configuration", className="text-danger mb-3"),
            html.P(f"Error: {str(e)}", className="mb-2"),
        ])
        return (
            html.Div(
                f"Failed to load configuration: {str(e)}", className="status-error"
            ),
            error_content,
            True,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    input_fields = []
    output_fields = []
    if 'input-fields-data' in form_values:
        input_fields = generate_dynamic_fields(form_values['input-fields-data'], 'input')
    if 'output-fields-data' in form_values:
        output_fields = generate_dynamic_fields(form_values['output-fields-data'], 'output')
    modal_content = html.Div([
        html.H5("Configuration Loaded Successfully!", className="text-success mb-3"),
        html.H6("Basic Configuration:", className="mb-2"),
        html.P(f"Dataset Path: {form_values.get('dataset-path', 'Not specified')}", className="mb-1"),
        html.P(f"Save Directory: {form_values.get('save-dir', 'Not specified')}", className="mb-1"),
        html.P(f"Experiment Name: {form_values.get('experiment-name', 'Not specified')}", className="mb-1"),
        html.P(f"Initial Prompt: {form_values.get('initial-prompt', 'Not specified')}", className="mb-1"),
        html.P(f"Task: {form_values.get('task', 'Not specified')}", className="mb-1"),
        html.P(f"Optimization Burden: {form_values.get('optim-burden', 'Not specified')}", className="mb-1"),
        html.P(f"Threads: {form_values.get('threads', 'Not specified')}", className="mb-1"),
        html.P(f"Demos: {form_values.get('demos', 'Not specified')}", className="mb-1"),
        html.P(f"Multiple Entities: {form_values.get('multiple', 'Not specified')}", className="mb-1"),
        html.P(f"AI Evaluation: {form_values.get('ai-evaluation', 'Not specified')}", className="mb-1"),
        html.Hr(className="my-3"),
        html.H6("Input Fields:", className="mb-2"),
        html.Div([
            html.Div([
                html.Div(f"Field {i+1}:", className="fw-bold mb-1"),
                html.P(f"Name: {field.get('name', 'N/A')}", className="mb-0 ms-3"),
                html.P(f"Type: {field.get('field_type', 'N/A')}", className="mb-0 ms-3"),
                html.P(f"Description: {field.get('description', 'N/A')}", className="mb-0 ms-3")
            ], className="mb-2")
            for i, field in enumerate(form_values.get('input-fields-data', []))
        ]) if form_values.get('input-fields-data') else html.P("No input fields configured", className="text-muted"),
        html.Hr(className="my-3"),
        html.H6("Output Fields:", className="mb-2"),
        html.Div([
            html.Div([
                html.Div(f"Field {i+1}:", className="fw-bold mb-1"),
                html.P(f"Name: {field.get('name', 'N/A')}", className="mb-0 ms-3"),
                html.P(f"Type: {field.get('field_type', 'N/A')}", className="mb-0 ms-3"),
                html.P(f"Description: {field.get('description', 'N/A')}", className="mb-0 ms-3")
            ], className="mb-2")
            for i, field in enumerate(form_values.get('output-fields-data', []))
        ]) if form_values.get('output-fields-data') else html.P("No output fields configured", className="text-muted"),
    ])
    status_msg = html.Div(
        "Configuration loaded successfully!", className="status-success"
    )
    multiple_value = form_values.get("multiple")
    if multiple_value is True:
        multiple_value = ["true"]
    else:
        multiple_value = []
    ai_evaluation_value = form_values.get("ai-evaluation")
    if ai_evaluation_value is True:
        ai_evaluation_value = ["true"]
    else:
        ai_evaluation_value = []
    return (
        status_msg,
        modal_content,
        True,
        form_values.get('dataset-path'),
        form_values.get('save-dir'),
        form_values.get('experiment-name'),
        form_values.get('initial-prompt'),
        form_values.get('task'),
        form_values.get('optim-burden'),
        form_values.get('threads'),
        form_values.get('demos'),
        multiple_value,
        ai_evaluation_value,
        input_fields if input_fields else None,
        output_fields if output_fields else None
    )


# Handle table extraction configuration writeback
@callback(
    [
        Output("table-extraction-config-load-status", "children", allow_duplicate=True),
        Output("table-extraction-config-modal-body", "children", allow_duplicate=True),
        Output("table-extraction-config-modal", "is_open", allow_duplicate=True),
        Output("extract-parsed-file-path", "value", allow_duplicate=True),
        Output("extract-save-folder-path", "value", allow_duplicate=True),
        Output("extract-num-threads", "value", allow_duplicate=True),
        Output("extract-output-fields-container", "children", allow_duplicate=True),
        Output("extract-classify-prompt", "value", allow_duplicate=True),
        Output("extract-extract-prompt", "value", allow_duplicate=True),
    ],
    Input({"type": "load-table-config", "index": ALL}, "n_clicks"),
    [State({"type": "load-table-config-file-path", "index": ALL}, "data")],
    prevent_initial_call=True,
)
def handle_table_extraction_writeback(n_clicks_list, taskcard_config_path_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", "", False, None, None, None, None, None, None
    try:
        form_values = get_form_values_for_module("table_extraction", taskcard_config_path_list)
    except ValueError as e:
        if str(e) == "NO_PATH":
            error_content = html.Div([
                html.H5("No Configuration File Path", className="text-warning mb-3"),
                html.P("No configuration file path was provided.", className="mb-2"),
            ])
            return "", error_content, True, None, None, None, None, None, None
        return (
            html.Div(f"Failed to load configuration: {str(e)}", className="status-error"),
            html.Div([
                html.H5("Failed to Load Configuration", className="text-danger mb-3"),
                html.P(f"Error: {str(e)}", className="mb-2"),
            ]),
            True,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    try:
        output_fields = []
        if "table-output-fields-data" in form_values:
            output_fields = generate_extract_output_fields(form_values["table-output-fields-data"])
        modal_content = html.Div(
            [
                html.H5(
                    "Configuration Loaded Successfully!", className="text-success mb-3"
                ),
                html.H6("Basic Configuration:", className="mb-2"),
                html.P(
                    f"Parsed File Path: {form_values.get('table-dataset-path', 'Not specified')}",
                    className="mb-1",
                ),
                html.P(
                    f"Save Folder Path: {form_values.get('table-save-dir', 'Not specified')}",
                    className="mb-1",
                ),
                html.P(
                    f"Number of Threads: {form_values.get('table-threads', 'Not specified')}",
                    className="mb-1",
                ),
                # Display Classification Prompt
                html.Hr(className="my-3"),
                html.H6("Classification Prompt:", className="mb-2"),
                html.P(
                    f"{form_values.get('table-classify-prompt', 'Not specified')}",
                    className="mb-1",
                ),
                # Display Extraction Prompt
                html.Hr(className="my-3"),
                html.H6("Extraction Prompt:", className="mb-2"),
                html.P(
                    f"{form_values.get('table-extract-prompt', 'Not specified')}",
                    className="mb-1",
                ),
                # Display Output Fields
                html.Hr(className="my-3"),
                html.H6("Output Fields:", className="mb-2"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(f"Field {i + 1}:", className="fw-bold mb-1"),
                                html.P(
                                    f"Name: {field.get('name', 'N/A')}",
                                    className="mb-0 ms-3",
                                ),
                                html.P(
                                    f"Type: {field.get('field_type', 'N/A')}",
                                    className="mb-0 ms-3",
                                ),
                                html.P(
                                    f"Description: {field.get('description', 'N/A')}",
                                    className="mb-0 ms-3",
                                ),
                                # 显示范围限制信息
                                html.P(
                                    f"Range Limit: {field.get('range_limit', 'None')}",
                                    className="mb-0 ms-3",
                                ),
                                # 显示字面量列表信息
                                html.P(
                                    f"Literal List: {', '.join(field.get('literal_list', []))}"
                                    if field.get("literal_list")
                                    else "Literal List: None",
                                    className="mb-0 ms-3",
                                ),
                            ],
                            className="mb-2",
                        )
                        for i, field in enumerate(
                            form_values.get("table-output-fields-data", [])
                        )
                    ]
                )
                if form_values.get("table-output-fields-data")
                else html.P("No output fields configured", className="text-muted"),
            ]
        )
        status_msg = html.Div(
            "Configuration loaded successfully!", className="status-success"
        )
        table_dataset_path = form_values.get("table-dataset-path")
        table_save_dir = form_values.get("table-save-dir")
        table_threads = form_values.get("table-threads")
        table_classify_prompt = form_values.get("table-classify-prompt")
        table_extract_prompt = form_values.get("table-extract-prompt")

        return (
            status_msg,
            modal_content,
            True,  # Open the modal
            table_dataset_path,
            table_save_dir,
            table_threads,
            output_fields if output_fields else None,
            table_classify_prompt,
            table_extract_prompt,
        )
    except Exception as e:
        error_content = html.Div([
            html.H5("Failed to Load Configuration", className="text-danger mb-3"),
            html.P(f"Error: {str(e)}", className="mb-2"),
        ])
        return (
            html.Div(f"Failed to load configuration: {str(e)}", className="status-error"),
            error_content,
            True,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# Handle build dataset configuration writeback
@callback(
    [
        Output("dataset-config-load-status", "children", allow_duplicate=True),
        Output("dataset-config-modal-body", "children", allow_duplicate=True),
        Output("dataset-config-modal", "is_open", allow_duplicate=True),
        Output("json-path", "value", allow_duplicate=True),
        Output("curated-dataset-path", "value", allow_duplicate=True),
        Output("save-directory", "value", allow_duplicate=True),
        Output("multiple-entities", "value", allow_duplicate=True),
        Output("article-field", "value", allow_duplicate=True),
        Output("article-parts", "value", allow_duplicate=True),
        Output("dataset-output-fields-container", "children", allow_duplicate=True),
    ],
    Input({"type": "load-dataset-config", "index": ALL}, "n_clicks"),
    [State({"type": "load-dataset-config-file-path", "index": ALL}, "data")],
    prevent_initial_call=True,
)
def handle_build_dataset_writeback(n_clicks_list, taskcard_config_path_list):
    """
    Handle build dataset configuration loading and writeback
    """
    # Get the triggered button context
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", None, None, None, None, None, None, None, None, None

    try:
        form_values = get_form_values_for_module("build_dataset", taskcard_config_path_list)
    except ValueError as e:
        if str(e) == "NO_PATH":
            return (
                "",
                html.Div(
                    "No configuration file path provided", className="alert alert-warning"
                ),
                True,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        error_content = html.Div([
            html.H5("Failed to Load Configuration", className="text-danger mb-3"),
            html.P(f"Error: {str(e)}", className="mb-2"),
        ])
        return (
            html.Div(
                f"Failed to load configuration: {str(e)}", className="status-error"
            ),
            error_content,
            True,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    fields = []
    if "fields-data" in form_values:
        fields = generate_dataset_output_fields(form_values["fields-data"])
    modal_content = html.Div(
        [
            html.H5(
                "Configuration Loaded Successfully!", className="text-success mb-3"
            ),
            html.H6("Basic Configuration:", className="mb-2"),
            html.P(
                f"JSON Files Path: {form_values.get('json-path', 'Not specified')}",
                className="mb-1",
            ),
            html.P(
                f"Curated Dataset Path: {form_values.get('curated-dataset-path', 'Not specified')}",
                className="mb-1",
            ),
            html.P(
                f"Save Directory: {form_values.get('save-dir', 'Not specified')}",
                className="mb-1",
            ),
            html.P(
                f"Multiple Entities: {form_values.get('multiple-entities', 'Not specified')}",
                className="mb-1",
            ),
            html.P(
                f"Article Field Name: {form_values.get('article-field', 'Not specified')}",
                className="mb-1",
            ),
            html.P(
                f"Article Parts: {form_values.get('article-parts', 'Not specified')}",
                className="mb-1",
            ),
            html.Hr(className="my-3"),
            html.H6("Fields:", className="mb-2"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(f"Field {i + 1}:", className="fw-bold mb-1"),
                            html.P(
                                f"Name: {field.get('name', 'N/A')}",
                                className="mb-0 ms-3",
                            ),
                            html.P(
                                f"Type: {field.get('field_type', 'N/A')}",
                                className="mb-0 ms-3",
                            ),
                            html.P(
                                f"Description: {field.get('description', 'N/A')}",
                                className="mb-0 ms-3",
                            ),
                        ],
                        className="mb-2",
                    )
                    for i, field in enumerate(form_values.get("fields-data", []))
                ]
            )
            if form_values.get("fields-data")
            else html.P("No fields configured", className="text-muted"),
        ]
    )
    status_msg = html.Div(
        "Configuration loaded successfully!", className="status-success"
    )
    multiple_value = form_values.get("multiple-entities", False)
    if multiple_value is True:
        multiple_value = ["true"]
    else:
        multiple_value = []
    return (
        status_msg,
        modal_content,
        True,
        form_values.get("json-path"),
        form_values.get("curated-dataset-path"),
        form_values.get("save-dir"),
        multiple_value,
        form_values.get("article-field"),
        form_values.get("article-parts"),
        fields if fields else None,
    )


@callback(
    [
        Output("original-config-load-status", "children", allow_duplicate=True),
        Output("original-config-modal-body", "children", allow_duplicate=True),
        Output("original-config-modal", "is_open", allow_duplicate=True),
        Output("dataset-path", "value", allow_duplicate=True),
        Output("save-dir", "value", allow_duplicate=True),
        Output("initial-prompt", "value", allow_duplicate=True),
        Output("judging", "value", allow_duplicate=True),
        Output("threads", "value", allow_duplicate=True),
        Output("multiple", "value", allow_duplicate=True),
        Output("input-fields-container", "children", allow_duplicate=True),
        Output("output-fields-container", "children", allow_duplicate=True),
    ],
    Input({"type": "load-original-config", "index": ALL}, "n_clicks"),
    [State({"type": "load-original-config-file-path", "index": ALL}, "data")],
    prevent_initial_call=True,
)
def handle_original_extraction_writeback(n_clicks_list, taskcard_config_path_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", None, None, None, None, None, None, None, None, None, None
    try:
        form_values = get_form_values_for_module("original_extraction", taskcard_config_path_list)
    except ValueError as e:
        if str(e) == "NO_PATH":
            return (
                "",
                html.Div("No configuration file path provided", className="alert alert-warning"),
                True,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        return (
            html.Div(f"Failed to load configuration: {str(e)}", className="status-error"),
            html.Div([
                html.H5("Failed to Load Configuration", className="text-danger mb-3"),
                html.P(f"Error: {str(e)}", className="mb-2"),
            ]),
            True,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    try:
        input_fields = []
        output_fields = []
        if "original-input-fields-data" in form_values:
            input_fields = generate_dynamic_fields(form_values["original-input-fields-data"], "input")
        if "original-output-fields-data" in form_values:
            output_fields = generate_dynamic_fields(form_values["original-output-fields-data"], "output")
        modal_content = html.Div(
            [
                html.H5(
                    "Configuration Loaded Successfully!", className="text-success mb-3"
                ),
                html.H6("Basic Configuration:", className="mb-2"),
                html.P(
                    f"Dataset Path: {form_values.get('original-dataset-path', 'Not specified')}",
                    className="mb-1",
                ),
                html.P(
                    f"Save Directory: {form_values.get('original-save-dir', 'Not specified')}",
                    className="mb-1",
                ),
                html.P(
                    f"Initial Prompt: {form_values.get('original-initial-prompt', 'Not specified')}",
                    className="mb-1",
                ),
                html.P(
                    f"Judging: {form_values.get('original-judging', 'Not specified')}",
                    className="mb-1",
                ),
                html.P(
                    f"Threads: {form_values.get('original-threads', 'Not specified')}",
                    className="mb-1",
                ),
                html.P(
                    f"Multiple: {form_values.get('original-multiple', 'Not specified')}",
                    className="mb-1",
                ),
                # Display Input Fields
                html.Hr(className="my-3"),
                html.H6("Input Fields:", className="mb-2"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(f"Field {i + 1}:", className="fw-bold mb-1"),
                                html.P(
                                    f"Name: {field.get('name', 'N/A')}",
                                    className="mb-0 ms-3",
                                ),
                                html.P(
                                    f"Type: {field.get('field_type', 'N/A')}",
                                    className="mb-0 ms-3",
                                ),
                                html.P(
                                    f"Description: {field.get('description', 'N/A')}",
                                    className="mb-0 ms-3",
                                ),
                            ],
                            className="mb-2",
                        )
                        for i, field in enumerate(
                            form_values.get("original-input-fields-data", [])
                        )
                    ]
                )
                if form_values.get("original-input-fields-data")
                else html.P("No input fields configured", className="text-muted"),
                # Display Output Fields
                html.Hr(className="my-3"),
                html.H6("Output Fields:", className="mb-2"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(f"Field {i + 1}:", className="fw-bold mb-1"),
                                html.P(
                                    f"Name: {field.get('name', 'N/A')}",
                                    className="mb-0 ms-3",
                                ),
                                html.P(
                                    f"Type: {field.get('field_type', 'N/A')}",
                                    className="mb-0 ms-3",
                                ),
                                html.P(
                                    f"Description: {field.get('description', 'N/A')}",
                                    className="mb-0 ms-3",
                                ),
                            ],
                            className="mb-2",
                        )
                        for i, field in enumerate(
                            form_values.get("original-output-fields-data", [])
                        )
                    ]
                )
                if form_values.get("original-output-fields-data")
                else html.P("No output fields configured", className="text-muted"),
            ]
        )
        status_msg = html.Div(
            "Configuration loaded successfully!", className="status-success"
        )
        multiple_value = form_values.get("original-multiple", False)
        if multiple_value is True:
            multiple_value = ["true"]
        else:
            multiple_value = []
        return (
            status_msg,
            modal_content,
            True,  # Open the modal
            form_values.get("original-dataset-path"),
            form_values.get("original-save-dir"),
            form_values.get("original-initial-prompt"),
            form_values.get("original-judging"),
            form_values.get("original-threads"),
            multiple_value,
            input_fields if input_fields else None,
            output_fields if output_fields else None,
        )
    except Exception as e:
        error_content = html.Div([
            html.H5("Failed to Load Configuration", className="text-danger mb-3"),
            html.P(f"Error: {str(e)}", className="mb-2"),
        ])
        return (
            html.Div(f"Failed to load configuration: {str(e)}", className="status-error"),
            error_content,
            True,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# Handle optimized extraction configuration writeback
@callback(
    [
        Output("optim-extraction-config-load-status", "children", allow_duplicate=True),
        Output("optim-extraction-config-modal-body", "children", allow_duplicate=True),
        Output("optim-extraction-config-modal", "is_open", allow_duplicate=True),
        Output("optim-load-dir", "value", allow_duplicate=True),
        Output("optim-dataset-path", "value", allow_duplicate=True),
        Output("optim-save-dir", "value", allow_duplicate=True),
        Output("optim-judging", "value", allow_duplicate=True),
        Output("optim-threads", "value", allow_duplicate=True),
    ],
    Input({"type": "load-optim-extraction-config", "index": ALL}, "n_clicks"),
    [State({"type": "load-optim-extraction-config-file-path", "index": ALL}, "data")],
    prevent_initial_call=True,
)
def handle_optimized_extraction_writeback(n_clicks_list, taskcard_config_path_list):
    """
    Handle optimized extraction configuration loading and writeback
    """
    # Get the triggered button context
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", "", False, None, None, None, None, None

    try:
        form_values = get_form_values_for_module("optimized_extraction", taskcard_config_path_list)
    except ValueError as e:
        if str(e) == "NO_PATH":
            error_content = html.Div(
                [
                    html.H5("No Configuration File Path", className="text-warning mb-3"),
                    html.P("No configuration file path was provided.", className="mb-2"),
                ]
            )
            return "", error_content, True, None, None, None, None, None
        error_content = html.Div(
            [
                html.H5("Failed to Load Configuration", className="text-danger mb-3"),
                html.P(f"Error: {str(e)}", className="mb-2"),
            ]
        )
        return (
            html.Div(
                f"Failed to load configuration: {str(e)}", className="status-error"
            ),
            error_content,
            True,
            None,
            None,
            None,
            None,
            None,
        )
    modal_content = html.Div(
        [
            html.H5(
                "Configuration Loaded Successfully!", className="text-success mb-3"
            ),
            html.H6("Loaded Configuration Values:", className="mb-3"),
            html.Div(
                [
                    html.P(
                        f"Load Directory: {form_values.get('optim-load-dir', 'Not specified')}",
                        className="mb-2",
                    ),
                    html.P(
                        f"Dataset Path: {form_values.get('optim-dataset-path', 'Not specified')}",
                        className="mb-2",
                    ),
                    html.P(
                        f"Save Directory: {form_values.get('optim-save-dir', 'Not specified')}",
                        className="mb-2",
                    ),
                    html.P(
                        f"Judging Mode: {form_values.get('optim-judging', 'Not specified')}",
                        className="mb-2",
                    ),
                    html.P(
                        f"Threads: {form_values.get('optim-threads', 'Not specified')}",
                        className="mb-2",
                    ),
                ],
                className="ms-3",
            ),
        ]
    )
    status_msg = html.Div(
        "Configuration loaded successfully!", className="status-success"
    )
    return (
        status_msg,
        modal_content,
        True,
        form_values.get("optim-load-dir"),
        form_values.get("optim-dataset-path"),
        form_values.get("optim-save-dir"),
        form_values.get("optim-judging"),
        form_values.get("optim-threads"),
    )


# Handle document parsing configuration writeback
@callback(
    [
        Output("parsing-config-load-status", "children", allow_duplicate=True),
        Output("parsing-config-modal-body", "children", allow_duplicate=True),
        Output("parsing-config-modal", "is_open", allow_duplicate=True),
        Output("folder-path", "value", allow_duplicate=True),
        Output("save-path", "value", allow_duplicate=True),
        Output("file-type", "value", allow_duplicate=True),
        Output("convert-mode", "value", allow_duplicate=True),
    ],
    Input({"type": "load-parsing-config", "index": ALL}, "n_clicks"),
    [State({"type": "load-parsing-config-file-path", "index": ALL}, "data")],
    prevent_initial_call=True,
)
def handle_document_parsing_writeback(n_clicks_list, taskcard_config_path_list):
    """
    Handle document parsing configuration loading and writeback
    """
    # Get the triggered button context
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", "", False, None, None, None, None

    try:
        form_values = get_form_values_for_module("document_parsing", taskcard_config_path_list)
    except ValueError as e:
        if str(e) == "NO_PATH":
            return (
                "",
                html.Div(
                    "No configuration file path provided", className="alert alert-warning"
                ),
                True,
                None,
                None,
                None,
                None,
            )
        error_content = html.Div(
            [
                html.H5("Failed to Load Configuration", className="text-danger mb-3"),
                html.P(f"Error: {str(e)}", className="mb-2"),
            ]
        )
        return (
            html.Div(f"Failed to load configuration: {str(e)}", className="status-error"),
            error_content,
            True,
            None,
            None,
            None,
            None,
        )

    modal_content = html.Div(
        [
            html.H5(
                "Configuration Loaded Successfully!", className="text-success mb-3"
            ),
            html.H6("Loaded Values:", className="mb-2"),
            html.P(
                f"Folder Path: {form_values.get('folder-path', 'Not specified')}",
                className="mb-1",
            ),
            html.P(
                f"Save Path: {form_values.get('save-path', 'Not specified')}",
                className="mb-1",
            ),
            html.P(
                f"File Type: {form_values.get('file-type', 'Not specified')}",
                className="mb-1",
            ),
            html.P(
                f"Convert Mode: {form_values.get('convert-mode', 'Not specified')}",
                className="mb-1",
            ),
        ]
    )

    status_msg = html.Div(
        "Configuration loaded successfully!", className="status-success"
    )
    return (
        status_msg,
        modal_content,
        True,
        form_values.get("folder-path"),
        form_values.get("save-path"),
        form_values.get("file-type"),
        form_values.get("convert-mode"),
    )


# Toggle parsing config modal
@callback(
    Output("parsing-config-modal", "is_open"),
    [
        Input("parsing-config-status-btn", "n_clicks"),
        Input("close-parsing-config-modal", "n_clicks"),
    ],
    [State("parsing-config-modal", "is_open")],
)
def toggle_parsing_config_modal(n_status, n_close, is_open):
    """
    Toggle the parsing config modal based on button clicks
    """
    if n_status or n_close:
        return not is_open
    return is_open


# Toggle optimized extraction config modal
@callback(
    Output("optim-extraction-config-modal", "is_open"),
    [
        Input("optim-extraction-config-status-btn", "n_clicks"),
        Input("close-optim-extraction-config-modal", "n_clicks"),
    ],
    [State("optim-extraction-config-modal", "is_open")],
)
def toggle_optim_extraction_config_modal(n_status, n_close, is_open):
    """
    Toggle the optimized extraction config modal based on button clicks
    """
    if n_status or n_close:
        return not is_open
    return is_open


# Toggle dataset config modal
@callback(
    Output("dataset-config-modal", "is_open"),
    [
        Input("dataset-config-status-btn", "n_clicks"),
        Input("close-dataset-config-modal", "n_clicks"),
    ],
    [State("dataset-config-modal", "is_open")],
)
def toggle_dataset_config_modal(n_status, n_close, is_open):
    """
    Toggle the dataset config modal based on button clicks
    """
    if n_status or n_close:
        return not is_open
    return is_open


@callback(
    Output("original-config-modal", "is_open"),
    [
        Input("original-config-status-btn", "n_clicks"),
        Input("close-original-config-modal", "n_clicks"),
    ],
    [State("original-config-modal", "is_open")],
)
def toggle_original_config_modal(n_status, n_close, is_open):
    if n_status or n_close:
        return not is_open
    return is_open

# Toggle optimization config modal
@callback(
    Output("optim-config-modal", "is_open"),
    [
        Input("close-optim-config-modal", "n_clicks"),
    ],
    [State("optim-config-modal", "is_open")],
)
def toggle_optim_config_modal(n_close, is_open):
    """
    Toggle the optimization config modal based on button clicks
    """
    if n_close:
        return False
    return is_open


# Toggle table parsing config modal
@callback(
    Output("table-parsing-config-modal", "is_open"),
    [
        Input("table-parsing-config-status-btn", "n_clicks"),
        Input("close-table-parsing-config-modal", "n_clicks"),
    ],
    [State("table-parsing-config-modal", "is_open")],
)
def toggle_table_parsing_config_modal(n_status, n_close, is_open):
    """
    Toggle the table parsing config modal based on button clicks
    """
    if n_status or n_close:
        return not is_open
    return is_open


# Handle table parsing configuration writeback
@callback(
    [
        Output("table-parsing-config-load-status", "children", allow_duplicate=True),
        Output("table-parsing-config-modal-body", "children", allow_duplicate=True),
        Output("table-parsing-config-modal", "is_open", allow_duplicate=True),
        Output("table-folder-path", "value", allow_duplicate=True),
        Output("table-save-path", "value", allow_duplicate=True),
        Output("table-file-type", "value", allow_duplicate=True),
    ],
    Input({"type": "load-table-parsing-config", "index": ALL}, "n_clicks"),
    [State({"type": "load-table-parsing-config-file-path", "index": ALL}, "data")],
    prevent_initial_call=True,
)
def handle_table_parsing_writeback(n_clicks_list, taskcard_config_path_list):
    """
    Handle table parsing configuration loading and writeback
    """
    # Get the triggered button context
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", "", False, None, None, None

    try:
        form_values = get_form_values_for_module("table_parsing", taskcard_config_path_list)
    except ValueError as e:
        if str(e) == "NO_PATH":
            error_content = html.Div(
                [
                    html.H5("No Configuration File Path", className="text-warning mb-3"),
                    html.P("No configuration file path was provided.", className="mb-2"),
                ]
            )
            return "", error_content, True, None, None, None
        error_content = html.Div(
            [
                html.H5("Failed to Load Configuration", className="text-danger mb-3"),
                html.P(f"Error: {str(e)}", className="mb-2"),
            ]
        )
        return (
            html.Div(f"加载配置失败: {str(e)}", className="status-error"),
            error_content,
            True,
            None,
            None,
            None,
        )

    modal_content = html.Div(
        [
            html.H5(
                "Configuration Loaded Successfully!", className="text-success mb-3"
            ),
            html.H6("Loaded Configuration Values:", className="mb-2"),
            html.P(
                f"Source Folder Path: {form_values.get('table-folder-path', 'Not specified')}",
                className="mb-1",
            ),
            html.P(
                f"Save Path: {form_values.get('table-save-path', 'Not specified')}",
                className="mb-1",
            ),
            html.P(
                f"File Type: {form_values.get('table-file-type', 'Not specified')}",
                className="mb-1",
            ),
        ]
    )

    status_msg = html.Div(
        "Configuration loaded successfully!", className="status-success"
    )
    return (
        status_msg,
        modal_content,
        True,
        form_values.get("table-folder-path"),
        form_values.get("table-save-path"),
        form_values.get("table-file-type"),
    )


def generic_writeback_input(config_file_path: str) -> Dict[str, Any]:
    """
    Generic input writeback function that reads a configuration file and returns the content of the data field
    
    Args:
        config_file_path: Path to the configuration file
        
    Returns:
        Content of the data field in the configuration file
        
    Raises:
        ValueError: When configuration file format is not supported or data field is missing
        FileNotFoundError: When configuration file does not exist
        json.JSONDecodeError: When JSON file format is invalid
        yaml.YAMLError: When YAML file format is invalid
    """
    # Check if file exists
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    
    # Get file extension
    _, file_extension = os.path.splitext(config_file_path)
    file_extension = file_extension.lower()
    
    try:
        # Read configuration file
        with open(config_file_path, 'r', encoding='utf-8') as f:
            if file_extension == '.json':
                config = json.load(f)
            elif file_extension in ['.yml', '.yaml']:
                import yaml
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Only JSON and YAML files are supported.")
        
        # Check if data field is included
        if 'data' not in config:
            raise ValueError("Configuration file is missing required 'data' key.")
        
        # Verify data field is a dictionary
        if not isinstance(config['data'], dict):
            raise ValueError("Configuration 'data' must be a dictionary.")
        
        return config['data']
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format in {config_file_path}: {str(e)}", e.doc, e.pos)
    except ImportError as e:
        raise ImportError(f"YAML support not available. Please install PyYAML: {str(e)}")
    except Exception as e:
        raise Exception(f"Error reading configuration file {config_file_path}: {str(e)}")


def get_actual_config_path_from_ctx(ctx, taskcard_config_path_list):
    """
    Extract the actual configuration file path from the callback context.
    
    Args:
        ctx: The callback context object
        taskcard_config_path_list: List of configuration file paths from TaskCards
        
    Returns:
        The actual configuration file path or None if not found
    """
    # Extract the triggered button ID
    triggered_prop = ctx.triggered[0]['prop_id']
    
    # Initialize taskcard_config_path
    taskcard_config_path = None
    
    # Check if this is a pattern-matching callback
    if '{"index"' in triggered_prop:
        # Extract the index from pattern-matching ID
        import json
        try:
            # Parse the dictionary string safely
            triggered_id_str = triggered_prop.split('.')[0]
            triggered_id_dict = json.loads(triggered_id_str)
            task_index = int(triggered_id_dict['index']) - 1
            # Find the corresponding config path for this task_id
            if task_index < len(taskcard_config_path_list):
                taskcard_config_path = taskcard_config_path_list[task_index]
        except (json.JSONDecodeError, ValueError, IndexError, KeyError) as e:
            # Handle parsing errors or index out of range
            print(f"Error parsing triggered ID: {e}")
            taskcard_config_path = None
    
    # Use config path from TaskCard if provided
    actual_config_path = taskcard_config_path
    return actual_config_path
import io


def _read_log_tail_incremental(file_path: str, from_pos: int | None, max_initial_bytes: int = 256 * 1024) -> tuple[str, int]:
    try:
        size = os.path.getsize(file_path)
        start = 0 if (from_pos is not None and from_pos >= 0) else max(0, size - max_initial_bytes)
        if from_pos is not None and from_pos >= 0:
            start = min(from_pos, size)
        with open(file_path, 'rb') as f:
            f.seek(start)
            chunk = f.read()
        text = chunk.decode('utf-8', errors='ignore')
        return text, size
    except Exception as e:
        return f"[failed to read log tail] {str(e)}", from_pos or 0


@callback(
    Output("realtime-log-modal", "is_open"),
    Output("realtime-log-file-label", "children"),
    Output("realtime-log-path", "data"),
    Output("realtime-log-buffer", "data"),
    Output("realtime-log-position", "data"),
    Output("realtime-log-interval", "disabled"),
    Output("realtime-log-content", "children"),
    Input({"type": "open-log-btn", "index": ALL}, "n_clicks"),
    Input({"type": "open-log-btn", "index": ALL}, "n_clicks_timestamp"),
    Input("close-realtime-log-modal", "n_clicks"),
    Input("realtime-log-interval", "n_intervals"),
    State({"type": "open-log-btn-file-path", "index": ALL}, "data"),
    State("realtime-log-modal", "is_open"),
    State("realtime-log-path", "data"),
    State("realtime-log-position", "data"),
    State("realtime-log-buffer", "data"),
    State("realtime-log-interval", "disabled"),
    prevent_initial_call=True,
)
def open_realtime_log_modal(all_clicks, all_click_timestamps, close_clicks, n_intervals, all_paths, is_open, cur_path, cur_pos, cur_buf, interval_disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    try:
        trigger = ctx.triggered[0]["prop_id"]
        prop = trigger.split(".")[0]
        # Close modal
        if prop == "close-realtime-log-modal":
            return False, None, None, "", 0, True, ""
        # Interval update
        if prop == "realtime-log-interval":
            if interval_disabled:
                raise dash.exceptions.PreventUpdate
            if not is_open or not cur_path:
                raise dash.exceptions.PreventUpdate
            text, new_pos = _read_log_tail_incremental(cur_path, cur_pos)
            new_buf = (cur_buf or "") + text
            max_buf_bytes = 1024 * 1024
            if len(new_buf.encode('utf-8', errors='ignore')) > max_buf_bytes:
                new_buf = new_buf[-max_buf_bytes:]
            label = f"File: {os.path.basename(cur_path)}"
            return True, label, cur_path, new_buf, new_pos, False, new_buf
        import json as _json
        # Open log modal from button
        try:
            btn = _json.loads(prop)
        except Exception:
            btn = {}
        if btn.get("type") != "open-log-btn":
            raise dash.exceptions.PreventUpdate
        idx = btn.get("index")
        click_val = ctx.triggered[0].get('value')
        # find timestamp for this index
        ts_for_idx = None
        for k in ctx.inputs.keys():
            if k.endswith('.n_clicks_timestamp'):
                try:
                    obj = _json.loads(k.split('.',1)[0])
                    if obj.get('type') == 'open-log-btn' and obj.get('index') == idx:
                        ts_for_idx = ctx.inputs.get(k)
                        break
                except Exception:
                    pass
        if not click_val or (isinstance(click_val, (int,float)) and click_val <= 0):
            raise dash.exceptions.PreventUpdate
        if not ts_for_idx:
            raise dash.exceptions.PreventUpdate
        keys = list(ctx.states.keys())
        target_path = None
        for k in keys:
            if not k.endswith('.data'):
                continue
            try:
                obj = _json.loads(k[:-5])
                if obj.get('type') == 'open-log-btn-file-path' and obj.get('index') == idx:
                    target_path = ctx.states[k]
                    break
            except Exception:
                pass
        if not target_path:
            return False, None, None, "", 0, True, ""
        tail_text, new_pos = _read_log_tail_incremental(target_path, None)
        # Trim buffer to avoid huge memory
        max_buf_bytes = 1024 * 1024
        buf = tail_text
        if len(buf.encode('utf-8', errors='ignore')) > max_buf_bytes:
            buf = buf[-max_buf_bytes:]
        label = f"File: {os.path.basename(target_path)}"
        return True, label, target_path, buf, new_pos, False, buf
    except dash.exceptions.PreventUpdate:
        raise dash.exceptions.PreventUpdate
    except Exception as e:
        return False, None, None, "", 0, True, ""
