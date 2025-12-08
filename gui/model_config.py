from dash import html, dcc
import dash_bootstrap_components as dbc
import os
import json

# Helper function to read configuration files
def read_config_file(usage_type):
    """Read model configuration file for specified usage type"""
    # Get the project root directory (parent of gui directory)
    project_root = os.path.join(os.path.dirname(__file__), '..')
    config_dir = os.path.join(project_root, "settings")
    config_path = os.path.join(config_dir, f"model_settings_{usage_type}.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading config file: {e}")
    return None

# Function to create configuration cards
def create_config_card(config_data, usage_type):
    """Create display card for specified configuration data"""
    if not config_data:
        return html.Div([
            html.H4(f"{usage_type.capitalize()} Configuration", className="config-card-title"),
            html.Div("No configuration available for this usage type.", className="no-config-message")
        ], className="config-card")
    
    # Format display configuration data and group into two columns
    config_pairs = []
    current_pair = []
    
    for key, value in config_data.items():
        # Skip api_key field - don't process or display it
        if key == 'api_key':
            continue
            
        # Convert key to more readable format
        display_key = key.replace('_', ' ').title()
        
        if value is None:
            display_value = 'Not set'
        else:
            display_value = value
        
        # Create configuration item
        config_item = html.Div([
            html.Div(display_key, className="config-item-key"),
            html.Div(str(display_value), className="config-item-value")
        ], className="config-item")
        
        # Add configuration item to current pair
        current_pair.append(config_item)
        
        # If current pair has two configuration items, or is the last configuration item, add to configuration pairs list
        if len(current_pair) == 2 or len(config_pairs) * 2 + len(current_pair) == len([k for k in config_data.keys() if k != 'api_key']):
            config_pairs.append(
                html.Div(current_pair, className="config-item-pair")
            )
            current_pair = []
    
    return html.Div([
        html.H4(f"{usage_type.capitalize()} Configuration", className="config-card-title"),
        html.Div(config_pairs, className="config-items")
    ], className="config-card")

# Model configuration page layout
def model_config_layout():
    # Get all possible model usage types
    usage_types = ["main", "prompt_generation", "judge", "coder"]
    
    # Create tabs and corresponding content for each usage type
    tabs_children = []
    tab_content = []
    
    for usage_type in usage_types:
        config_data = read_config_file(usage_type)
        
        # Create tab
        tabs_children.append({
            "label": usage_type.replace('_', ' ').title(),
            "value": usage_type
        })
        
        # Create tab content
        tab_content.append(
            dcc.Tab(
                label=usage_type.replace('_', ' ').title(),
                value=usage_type,
                children=[
                    create_config_card(config_data, usage_type)
                ]
            )
        )
    
    layout = html.Div([
        # New Model Settings module using tab pagination
        html.Div([
            html.H3("Model Settings", className="module-title"),
            dcc.Tabs(
                id="model-settings-tabs",
                value="main",
                children=tab_content,
                className="model-settings-tabs"
            )
        ], className="module-container"),

        html.Div([
            html.H2("Model Configuration", className="module-title"),
            
            # Model identification section
            html.H3("Model Identification", className="section-title"),
            html.Div([
                html.Label("Model Name", className="form-label"),
                dcc.Input(
                    id="model-name",
                    type="text",
                    placeholder="gpt-4.1",
                    value="",
                    className="form-input form-input-limited"
                )
            ], className="form-group"),
            html.Div([
                html.Label("Model Type", className="form-label"),
                dcc.Dropdown(
                    id="model-type",
                    options=[
                        {"label": "OpenAI", "value": "openai"},
                        {"label": "vLLM", "value": "vllm"},
                        {"label": "Ollama", "value": "ollama"},
                        {"label": "Qwen", "value": "qwen"},
                        {"label": "DeepSeek", "value": "deepseek"},
                        {"label": "Gemini", "value": "gemini"},
                        {"label": "Anthropic", "value": "anthropic"},
                        {"label": "OpenRouter", "value": "openrouter"},
                        {"label": "SGLang", "value": "sglang"},
                        {"label": "Custom (OpenAI Compatible)", "value": "custom"}
                    ],
                    value="openai",
                    clearable=False,
                    className="form-select form-select-limited"
                )
            ], className="form-group"),
            
            # API configuration section
            html.H3("API Configuration", className="section-title"),
            html.Div([
                html.Div([
                    html.Label("API Base URL", className="form-label"),
                    dcc.Input(
                        id="api-base",
                        type="text",
                        placeholder="API endpoint URL e.g. https://api.openai.com/v1",
                        value="",
                        className="form-input form-input-limited"
                    )
                ], className="form-group"),
                html.Div([
                    html.Label("API Key", className="form-label"),
                    dcc.Input(
                        id="api-key",
                        type="password",
                        placeholder="YOUR-API-KEY",
                        className="form-input form-input-limited"
                    )
                ], className="form-group")
            ], className="grid"),
            
            # API Key说明
            html.Div([
                html.P("API Key Information:", className="help-text-title"),
                html.Ul([
                    html.Li("API Keys are encrypted and stored in configuration files by default"),
                    html.Li("You can also leave the API Key field empty and use the environment variable like OPENAI_API_KEY instead"),
                ], className="help-text-list")
            ], className="help-text-container"),
            
            # Model usage configuration section
            html.H3("Model Usage", className="section-title"),
            html.Div([
                html.Div([
                    html.Label("Model Usage Type", className="form-label"),
                    dcc.Dropdown(
                        id="model-usage",
                        options=[
                            {"label": "Main", "value": "main"},
                            {"label": "Prompt Generation", "value": "prompt_generation"},
                            {"label": "Judge", "value": "judge"},
                            {"label": "Coder", "value": "coder"}
                        ],
                        value="main",
                        clearable=False,
                        className="form-select form-select-limited"
                    )
                ], className="form-group"),
                html.Div([
                    html.P("Option Descriptions:", className="help-text-title"),
                    html.Ul([
                        html.Li("main: Primary model for general tasks, if the model for prompt generation/judge/coding is not specified, this model will be used instead"),
                        html.Li("prompt_generation: Model for generating prompts"),
                        html.Li("judge: Model for evaluation and scoring"),
                        html.Li("coder: Model for code generation and programming tasks")
                    ], className="help-text-list")
                ], className="help-text-container")
            ]),
            
            # Sampling parameters section
            html.H3("Sampling Parameters", className="section-title"),
            # First two parameters maintain grid layout
            html.Div([
                html.Div([
                    html.Label("Temperature", className="form-label"),
                    dcc.Input(
                        id="temperature",
                        type="number",
                        min=0,
                        max=2,
                        step=0.1,
                        value=0.0,
                        className="form-input form-input-limited"
                    )
                ], className="form-group"),
                html.Div([
                    html.Label("Max Tokens", className="form-label"),
                    dcc.Input(
                        id="max-tokens",
                        type="number",
                        value=2500,
                        className="form-input form-input-limited"
                    )
                ], className="form-group")
            ], className="grid"),
            # Last three parameters each occupy one line
            html.Div([
                html.Label("Top-p", className="form-label"),
                dcc.Input(
                    id="top-p",
                    type="number",
                    min=0,
                    max=1,
                    step=0.01,
                    placeholder="0.0-1.0 or leave empty",
                    className="form-input form-input-limited"
                ),
                html.Div("Optional, for OpenAI compatible APIs", className="form-help")
            ], className="form-group"),
            html.Div([
                html.Label("Top-k", className="form-label"),
                dcc.Input(
                    id="top-k",
                    type="number",
                    min=1,
                    placeholder="Positive integer or leave empty",
                    className="form-input form-input-limited"
                ),
                html.Div("Optional, for vllm, sglang, openrouter, etc.", className="form-help")
            ], className="form-group"),
            html.Div([
                html.Label("Min-p", className="form-label"),
                dcc.Input(
                    id="min-p",
                    type="number",
                    min=0,
                    max=1,
                    step=0.01,
                    placeholder="0.0-1.0 or leave empty",
                    className="form-input form-input-limited"
                ),
                html.Div("Optional, for vllm, sglang, openrouter, etc.", className="form-help")
            ], className="form-group"),
            
            # Action buttons
            html.Div([
                html.Button("Save Configuration", id="save-config", className="btn btn-primary"),
                html.Button("Test Connection", id="test-connection", className="btn btn-secondary")
            ], className="button-group"),
            
            # Status message
            html.Div(id="config-status", className="status-message"),
            dbc.Modal(
                [
                    dbc.ModalHeader("Model Configuration Status"),
                    dbc.ModalBody(id="model-config-modal-body"),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-model-config-modal", className="ml-auto")
                    ),
                ],
                id="model-config-modal",
                is_open=False,
                size="lg"
            )
        ], className="module-container")
    ])
    
    return layout