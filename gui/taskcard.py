import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, MATCH
import dash
import json
import os

class TaskCard:
    def __init__(self, task_id, name, create_time, status, content=None, config_path=None, writeback_handler=None):
        """
        Initialize task card
        
        Parameters:
        - task_id: Unique identifier for the task
        - name: Task name
        - create_time: Creation time
        - status: Task status
        - content: Content displayed when the card is expanded
        - config_path: Path to configuration file (optional)
        - writeback_handler: Writeback handler function (optional)
        """
        self.task_id = task_id
        self.name = name
        self.create_time = create_time
        self.status = status
        self.content = content
        self.config_path = config_path
        self.writeback_handler = writeback_handler
        
        self.status_color = self._get_status_color(status)
        
    def _get_status_color(self, status):
        """Return corresponding color based on status"""
        status_colors = {
            # New status values
            "running": "primary",
            "succeed": "success",
            "failed": "danger",
            "cancelled": "secondary",
            
            # Legacy status values (for backward compatibility)
            "Completed": "success",
            "In Progress": "primary", 
            "Pending": "warning",
            "Cancelled": "danger",
            "Paused": "secondary"
        }
        return status_colors.get(status, "secondary")
    
    def render(self):
        """Render collapsible task card"""
        card_id = f"task-card-{self.task_id}"
        collapse_id = {"type": "collapse", "index": self.task_id}
        button_id = {"type": "toggle-btn", "index": self.task_id}
        
        has_content = self.content is not None
        
        if has_content:
            card_header = dbc.CardHeader([
                dbc.Row([
                    dbc.Col([
                        html.Small(self.name, className="me-2"),
                        html.Small(f"Created: {self.create_time}", className="text-muted")
                    ], width=8, className="d-flex align-items-center"),
                    
                    dbc.Col([
                        html.Div([
                            dbc.Badge(self.status, color=self.status_color, className="me-2", id={"type": "status-badge", "index": self.task_id}),
                            (dcc.Loading(
                                dbc.Button(
                                    "Cancel task",
                                    id={"type": "cancel-task-btn", "index": self.task_id},
                                    color="danger",
                                    size="sm",
                                    disabled=False,
                                    className="me-2",
                                    style={"fontSize": "0.75rem", "padding": "0.2rem 0.5rem", "border": "1px solid #dc3545", "backgroundColor": "#dc3545", "color": "white", "backgroundImage": "none"}
                                ),
                                type="circle"
                            ) if self.status == "running" else None),
                            html.Button(
                                html.I(className="fas fa-chevron-down", style={"color": "black", "fontSize": "12px"}),
                                id=button_id,
                                className="btn btn-sm btn-outline-secondary",
                                style={"border": "1px solid black", "backgroundColor": "transparent", "backgroundImage": "none", "transition": "transform 0.3s ease", "padding": "2px 6px", "margin": "0", "width": "24px", "height": "24px", "display": "flex", "alignItems": "center", "justifyContent": "center"}
                            )
                        ], className="d-flex justify-content-end align-items-center w-100")
                    ], width=4, className="d-flex justify-content-end align-items-center")
                ])
            ], className="py-2")
            
            content_with_button = self.content
            if self.config_path and self.writeback_handler:
                button_id = None
                if self.writeback_handler == 'model-config':
                    button_id = 'load-model-config'
                elif self.writeback_handler == 'run-optimization':
                    button_id = 'load-optim-config'
                elif self.writeback_handler == 'optimized-extraction':
                    button_id = 'load-optim-extraction-config'
                elif self.writeback_handler == 'table-extraction':
                    button_id = 'load-table-config'
                elif self.writeback_handler == 'build-dataset':
                    button_id = 'load-dataset-config'
                elif self.writeback_handler == 'original-extraction':
                    button_id = 'load-original-config'
                elif self.writeback_handler == 'doc-parsing':
                    button_id = 'load-parsing-config'
                elif self.writeback_handler == 'table-parsing':
                    button_id = 'load-table-parsing-config'
                elif self.writeback_handler == 'prompt-optimization':
                    button_id = 'load-optim-config'
                
                if button_id:
                    task_id_suffix = self.task_id.split("-")[-1]
                    
                    use_config_button = dbc.Button(
                        "Use this config",
                        id={'type': button_id, 'index': task_id_suffix},
                        color="light",
                        size="sm",
                        style={"fontSize": "0.75rem", "padding": "0.2rem 0.5rem", "border": "1px solid #667eea", "backgroundColor": "#667eea", "color": "white", "backgroundImage": "none"}
                    )
                    
                    config_path_store = dcc.Store(
                        id={'type': f'{button_id}-file-path', 'index': task_id_suffix},
                        data=self.config_path
                    )
                    
                    if isinstance(self.content, list):
                        content_with_button = [use_config_button, config_path_store] + self.content
                    else:
                        content_with_button = [use_config_button, config_path_store, self.content]
                else:
                    content_with_button = self.content
            
            card_body = dbc.Collapse(
                dbc.CardBody((
                    [
                        html.Div(id={"type": "cancel-error", "index": self.task_id}),
                        dcc.Store(id={"type": "cancel-log-path", "index": self.task_id}, data=self.config_path)
                    ] + (content_with_button if isinstance(content_with_button, list) else [content_with_button])
                )),
                id=collapse_id,
                is_open=False
            )
            
            card = dbc.Card(
                [card_header, card_body],
                className="mb-3 task-card",
                style={"width": "100%"}
            )
        else:
            card_header = dbc.CardHeader([
                dbc.Row([
                    dbc.Col([
                        html.Small(self.name, className="me-2"),
                        html.Small(f"Created: {self.create_time}", className="text-muted")
                    ], width=8, className="d-flex align-items-center"),
                    
                    dbc.Col([
                        html.Div([
                            dbc.Badge(self.status, color=self.status_color, id={"type": "status-badge", "index": self.task_id}),
                            (dcc.Loading(
                                dbc.Button(
                                    "Cancel task",
                                    id={"type": "cancel-task-btn", "index": self.task_id},
                                    color="danger",
                                    size="sm",
                                    disabled=False,
                                    className="me-2",
                                    style={"fontSize": "0.75rem", "padding": "0.2rem 0.5rem", "border": "1px solid #dc3545", "backgroundColor": "#dc3545", "color": "white", "backgroundImage": "none"}
                                ),
                                type="circle"
                            ) if self.status == "running" else None),
                            html.Button(
                                html.I(className="fas fa-chevron-down", style={"color": "black", "fontSize": "12px"}),
                                id={"type": "toggle-btn", "index": self.task_id},
                                className="btn btn-sm btn-outline-secondary",
                                style={"border": "1px solid black", "backgroundColor": "transparent", "backgroundImage": "none", "transition": "transform 0.3s ease", "padding": "2px 6px", "margin": "0", "width": "24px", "height": "24px", "display": "flex", "alignItems": "center", "justifyContent": "center"}
                            )
                        ], className="d-flex justify-content-end align-items-center w-100")
                    ], width=4, className="d-flex justify-content-end align-items-center")
                ])
            ], className="py-2")
            
            card = dbc.Card(
                [
                    card_header,
                    dbc.CardBody([
                        html.Div(id={"type": "cancel-error", "index": self.task_id}),
                        dcc.Store(id={"type": "cancel-log-path", "index": self.task_id}, data=self.config_path)
                    ])
                ],
                className="mb-3 task-card",
                style={"width": "100%"}
            )
        
        return card
    


def create_task_card_list(task_data_list):
    """
    Create task card list
    
    Parameters:
    - task_data_list: List of dictionaries containing task data, each dictionary should contain:
        - id: Task ID
        - name: Task name
        - create_time: Creation time
        - status: Task status
        - content: Card content (optional)
        - config_path: Path to configuration file (optional)
        - writeback_handler: Writeback handler function (optional)
    
    Returns:
    - html.Div component containing all task cards
    """
    cards = []
    for task_data in task_data_list:
        # Only pass content when it exists and is not empty
        content = task_data.get("content")
        if content is None:
            content = None
            
        card = TaskCard(
            task_id=task_data.get("id"),
            name=task_data.get("name", "Unnamed Task"),
            create_time=task_data.get("create_time", "Unknown Time"),
            status=task_data.get("status", "Unknown Status"),
            content=content,
            config_path=task_data.get("config_path"),
            writeback_handler=task_data.get("writeback_handler")
        )
        cards.append(card.render())
    
    return html.Div(cards, className="task-card-container")


def load_data_from_json_and_format(json_file_path):
    """
    Load data from the "data" field in a JSON file and format it using markdown
    
    Parameters:
    - json_file_path: Path to the JSON file
    
    Returns:
    - html.Div component containing formatted data content
    """
    try:
        # Check if file exists
        if not os.path.exists(json_file_path):
            return html.Div([
                html.H4("File does not exist", className="text-danger"),
                html.P(f"File path: {json_file_path}")
            ])
        
        # Read JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract data field
        data_content = data.get('data', {})
        
        if not data_content:
            return html.Div([
                html.H4("Data is empty", className="text-warning"),
                html.P("No valid data field found in the JSON file")
            ])
        
        # Convert data to markdown format string
        markdown_content = format_data_to_markdown(data_content)
        
        # Use dcc.Markdown component to render markdown content
        log_path = os.path.splitext(json_file_path)[0] + '.log'
        log_name = os.path.basename(log_path)
        run_dir = os.path.basename(os.path.dirname(json_file_path))
        log_exists = os.path.exists(log_path)
        return html.Div([
            html.H5("Task Config Details", className="mb-3 mt-3", style={"fontSize": "0.9rem"}),
            dcc.Markdown(markdown_content, className="markdown-content"),
            html.Hr(),
            html.H5("Task Log Path", className="mb-3 mt-3", style={"fontSize": "0.9rem"}),
            html.Div([
                html.Span(log_name if log_exists else f"{log_name} (Not found)", className="me-2", style={"fontSize": "12px"}),
                (dbc.Button(
                    "View the task log",
                    id={"type": "open-log-btn", "index": run_dir},
                    color="light",
                    size="sm",
                    disabled=(not log_exists),
                    style={"fontSize": "0.75rem", "padding": "0.2rem 0.5rem", "border": "1px solid #667eea", "backgroundColor": "#667eea", "color": "white", "backgroundImage": "none"}
                ) if True else None),
                dcc.Store(id={"type": "open-log-btn-file-path", "index": run_dir}, data=log_path)
            ], className="d-flex align-items-center")
        ])
        
    except json.JSONDecodeError as e:
        return html.Div([
            html.H4("JSON parsing error", className="text-danger"),
            html.P(f"Error message: {str(e)}"),
            html.P(f"File path: {json_file_path}")
        ])
    except Exception as e:
        return html.Div([
            html.H4("Error occurred while loading data", className="text-danger"),
            html.P(f"Error message: {str(e)}"),
            html.P(f"File path: {json_file_path}")
        ])


def format_data_to_markdown(data_dict):
    """
    Convert dictionary data to markdown format string
    
    Parameters:
    - data_dict: Dictionary containing data
    
    Returns:
    - Markdown format string
    """
    markdown_lines = []
    
    for key, value in data_dict.items():
        key_display = get_key_display_name(key)
        
        if isinstance(value, dict):
            markdown_lines.append(f"### {key_display}")
            nested_markdown = format_data_to_markdown(value)
            for line in nested_markdown.split('\n'):
                if line.strip():
                    markdown_lines.append(f"    {line}")
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                markdown_lines.append(f"###### {key_display}")
                for i, item in enumerate(value, 1):
                    nested_markdown = format_data_to_markdown(item)
                    nested_lines = [line.strip() for line in nested_markdown.split('\n') if line.strip()]
                    if nested_lines:
                        markdown_lines.append(f"- {nested_lines[0]}")
                        for line in nested_lines[1:]:
                            markdown_lines.append(f"  {line}")
                    else:
                        markdown_lines.append("- (Empty)")
            else:
                markdown_lines.append(f"**{key_display}**:")
                for item in value:
                    markdown_lines.append(f"- {item}")
        else:
            if isinstance(value, bool):
                value = "Yes" if value else "No"
            if value == "":
                value = "None"
            markdown_lines.append(f"**{key_display}**: {value}")
    
    return '\n\n'.join(markdown_lines)


def get_key_display_name(key):
    """
    Convert key name to a more friendly display name
    
    Parameters:
    - key: Original key name
    
    Returns:
    - Display name
    """
    display_names = {
        'folder_path': 'Folder Path',
        'save_path': 'Save Path',
        'file_type': 'File Type',
        'convert_mode': 'Convert Mode',
        'message': 'Message',
        'result': 'Result',
        'details': 'Details',
        'dataset_file': 'Dataset File',
        'error': 'Error Message',
        'dataset': 'Dataset',
        'save_dir': 'Save Directory',
        'inputFields': 'Input Fields',
        'outputFields': 'Output Fields',
        'initial_prompt': 'Initial Prompt',
        'judging': 'Judging',
        'task': 'Task',
        'threads': 'Threads',
        'multiple': 'Multiple',
        'name': 'Name',
        'field_type': 'Field Type',
        'description': 'Description'
    }
    return display_names.get(key, key.replace('_', ' ').title())


def generate_task_data_from_runs(base_path, n, task_type="doc-parsing"):
    """
    Generate task card data list by selecting the latest n JSON files from subdirectories
    
    Parameters:
    - base_path: Base path, e.g., "gui/runs/doc_parsing"
    - n: Number of latest tasks to select
    - task_type: Task type for generating task ID prefix, default is "doc-parsing"
    
    Returns:
    - List of task data, each containing id, name, create_time, status, content
    """
    import glob
    import re
    from datetime import datetime
    
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.getcwd(), base_path)
    
    if not os.path.exists(base_path):
        print(f"Warning: Path {base_path} does not exist")
        return []
    
    subdirs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
    
    def extract_timestamp_from_path(path):
        """Extract timestamp from path"""
        match = re.search(r'(?:run_)?(\d{8})_(\d{6})', os.path.basename(path))
        if match:
            date_str = match.group(1) + match.group(2)
            try:
                return datetime.strptime(date_str, '%Y%m%d%H%M%S')
            except ValueError:
                return datetime.min
        return datetime.min
    
    subdirs.sort(key=extract_timestamp_from_path, reverse=True)
    
    latest_subdirs = subdirs[:n]
    
    actual_count = len(latest_subdirs)
    if actual_count < n:
        pass
        
    
    task_data_list = []
    
    for i, subdir in enumerate(latest_subdirs):
        json_files = glob.glob(os.path.join(subdir, "*.json"))
        
        if not json_files:
            continue
        
        json_file = json_files[0]
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            name = json_data.get('name', 'Unknown Task')
            create_time = json_data.get('created_time', 'Unknown Time')
            status = json_data.get('status', 'unknown')
            
            if create_time != 'Unknown Time':
                try:
                    dt = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                    create_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, AttributeError):
                    pass
            
            task_data = {
                "id": f"{task_type}-{i+1}",
                "name": name,
                "create_time": create_time,
                "status": status,
                "content": load_data_from_json_and_format(json_file),
                "config_path": json_file,
                "writeback_handler": task_type
            }
            
            task_data_list.append(task_data)
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading JSON file {json_file}: {e}")
            continue
    
    return task_data_list

