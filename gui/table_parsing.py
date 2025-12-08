from dash import html, dcc
import dash_bootstrap_components as dbc
from taskcard import TaskCard, create_task_card_list, load_data_from_json_and_format, generate_task_data_from_runs

# Build Table Files Parsing page layout
def table_parsing_layout():
    return html.Div([
        html.Div([
            # Tabbed Pages
            dcc.Tabs(
                id="table-parsing-tabs",
                value="table-files-tab",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        label="Table Files Parsing",
                        value="table-files-tab",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=[
                            # Table Files Parsing tab content - existing page content
                            html.Div([
                                html.Div([
                                    # === TASK CONFIGURATION ===
                                    html.Div([
                                        html.H3("Task Name", className="section-title"),
                                        html.Div([
                                            html.Label("Task Name", className="form-label"),
                                            dcc.Input(
                                                id="task-name-table-parsing",
                                                placeholder="Enter a name for this task",
                                                value="",
                                                className="form-input"
                                            ),
                                            html.Small("A descriptive name for this table parsing task", className="form-hint")
                                        ], className="form-group")
                                    ], className="section-container"),

                                    # === FILE SOURCE CONFIGURATION ===
                                    html.Div([
                                        html.H3("File Source Configuration", className="section-title"),
                                        html.Div([
                                            html.Label("Source Folder Path", className="form-label"),
                                            dcc.Input(
                                                id="table-folder-path",
                                                placeholder="/path/to/your/source/files",
                                                value="",
                                                className="form-input"
                                            ),
                                            html.Small("Path to the folder containing source files with tables to parse", className="form-hint")
                                        ], className="form-group")
                                    ], className="section-container"),

                                    # === OUTPUT CONFIGURATION ===
                                    html.Div([
                                        html.H3("Output Configuration", className="section-title"),
                                        # Configuration loading status display (changed to pop-up trigger)
                                        html.Div([
                                            dbc.Button(
                                                "Configuration Status",
                                                id="table-parsing-config-status-btn",
                                                color="info",
                                                size="sm",
                                                style={"marginBottom": "20px", "display": "none"}
                                            ),
                                            html.Div(id="table-parsing-config-load-status", style={"display": "none"}),
                                        ]),
                                        html.Div([
                                            html.Label("Save Path", className="form-label"),
                                            dcc.Input(
                                                id="table-save-path",
                                                placeholder="/path/to/save/results",
                                                value="",
                                                className="form-input"
                                            ),
                                            html.Small("Path to save parsed TSV results", className="form-hint")
                                        ], className="form-group")
                                    ], className="section-container"),

                                    # === NON TABULAR FILE TYPE CONFIGURATION ===
                                    html.Div([
                                        html.H3("Non Tabular File Type Configuration", className="section-title"),
                                        html.Div([
                                            html.Label("File Type", className="form-label"),
                                            dcc.Dropdown(
                                                id="table-file-type",
                                                options=[
                                                    {"label": "PDF", "value": "PDF"},
                                                    {"label": "ScienceDirect", "value": "scienceDirect"},
                                                    {"label": "PMC", "value": "PMC"},
                                                    {"label": "Arxiv", "value": "Arxiv"}
                                                ],
                                                value="PDF",
                                                clearable=False,
                                                className="form-select"
                                            ),
                                            html.Div([
                                                html.P(html.Strong("This configuration is used to parse tables from non-tabular files (such as PDF files of the main text of literature). Please note that due to the limitations of parsing tools, the parsing results may contain errors, so this method is not recommended for non-tabular file parsing."), className="help-text-description"),
                                                html.P("File Type Descriptions:", className="help-text-title"),
                                                html.Ul([
                                                    html.Li("PDF: Standard PDF files containing tables"),
                                                    html.Li("ScienceDirect: ScienceDirect formatted XML files"),
                                                    html.Li("PMC: PubMed Central formatted XML files"),
                                                    html.Li("Arxiv: ArXiv preprint formatted XML files")
                                                ], className="help-text-list")
                                            ], className="help-text-container")
                                        ], className="form-group")
                                    ], className="section-container"),



                                    # === ACTION BUTTONS ===
                                    html.Div([
                                        html.Button(
                                            "Start Table Parsing",
                                            id="start-table-parsing",
                                            className="btn btn-primary",
                                            style={"marginTop": "20px", "height": "40px", "padding": "0 20px", "fontSize": "14px"}
                                        )
                                    ], className="section-container"),

                                    # === RESULTS DISPLAY ===
                                    html.Div([
                                        html.Div(id="table-files-parsing-results", className="results-container")
                                    ])
                                ], className="module-container")
                            ]),
                            
                            # === RECENT TASKS (Independent Section) ===
                            html.Div([
                                html.H3("Recent Tasks", className="section-title"),
                                html.Div([
                                    html.P("The following are the most recent table parsing tasks, showing task name, creation time, and status information.", 
                                          className="help-text-description", style={"marginBottom": "15px"})
                                ]),
                                # Dynamically load the latest 10 task examples
                                create_task_card_list(generate_task_data_from_runs("runs/table_parsing", 10, "table-parsing"))
                            ], className="module-container", style={"marginTop": "30px"}),
                            
                            # Table parsing configuration loading status modal
                            dbc.Modal(
                                [
                                    dbc.ModalHeader("Table Parsing Configuration Load Status"),
                                    dbc.ModalBody(id="table-parsing-config-modal-body"),
                                    dbc.ModalFooter(
                                        dbc.Button("Close", id="close-table-parsing-config-modal", className="ml-auto")
                                    ),
                                ],
                                id="table-parsing-config-modal",
                                is_open=False,
                                size="lg"
                            )
                        ]
                    ),
                    
                    dcc.Tab(
                        label="Extraction From Tables",
                        value="extraction-tables-tab",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=[
                            # Extraction From Tables tab content
                            html.Div([
                                html.Div([
                                    # === TASK CONFIGURATION ===
                                    html.Div([
                                        html.H3("Task Name", className="section-title"),
                                        html.Div([
                                            html.Label("Task Name", className="form-label"),
                                            dcc.Input(
                                                id="task-name-table-extraction",
                                                placeholder="Enter a name for this task",
                                                value="",
                                                className="form-input"
                                            ),
                                            html.Small("A descriptive name for this table extraction task", className="form-hint")
                                        ], className="form-group")
                                    ], className="section-container"),

                                    # === PARSED FILE SOURCE CONFIGURATION ===
                                    html.Div([
                                        html.H3("Parsed File Source Configuration", className="section-title"),
                                        html.Div([
                                            html.Label("Parsed File Path", className="form-label"),
                                            dcc.Input(
                                                id="extract-parsed-file-path",
                                                placeholder="/path/to/parsed/files",
                                                value="",
                                                className="form-input"
                                            ),
                                            html.Small("Save Path of the Table Files Parsing step", className="form-hint")
                                        ], className="form-group")
                                    ], className="section-container"),

                                    # === OUTPUT CONFIGURATION ===
                                    html.Div([
                                        html.H3("Output Configuration", className="section-title"),
                                        # Configuration loading status display (changed to pop-up trigger)
                                        html.Div([
                                            html.Button(
                                                "Display Current Config Status",
                                                id="table-extraction-config-status-btn",
                                                className="btn btn-outline-info",
                                                style={"marginBottom": "10px", "display": "none"}
                                            ),
                                            html.Div(id="table-extraction-config-load-status", style={"display": "none"}),
                                        ]),
                                        html.Div([
                                            html.Label("Save Folder Path", className="form-label"),
                                            dcc.Input(
                                                id="extract-save-folder-path",
                                                placeholder="/path/to/save/results",
                                                value="",
                                                className="form-input"
                                            ),
                                            html.Small([
                                                "Path to save extracted table results. ",
                                                html.Strong(
                                                    "Must be a new path that does not contain existing files.",
                                                    style={"color": "#FF0000"}
                                                )
                                            ], className="form-hint")
                                        ], className="form-group")
                                    ], className="section-container"),

                                    # === OUTPUT FIELDS CONFIGURATION ===
                                    html.Div([
                                        html.H3("Output Fields Configuration", className="section-title"),
                                        html.Small([
                                            "Output fields are the properties contained in the extracted entities. These fields will be used to define what information you want to extract from the input data."
                                        ], className="form-hint", style={"lineHeight": "1.5", "border": "1px solid #e1e5e9", "borderLeft": "4px solid #3b82f6", "borderRadius": "8px", "padding": "16px", "backgroundColor": "#f8f9fa", "marginBottom": "15px"}),
                                        html.Button(
                                            "Add OutputField",
                                            id="add-extract-output-field",
                                            className="btn",
                                            style={"marginBottom": "15px", "height": "32px", "padding": "0 12px", "fontSize": "12px"}
                                        ),
                                        html.Div(id="extract-output-fields-container", children=[
                                            html.Div([
                                                html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}, children=[
                                                    html.H4("Output Field", style={"margin": "0", "fontSize": "16px"}),
                                                    html.Button(
                                                        "×",
                                                        id={'type': 'delete-extract-output-field', 'index': 'extract-output-field-initial'},
                                                        className="delete-button",
                                                        style={
                                                            "background": "none",
                                                            "border": "none",
                                                            "color": "#6b7280",
                                                            "fontSize": "20px",
                                                            "cursor": "pointer",
                                                            "padding": "0 5px",
                                                            "lineHeight": "1",
                                                            "borderRadius": "4px"
                                                        }
                                                    )
                                                ]),
                                                html.Div(style={"display": "flex", "gap": "20px"}, children=[
                                                    html.Div(style={"flex": 1}, children=[
                                                        html.Label("Output Field Name", className="form-label"),
                                                        dcc.Input(
                                                            id={'type': 'extract-output-name', 'index': 'extract-output-field-initial'},
                                                            placeholder="output_field",
                                                            className="form-input",
                                                            style={"height": "38px"}
                                                        )
                                                    ]),
                                                    html.Div(style={"flex": 1}, children=[
                                                        html.Label("Output Field Type", className="form-label"),
                                                        dcc.Dropdown(
                                                            id={'type': 'extract-output-type', 'index': 'extract-output-field-initial'},
                                                            options=[
                                                                {"label": "String", "value": "str"},
                                                                {"label": "Integer", "value": "int"},
                                                                {"label": "Float", "value": "float"},
                                                                {"label": "Boolean", "value": "bool"},
                                                                {"label": "List", "value": "list"},
                                                                {"label": "Literal", "value": "literal"},
                                                                {"label": "List Literal", "value": "list_literal"}
                                                            ],
                                                            value="str",
                                                            clearable=False,
                                                            style={"height": "38px", "border": "1px solid #e1e5e9", "border-radius": "6px"}
                                                        )
                                                    ])
                                                ]),
                                                html.Div([
                                                    html.Label("Output Field Description", className="form-label"),
                                                    dcc.Input(
                                                        id={'type': 'extract-output-description', 'index': 'extract-output-field-initial'},
                                                        placeholder="Description of what to extract",
                                                        className="form-input"
                                                    )
                                                ], className="form-group"),
                                                html.Div(style={"display": "flex", "gap": "20px"}, children=[
                                                    html.Div(style={"flex": 1}, children=[
                                                        dcc.Checklist(
                                                            id={'type': 'extract-output-add-range', 'index': 'extract-output-field-initial'},
                                                            options=[{"label": "Add Range Limits", "value": "true"}],
                                                            className="form-checkbox"
                                                        )
                                                    ]),
                                                    html.Div(style={"flex": 1}, children=[
                                                        dcc.Checklist(
                                                            id={'type': 'extract-output-add-literal', 'index': 'extract-output-field-initial'},
                                                            options=[{"label": "Add Literal List", "value": "true"}],
                                                            className="form-checkbox"
                                                        )
                                                    ])
                                                ]),
                                                html.Div(id={'type': 'extract-output-range-container', 'index': 'extract-output-field-initial'}, style={"display": "none"}, children=[
                                                    html.Div([
                                                        html.Label("Range Minimum", className="form-label"),
                                                        dcc.Input(
                                                            id={'type': 'extract-output-range-min', 'index': 'extract-output-field-initial'},
                                                            type="number",
                                                            placeholder="Min value",
                                                            className="form-input"
                                                        )
                                                    ], className="form-group"),
                                                    html.Div([
                                                        html.Label("Range Maximum", className="form-label"),
                                                        dcc.Input(
                                                            id={'type': 'extract-output-range-max', 'index': 'extract-output-field-initial'},
                                                            type="number",
                                                            placeholder="Max value",
                                                            className="form-input"
                                                        )
                                                    ], className="form-group")
                                                ]),
                                                html.Div(id={'type': 'extract-output-literal-container', 'index': 'extract-output-field-initial'}, style={"display": "none"}, children=[
                                                    html.Div([
                                                        html.Label("Literal List (comma separated)", className="form-label"),
                                                        dcc.Input(
                                                            id={'type': 'extract-output-literal-list', 'index': 'extract-output-field-initial'},
                                                            placeholder="option1,option2,option3",
                                                            className="form-input"
                                                        )
                                                    ], className="form-group")
                                                ])
                                            ], id="extract-output-field-initial", className="section-container", style={"border": "1px solid #e1e5e9", "border-radius": "8px", "padding": "16px", "margin-bottom": "20px"})
                                        ])
                                    ], className="section-container"),

                                    # === PROMPT CONFIGURATION ===
                                    html.Div([
                                        html.H3("Prompt Configuration", className="section-title"),
                                        html.Div([
                                            html.Label("Classification Prompt", className="form-label"),
                                            dcc.Textarea(
                                                id="extract-classify-prompt",
                                                placeholder="Please classify the table content",
                                                value="Please classify the table content",
                                                className="form-textarea",
                                                style={"height": "80px"}
                                            ),
                                            html.Small("Prompt for table classification", className="form-hint")
                                        ], className="form-group"),
                                        html.Div([
                                            html.Label("Extraction Prompt", className="form-label"),
                                            dcc.Textarea(
                                                id="extract-extract-prompt",
                                                placeholder="Please extract the data according to the specified fields",
                                                value="Please extract the data according to the specified fields",
                                                className="form-textarea",
                                                style={"height": "80px"}
                                            ),
                                            html.Small("Prompt for data extraction", className="form-hint")
                                        ], className="form-group")
                                    ], className="section-container"),

                                    # === PROCESSING CONFIGURATION ===
                                    html.Div([
                                        html.H3("Processing Configuration", className="section-title"),
                                        html.Div([
                                            html.Label("Number of Threads", className="form-label"),
                                            dcc.Input(
                                                id="extract-num-threads",
                                                type="number",
                                                placeholder="6",
                                                value=6,
                                                min=1,
                                                max=32,
                                                className="form-input"
                                            ),
                                            html.Small("Number of threads to use for processing (1-32)", className="form-hint")
                                        ], className="form-group")
                                    ], className="section-container"),

                                    # === ACTION BUTTONS ===
                                    html.Div([
                                        html.Button(
                                            "Start Table Extraction",
                                            id="start-table-extraction",
                                            className="btn btn-primary",
                                            style={"marginTop": "20px", "height": "40px", "padding": "0 20px", "fontSize": "14px"}
                                        )
                                    ], className="section-container"),

                                    # === RESULTS DISPLAY ===
                                    html.Div([
                                        html.Div(id="table-extraction-results", className="results-container")
                                    ])
                                ], className="module-container")
                            ]),
                            
                            # === RECENT TASKS (Independent Section) ===
                            html.Div([
                                html.H3("Recent Tasks", className="section-title"),
                                html.Div([
                                    html.P("The following are the most recent table extraction tasks, showing task name, creation time, and status information.", 
                                          className="help-text-description", style={"marginBottom": "15px"})
                                ]),
                                # Dynamically load the latest 10 task examples
                                create_task_card_list(generate_task_data_from_runs("runs/table_extraction", 10, "table-extraction"))
                            ], className="module-container", style={"marginTop": "30px"}),
                            
                            # Table extraction configuration loading status modal
                            dbc.Modal(
                                [
                                    dbc.ModalHeader("Table Extraction Configuration Load Status"),
                                    dbc.ModalBody(id="table-extraction-config-modal-body"),
                                    dbc.ModalFooter(
                                        dbc.Button("Close", id="close-table-extraction-config-modal", className="ml-auto")
                                    ),
                                ],
                                id="table-extraction-config-modal",
                                is_open=False,
                                size="lg"
                            )
                        ]
                    )
                ]
            )
        ])
    ])
