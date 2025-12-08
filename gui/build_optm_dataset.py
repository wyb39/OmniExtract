from dash import html, dcc
import dash_bootstrap_components as dbc
from taskcard import TaskCard, create_task_card_list, load_data_from_json_and_format, generate_task_data_from_runs

# Build Optm Dataset Page Layout
def build_optm_dataset_layout():
    return html.Div([
        html.Div([
            html.H2("Build Optimization Dataset", className="module-title"),
            
            # === TASK NAME ===
            html.Div([
                html.H3("Task Name", className="section-title"),
                html.Div([
                    html.Label("Task Name", className="form-label"),
                    dcc.Input(
                        id="task-name-dataset",
                        placeholder="Enter a name for this task",
                        className="form-input"
                    ),
                    html.Small("Specify a name for this build dataset task. This will be used to identify the task in the recent tasks list.", className="form-hint")
                ], className="form-group")
            ], className="section-container"),
            
            # === REQUIRED PATHS ===
            html.Div([
                html.H3("Required Paths", className="section-title"),
                html.Div([
                    html.Label("JSON Files Path", className="form-label"),
                    dcc.Input(
                        id="json-path",
                        placeholder="path/to/your/json/files",
                        className="form-input"
                    ),
                    html.Small([
                        "Path to the JSON files containing article data. The json files could be converted using file_to_json function.",
                        html.Br(),
                        html.Br(),
                        html.Strong("⚠️ Special Note: ", style={"color": "#dc2626"}),
                        "To ensure information accuracy, the Document Parsing function will ",
                        html.Strong("NOT"),
                        " parse Title and Abstract from documents (as they can be obtained more accurately from websites). ",
                        "If you need to extract this information, please add ",
                        html.Code("Title", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        " and ",
                        html.Code("Abstract", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        " fields to your JSON files.",
                        html.Br(),
                        html.Br(),
                        "Format requirements: Each JSON file must follow one of these structures:",
                        html.Br(),
                        html.Code("{", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        html.Br(),
                        html.Code('"Introduction": "xxx",', style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px", "marginLeft": "20px"}),
                        html.Br(),
                        html.Code('"Method": "xxx",', style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px", "marginLeft": "20px"}),
                        html.Br(),
                        html.Code('"Result": "xxx",', style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px", "marginLeft": "20px"}),
                        html.Br(),
                        html.Code('"Discussion": "xxx",', style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px", "marginLeft": "20px"}),
                        html.Br(),
                        html.Code('"...": "..."', style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px", "marginLeft": "20px"}),
                        html.Br(),
                        html.Code("}", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        html.Br(),
                        html.Br(),
                        "Or alternatively:",
                        html.Br(),
                        html.Code("{", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        html.Br(),
                        html.Code('"Document": "xxx"', style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px", "marginLeft": "20px"}),
                        html.Br(),
                        html.Code("}", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"})
                    ], className="form-hint", style={"lineHeight": "1.5", "border": "1px solid #e1e5e9", "borderLeft": "4px solid #3b82f6", "borderRadius": "8px", "padding": "16px", "backgroundColor": "#f8f9fa"})
                ], className="form-group"),
                html.Div([
                    html.Label("Curated Dataset Path", className="form-label"),
                    dcc.Input(
                        id="curated-dataset-path",
                        placeholder="path/to/your/curated/dataset.json",
                        className="form-input"
                    ),
                    html.Small([
                        "Path to the curated dataset file. Supported formats: JSON, CSV, TSV, Excel.",
                        html.Br(),
                        html.Br(),
                        html.Span("Required:", style={"color": "#dc2626", "fontWeight": "bold"}),
                        " Your dataset must have a column whose name exactly matches ",
        				html.Strong("Article Field Name"),
                        ".",
                        html.Br(),
                        "The values in this column should be the article file name ",
                        html.Strong("without extension"),
                        " (e.g., ",
                        html.Code("paper_123", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        " for ",
                        html.Code("paper_123.json", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        ").",
                        html.Br(),
                        html.Br(),
                        "Example: If ",
                        html.Code("Article Field Name", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        " is ",
                        html.Code("article_field", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        ", your dataset needs a column ",
                        html.Code("article_field", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        " with values like ",
                        html.Code("paper_123", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        ", ",
                        html.Code("paper_456", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        "."
                    ], className="form-hint", style={"lineHeight": "1.5", "border": "1px solid #e1e5e9", "borderLeft": "4px solid #3b82f6", "borderRadius": "8px", "padding": "16px", "backgroundColor": "#f8f9fa"})
                ], className="form-group"),
                html.Div([
                    html.Label("Save Directory", className="form-label"),
                    dcc.Input(
                        id="save-directory",
                        placeholder="path/to/save/directory",
                        className="form-input"
                    ),
                    html.Small([
                         "Directory where the built optimization dataset will be saved. The directory will be created if it doesn't exist.",
                          html.Br(),
                          html.Br(),
                          "The built optimization dataset will be saved as ",
                          html.Code("_optim_dataset.json", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                          " in the specified directory."
                     ], className="form-hint", style={"lineHeight": "1.5", "border": "1px solid #e1e5e9", "borderLeft": "4px solid #3b82f6", "borderRadius": "8px", "padding": "16px", "backgroundColor": "#f8f9fa"})
                ], className="form-group")
            ], className="section-container"),

            # === FIELDS CONFIGURATION ===
            html.Div([
                html.H3("Fields Configuration", className="section-title"),
                html.Div([
                    html.Button(
                        "Display Current Config Status",
                        id="dataset-config-status-btn",
                        className="btn btn-outline-info",
                        style={"marginBottom": "10px", "display": "none"}
                    ),
                    html.Button(
                        "Load Dataset Config",
                        id={'type': 'load-dataset-config', 'index': 1},
                        className="btn btn-outline-primary",
                        style={"marginBottom": "10px", "marginLeft": "10px", "display": "none"}
                    ),
                    html.Div(id="dataset-config-load-status", style={"display": "none"}),
                ]),
                html.Div([
                    html.Small([
                        "Add one or more fields to describe what information to extract from each article.",
                        html.Br(),
                        html.Br(),
                        "Each field you add will appear as a key in the output file ",
                        html.Code("_optim_dataset.json", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        ". The ",
                        html.Strong("Field Name"),
                        " becomes the exact key used in the output and should match your curated dataset column name when applicable.",
                                            ], className="form-hint", style={"lineHeight": "1.5", "border": "1px solid #e1e5e9", "borderLeft": "4px solid #3b82f6", "borderRadius": "8px", "padding": "16px", "backgroundColor": "#f8f9fa"})
                ]),
                html.Button(
                    "Add Field",
                    id="add-dataset-output-field",
                    className="btn",
                    style={"marginBottom": "15px", "height": "32px", "padding": "0 12px", "fontSize": "12px"}
                ),
                html.Div(id="dataset-output-fields-container", children=[
                    html.Div([
                        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}, children=[
                            html.H4("Field", style={"margin": "0", "fontSize": "16px"}),
                            html.Button(
                                "×",
                                id={'type': 'delete-dataset-output-field', 'index': 'dataset-output-field-initial'},
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
                                html.Label("Field Name", className="form-label"),
                                dcc.Input(
                                    id={'type': 'dataset-output-name', 'index': 'dataset-output-field-initial'},
                                    placeholder="field_name",
                                    className="form-input",
                                    style={"height": "38px"}
                                )
                            ]),
                            html.Div(style={"flex": 1}, children=[
                                html.Label("Field Type", className="form-label"),
                                dcc.Dropdown(
                                    id={'type': 'dataset-output-type', 'index': 'dataset-output-field-initial'},
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
                            html.Label("Field Description", className="form-label"),
                            dcc.Input(
                                id={'type': 'dataset-output-description', 'index': 'dataset-output-field-initial'},
                                placeholder="Description of this field",
                                className="form-input"
                            )
                        ], className="form-group"),
                        html.Div(style={"display": "flex", "gap": "20px"}, children=[
                            html.Div(style={"flex": 1}, children=[
                                dcc.Checklist(
                                    id={'type': 'dataset-output-add-range', 'index': 'dataset-output-field-initial'},
                                    options=[{"label": "Add Range Limits", "value": "true"}],
                                    className="form-checkbox"
                                )
                            ]),
                            html.Div(style={"flex": 1}, children=[
                                dcc.Checklist(
                                    id={'type': 'dataset-output-add-literal', 'index': 'dataset-output-field-initial'},
                                    options=[{"label": "Add Literal List", "value": "true"}],
                                    className="form-checkbox"
                                )
                            ])
                        ]),
                        html.Div(id={'type': 'dataset-output-range-container', 'index': 'dataset-output-field-initial'}, style={"display": "none"}, children=[
                            html.Div([
                                html.Label("Range Minimum", className="form-label"),
                                dcc.Input(
                                    id={'type': 'dataset-output-range-min', 'index': 'dataset-output-field-initial'},
                                    type="number",
                                    placeholder="Min value",
                                    className="form-input"
                                )
                            ], className="form-group"),
                            html.Div([
                                html.Label("Range Maximum", className="form-label"),
                                dcc.Input(
                                    id={'type': 'dataset-output-range-max', 'index': 'dataset-output-field-initial'},
                                    type="number",
                                    placeholder="Max value",
                                    className="form-input"
                                )
                            ], className="form-group")
                        ]),
                        html.Div(id={'type': 'dataset-output-literal-container', 'index': 'dataset-output-field-initial'}, style={"display": "none"}, children=[
                            html.Div([
                                html.Label("Literal List (comma separated)", className="form-label"),
                                dcc.Input(
                                    id={'type': 'dataset-output-literal-list', 'index': 'dataset-output-field-initial'},
                                    placeholder="option1,option2,option3",
                                    className="form-input"
                                )
                            ], className="form-group")
                        ])
                    ], id="dataset-output-field-initial", className="section-container", style={"border": "1px solid #e1e5e9", "border-radius": "8px", "padding": "16px", "margin-bottom": "20px"})
                ])
            ], className="section-container"),

            # === EXTRACTION MODE ===
            html.Div([
                html.H3("Extraction Mode", className="section-title"),
                html.H4("Multiple Entities", className="subsection-title", style={"marginTop": "15px", "marginBottom": "10px", "fontSize": "16px", "color": "#374151"}),
                html.Div([
                    dcc.Checklist(
                        id="multiple-entities",
                        options=[{"label": "Set true for extracting multiple entities from the document", "value": "true"}],
                        value=[],
                        inline=True
                    )
                ], className="form-group")
            ], className="section-container"),

            # === ARTICLE CONFIGURATION ===
            html.Div([
                html.H3("Article Configuration", className="section-title"),
                html.Div([
                    html.Label("Article Field Name", className="form-label"),
                    dcc.Input(
                        id="article-field",
                        placeholder="article_field",
                        value="article_field",
                        className="form-input"
                    ),
                    html.Small("Field name for the article file name to extract", className="form-hint")
                ], className="form-group"),
                html.Div([
                        html.Label("Article Parts to Extract From (Optional)", className="form-label"),
                        dcc.Dropdown(
                            id="article-parts",
                            options=[
                                {"label": "Title", "value": "Title"},
                                {"label": "Abstract", "value": "Abstract"},
                                {"label": "Introduction", "value": "Introduction"},
                                {"label": "Method", "value": "Method"},
                                {"label": "Result", "value": "Result"},
                                {"label": "Discussion", "value": "Discussion"},
                                {"label": "Conclusion", "value": "Conclusion"}
                            ],
                            multi=True,
                            className="form-select"
                        ),
                        html.Small("If not specified, the entire article will be used", className="form-hint")
                    ], className="form-group")
            ], className="section-container"),

            # === ACTION BUTTONS ===
            html.Div([
                html.Button(
                    "Start Building Dataset",
                    id="start-building-dataset",
                    className="btn btn-primary",
                    style={"marginTop": "20px", "height": "40px", "padding": "0 20px", "fontSize": "14px"}
                )
            ], className="section-container"),

            # === RESULTS DISPLAY ===
            html.Div([
                html.Div(id="build-results", className="results-container")
            ])
        ], className="module-container"),
        
        # === RECENT BUILD DATASET TASKS ===
        html.Div([
            html.H3("Recent Build Dataset Tasks", className="section-title"),
            html.Div([
                html.P("The following are the most recent build dataset tasks, showing task name, creation time, and status information.", 
                      className="help-text-description", style={"marginBottom": "15px"})
            ]),
            create_task_card_list(generate_task_data_from_runs("runs/build_optim_dataset", 10, "build-dataset"))
        ], className="module-container", style={"marginTop": "30px"}),
        
        # === CONFIGURATION MODAL ===
        html.Div([
            dcc.Store(id='dataset-config-load-status', data=''),
            dcc.Store(id='dataset-config-modal-body', data=''),
            dcc.Store(id={'type': 'load-dataset-config-file-path', 'index': 1}, data=''),
            
            # Configuration Load Modal
            dbc.Modal([
                dbc.ModalHeader("Configuration Load Status"),
                dbc.ModalBody(id="dataset-config-modal-body"),
                dbc.ModalFooter([
                    dbc.Button(
                        "Close",
                        id="close-dataset-config-modal",
                        className="btn btn-secondary"
                    )
                ])
            ],
            id="dataset-config-modal",
            is_open=False,
            size="lg"
            )
        ])
    ])
