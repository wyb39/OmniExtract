from dash import html, dcc
import dash_bootstrap_components as dbc
from taskcard import TaskCard, create_task_card_list, load_data_from_json_and_format, generate_task_data_from_runs

# Construct the Document Parsing page layout
def document_parsing_layout():
    return html.Div([
        html.Div([
            html.H2("Document Parsing", className="module-title"),
            
            # === TASK NAME CONFIGURATION ===
            html.Div([
                html.H3("Task Name", className="section-title"),
                html.Div([
                    html.Label("Task Name", className="form-label"),
                    dcc.Input(
                        id="task-name",
                        placeholder="Enter a name for this task",
                        value="",
                        className="form-input"
                    ),
                    html.Small("Provide a descriptive name for this document parsing task", className="form-hint")
                ], className="form-group")
            ], className="section-container"),

            # === FILE SOURCE CONFIGURATION ===
            html.Div([
                html.H3("File Source Configuration", className="section-title"),
                html.Div([
                    html.Label("Source Folder Path", className="form-label"),
                    dcc.Input(
                        id="folder-path",
                        placeholder="path/to/your/source/files",
                        value="",
                        className="form-input"
                    ),
                    html.Div([
                        html.P("Directory Structure Example:", className="help-text-title"),
                        html.Pre([
                            html.Code([
                                "source_directory/\n",
                                "├── document1.pdf\n",
                                "├── document2.pdf\n",
                                "├── document3.pdf\n",
                                "└── document4.pdf"
                            ], style={"fontFamily": "monospace", "fontSize": "14px", "lineHeight": "1.4"})
                        ], style={"backgroundColor": "#f8f9fa", "padding": "10px", "borderRadius": "4px", "margin": "10px 0"}),
                        html.P("Enter the path to the source directory containing multiple PDF files like the example above.", className="help-text-description")
                    ], className="help-text-container")
                ], className="form-group")
            ], className="section-container"),

            # === OUTPUT CONFIGURATION ===
            html.Div([
                html.H3("Output Configuration", className="section-title"),
                # Configure loading status display (changed to modal trigger)
                html.Div([
                    html.Button(
                        "Display Current Config Status",
                        id="parsing-config-status-btn",
                        className="btn btn-outline-info",
                        style={"marginBottom": "10px", "display": "none"}
                    ),
                    html.Div(id="parsing-config-load-status", style={"display": "none"}),
                ]),
                # Save path configuration
                html.Div([
                    html.Label("Save Path", className="form-label"),
                    dcc.Input(
                        id="save-path",
                        placeholder="path/to/save/json/files",
                        value="",
                        className="form-input"
                    ),
                    html.Div([
                        html.P("Files Will Be Saved In This Structure:", className="help-text-title"),
                        html.Pre([
                            html.Code([
                                "output_directory/\n",
                                "├── document1.json\n",
                                "├── document2.json\n",
                                "├── document3.json\n",
                                "├── document4.json\n",
                                "└── ", html.Span("_dataset.json", style={"color": "#d32f2f", "fontWeight": "bold"}), " (contains all JSON files for ", html.Strong("Document Extraction module"), ")"
                            ], style={"fontFamily": "monospace", "fontSize": "14px", "lineHeight": "1.4"})
                        ], style={"backgroundColor": "#f8f9fa", "padding": "10px", "borderRadius": "4px", "margin": "10px 0"}),
                        html.P(["Enter the path output_directory where JSON files will be saved. The ", html.Span("_dataset.json", style={"color": "#d32f2f", "fontWeight": "bold"}), " file contains all converted documents and can be used in the ", html.Strong("Document Extraction module"), "."], className="help-text-description")
                    ], className="help-text-container")
                ], className="form-group")
            ], className="section-container"),

            # === FILE TYPE CONFIGURATION ===
            html.Div([
                html.H3("File Type Configuration", className="section-title"),
                html.Div([
                    html.Label("File Type", className="form-label"),
                    dcc.Dropdown(
                        id="file-type",
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
                    html.Small("Specify the type of files to be processed", className="form-hint")
                ], className="form-group")
            ], className="section-container"),

            # === CONVERSION MODE ===
            html.Div([
                html.H3("Conversion Mode", className="section-title"),
                html.Div([
                    html.Label("Conversion Mode", className="form-label"),
                    dcc.Dropdown(
                        id="convert-mode",
                        options=[
                            {"label": "By Part (divide document by article parts)", "value": "byPart"},
                            {"label": "Whole Document (convert entire document as one)", "value": "wholeDoc"}
                        ],
                        value="byPart",
                        clearable=False,
                        className="form-select"
                    ),
                    html.Small([
                        "Choose the conversion strategy",
                        html.Br(),
                        html.Br(),
                        "Parsed File Data Structure Description:",
                        html.Br(),
                        html.Br(),
                        "1. JSON Structure Generated in By Part Mode:",
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
                        html.Code('}', style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        html.Br(),
                        html.Br(),
                        "2. JSON Structure Generated in Whole Document Mode:",
                        html.Br(),
                        html.Code("{", style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        html.Br(),
                        html.Code('"Document": "xxx"', style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px", "marginLeft": "20px"}),
                        html.Br(),
                        html.Code('}', style={"backgroundColor": "#f3f4f6", "padding": "2px 4px", "borderRadius": "3px"}),
                        html.Br(),
                        html.Br(),
                        html.Strong("⚠️ Note: ", style={"color": "#dc2626"}),
                        "The Document Parsing function will NOT parse Title and Abstract (as they can be obtained more accurately from websites)."
                    ], className="form-hint", style={"lineHeight": "1.5", "border": "1px solid #e1e5e9", "borderLeft": "4px solid #3b82f6", "borderRadius": "8px", "padding": "16px", "backgroundColor": "#f8f9fa"})
                ], className="form-group")
            ], className="section-container"),

            # === ACTION BUTTONS ===
            html.Div([
                html.Button(
                    "Start Document Parsing",
                    id="start-document-parsing",
                    className="btn btn-primary",
                    style={"marginTop": "20px", "height": "40px", "padding": "0 20px", "fontSize": "14px"}
                )
            ], className="section-container"),

            # === RESULTS DISPLAY ===
            html.Div([
                html.Div(id="parsing-results", className="results-container")
            ], className="section-container")
        ], className="module-container"),

        
        
        # === RECENT TASKS (Independent Section) ===
        html.Div([
            html.H3("Recent Tasks", className="section-title"),
            html.Div([
                html.P("The following are the most recent document parsing tasks, showing task name, creation time, and status information.", className="help-text-description", style={"marginBottom": "15px"})
            ]),
            # Dynamically load the latest 10 task examples
            create_task_card_list(generate_task_data_from_runs("runs/doc_parsing", 10, "doc-parsing"))
        ], className="module-container", style={"marginTop": "30px"}),
        
        dbc.Modal(
            [
                dbc.ModalHeader("Configuration Load Status"),
                dbc.ModalBody(id="parsing-config-modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-parsing-config-modal", className="ml-auto")
                ),
            ],
            id="parsing-config-modal",
            is_open=False,
            size="lg"
        )
    ])