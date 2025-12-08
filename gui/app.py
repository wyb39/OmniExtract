import dash
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import os
# Initialize Dash application
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
# Suppress initial layout validation errors
app.config.suppress_callback_exceptions = True
app.title = "OmniExtract"
server = app.server  # For production deployment

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
<link rel=\"icon\" href=\"/assets/img/logo.png\" type=\"image/png\"> 
{%css%}
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
"""

# Application layout
app.layout = html.Div([
    # Navigation bar
    html.Nav([
        html.Div([
            dcc.Link(
                href="/",  # Link to the home page
                children=[
                    html.Div([
                        html.Img(src="/assets/img/logo.png", className="logo"),
                        html.H1("OmniExtract", className="app-title")
                    ], className="logo-container")
                ],
                style={"textDecoration": "none", "color": "inherit"}
            ),
            html.Div([
                dcc.Link("Model Configuration", href="/model-config", className="nav-link", id="nav-model-config"),
                dcc.Link("Document Parsing", href="/document-parsing", className="nav-link", id="nav-document-parsing"),
                dcc.Link("Document Extraction", href="/doc-extraction", className="nav-link", id="nav-doc-extraction"),
                dcc.Link("Build Optimization Dataset", href="/build-optm-dataset", className="nav-link", id="nav-build-optm-dataset"),
                dcc.Link("Prompt Optimization", href="/prompt-optimization", className="nav-link", id="nav-prompt-optimization"),
                dcc.Link("Table Extraction", href="/table-parsing", className="nav-link", id="nav-table-parsing"),
            ], className="nav-links")
        ], className="nav-container")
    ], className="navbar"),
    
    # Page content area
    html.Div(id="page-content", className="page-content"),
    
    # For URL routing
    dcc.Location(id="url", refresh=False),
    
    # Use external CSS file
    html.Link(
        rel="stylesheet",
        href="/assets/style.css"
    ),
    
    # Add Font Awesome icon library
    html.Link(
        rel="stylesheet",
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    )
    ,
    dcc.Store(id="realtime-log-path"),
    dcc.Store(id="realtime-log-buffer"),
    dcc.Store(id="realtime-log-position"),
    dcc.Interval(id="realtime-log-interval", interval=1000, n_intervals=0, disabled=True),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("View the real-time log")),
        dbc.ModalBody(
            html.Div([
                html.Div(id="realtime-log-file-label", className="text-muted", style={"fontSize": "12px", "marginBottom": "8px"}),
                html.Pre(id="realtime-log-content", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace", "fontSize": "12px", "margin": 0}),
            ], id="realtime-log-content-container", style={"maxHeight": "70vh", "overflowY": "auto", "border": "1px solid #e1e5e9", "borderRadius": "6px", "padding": "12px", "backgroundColor": "#f9fafb", "scrollBehavior": "smooth"})
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-realtime-log-modal", color="secondary", size="sm")
        )
    ], id="realtime-log-modal", is_open=False, size="xl")
])

# Import all callback functions
import callbacks

# Route callback functions are already defined in callbacks.py

# Run application
if __name__ == "__main__":
    # Cancel all running tasks before starting the service
    from call_cli import cancel_all_running_tasks
    cancelled_count = cancel_all_running_tasks()
    
    app.run(debug=True, host=os.environ.get("HOST", "127.0.0.1"), port=os.environ.get("PORT", "8050"))