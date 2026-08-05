"""
gui_v2 — Jinja/Flask rewrite of the Dash-based gui.

Run from the project root:

    python -m gui_v2.app
    # or
    python gui_v2/app.py

Then open http://localhost:8051

Design notes:
  * No Dash. Pure Flask + Jinja2 templates.
  * Pages are server-rendered; forms POST JSON to /api/<module>/run endpoints
    which spawn a subprocess (reusing gui.process_manager + gui.call_cli) and
    return a status summary rendered into the page.
  * Recent-task cards are rendered server-side from gui_v2/runs/<module>.
  * Task cards support: expand/collapse, cancel (if running), view real-time
    .log, and "use this config" writeback (fills the form via /api/task/config).
"""
import os
import sys

# Put the project root AND the legacy gui dir on sys.path so that:
#   * gui / gui_v2 / src are importable as packages
#   * gui's internal flat imports (e.g. `from process_manager import ...`) resolve
# Works whether launched via `python gui_v2/app.py` or `python -m gui_v2.app`.
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_GUI_DIR = os.path.join(_PROJECT_ROOT, "gui")
for _p in (_PROJECT_ROOT, _GUI_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
from flask import Flask, render_template, request, jsonify, abort

# Package-qualified imports (no flat imports) prevent collisions with the
# legacy Dash gui package (e.g. gui/taskcard.py).
from gui_v2 import task_service, taskcard
from gui.call_cli import cancel_all_running_tasks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["JSON_AS_ASCII"] = False


# ----- Jinja filters / globals used by templates -----
_KEY_DISPLAY = {
    "folder_path": "Folder Path", "save_path": "Save Path", "file_type": "File Type",
    "convert_mode": "Convert Mode", "message": "Message", "result": "Result", "details": "Details",
    "dataset_file": "Dataset File", "error": "Error Message", "dataset": "Dataset",
    "save_dir": "Save Directory", "inputFields": "Input Fields", "outputFields": "Output Fields",
    "initial_prompt": "Initial Prompt", "judging": "Judging", "task": "Task", "threads": "Threads",
    "multiple": "Multiple", "name": "Name", "field_type": "Field Type", "description": "Description",
    "optim_burden": "Optimization Burden", "demos": "Demos", "ai_evaluation": "AI Evaluation",
    "load_dir": "Load Directory", "json_path": "JSON Files Path",
    "article_field": "Article Field", "article_parts": "Article Parts", "fields": "Fields",
    "parsed_file_path": "Parsed File Path", "save_folder_path": "Save Folder Path",
    "classify_prompt": "Classify Prompt", "extract_prompt": "Extract Prompt",
    "num_threads": "Number of Threads", "non_tabular_file_format": "Non-tabular File Format",
    "file_folder_path": "File Folder Path", "extract_directly": "Extract Directly",
}


@app.template_filter("basename")
def _basename(path):
    return os.path.basename(path or "")


@app.template_global("key_display")
def _key_display(key):
    return _KEY_DISPLAY.get(key, str(key).replace("_", " ").title())


# ---------------------------------------------------------------------------
# page routes
# ---------------------------------------------------------------------------
NAV = [
    ("model-config", "Model Configuration", "fa-sliders"),
    ("document-parsing", "Document Parsing", "fa-file-lines"),
    ("doc-extraction", "Document Extraction", "fa-wand-magic-sparkles"),
    ("build-optm-dataset", "Build Optimization Dataset", "fa-database"),
    ("prompt-optimization", "Prompt Optimization", "fa-bolt"),
    ("table-parsing", "Table Extraction", "fa-table"),
]


@app.context_processor
def inject_globals():
    return {"nav_items": NAV, "has_main_config": taskcard.has_main_config()}


@app.route("/")
def page_index():
    return render_template("index.html", active="index")


@app.route("/model-config")
def page_model_config():
    return render_template("model_config.html", active="model-config")


@app.route("/document-parsing")
def page_document_parsing():
    tasks = taskcard.generate_task_data_from_runs("doc-parsing", 10)
    return render_template("document_parsing.html", active="document-parsing", tasks=tasks,
                           module_key="doc-parsing")


@app.route("/doc-extraction")
def page_doc_extraction():
    orig = taskcard.generate_task_data_from_runs("original-extraction", 10)
    optm = taskcard.generate_task_data_from_runs("optimized-extraction", 10)
    return render_template("doc_extraction.html", active="doc-extraction",
                           original_tasks=orig, optimized_tasks=optm)


@app.route("/build-optm-dataset")
def page_build_optm_dataset():
    tasks = taskcard.generate_task_data_from_runs("build-dataset", 10)
    return render_template("build_optm_dataset.html", active="build-optm-dataset",
                           tasks=tasks, module_key="build-dataset")


@app.route("/prompt-optimization")
def page_prompt_optimization():
    tasks = taskcard.generate_task_data_from_runs("prompt-optimization", 10)
    return render_template("prompt_optimization.html", active="prompt-optimization",
                           tasks=tasks, module_key="prompt-optimization")


@app.route("/table-parsing")
def page_table_parsing():
    tp = taskcard.generate_task_data_from_runs("table-parsing", 10)
    te = taskcard.generate_task_data_from_runs("table-extraction", 10)
    return render_template("table_parsing.html", active="table-parsing",
                           table_parsing_tasks=tp, table_extraction_tasks=te)


# ---------------------------------------------------------------------------
# task run API
# ---------------------------------------------------------------------------
@app.route("/api/task/run/<module_key>", methods=["POST"])
def api_run_task(module_key):
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        result = task_service.dispatch(module_key, payload)
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
    return jsonify(result)


@app.route("/api/task/cancel", methods=["POST"])
def api_cancel_task():
    data = request.get_json(silent=True) or request.form.to_dict()
    log_path = data.get("log_path")
    if not log_path:
        return jsonify({"ok": False, "error": "log_path required"}), 400
    ok = task_service.cancel(log_path)
    return jsonify({"ok": ok})


@app.route("/api/task/config", methods=["GET"])
def api_task_config():
    """Return a writeback payload (form values) for a given run JSON path."""
    path = request.args.get("path")
    module_key = request.args.get("module")
    if not path or not module_key:
        return jsonify({"ok": False, "error": "path and module are required"}), 400
    jdata = task_service.read_run_json(path)
    if jdata is None:
        return jsonify({"ok": False, "error": "cannot read run json"}), 404
    return jsonify({"ok": True, "values": task_service.writeback_for(module_key, jdata.get("data", {}))})


@app.route("/api/task/log", methods=["GET"])
def api_task_log():
    path = request.args.get("path")
    if not path:
        return jsonify({"ok": False, "error": "path required"}), 400
    content = task_service.read_log_tail(path)
    if content is None:
        return jsonify({"ok": False, "error": "log not found"}), 404
    return jsonify({"ok": True, "content": content, "size": len(content)})


# ---------------------------------------------------------------------------
# model config API (synchronous)
# ---------------------------------------------------------------------------
@app.route("/api/model/settings", methods=["GET"])
def api_model_settings():
    usage = request.args.get("model_usage", "main")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(project_root, "settings", f"model_settings_{usage}.json")
    if not os.path.exists(cfg_path):
        return jsonify({"setting_status": False})
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"setting_status": False, "error": str(e)}), 500


@app.route("/api/model/modify", methods=["POST"])
def api_model_modify():
    payload = request.get_json(silent=True) or {}
    for key in ("temperature", "max_tokens"):
        if key in payload and payload[key] not in (None, ""):
            try:
                payload[key] = float(payload[key])
                if payload[key] == int(payload[key]):
                    payload[key] = int(payload[key])
            except (TypeError, ValueError):
                pass
    for key in ("top_p", "min_p"):
        if payload.get(key) not in (None, ""):
            try:
                payload[key] = float(payload[key])
            except (TypeError, ValueError):
                pass
        else:
            payload[key] = None
    if payload.get("top_k") not in (None, ""):
        try:
            payload["top_k"] = int(payload["top_k"])
        except (TypeError, ValueError):
            pass
    else:
        payload["top_k"] = None
    result = task_service.save_model_config(payload)
    code = 200 if result.get("ok") else 500
    return jsonify(result), code


@app.route("/api/model/test", methods=["POST"])
def api_model_test():
    payload = request.get_json(silent=True) or {}
    result = task_service.test_model_connection(payload)
    code = 200 if result.get("ok") else 500
    return jsonify(result), code


@app.errorhandler(404)
def not_found(_e):
    return render_template("base.html", active=""), 404


def main():
    # Mark any tasks left "running" from a previous (crashed) session as cancelled.
    cancel_all_running_tasks(os.path.join(BASE_DIR, "runs"))

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8051"))
    use_prod = os.environ.get("GUI_V2_PROD", "1") == "1"
    if use_prod:
        try:
            from waitress import serve
            print(f"gui_v2 (waitress) serving on http://{host}:{port}")
            serve(app, host=host, port=port)
            return
        except Exception as e:
            print(f"waitress unavailable ({e}), falling back to Flask dev server")
    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    main()
