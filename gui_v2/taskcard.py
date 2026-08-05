"""
Task card data generation for gui_v2 (Jinja-based, no Dash dependency).

Ported from gui/taskcard.py but returns plain dicts that templates can render.
Each task card dict has:
    id, name, create_time, status, config_path (run json path),
    log_path, log_exists, data (the run's "data" field, for display).
"""
import glob
import json
import os
import re
from datetime import datetime

# Map each gui_v2 module key -> (runs subdirectory under gui_v2/runs, human title)
MODULE_RUN_DIRS = {
    "doc-parsing": ("doc_parsing", "Document Parsing"),
    "original-extraction": ("doc_extraction/original", "Document Extraction \u00b7 Original"),
    "optimized-extraction": ("doc_extraction/optm", "Document Extraction \u00b7 Optimized"),
    "prompt-optimization": ("prompt_optimization", "Prompt Optimization"),
    "build-dataset": ("build_optim_dataset", "Build Optimization Dataset"),
    "table-parsing": ("table_parsing", "Table Files Parsing"),
    "table-extraction": ("table_extraction", "Table Extraction"),
}


def runs_root():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def module_runs_dir(module_key):
    sub = MODULE_RUN_DIRS.get(module_key, (module_key, module_key))[0]
    return os.path.join(runs_root(), sub)


def _extract_timestamp(path):
    m = re.search(r"(?:run_)?(\d{8})_(\d{6})", os.path.basename(path))
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return datetime.min
    return datetime.min


def _fmt_create_time(raw):
    if not raw or raw == "Unknown Time":
        return "Unknown Time"
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return raw


def generate_task_data_from_runs(module_key, n=10):
    """Return the latest n task card dicts for a module."""
    base_path = module_runs_dir(module_key)
    if not os.path.isdir(base_path):
        return []

    subdirs = []
    for item in os.listdir(base_path):
        p = os.path.join(base_path, item)
        if os.path.isdir(p):
            subdirs.append(p)
    subdirs.sort(key=_extract_timestamp, reverse=True)

    tasks = []
    for i, subdir in enumerate(subdirs[:n]):
        json_files = glob.glob(os.path.join(subdir, "*.json"))
        if not json_files:
            continue
        json_file = json_files[0]
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                jdata = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        log_path = os.path.splitext(json_file)[0] + ".log"
        tasks.append({
            "id": f"{module_key}-{i + 1}",
            "name": jdata.get("name", "Unnamed Task"),
            "create_time": _fmt_create_time(jdata.get("created_time")),
            "status": jdata.get("status", "unknown"),
            "config_path": json_file,
            "log_path": log_path,
            "log_exists": os.path.exists(log_path),
            "data": jdata.get("data", {}) or {},
            "module_key": module_key,
        })
    return tasks


def has_main_config():
    """Check whether a main model settings file exists in the project settings/ dir."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.exists(os.path.join(project_root, "settings", "model_settings_main.json"))
