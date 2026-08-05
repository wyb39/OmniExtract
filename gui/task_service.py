"""
Task service for gui (Flask/Jinja, no Dash).

Handles:
  * launching extraction/parsing/optimization tasks as subprocesses
  * canceling a running task
  * reading task run JSON (status / data)
  * reading real-time .log files
  * building a writeback payload (config -> form values) from a run JSON

It reuses the Dash-free utilities already present in the repo:
  * gui.call_cli          (create_initial_log, run_task_and_update_log, cancel_task)
  * gui.process_manager   (process_manager.start_python_code)
  * gui.yml_generation.*  (YAML config writers)
  * src.cli.cli_handler   (the actual task callables)

All run artifacts are stored under  gui/runs/<module>/run_<timestamp>/
"""
import json
import os
import sys
from datetime import datetime

# The gui package uses FLAT imports internally (e.g. gui/call_cli.py does
# `from process_manager import process_manager`), so the gui directory itself
# must be on sys.path for `gui.call_cli` to load.
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_GUI_DIR = os.path.join(_PROJECT_ROOT, "gui")
for _p in (_PROJECT_ROOT, _GUI_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gui.call_cli import create_initial_log, cancel_task, run_task_and_update_log
from gui.process_manager import process_manager
from gui.yml_generation.yml_document_parsing import save_document_parsing_config_to_yaml
from gui.yml_generation.yml_doc_extraction import (
    generate_novel_config_yaml,
    generate_optimized_config_from_callback,
)
from gui.yml_generation.yml_prompt_generation import save_prompt_optimization_config_to_yaml
from gui.yml_generation.yml_table_parsing import (
    save_table_parsing_config_to_yaml,
    save_extract_table_config_to_yaml,
)
from gui.yml_generation.yml_build_optm_dataset import generate_build_optm_dataset_yml_from_dash_callback
from gui.yml_generation.yml_model_config import save_model_config_to_yaml


# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------
def runs_root():
    return os.path.join(_GUI_DIR, "runs")


def _project_root():
    return _PROJECT_ROOT


def _src_path():
    return os.path.join(_project_root(), "src")


def _make_run_subdir(module_subdir):
    """Create and return a fresh runs/<module_subdir>/run_<timestamp> directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = os.path.join(runs_root(), module_subdir, f"run_{timestamp}")
    os.makedirs(sub_dir, exist_ok=True)
    return sub_dir, timestamp


# ----------------------------------------------------------------------
# field parsing (from the JSON posted by the browser)
# ----------------------------------------------------------------------
def parse_fields(field_list, defaults_key="input_text"):
    """Normalize a list of field dicts coming from the browser into the
    config schema used by the task callables.

    Each browser field looks like:
        {name, type, description, hasRange, rangeMin, rangeMax,
         hasLiteral, literalList}
    """
    out = []
    for fld in field_list or []:
        name = (fld.get("name") or "").strip()
        if not name:
            continue
        item = {
            "name": name,
            "field_type": fld.get("type") or "str",
            "description": fld.get("description") or "",
        }
        if fld.get("hasRange") and fld.get("rangeMin") is not None and fld.get("rangeMax") is not None:
            try:
                item["range_min"] = float(fld["rangeMin"])
                item["range_max"] = float(fld["rangeMax"])
            except (TypeError, ValueError):
                pass
        if fld.get("hasLiteral") and fld.get("literalList"):
            item["literal_list"] = [s.strip() for s in str(fld["literalList"]).split(",") if s.strip()]
        out.append(item)
    return out


# ----------------------------------------------------------------------
# subprocess launching helper
# ----------------------------------------------------------------------
def _launch(callable_import, config, log_path):
    """Spawn a detached python subprocess that runs:

        from gui.call_cli import run_task_and_update_log
        from <callable_import[0]> import <callable_import[1]>
        run_task_and_update_log(<callable>, config, log_path)

    The cwd is set to the project root so `gui` and `src` are importable as
    (namespace) packages.
    """
    proj = _project_root()
    srcp = _src_path()
    proj_esc = proj.replace("\\", "\\\\")
    srcp_esc = srcp.replace("\\", "\\\\")
    gui_esc = _GUI_DIR.replace("\\", "\\\\")
    log_path_fixed = log_path.replace("\\", "\\\\")
    config_repr = repr(config)

    module_path, attr_name = callable_import
    python_code = (
        "import sys, os\n"
        f"sys.path.insert(0, r'{proj_esc}')\n"
        f"sys.path.insert(0, r'{srcp_esc}')\n"
        f"sys.path.insert(0, r'{gui_esc}')\n"
        f"os.chdir(r'{proj_esc}')\n"
        "from gui.call_cli import run_task_and_update_log\n"
        f"from {module_path} import {attr_name}\n"
        f"config = {config_repr}\n"
        f"result = run_task_and_update_log(callable_obj={attr_name}, data=config, log_path=r'{log_path_fixed}')\n"
        "print(result)\n"
    )
    proc_log_filename = os.path.basename(log_path).replace(".json", ".log")
    process_manager.start_python_code(
        python_code=python_code,
        key=log_path,
        capture_output=False,
        text=True,
        log_to_key_dir=True,
        log_filename=proc_log_filename,
    )


# ----------------------------------------------------------------------
# module-specific run handlers
# ----------------------------------------------------------------------
def run_document_parsing(payload):
    folder_path = (payload.get("folder_path") or "").strip()
    save_path = (payload.get("save_path") or "").strip()
    if not folder_path:
        return {"ok": False, "error": "Source Folder Path is required!"}
    if not save_path:
        return {"ok": False, "error": "Save Path is required!"}
    os.makedirs(save_path, exist_ok=True)

    file_type = payload.get("file_type") or "PDF"
    convert_mode = payload.get("convert_mode") or "byPart"
    task_name = (payload.get("task_name") or "").strip()

    sub_dir, ts = _make_run_subdir("doc_parsing")
    yaml_filename = f"file_to_json_config_{ts}.yml"
    config = {
        "folder_path": folder_path,
        "save_path": save_path,
        "file_type": file_type,
        "convert_mode": convert_mode,
    }
    save_document_parsing_config_to_yaml(
        folder_path=folder_path, save_path=save_path, file_type=file_type,
        convert_mode=convert_mode, output_dir=sub_dir, filename=yaml_filename,
    )
    log_path = os.path.join(sub_dir, f"file_to_json_{ts}.json")
    create_initial_log(name=task_name or ts, data=config, log_path=log_path)
    _launch(("src.cli.cli_handler", "run_file_to_json"), config, log_path)
    return {"ok": True, "timestamp": ts, "config": config, "module": "doc-parsing"}


def run_original_extraction(payload):
    dataset_path = (payload.get("dataset_path") or "").strip()
    save_dir = (payload.get("save_dir") or "").strip()
    if not dataset_path:
        return {"ok": False, "error": "Dataset Path is required!"}
    if not save_dir:
        return {"ok": False, "error": "Save Directory is required!"}

    input_fields = parse_fields(payload.get("inputFields"))
    output_fields = parse_fields(payload.get("outputFields"))
    if not input_fields:
        input_fields = [{"name": "input_text", "field_type": "str", "description": "Input field"}]
    if not output_fields:
        output_fields = [{"name": "extracted_info", "field_type": "str", "description": "Extracted data"}]

    initial_prompt = payload.get("initial_prompt") or "Extract the target information from the given text"
    judging = payload.get("judging") or ""
    task = payload.get("task") or "Extraction"
    threads = int(payload.get("threads") or 6)
    multiple = bool(payload.get("multiple"))
    task_name = (payload.get("task_name") or "").strip()

    sub_dir, ts = _make_run_subdir(os.path.join("doc_extraction", "original"))
    yaml_path = os.path.join(sub_dir, f"config_{ts}.yml")
    generate_novel_config_yaml(
        dataset_path=dataset_path, save_dir=save_dir, input_fields=input_fields,
        output_fields=output_fields, initial_prompt=initial_prompt,
        judging=judging, task=task, threads=threads, multiple=multiple, output_path=yaml_path,
    )
    config = {
        "dataset": dataset_path, "save_dir": save_dir, "inputFields": input_fields,
        "outputFields": output_fields, "initial_prompt": initial_prompt,
        "judging": judging, "task": task, "threads": threads, "multiple": multiple,
    }
    log_path = os.path.join(sub_dir, f"original_extraction_{ts}.json")
    create_initial_log(name=task_name or ts, data=config, log_path=log_path)
    _launch(("src.cli.cli_handler", "run_pred_original"), config, log_path)
    return {"ok": True, "timestamp": ts, "config": config, "module": "original-extraction"}


def run_optimized_extraction(payload):
    load_dir = (payload.get("load_dir") or "").strip()
    dataset_path = (payload.get("dataset_path") or "").strip()
    save_dir = (payload.get("save_dir") or "").strip()
    if not load_dir:
        return {"ok": False, "error": "Load Directory is required!"}
    if not dataset_path:
        return {"ok": False, "error": "Dataset Path is required!"}
    if not save_dir:
        return {"ok": False, "error": "Save Directory is required!"}

    judging = payload.get("judging") or ""
    threads = int(payload.get("threads") or 6)
    task_name = (payload.get("task_name") or "").strip()

    sub_dir, ts = _make_run_subdir(os.path.join("doc_extraction", "optm"))
    yaml_filename = f"optimized_extraction_config_{ts}.yml"
    yaml_output_path = os.path.join(sub_dir, yaml_filename)
    generate_optimized_config_from_callback(
        load_dir=load_dir, dataset=dataset_path, save_dir=save_dir,
        judging=judging, threads=threads, output_file="result.json", output_path=yaml_output_path,
    )
    config = {
        "load_dir": load_dir, "dataset": dataset_path, "save_dir": save_dir,
        "judging": judging, "threads": threads, "output_file": "result.json",
    }
    log_path = os.path.join(sub_dir, f"pred_optimized_{ts}.json")
    create_initial_log(name=task_name or ts, data=config, log_path=log_path)
    _launch(("src.cli.cli_handler", "run_pred_optimized"), config, log_path)
    return {"ok": True, "timestamp": ts, "config": config, "module": "optimized-extraction"}


def run_build_dataset(payload):
    json_path = (payload.get("json_path") or "").strip()
    curated_dataset_path = (payload.get("curated_dataset_path") or "").strip()
    save_directory = (payload.get("save_directory") or "").strip()
    if not json_path:
        return {"ok": False, "error": "JSON Files Path is required!"}
    if not curated_dataset_path:
        return {"ok": False, "error": "Curated Dataset Path is required!"}
    if not save_directory:
        return {"ok": False, "error": "Save Directory is required!"}

    fields_data = parse_fields(payload.get("fields"))
    multiple = bool(payload.get("multiple"))
    article_field = payload.get("article_field") or "article_field"
    article_parts = payload.get("article_parts") or []
    task_name = (payload.get("task_name") or "").strip()

    sub_dir, ts = _make_run_subdir("build_optim_dataset")
    yml_filename = f"build_optm_set_config_{ts}.yml"
    yml_output_path = os.path.join(sub_dir, yml_filename)
    yml_content = generate_build_optm_dataset_yml_from_dash_callback(
        json_path=json_path, curated_dataset_path=curated_dataset_path,
        fields_data=fields_data, multiple_entities=multiple,
        article_field=article_field, article_parts=article_parts, save_directory=save_directory,
    )
    with open(yml_output_path, "w", encoding="utf-8") as f:
        f.write(yml_content)

    config = {
        "json_path": json_path, "dataset": curated_dataset_path, "save_dir": save_directory,
        "fields": fields_data, "multiple": multiple, "article_field": article_field,
        "article_parts": article_parts,
    }
    log_path = os.path.join(sub_dir, f"build_dataset_{ts}.json")
    create_initial_log(name=task_name or ts, data=config, log_path=log_path)
    _launch(("src.cli.cli_handler", "run_build_optm_set"), config, log_path)
    return {"ok": True, "timestamp": ts, "config": config, "module": "build-dataset"}


def run_prompt_optimization(payload):
    dataset_path = (payload.get("dataset_path") or "").strip()
    save_dir = (payload.get("save_dir") or "").strip()
    if not dataset_path:
        return {"ok": False, "error": "Dataset Path is required!"}
    if not save_dir:
        return {"ok": False, "error": "Save Directory is required!"}

    input_fields = parse_fields(payload.get("inputFields"))
    output_fields = parse_fields(payload.get("outputFields"))
    initial_prompt = payload.get("initial_prompt") or "Extract the target information from the given text"
    task = payload.get("task") or "Extraction"
    optim_burden = payload.get("optim_burden") or "medium"
    threads = int(payload.get("threads") or 6)
    demos = int(payload.get("demos") or 1)
    multiple = bool(payload.get("multiple"))
    ai_evaluation = bool(payload.get("ai_evaluation"))
    task_name = (payload.get("task_name") or "").strip()

    sub_dir, ts = _make_run_subdir("prompt_optimization")
    yaml_filename = f"optim_{ts}.yml"
    save_prompt_optimization_config_to_yaml(
        dataset_path=dataset_path, save_dir=save_dir, experiment_name=task_name,
        input_fields_data=input_fields, output_fields_data=output_fields,
        initial_prompt=initial_prompt, task=task, optim_burden=optim_burden,
        threads=threads, demos=demos, multiple=multiple, ai_evaluation=ai_evaluation,
        recall_prior=False, output_dir=sub_dir, filename=yaml_filename,
    )
    config = {
        "inputFields": input_fields, "outputFields": output_fields, "initial_prompt": initial_prompt,
        "dataset": dataset_path, "save_dir": save_dir, "task": task, "optim_burden": optim_burden,
        "threads": threads, "demos": demos, "multiple": multiple, "ai_evaluation": ai_evaluation,
    }
    log_path = os.path.join(sub_dir, f"optim_{ts}.json")
    create_initial_log(name=task_name or ts, data=config, log_path=log_path)
    _launch(("src.cli.cli_handler", "run_optim_custom"), config, log_path)
    return {"ok": True, "timestamp": ts, "config": config, "module": "prompt-optimization"}


def run_table_parsing(payload):
    folder_path = (payload.get("file_folder_path") or "").strip()
    save_path = (payload.get("save_folder_path") or "").strip()
    if not folder_path:
        return {"ok": False, "error": "Source Folder Path is required!"}
    if not save_path:
        return {"ok": False, "error": "Save Path is required!"}
    os.makedirs(save_path, exist_ok=True)

    file_type = payload.get("non_tabular_file_format") or "PDF"
    task_name = (payload.get("task_name") or "").strip()

    sub_dir, ts = _make_run_subdir("table_parsing")
    yaml_filename = f"parse_table_to_tsv_config_{ts}.yml"
    save_table_parsing_config_to_yaml(
        table_folder_path=folder_path, table_save_path=save_path, table_file_type=file_type,
        output_dir=sub_dir, filename=yaml_filename,
    )
    config = {
        "file_folder_path": folder_path, "save_folder_path": save_path,
        "non_tabular_file_format": file_type,
    }
    log_path = os.path.join(sub_dir, f"parse_table_to_tsv_{ts}.json")
    create_initial_log(name=task_name or ts, data=config, log_path=log_path)
    _launch(("src.cli.cli_handler", "run_parse_table_to_tsv"), config, log_path)
    return {"ok": True, "timestamp": ts, "config": config, "module": "table-parsing"}


def run_table_extraction(payload):
    parse_file_path = (payload.get("parsed_file_path") or "").strip()
    save_dir = (payload.get("save_folder_path") or "").strip()
    if not parse_file_path:
        return {"ok": False, "error": "Parsed File Path is required!"}
    if not save_dir:
        return {"ok": False, "error": "Save Folder Path is required!"}

    output_fields = parse_fields(payload.get("outputFields"))
    classification_prompt = payload.get("classify_prompt") or "Please classify the table content"
    extraction_prompt = payload.get("extract_prompt") or "Please extract the data according to the specified fields"
    threads = int(payload.get("num_threads") or 6)
    task_name = (payload.get("task_name") or "").strip()

    sub_dir, ts = _make_run_subdir("table_extraction")
    yaml_filename = f"extract_table_service_config_{ts}.yml"
    save_extract_table_config_to_yaml(
        extract_parsed_file_path=parse_file_path, extract_save_folder_path=save_dir,
        extract_output_fields=output_fields, extract_classify_prompt=classification_prompt,
        extract_extract_prompt=extraction_prompt, extract_num_threads=threads,
        output_dir=sub_dir, filename=yaml_filename,
    )
    config = {
        "parsed_file_path": parse_file_path, "save_folder_path": save_dir,
        "outputFields": output_fields, "classify_prompt": classification_prompt,
        "extract_prompt": extraction_prompt, "extract_directly": False, "num_threads": threads,
    }
    log_path = os.path.join(sub_dir, f"extract_table_service_{ts}.json")
    create_initial_log(name=task_name or ts, data=config, log_path=log_path)
    _launch(("src.cli.cli_handler", "run_extract_table_service"), config, log_path)
    return {"ok": True, "timestamp": ts, "config": config, "module": "table-extraction"}


# model config — synchronous (no subprocess), mirrors gui behavior
def save_model_config(payload):
    from src.cli.cli_handler import modify_model
    complete = dict(payload)
    safe = {k: v for k, v in payload.items() if k != "api_key"}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_base = os.path.join(runs_root(), "model_config")
    sub_dir = os.path.join(runs_base, f"run_{ts}")
    os.makedirs(sub_dir, exist_ok=True)
    yaml_filename = f"{payload.get('model_usage', 'main')}_config_{ts}.yml"
    yaml_output_path = save_model_config_to_yaml(
        model_name=payload.get("model_name"), model_type=payload.get("model_type"),
        api_base=payload.get("api_base"), api_key=None, model_usage=payload.get("model_usage"),
        temperature=payload.get("temperature"), max_tokens=payload.get("max_tokens") or 2500,
        top_p=payload.get("top_p"), top_k=payload.get("top_k"), min_p=payload.get("min_p"),
        output_dir=sub_dir, filename=yaml_filename,
    )
    log_path = os.path.join(sub_dir, f"modify_model_{ts}.json")
    create_initial_log(name=ts, data=safe, log_path=log_path)
    result = run_task_and_update_log(callable_obj=modify_model, data=complete, log_path=log_path)
    return {"ok": result.get("status") == "succeed", "yaml_path": yaml_output_path,
            "result": result, "timestamp": ts}


def test_model_connection(payload):
    from src.cli.cli_handler import run_model_test_call
    cfg = dict(payload)
    cfg.setdefault("prompt", "Hello")
    try:
        result = run_model_test_call(cfg)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ----------------------------------------------------------------------
# status / cancel / read helpers
# ----------------------------------------------------------------------
RUN_HANDLERS = {
    "doc-parsing": run_document_parsing,
    "original-extraction": run_original_extraction,
    "optimized-extraction": run_optimized_extraction,
    "build-dataset": run_build_dataset,
    "prompt-optimization": run_prompt_optimization,
    "table-parsing": run_table_parsing,
    "table-extraction": run_table_extraction,
}


def dispatch(module_key, payload):
    handler = RUN_HANDLERS.get(module_key)
    if not handler:
        return {"ok": False, "error": f"Unknown module: {module_key}"}
    return handler(payload)


def cancel(log_path):
    return cancel_task(log_path)


def read_run_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_log_tail(path, max_chars=200000):
    if not os.path.exists(path):
        return None
    try:
        size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if size > max_chars:
                f.seek(size - max_chars)
                f.readline()
            return f.read()
    except Exception:
        return None


# ----------------------------------------------------------------------
# writeback — turn a stored run config (data dict) into form values
# ----------------------------------------------------------------------
def writeback_for(module_key, data):
    """Return a dict of {form_field_id: value} suitable for filling the form.
    Field-array keys (inputFields / outputFields / fields) are returned as-is
    so the browser can rebuild the dynamic field blocks.
    """
    d = data or {}
    if module_key == "doc-parsing":
        return {
            "task_name": "", "folder_path": d.get("folder_path", ""),
            "save_path": d.get("save_path", ""), "file_type": d.get("file_type", "PDF"),
            "convert_mode": d.get("convert_mode", "byPart"),
        }
    if module_key == "original-extraction":
        return {
            "task_name": "", "dataset_path": d.get("dataset", ""), "save_dir": d.get("save_dir", ""),
            "initial_prompt": d.get("initial_prompt", ""), "judging": d.get("judging", ""),
            "task": d.get("task", "Extraction"), "threads": d.get("threads", 6),
            "multiple": bool(d.get("multiple")),
            "inputFields": d.get("inputFields", []), "outputFields": d.get("outputFields", []),
        }
    if module_key == "optimized-extraction":
        return {
            "task_name": "", "load_dir": d.get("load_dir", ""), "dataset_path": d.get("dataset", ""),
            "save_dir": d.get("save_dir", ""), "judging": d.get("judging", ""),
            "threads": d.get("threads", 6),
        }
    if module_key == "prompt-optimization":
        return {
            "task_name": "", "dataset_path": d.get("dataset", ""), "save_dir": d.get("save_dir", ""),
            "initial_prompt": d.get("initial_prompt", ""), "task": d.get("task", "Extraction"),
            "optim_burden": d.get("optim_burden", "medium"), "threads": d.get("threads", 6),
            "demos": d.get("demos", 1), "multiple": bool(d.get("multiple")),
            "ai_evaluation": bool(d.get("ai_evaluation")),
            "inputFields": d.get("inputFields", []), "outputFields": d.get("outputFields", []),
        }
    if module_key == "build-dataset":
        return {
            "task_name": "", "json_path": d.get("json_path", ""),
            "curated_dataset_path": d.get("dataset", ""), "save_directory": d.get("save_dir", ""),
            "multiple": bool(d.get("multiple")), "article_field": d.get("article_field", "article_field"),
            "article_parts": d.get("article_parts", []), "fields": d.get("fields", []),
        }
    if module_key == "table-parsing":
        return {
            "task_name": "", "file_folder_path": d.get("file_folder_path", ""),
            "save_folder_path": d.get("save_folder_path", ""),
            "non_tabular_file_format": d.get("non_tabular_file_format", "PDF"),
        }
    if module_key == "table-extraction":
        return {
            "task_name": "", "parsed_file_path": d.get("parsed_file_path", ""),
            "save_folder_path": d.get("save_folder_path", ""),
            "classify_prompt": d.get("classify_prompt", ""),
            "extract_prompt": d.get("extract_prompt", ""), "num_threads": d.get("num_threads", 6),
            "outputFields": d.get("outputFields", []),
        }
    return {}
