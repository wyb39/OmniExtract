"""HTTP endpoints and process orchestration for web workflows."""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import uuid
import zipfile
from datetime import datetime
from multiprocessing import Process
from typing import Any, Dict, Iterable

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import baseUtil
from workflow_service import (
    run_workflow_doc_extraction,
    run_workflow_doc_extraction_optimized,
    run_workflow_prompt_optimization,
    run_workflow_table_extraction,
)


router = APIRouter()
root_dir = os.path.abspath(baseUtil.get_root_path())
process_dir = os.path.join(root_dir, "process")
_optimization_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
_worker_lock = threading.Lock()
_workers_started = False


def start_workflow_workers(worker_count: int = 3) -> None:
    """Start the prompt-optimization worker pool once per application."""

    global _workers_started
    with _worker_lock:
        if _workers_started:
            return
        for _ in range(max(1, worker_count)):
            threading.Thread(target=_optimization_worker, daemon=True).start()
        _workers_started = True


def _optimization_worker() -> None:
    while True:
        kwargs = _optimization_queue.get()
        try:
            process = Process(target=run_workflow_prompt_optimization, kwargs=kwargs)
            process.start()
            process.join()
        finally:
            _optimization_queue.task_done()


def _new_workspace() -> tuple[str, str]:
    os.makedirs(process_dir, exist_ok=True)
    workflow_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    workspace = os.path.join(process_dir, workflow_id)
    os.makedirs(workspace, exist_ok=False)
    with open(os.path.join(workspace, "workflow_status.json"), "w", encoding="utf-8") as handle:
        json.dump({"status": "queued", "workflow_id": workflow_id}, handle)
    return workflow_id, workspace


def _save_upload(upload: UploadFile, destination: str) -> None:
    with open(destination, "wb") as target:
        shutil.copyfileobj(upload.file, target)


def _parse_config(config: str) -> Dict[str, Any]:
    try:
        value = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON workflow configuration") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="Workflow configuration must be a JSON object")
    return value


def _validate_zip(path: str, file_type: str | None = None, require_document: bool = False) -> None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            if archive.testzip() is not None or not archive.namelist():
                raise HTTPException(status_code=400, detail="Invalid or empty zip file")
            if not require_document:
                return
            names = [name for name in archive.namelist() if not name.endswith("/") and not os.path.basename(name).startswith(".")]
            extensions = {os.path.splitext(name)[1].lower() for name in names}
            if file_type == "PDF" and ".pdf" not in extensions:
                raise HTTPException(status_code=400, detail="File type is PDF but no PDF files were found")
            if file_type in {"scienceDirect", "PMC"} and ".xml" not in extensions:
                raise HTTPException(status_code=400, detail=f"File type is {file_type} but no XML files were found")
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Invalid zip file") from exc


def _validate_input_fields(config: Dict[str, Any]) -> None:
    convert_mode = config.get("convertMode")
    fields = config.get("inputFields", [])
    valid_parts = {"Introduction", "Method", "Result", "Discussion", "Conclusion"}
    if convert_mode == "byPart" and any(field.get("name") not in valid_parts for field in fields):
        raise HTTPException(status_code=400, detail="Input fields must be valid article parts for byPart mode")
    if convert_mode == "wholeDoc" and (len(fields) != 1 or fields[0].get("name") != "Document"):
        raise HTTPException(status_code=400, detail="wholeDoc mode requires one input field named Document")


def _launch(target: Any, kwargs: Dict[str, Any]) -> None:
    try:
        process = Process(target=target, kwargs=kwargs)
        process.start()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to start workflow: {exc}") from exc


def _started_response(workflow_id: str, detail: str) -> Dict[str, Any]:
    return {"status_code": 200, "detail": detail, "workflow_id": workflow_id}


@router.post("/api/run_workflow_doc_extraction")
async def run_workflow_doc_extraction_api(
    source_zip: UploadFile = File(...),
    config: str = Form(...),
):
    config_data = _parse_config(config)
    _validate_input_fields(config_data)
    workflow_id, workspace = _new_workspace()
    zip_path = os.path.join(workspace, "upload.zip")
    _save_upload(source_zip, zip_path)
    file_type = config_data.get("fileType", "PDF")
    _validate_zip(zip_path, file_type, require_document=True)
    _launch(
        run_workflow_doc_extraction,
        {
            "task_name": config_data.get("taskName", workflow_id),
            "contact_email": config_data.get("contactEmail", ""),
            "file_type": file_type,
            "zip_file_path": zip_path,
            "convert_mode": config_data.get("convertMode", "byPart"),
            "input_fields": config_data.get("inputFields", []),
            "output_fields": config_data.get("outputFields", []),
            "base_path": workspace,
            "initial_prompt": config_data.get("initialPrompt", ""),
            "judging_mode": config_data.get("judgingMode", "confidence"),
            "threads": int(config_data.get("threads", 6)),
            "multiple_entities": bool(config_data.get("multipleEntities", False)),
        },
    )
    return _started_response(workflow_id, "Document extraction workflow started")


@router.post("/api/run_workflow_table_extraction")
async def run_workflow_table_extraction_api(
    source_zip: UploadFile = File(...),
    config: str = Form(...),
):
    config_data = _parse_config(config)
    workflow_id, workspace = _new_workspace()
    zip_path = os.path.join(workspace, "upload.zip")
    _save_upload(source_zip, zip_path)
    _validate_zip(zip_path)
    _launch(
        run_workflow_table_extraction,
        {
            "task_name": config_data.get("taskName", workflow_id),
            "contact_email": config_data.get("contactEmail", ""),
            "file_type": config_data.get("fileType", "PDF"),
            "zip_file_path": zip_path,
            "output_fields": config_data.get("outputFields", []),
            "base_path": workspace,
            "classify_prompt": config_data.get("classifyPrompt", ""),
            "extract_prompt": config_data.get("extractPrompt", ""),
            "threads": int(config_data.get("threads", 6)),
        },
    )
    return _started_response(workflow_id, "Table extraction workflow started")


@router.post("/api/run_workflow_prompt_optimization")
async def run_workflow_prompt_optimization_api(
    source_zip: UploadFile = File(...),
    dataset_file: UploadFile = File(...),
    config: str = Form(...),
):
    config_data = _parse_config(config)
    _validate_input_fields(config_data)
    workflow_id, workspace = _new_workspace()
    zip_path = os.path.join(workspace, "upload.zip")
    dataset_path = os.path.join(workspace, "dataset.json")
    _save_upload(source_zip, zip_path)
    _save_upload(dataset_file, dataset_path)
    file_type = config_data.get("fileType", "PDF")
    _validate_zip(zip_path, file_type, require_document=True)
    start_workflow_workers()
    _optimization_queue.put(
        {
            "task_name": config_data.get("taskName", workflow_id),
            "contact_email": config_data.get("contactEmail", ""),
            "file_type": file_type,
            "zip_file_path": zip_path,
            "dataset_file_path": dataset_path,
            "convert_mode": config_data.get("convertMode", "byPart"),
            "input_fields": config_data.get("inputFields", []),
            "output_fields": config_data.get("outputFields", []),
            "base_path": workspace,
            "initial_prompt": config_data.get("initialPrompt", ""),
            "demos": int(config_data.get("demos", 1)),
            "article_field": config_data.get("articleField", "article_field"),
            "multiple_entities": bool(config_data.get("multipleEntities", False)),
        }
    )
    return _started_response(workflow_id, "Prompt optimization workflow queued")


@router.post("/api/run_workflow_doc_extraction_optimized")
async def run_workflow_doc_extraction_optimized_api(
    config_zip: UploadFile = File(...),
    source_zip: UploadFile = File(...),
    config: str = Form(...),
):
    config_data = _parse_config(config)
    workflow_id, workspace = _new_workspace()
    source_zip_path = os.path.join(workspace, "upload_data.zip")
    config_zip_path = os.path.join(workspace, "upload_config.zip")
    _save_upload(source_zip, source_zip_path)
    _save_upload(config_zip, config_zip_path)
    file_type = config_data.get("fileType", "PDF")
    _validate_zip(source_zip_path, file_type, require_document=True)
    _validate_zip(config_zip_path)
    _launch(
        run_workflow_doc_extraction_optimized,
        {
            "task_name": config_data.get("taskName", workflow_id),
            "contact_email": config_data.get("contactEmail", ""),
            "file_type": file_type,
            "zip_file_path": source_zip_path,
            "config_zip_path": config_zip_path,
            "convert_mode": config_data.get("convertMode", "byPart"),
            "base_path": workspace,
            "judging_mode": config_data.get("judging", "confidence"),
            "threads": int(config_data.get("threads", 6)),
        },
    )
    return _started_response(workflow_id, "Optimized document extraction workflow started")


@router.get("/api/workflow/{workflow_id}/status")
def workflow_status(workflow_id: str):
    status_path = os.path.join(process_dir, workflow_id, "workflow_status.json")
    if not os.path.isfile(status_path):
        raise HTTPException(status_code=404, detail="Workflow not found")
    with open(status_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@router.get("/api/download/{module}/{filename}")
async def download_file(module: str, filename: str):
    allowed_modules = {"doc_extraction", "prompt_optimization", "table_extraction", "performance_evaluation_dataset"}
    if module not in allowed_modules or os.path.basename(filename) != filename:
        raise HTTPException(status_code=400, detail="Invalid download path")
    file_path = os.path.abspath(os.path.join(root_dir, "upload_data", module, filename))
    upload_root = os.path.abspath(os.path.join(root_dir, "upload_data", module))
    if os.path.commonpath([upload_root, file_path]) != upload_root or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename)
