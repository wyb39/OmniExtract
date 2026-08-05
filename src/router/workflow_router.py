"""HTTP endpoints and process orchestration for web workflows."""

from __future__ import annotations

import json
import hashlib
import hmac
import os
import queue
import secrets
import shutil
import threading
import zipfile
from datetime import datetime
from multiprocessing import Process
from typing import Any, Dict, Iterable

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

from src.common import baseUtil
from src.workflow.workflow_service import (
    new_workspace,
    run_workflow_doc_extraction,
    run_workflow_doc_extraction_optimized,
    run_workflow_prompt_optimization,
    run_workflow_table_extraction,
)


router = APIRouter()
root_dir = os.path.abspath(baseUtil.get_root_path())
process_dir = os.path.join(root_dir, "process")
templates = Jinja2Templates(directory=os.path.join(root_dir, "ui_jinja", "templates"))
templates_v2 = Jinja2Templates(directory=os.path.join(root_dir, "ui_jinja_v2", "templates"))
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


def _new_workspace() -> tuple[str, str, str]:
    workflow_id, workspace = new_workspace()
    access_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
    with open(os.path.join(workspace, "workflow_access.json"), "w", encoding="utf-8") as handle:
        json.dump({"token_hash": token_hash}, handle)
    with open(os.path.join(workspace, "workflow_status.json"), "w", encoding="utf-8") as handle:
        json.dump({"status": "queued", "workflow_id": workflow_id}, handle)
    return workflow_id, workspace, access_token


def _save_upload(upload: UploadFile, destination: str) -> None:
    with open(destination, "wb") as target:
        shutil.copyfileobj(upload.file, target)


def _resolve_input_file(
    upload: UploadFile | None,
    path: str | None,
    destination: str,
    *,
    is_zip: bool = False,
    file_type: str | None = None,
    require_document: bool = False,
) -> str:
    """Resolve a workflow input from either an uploaded file or a server-side path.

    ``upload`` takes precedence when provided; otherwise ``path`` is used in
    place (read-only). At least one of the two must be supplied so existing UI
    uploads keep working while scripted callers may pass a file path instead.
    """
    if upload is not None:
        _save_upload(upload, destination)
        resolved = destination
    elif path:
        resolved = os.path.abspath(path)
        if not os.path.isfile(resolved):
            raise HTTPException(status_code=400, detail=f"Input file not found: {path}")
    else:
        raise HTTPException(
            status_code=400,
            detail="No input file provided; supply either an uploaded file or a file path",
        )
    if is_zip:
        _validate_zip(resolved, file_type, require_document)
    return resolved


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


def _workflow_url(request: Request, workflow_id: str, access_token: str, route_name: str) -> str:
    return str(request.url_for(route_name, workflow_id=workflow_id).include_query_params(token=access_token))


def _started_response(request: Request, workflow_id: str, access_token: str, detail: str) -> Dict[str, Any]:
    return {
        "status_code": 200,
        "detail": detail,
        "workflow_id": workflow_id,
        "workflow_url": _workflow_url(request, workflow_id, access_token, "workflow_status_page"),
        "status_url": _workflow_url(request, workflow_id, access_token, "workflow_status"),
    }


def _workspace(workflow_id: str) -> str:
    if os.path.basename(workflow_id) != workflow_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    path = os.path.abspath(os.path.join(process_dir, workflow_id))
    if not _is_within(os.path.abspath(process_dir), path):
        raise HTTPException(status_code=404, detail="Workflow not found")
    return path


def _is_within(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


def _authorize(workflow_id: str, token: str | None) -> str:
    workspace = _workspace(workflow_id)
    if not token:
        raise HTTPException(status_code=403, detail="A workflow access token is required")
    access_path = os.path.join(workspace, "workflow_access.json")
    try:
        with open(access_path, "r", encoding="utf-8") as handle:
            expected = json.load(handle).get("token_hash", "")
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="Workflow not found")
    actual = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise HTTPException(status_code=403, detail="Invalid workflow access token")
    return workspace


def _public_status(workflow_id: str, raw: Dict[str, Any], request: Request, token: str) -> Dict[str, Any]:
    result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
    workspace = _workspace(workflow_id)
    artifacts = []
    for key in (
        "result_zip",
        "optimization_config_zip",
        "format_tables_zip",
        "processing_report",
    ):
        value = result.get(key)
        if not isinstance(value, str):
            continue
        absolute = os.path.abspath(value)
        if not _is_within(workspace, absolute) or not os.path.isfile(absolute):
            continue
        relative = os.path.relpath(absolute, workspace).replace(os.sep, "/")
        artifact_url = str(
            request.url_for(
                "workflow_artifact",
                workflow_id=workflow_id,
                artifact_path=relative,
            ).include_query_params(token=token)
        )
        artifacts.append({
            "name": os.path.basename(relative),
            "path": relative,
            "url": artifact_url,
        })
    public = {key: raw[key] for key in ("status", "workflow_id", "workflow_type", "task_name", "updated_at", "error") if key in raw}
    public["workflow_id"] = workflow_id
    report_path = result.get("processing_report")
    if isinstance(report_path, str) and os.path.isfile(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            public["processing_status"] = report.get("processing_status")
            public["failed_documents"] = report.get("failed_documents", [])
        except (OSError, json.JSONDecodeError):
            public["processing_status"] = result.get("processing_status", "failed")
            public["failed_documents"] = []
    elif "processing_status" in result:
        public["processing_status"] = result["processing_status"]
        public["failed_documents"] = []
    public["artifacts"] = artifacts
    return public


@router.post("/api/run_workflow_doc_extraction")
async def run_workflow_doc_extraction_api(
    request: Request,
    source_zip: UploadFile | None = File(None),
    source_zip_path: str = Form(""),
    config: str = Form(...),
):
    config_data = _parse_config(config)
    _validate_input_fields(config_data)
    workflow_id, workspace, access_token = _new_workspace()
    file_type = config_data.get("fileType", "PDF")
    zip_path = _resolve_input_file(
        source_zip,
        source_zip_path or None,
        os.path.join(workspace, "upload.zip"),
        is_zip=True,
        file_type=file_type,
        require_document=True,
    )
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
    return _started_response(request, workflow_id, access_token, "Document extraction workflow started")


@router.post("/api/run_workflow_table_extraction")
async def run_workflow_table_extraction_api(
    request: Request,
    source_zip: UploadFile | None = File(None),
    source_zip_path: str = Form(""),
    config: str = Form(...),
):
    config_data = _parse_config(config)
    workflow_id, workspace, access_token = _new_workspace()
    zip_path = _resolve_input_file(
        source_zip,
        source_zip_path or None,
        os.path.join(workspace, "upload.zip"),
        is_zip=True,
    )
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
    return _started_response(request, workflow_id, access_token, "Table extraction workflow started")


@router.post("/api/run_workflow_prompt_optimization")
async def run_workflow_prompt_optimization_api(
    request: Request,
    source_zip: UploadFile | None = File(None),
    source_zip_path: str = Form(""),
    dataset_file: UploadFile | None = File(None),
    dataset_file_path: str = Form(""),
    config: str = Form(...),
):
    config_data = _parse_config(config)
    _validate_input_fields(config_data)
    workflow_id, workspace, access_token = _new_workspace()
    file_type = config_data.get("fileType", "PDF")
    zip_path = _resolve_input_file(
        source_zip,
        source_zip_path or None,
        os.path.join(workspace, "upload.zip"),
        is_zip=True,
        file_type=file_type,
        require_document=True,
    )
    dataset_path = _resolve_input_file(
        dataset_file,
        dataset_file_path or None,
        os.path.join(workspace, "dataset.json"),
    )
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
    return _started_response(request, workflow_id, access_token, "Prompt optimization workflow queued")


@router.post("/api/run_workflow_doc_extraction_optimized")
async def run_workflow_doc_extraction_optimized_api(
    request: Request,
    config_zip: UploadFile | None = File(None),
    config_zip_path: str = Form(""),
    source_zip: UploadFile | None = File(None),
    source_zip_path: str = Form(""),
    config: str = Form(...),
):
    config_data = _parse_config(config)
    workflow_id, workspace, access_token = _new_workspace()
    file_type = config_data.get("fileType", "PDF")
    source_zip_path_resolved = _resolve_input_file(
        source_zip,
        source_zip_path or None,
        os.path.join(workspace, "upload_data.zip"),
        is_zip=True,
        file_type=file_type,
        require_document=True,
    )
    config_zip_path_resolved = _resolve_input_file(
        config_zip,
        config_zip_path or None,
        os.path.join(workspace, "upload_config.zip"),
        is_zip=True,
    )
    _launch(
        run_workflow_doc_extraction_optimized,
        {
            "task_name": config_data.get("taskName", workflow_id),
            "contact_email": config_data.get("contactEmail", ""),
            "file_type": file_type,
            "zip_file_path": source_zip_path_resolved,
            "config_zip_path": config_zip_path_resolved,
            "convert_mode": config_data.get("convertMode", "byPart"),
            "base_path": workspace,
            "judging_mode": config_data.get("judging", "confidence"),
            "threads": int(config_data.get("threads", 6)),
        },
    )
    return _started_response(request, workflow_id, access_token, "Optimized document extraction workflow started")


@router.get("/api/workflow/{workflow_id}/status")
def workflow_status(request: Request, workflow_id: str, token: str | None = None):
    workspace = _authorize(workflow_id, token)
    status_path = os.path.join(workspace, "workflow_status.json")
    if not os.path.isfile(status_path):
        raise HTTPException(status_code=404, detail="Workflow not found")
    with open(status_path, "r", encoding="utf-8") as handle:
        return _public_status(workflow_id, json.load(handle), request, token or "")


@router.get("/workflow/{workflow_id}", name="workflow_status_page")
def workflow_status_page(request: Request, workflow_id: str, token: str | None = None):
    workspace = _authorize(workflow_id, token)
    status_path = os.path.join(workspace, "workflow_status.json")
    if not os.path.isfile(status_path):
        raise HTTPException(status_code=404, detail="Workflow not found")
    with open(status_path, "r", encoding="utf-8") as handle:
        status = _public_status(workflow_id, json.load(handle), request, token or "")
    return templates.TemplateResponse(
        "workflow_status.html",
        {"request": request, "status": status, "workflow_id": workflow_id},
    )


# v2 preview of the status page. Identical data and token authorization as the
# route above; only the rendered template differs.
@router.get("/v2/workflow/{workflow_id}", name="workflow_status_page_v2")
def workflow_status_page_v2(request: Request, workflow_id: str, token: str | None = None):
    workspace = _authorize(workflow_id, token)
    status_path = os.path.join(workspace, "workflow_status.json")
    if not os.path.isfile(status_path):
        raise HTTPException(status_code=404, detail="Workflow not found")
    with open(status_path, "r", encoding="utf-8") as handle:
        status = _public_status(workflow_id, json.load(handle), request, token or "")
    return templates_v2.TemplateResponse(
        "workflow_status.html",
        {"request": request, "status": status, "workflow_id": workflow_id},
    )


@router.get("/api/workflow/{workflow_id}/artifact/{artifact_path:path}", name="workflow_artifact")
def workflow_artifact(request: Request, workflow_id: str, artifact_path: str, token: str | None = None):
    workspace = _authorize(workflow_id, token)
    relative = artifact_path.replace("\\", "/").lstrip("/")
    candidate = os.path.abspath(os.path.join(workspace, relative))
    if not _is_within(workspace, candidate) or not os.path.isfile(candidate):
        raise HTTPException(status_code=404, detail="Artifact not found")
    with open(os.path.join(workspace, "workflow_status.json"), "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    public = _public_status(workflow_id, raw, request, token or "")
    if not any(item["path"] == relative for item in public["artifacts"]):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path=candidate, filename=os.path.basename(candidate))


@router.get("/api/download/{module}/{filename}")
async def download_file(module: str, filename: str):
    allowed_modules = {"doc_extraction", "prompt_optimization", "table_extraction"}
    if module not in allowed_modules or os.path.basename(filename) != filename:
        raise HTTPException(status_code=400, detail="Invalid download path")
    file_path = os.path.abspath(os.path.join(root_dir, "upload_data", module, filename))
    upload_root = os.path.abspath(os.path.join(root_dir, "upload_data", module))
    if os.path.commonpath([upload_root, file_path]) != upload_root or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename)
