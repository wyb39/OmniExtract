"""Background workflow implementations used by the web API.

The workflows intentionally call the same service functions as the CLI.  PDF
input therefore follows the production Hybrid/OpenDoc parser configured in
``articleUtil``; there is no separate parser-specific web path here.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Tuple

from loguru import logger

from src.common import baseUtil

from src.utils.evalUtil import PredictionSettings
from src.utils.optimUtil import DspyField, OptimSettings
from src.service.service import (
    build_optm_set,
    extract_table_service,
    file_to_json,
    optim_custom,
    parse_table_to_tsv,
    pred,
)
from src.common.error_handling import (
    REPORT_FILENAME,
    ReportedTaskError,
    merge_report_files,
    read_report,
    write_failure_and_wrap,
    write_report,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_workspace() -> Tuple[str, str]:
    """Create a fresh workspace directory under ``process/``.

    Returns ``(workflow_id, workspace_path)``. The workspace is an empty folder
    that workflow runs use as ``base_path`` for status files and artifacts.
    Shared by the HTTP router (which then adds access tokens) and the CLI.
    """
    root_dir = os.path.abspath(baseUtil.get_root_path())
    process_dir = os.path.join(root_dir, "process")
    os.makedirs(process_dir, exist_ok=True)
    workflow_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    workspace = os.path.join(process_dir, workflow_id)
    os.makedirs(workspace, exist_ok=False)
    return workflow_id, workspace


def _write_status(base_path: str, status: str, **details: Any) -> None:
    os.makedirs(base_path, exist_ok=True)
    payload = {"status": status, "updated_at": _now(), **details}
    with open(os.path.join(base_path, "workflow_status.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _notify(
    workflow_type: str,
    contact_email: str,
    task_name: str,
    result: Dict[str, Any],
    attachments: Iterable[str] = (),
) -> None:
    if not contact_email:
        return
    try:
        from src.workflow.workflow_notifications import send_workflow_notification

        send_workflow_notification(
            recipient_email=contact_email,
            workflow_type=workflow_type,
            task_name=task_name,
            result=result,
            attachment_paths=list(attachments),
        )
    except Exception as exc:  # notification failure must not fail a workflow
        logger.warning("Workflow notification failed: {}", exc)


def _safe_extract(
    archive_path: str,
    target_dir: str,
    allowed_extensions: Iterable[str] | None = None,
) -> List[str]:
    """Extract an uploaded archive without allowing path traversal."""

    allowed = {ext.lower() for ext in allowed_extensions} if allowed_extensions else None
    extracted: List[str] = []
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise zipfile.BadZipFile(f"Corrupt file in archive: {bad_file}")
        root = os.path.abspath(target_dir)
        for info in archive.infolist():
            if info.is_dir():
                continue
            filename = os.path.basename(info.filename)
            if not filename or filename.startswith(".") or "__MACOSX" in info.filename:
                continue
            extension = os.path.splitext(filename)[1].lower()
            if allowed is not None and extension not in allowed:
                continue
            destination = os.path.abspath(os.path.join(root, filename))
            if os.path.commonpath([root, destination]) != root:
                raise ValueError("Archive contains an unsafe path")
            with archive.open(info) as source, open(destination, "wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(destination)
    return extracted


def _transform_field(field_data: Dict[str, Any]) -> Dict[str, Any]:
    field = {
        "name": field_data.get("name"),
        "field_type": field_data.get("type", field_data.get("field_type", "str")),
        "description": field_data.get("description", ""),
    }
    if field_data.get("hasRange") or "range_min" in field_data or "range_max" in field_data:
        for frontend_key, model_key in (("rangeMin", "range_min"), ("rangeMax", "range_max")):
            value = field_data.get(frontend_key, field_data.get(model_key))
            if value not in (None, ""):
                try:
                    field[model_key] = float(value)
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid numeric value for {frontend_key}")
    literal_value = field_data.get("literalList", field_data.get("literal_list"))
    if field_data.get("hasLiteral") or literal_value:
        if isinstance(literal_value, str):
            literal_value = [item.strip() for item in literal_value.split(",") if item.strip()]
        field["literal_list"] = literal_value or []
    return field


def _fields(values: Iterable[Dict[str, Any]]) -> List[DspyField]:
    return [DspyField(**_transform_field(value)) for value in values]


def _zip_files(source_dir: str, archive_path: str) -> str:
    archive_absolute = os.path.abspath(archive_path)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(source_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                if os.path.abspath(full_path) == archive_absolute:
                    continue
                archive.write(full_path, os.path.relpath(full_path, source_dir))
    return archive_path


def _run_workflow(
    workflow_type: str,
    task_name: str,
    contact_email: str,
    base_path: str,
    work: Callable[[], Dict[str, Any]],
    attachments: Callable[[Dict[str, Any]], Iterable[str]] = lambda result: (),
) -> Dict[str, Any]:
    workflow_id = os.path.basename(base_path)
    _write_status(base_path, "running", workflow_id=workflow_id, workflow_type=workflow_type, task_name=task_name)
    try:
        result = dict(work())
        result.setdefault("status", "success")
        result["workflow_id"] = workflow_id
        result["task_created_time"] = workflow_id
        report_path = result.get("processing_report")
        if isinstance(report_path, str) and os.path.isfile(report_path):
            report = read_report(report_path).with_workflow_id(workflow_id)
            write_report(report, report_path)
            result["processing_status"] = report.processing_status
        else:
            result.setdefault("processing_status", "success")
        workflow_status = (
            "failed"
            if result["processing_status"] == "failed"
            else "completed"
        )
        result["status"] = result["processing_status"]
        _write_status(
            base_path,
            workflow_status,
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            task_name=task_name,
            result=result,
        )
        attachment_paths = list(attachments(result))
        if isinstance(report_path, str) and os.path.isfile(report_path):
            attachment_paths.append(report_path)
        _notify(workflow_type, contact_email, task_name, result, attachment_paths)
        return result
    except Exception as exc:
        if isinstance(exc, ReportedTaskError) and os.path.isfile(exc.report_path):
            report_path = exc.report_path
            report = read_report(report_path).with_workflow_id(workflow_id)
            write_report(report, report_path)
        else:
            wrapped = write_failure_and_wrap(
                exc,
                base_path,
                workflow_id=workflow_id,
                stage=workflow_type,
            )
            report_path = wrapped.report_path
            report = read_report(report_path)
        # A later task-fatal error must not hide document failures already
        # recorded by parsing/prediction stages.  Consolidate every report in
        # the workspace into the single failed-workflow artifact.
        report_paths = []
        for root, _, files in os.walk(base_path):
            if REPORT_FILENAME in files:
                report_paths.append(os.path.join(root, REPORT_FILENAME))
        if report_path not in report_paths:
            report_paths.append(report_path)
        report_path = merge_report_files(
            report_paths,
            os.path.join(base_path, REPORT_FILENAME),
            workflow_id=workflow_id,
            terminal_report_path=report_path,
        )
        report = read_report(report_path)
        failed_result = {
            "status": "failed",
            "workflow_id": workflow_id,
            "processing_status": report.processing_status,
            "processing_report": report_path,
        }
        _write_status(
            base_path,
            "failed",
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            task_name=task_name,
            error=str(exc),
            result=failed_result,
        )
        _notify(
            workflow_type,
            contact_email,
            task_name,
            failed_result,
            [report_path],
        )
        logger.exception("{} workflow failed", workflow_type)
        raise


def run_workflow_doc_extraction(
    task_name: str,
    contact_email: str,
    file_type: str,
    zip_file_path: str,
    convert_mode: str,
    input_fields: List[Dict[str, Any]],
    output_fields: List[Dict[str, Any]],
    base_path: str,
    initial_prompt: str = "",
    judging_mode: str = "confidence",
    threads: int = 6,
    multiple_entities: bool = False,
) -> Dict[str, Any]:
    def work() -> Dict[str, Any]:
        source_dir = os.path.join(base_path, "source_file")
        parsed_dir = os.path.join(base_path, "parsed")
        target_dir = os.path.join(base_path, "target")
        _safe_extract(zip_file_path, source_dir, {".pdf", ".xml"})
        parse_result = file_to_json(source_dir, parsed_dir, file_type, convert_mode)
        dataset_file = parse_result["details"]["dataset_file"]
        settings = PredictionSettings(
            inputFields=_fields(input_fields),
            outputFields=_fields(output_fields),
            dataset=dataset_file,
            save_dir=target_dir,
            output_file="result.json",
            task="Extraction",
            initial_prompt=initial_prompt,
            judging=judging_mode,
            threads=threads,
            multiple=multiple_entities,
        )
        prediction_result = pred(settings)
        result_file = os.path.join(target_dir, "result.json")
        processing_report = merge_report_files(
            [
                parse_result.get("processing_report"),
                prediction_result.get("processing_report"),
            ],
            os.path.join(target_dir, REPORT_FILENAME),
            workflow_id=os.path.basename(base_path),
        )
        result_zip = _zip_files(target_dir, os.path.join(target_dir, "result.zip"))
        return {
            "result_file": result_file,
            "result_zip": result_zip,
            "prediction_result": prediction_result,
            "processing_report": processing_report,
        }

    return _run_workflow("doc_extraction", task_name, contact_email, base_path, work, lambda result: [result["result_zip"]])


def run_workflow_prompt_optimization(
    task_name: str,
    contact_email: str,
    file_type: str,
    zip_file_path: str,
    dataset_file_path: str,
    convert_mode: str,
    input_fields: List[Dict[str, Any]],
    output_fields: List[Dict[str, Any]],
    base_path: str,
    initial_prompt: str = "",
    demos: int = 1,
    article_field: str = "article_field",
    multiple_entities: bool = False,
) -> Dict[str, Any]:
    def work() -> Dict[str, Any]:
        source_dir = os.path.join(base_path, "source_file")
        parsed_dir = os.path.join(base_path, "parsed")
        optimized_dir = os.path.join(base_path, "optimized_prompt")
        _safe_extract(zip_file_path, source_dir, {".pdf", ".xml"})
        parse_result = file_to_json(
            source_dir,
            parsed_dir,
            file_type,
            convert_mode,
        )
        input_dspy_fields = _fields(input_fields)
        output_dspy_fields = _fields(output_fields)
        article_parts = [field.name for field in input_dspy_fields] if convert_mode == "byPart" else None
        dataset_result = build_optm_set(
            json_path=parsed_dir,
            dataset=dataset_file_path,
            save_dir=optimized_dir,
            fields=output_dspy_fields,
            multiple=multiple_entities,
            article_field=article_field,
            article_parts=article_parts,
        )
        optim_dataset = os.path.join(optimized_dir, "_optim_dataset.json")
        settings = OptimSettings(
            inputFields=input_dspy_fields,
            outputFields=output_dspy_fields,
            dataset=optim_dataset,
            save_dir=optimized_dir,
            task="Extraction",
            initial_prompt=initial_prompt,
            demos=demos,
            multiple=multiple_entities,
            threads=6,
            ai_evaluation=True,
        )
        optimization_result = optim_custom(settings)
        processing_report = merge_report_files(
            [
                parse_result.get("processing_report"),
                dataset_result.get("processing_report"),
                optimization_result.get("processing_report"),
            ],
            os.path.join(optimized_dir, REPORT_FILENAME),
            workflow_id=os.path.basename(base_path),
        )
        optimization_zip = _zip_files(optimized_dir, os.path.join(optimized_dir, "optimization_config.zip"))
        return {
            "optimized_prompt_dir": optimized_dir,
            "optimization_config_zip": optimization_zip,
            "result": optimization_result,
            "processing_report": processing_report,
        }

    return _run_workflow("prompt_optimization", task_name, contact_email, base_path, work, lambda result: [result["optimization_config_zip"]])


def run_workflow_table_extraction(
    task_name: str,
    contact_email: str,
    file_type: str,
    zip_file_path: str,
    output_fields: List[Dict[str, Any]],
    base_path: str,
    classify_prompt: str,
    extract_prompt: str,
    threads: int = 6,
) -> Dict[str, Any]:
    def work() -> Dict[str, Any]:
        source_dir = os.path.join(base_path, "source_file")
        parsed_dir = os.path.join(base_path, "parsed")
        result_dir = os.path.join(base_path, "result")
        _safe_extract(zip_file_path, source_dir)
        parse_result = parse_table_to_tsv(
            source_dir,
            parsed_dir,
            non_tabular_file_format=file_type,
            verbose=True,
        )
        extraction_result = extract_table_service(
            parsed_file_path=parsed_dir,
            save_folder_path=result_dir,
            outputFields=_fields(output_fields),
            classify_prompt=classify_prompt,
            extract_prompt=extract_prompt,
            num_threads=threads,
        )
        processing_report = merge_report_files(
            [
                parse_result.get("processing_report"),
                extraction_result.get("processing_report"),
            ],
            os.path.join(result_dir, REPORT_FILENAME),
            workflow_id=os.path.basename(base_path),
        )
        format_tables_zip = os.path.join(result_dir, "format_tables.zip")
        _zip_files(result_dir, format_tables_zip)
        return {
            "result_dir": result_dir,
            "format_tables_zip": format_tables_zip,
            "processing_report": processing_report,
        }

    return _run_workflow("table_extraction", task_name, contact_email, base_path, work, lambda result: [result["format_tables_zip"]])


def run_workflow_doc_extraction_optimized(
    task_name: str,
    contact_email: str,
    file_type: str,
    zip_file_path: str,
    config_zip_path: str,
    convert_mode: str,
    base_path: str,
    judging_mode: str = "confidence",
    threads: int = 6,
) -> Dict[str, Any]:
    def work() -> Dict[str, Any]:
        source_dir = os.path.join(base_path, "source_file")
        parsed_dir = os.path.join(base_path, "parsed")
        target_dir = os.path.join(base_path, "target")
        config_dir = os.path.join(base_path, "config")
        _safe_extract(zip_file_path, source_dir, {".pdf", ".xml"})
        parse_result = file_to_json(source_dir, parsed_dir, file_type, convert_mode)
        dataset_file = parse_result["details"]["dataset_file"]
        _safe_extract(config_zip_path, config_dir)
        optim_settings_path = _find_file(config_dir, "optim_settings.json")
        prompt_path = _find_file(config_dir, "optim_prompt.json")
        with open(optim_settings_path, "r", encoding="utf-8") as handle:
            settings_data = json.load(handle)
        prediction_settings = PredictionSettings(
            inputFields=_fields(settings_data.get("inputFields", [])),
            outputFields=_fields(settings_data.get("outputFields", [])),
            dataset=dataset_file,
            save_dir=target_dir,
            output_file="result.json",
            task=settings_data.get("task", "Extraction"),
            initial_prompt=settings_data.get("initial_prompt", ""),
            judging=judging_mode,
            threads=threads,
            multiple=settings_data.get("multiple", False),
        )
        prediction_result = pred(prediction_settings, prompt_dir=prompt_path)
        result_file = os.path.join(target_dir, "result.json")
        processing_report = merge_report_files(
            [
                parse_result.get("processing_report"),
                prediction_result.get("processing_report"),
            ],
            os.path.join(target_dir, REPORT_FILENAME),
            workflow_id=os.path.basename(base_path),
        )
        result_zip = _zip_files(target_dir, os.path.join(target_dir, "result.zip"))
        return {
            "result_file": result_file,
            "result_zip": result_zip,
            "prediction_result": prediction_result,
            "processing_report": processing_report,
        }

    return _run_workflow("doc_extraction_optimized", task_name, contact_email, base_path, work, lambda result: [result["result_zip"]])


def _find_file(root_dir: str, filename: str) -> str:
    for root, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found in {root_dir}")
