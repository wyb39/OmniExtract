"""Production error reporting and document-level batch isolation.

This module deliberately contains the complete public error contract.  Keep
integration code in service/workflow modules thin: a user-facing report has
exactly ``workflow_id``, ``processing_status`` and ``failed_documents``.
Tracebacks remain in application logs and are never copied into the report.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from token_usage import TokenUsage, current_token_usage


REPORT_FILENAME = "processing_report.json"
_T = TypeVar("_T")
_R = TypeVar("_R")

_SECRETS = (
    re.compile(
        r"(?i)\b(api[_-]?key|token|authorization)\s*[:=]\s*"
        r"(?:(?:bearer|basic|token)\s+)?[^\s,;]+"
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def _public_text(value: Any, limit: int = 500) -> str:
    """Remove common credentials and bound text copied from an exception."""

    text = str(value).strip() or type(value).__name__
    for pattern in _SECRETS:
        text = pattern.sub(
            lambda match: (
                f"{match.group(1)}=<redacted>"
                if match.lastindex
                else "<redacted>"
            ),
            text,
        )
    return text[:limit]


@dataclass(frozen=True)
class Issue:
    """One actionable problem that changed a single document/task result."""

    stage: str
    code: str
    message: str
    action: str
    retryable: bool = False
    component_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "action": self.action,
            "retryable": self.retryable,
        }
        if self.component_id:
            value["component_id"] = self.component_id
        return value


class DocumentStageError(RuntimeError):
    """Carry the exact document stage across the generic batch executor."""

    def __init__(self, stage: str, error: BaseException) -> None:
        self.stage = stage
        self.original = error
        super().__init__(str(error))


def map_exception(
    error: BaseException,
    stage: str,
    *,
    component_id: str | None = None,
) -> Issue:
    """Map implementation exceptions to stable, concise user guidance."""

    if isinstance(error, DocumentStageError):
        stage = error.stage
        error = error.original
    text = _public_text(error)
    lowered = f"{type(error).__name__}: {text}".lower()
    retryable = False

    if stage == "optimization" and isinstance(error, FileNotFoundError):
        code, action = "OPTIM_DATASET_NOT_FOUND", "Set 'dataset' to an existing JSON, CSV, TSV or XLSX file."
    elif stage == "optimization" and any(
        token in lowered
        for token in ("at least two", "fewer than two", "too small")
    ):
        code, action = "OPTIM_DATASET_TOO_SMALL", "Provide at least two complete, valid optimization records."
    elif stage == "optimization" and any(
        token in lowered for token in ("dataset is empty", "no usable records")
    ):
        code, action = "OPTIM_DATASET_EMPTY", "Add at least two complete records to the optimization dataset."
    elif isinstance(error, FileNotFoundError):
        code, action = "SOURCE_NOT_FOUND", "Check that the input file exists and upload it again."
    elif isinstance(error, PermissionError):
        code, action = "FILE_ACCESS_DENIED", "Check file and output-directory permissions, then retry."
    elif isinstance(error, (TimeoutError,)) or "timed out" in lowered or "timeout" in lowered:
        code, action, retryable = "MODEL_TIMEOUT", "Check the model service and retry with fewer threads.", True
    elif any(token in lowered for token in ("rate limit", "ratelimit", "429", "too many requests")):
        code, action, retryable = "MODEL_RATE_LIMITED", "Wait and retry, or reduce the number of threads.", True
    elif any(token in lowered for token in ("401", "403", "unauthorized", "authentication", "api key")):
        code, action = "MODEL_AUTH_FAILED", "Check the configured model credentials and permissions."
    elif any(token in lowered for token in ("503", "service unavailable", "connection refused", "connection error")):
        code, action, retryable = "MODEL_UNAVAILABLE", "Check the API base and provider status, then retry.", True
    elif isinstance(error, (json.JSONDecodeError, UnicodeError)):
        code, action = "SOURCE_INVALID", "Check the file format and encoding, then upload a valid file."
    elif isinstance(error, OSError) and stage == "result_save":
        code, action = "OUTPUT_WRITE_FAILED", "Check output-directory permissions, free space and file locks."
    else:
        defaults = {
            "document_parse": ("DOCUMENT_PARSE_FAILED", "Check that the document is valid and matches the selected format."),
            "markdown_convert": ("MARKDOWN_GENERATION_FAILED", "Check the source document or try another supported parser."),
            "json_convert": ("JSON_CONVERSION_FAILED", "Check the generated Markdown and section structure."),
            "prediction": ("PREDICTION_FAILED", "Check the input fields and model service, then retry this document."),
            "judgement": ("JUDGEMENT_FAILED", "Check the judge model settings; the prediction result may still be usable."),
            "table_parse": ("TABLE_PARSE_FAILED", "Check the document/table format and retry this document."),
            "table_extract": ("TABLE_EXTRACTION_FAILED", "Check the table prompts and model service, then retry."),
            "optimization": ("OPTIMIZATION_FAILED", "Check the dataset and optim.log, then retry with fewer threads or demos."),
            "result_save": ("OUTPUT_WRITE_FAILED", "Check output-directory permissions, free space and file locks."),
        }
        code, action = defaults.get(
            stage,
            ("TASK_FAILED", "Check the task input and application log, then retry."),
        )

    return Issue(
        stage=stage,
        code=code,
        message=text,
        action=action,
        retryable=retryable,
        component_id=component_id,
    )


@dataclass
class ProcessingReport:
    """Mutable in-memory aggregate; serialization intentionally stays minimal."""

    workflow_id: str
    succeeded: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    _failed: dict[str, list[Issue]] = field(default_factory=dict)
    _status_override: str | None = field(default=None, repr=False)

    def success(self, count: int = 1) -> None:
        self.succeeded += max(0, int(count))

    def failure(self, document_id: str, issue: Issue) -> None:
        identity = str(document_id).strip() or "workflow"
        self._failed.setdefault(identity, []).append(issue)

    @property
    def processing_status(self) -> str:
        if self._status_override is not None:
            return self._status_override
        if not self._failed:
            return "success"
        return "partial" if self.succeeded else "failed"

    def merge(self, other: "ProcessingReport") -> "ProcessingReport":
        self.succeeded += other.succeeded
        self.token_usage.add(other.token_usage)
        for document_id, issues in other._failed.items():
            self._failed.setdefault(document_id, []).extend(issues)
        return self

    def with_workflow_id(self, workflow_id: str) -> "ProcessingReport":
        self.workflow_id = str(workflow_id)
        return self

    def force_status(self, status: str) -> "ProcessingReport":
        """Set status after an end-to-end pipeline decision."""

        if status not in {"success", "partial", "failed"}:
            raise ValueError(f"Unsupported processing status: {status}")
        self._status_override = status
        return self

    def to_dict(self) -> dict[str, Any]:
        # Dict insertion order is part of the public JSON presentation.
        return {
            "workflow_id": self.workflow_id,
            "processing_status": self.processing_status,
            "failed_documents": [
                {
                    "document_id": document_id,
                    "issues": [issue.to_dict() for issue in issues],
                }
                for document_id, issues in self._failed.items()
            ],
            "token_usage": self.token_usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcessingReport":
        report = cls(str(value.get("workflow_id", "task")))
        report.token_usage = TokenUsage.from_report_dict(value.get("token_usage"))
        status = value.get("processing_status")
        if status in {"success", "partial"}:
            report.success()
        for item in value.get("failed_documents", []):
            if not isinstance(item, Mapping):
                continue
            for issue in item.get("issues", []):
                if not isinstance(issue, Mapping):
                    continue
                report.failure(
                    str(item.get("document_id", "workflow")),
                    Issue(
                        stage=str(issue.get("stage", "task")),
                        code=str(issue.get("code", "TASK_FAILED")),
                        message=str(issue.get("message", "Task failed")),
                        action=str(issue.get("action", "Check the application log and retry.")),
                        retryable=bool(issue.get("retryable", False)),
                        component_id=(
                            str(issue["component_id"])
                            if issue.get("component_id") is not None
                            else None
                        ),
                    ),
                )
        return report


def report_for_failure(
    workflow_id: str,
    error: BaseException,
    stage: str,
    *,
    document_id: str = "workflow",
) -> ProcessingReport:
    report = ProcessingReport(workflow_id)
    report.failure(document_id, map_exception(error, stage))
    return report


def write_report(
    report: ProcessingReport | Mapping[str, Any],
    destination: str | Path,
) -> str:
    """Atomically persist a report so users never receive a partial JSON file."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict() if isinstance(report, ProcessingReport) else dict(report)
    tracked_usage = current_token_usage()
    if tracked_usage is not None:
        if isinstance(report, ProcessingReport):
            report.token_usage = tracked_usage
            payload = report.to_dict()
        else:
            payload["token_usage"] = tracked_usage.to_dict()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return str(path)


def read_report(path: str | Path) -> ProcessingReport:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Processing report must be a JSON object")
    return ProcessingReport.from_dict(value)


def merge_report_files(
    report_paths: Iterable[str | Path | None],
    destination: str | Path,
    *,
    workflow_id: str,
    terminal_report_path: str | Path | None = None,
) -> str:
    """Merge issues while deriving status from the terminal processing stage.

    Earlier parsing success only means a document reached the next stage.  It
    must not turn an all-failed prediction/extraction stage into ``partial``.
    Conversely, terminal success plus an earlier document failure is partial.
    """

    merged = ProcessingReport(workflow_id)
    loaded: list[tuple[Path, ProcessingReport]] = []
    for value in report_paths:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            continue
        report = read_report(path)
        loaded.append((path, report))
        merged.merge(report)
    if not loaded:
        merged.success()
    else:
        terminal: ProcessingReport | None = None
        if terminal_report_path is not None:
            expected = Path(terminal_report_path).expanduser().resolve()
            terminal = next(
                (report for path, report in loaded if path == expected),
                None,
            )
        if terminal is None:
            terminal = loaded[-1][1]

        if terminal.processing_status == "failed":
            merged.force_status("failed")
        elif merged._failed:
            merged.force_status("partial")
        else:
            merged.force_status("success")
    merged.with_workflow_id(workflow_id)
    return write_report(merged, destination)


@dataclass(frozen=True)
class IsolatedResult:
    """Outcome of one item; failures carry no value and never abort siblings."""

    item_id: str
    value: Any = None
    issue: Issue | None = None


def run_isolated(
    items: Sequence[_T] | Iterable[_T],
    worker: Callable[[_T], _R],
    identify: Callable[[_T], str],
    *,
    stage: str,
    workflow_id: str,
    max_workers: int = 1,
) -> tuple[list[IsolatedResult], ProcessingReport]:
    """Execute every item and convert each ordinary exception into one issue."""

    values = list(items)
    if max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    report = ProcessingReport(workflow_id)
    outcomes: list[IsolatedResult | None] = [None] * len(values)

    def execute(index: int, item: _T) -> tuple[int, IsolatedResult]:
        item_id = str(identify(item))
        try:
            return index, IsolatedResult(item_id=item_id, value=worker(item))
        except Exception as error:
            return index, IsolatedResult(
                item_id=item_id,
                issue=map_exception(error, stage),
            )

    if max_workers == 1:
        completed = (execute(index, item) for index, item in enumerate(values))
        for index, outcome in completed:
            outcomes[index] = outcome
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    copy_context().run,
                    execute,
                    index,
                    item,
                ): index
                for index, item in enumerate(values)
            }
            for future in as_completed(futures):
                index, outcome = future.result()
                outcomes[index] = outcome

    final: list[IsolatedResult] = []
    for index, outcome in enumerate(outcomes):
        if outcome is None:  # defensive: one input must always yield one result
            item_id = str(identify(values[index]))
            outcome = IsolatedResult(
                item_id=item_id,
                issue=Issue(
                    stage=stage,
                    code="INTERNAL_DOCUMENT_ERROR",
                    message="The document worker returned no result.",
                    action="Retry the document and provide the application log if it repeats.",
                ),
            )
        final.append(outcome)
        if outcome.issue is None:
            report.success()
        else:
            report.failure(outcome.item_id, outcome.issue)
    return final, report


class ReportedTaskError(RuntimeError):
    """Raised after a task-fatal error has already been written for the user."""

    def __init__(self, error: BaseException, report_path: str, issue: Issue) -> None:
        self.report_path = report_path
        self.issue = issue
        super().__init__(issue.message)


def write_failure_and_wrap(
    error: BaseException,
    output_directory: str | Path,
    *,
    workflow_id: str,
    stage: str,
    document_id: str = "workflow",
) -> ReportedTaskError:
    """Persist a fatal task error beside its expected output before re-raising."""

    issue = map_exception(error, stage)
    report = ProcessingReport(workflow_id)
    report.failure(document_id, issue)
    report_path = write_report(report, Path(output_directory) / REPORT_FILENAME)
    return ReportedTaskError(error, report_path, issue)


__all__ = [
    "DocumentStageError",
    "Issue",
    "IsolatedResult",
    "ProcessingReport",
    "REPORT_FILENAME",
    "ReportedTaskError",
    "map_exception",
    "merge_report_files",
    "read_report",
    "report_for_failure",
    "run_isolated",
    "write_failure_and_wrap",
    "write_report",
]
