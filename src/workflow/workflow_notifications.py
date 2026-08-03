"""Optional email notifications for completed workflows.

Notifications are disabled unless SMTP credentials are supplied through
environment variables. No credentials are stored in the repository.
"""

from __future__ import annotations

import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Iterable

from loguru import logger


def send_workflow_notification(
    recipient_email: str,
    workflow_type: str,
    task_name: str,
    result: Dict[str, Any],
    attachment_paths: Iterable[str] = (),
) -> bool:
    sender = os.getenv("OMNI_EXTRACT_SMTP_SENDER")
    password = os.getenv("OMNI_EXTRACT_SMTP_PASSWORD")
    if not sender or not password:
        logger.info("Workflow email skipped: SMTP credentials are not configured")
        return False

    server_name = os.getenv("OMNI_EXTRACT_SMTP_SERVER", "mail.cstnet.cn")
    port = int(os.getenv("OMNI_EXTRACT_SMTP_PORT", "994"))
    use_ssl = os.getenv("OMNI_EXTRACT_SMTP_USE_SSL", "true").lower() not in {"0", "false", "no"}
    status = result.get("status", "unknown")
    subject = f"Workflow {workflow_type} '{task_name}' finished with status {status}"
    body = "\n".join(
        [
            f"Workflow: {workflow_type}",
            f"Task name: {task_name}",
            f"Status: {status}",
            f"Workflow ID: {result.get('workflow_id', '')}",
            str(result.get("error", "")) if status == "failed" else "",
        ]
    )

    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain", "utf-8"))
    for path in attachment_paths:
        if not path or not os.path.isfile(path):
            continue
        with open(path, "rb") as handle:
            part = MIMEApplication(handle.read(), Name=os.path.basename(path))
        part["Content-Disposition"] = f'attachment; filename="{os.path.basename(path)}"'
        message.attach(part)

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(server_name, port) as smtp:
                smtp.login(sender, password)
                smtp.send_message(message)
        else:
            with smtplib.SMTP(server_name, port) as smtp:
                smtp.starttls()
                smtp.login(sender, password)
                smtp.send_message(message)
        return True
    except Exception as exc:
        logger.warning("Workflow email could not be sent: {}", exc)
        return False

