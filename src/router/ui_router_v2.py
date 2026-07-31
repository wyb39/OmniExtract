"""Jinja2 web UI v2 request endpoints.

This is a parallel, read-only-preview variant of ``ui_router``: it serves the
redesigned templates from ``ui_jinja_v2/templates`` under the ``/v2`` prefix.
The original routes and templates in ``ui_jinja`` remain untouched, and both
UIs share the exact same API endpoints and workflow backend.
"""

import os

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

import baseUtil


root_dir = os.path.abspath(baseUtil.get_root_path())
templates_v2 = Jinja2Templates(directory=os.path.join(root_dir, "ui_jinja_v2", "templates"))

# Included by app.py with the /omniextract/v2 prefix.
router = APIRouter()


@router.get("/")
def index_v2(request: Request):
    return templates_v2.TemplateResponse("index.html", {"request": request})


@router.get("/model-config")
def model_config_v2(request: Request):
    return templates_v2.TemplateResponse("model_config.html", {"request": request})


@router.get("/doc-extraction-original")
def doc_extraction_original_v2(request: Request):
    return templates_v2.TemplateResponse("doc_extraction_original.html", {"request": request})


@router.get("/doc-extraction-optimized")
def doc_extraction_optimized_v2(request: Request):
    return templates_v2.TemplateResponse("doc_extraction_optimized.html", {"request": request})


@router.get("/prompt-optimization")
def prompt_optimization_v2(request: Request):
    return templates_v2.TemplateResponse("prompt_optimization.html", {"request": request})


@router.get("/table-parsing")
def table_parsing_v2(request: Request):
    return templates_v2.TemplateResponse("table_extraction.html", {"request": request})
