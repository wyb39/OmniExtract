"""Jinja2 web UI request endpoints.

The FastAPI application owns middleware and static-file mounting, while this
module owns all page responses for the migrated Jinja UI.
"""

import os

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from src.common import baseUtil


root_dir = os.path.abspath(baseUtil.get_root_path())
templates = Jinja2Templates(directory=os.path.join(root_dir, "ui_jinja", "templates"))

# Included by app.py with the /omniextract prefix.
router = APIRouter()


@router.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/model-config")
def model_config(request: Request):
    return templates.TemplateResponse("model_config.html", {"request": request})


@router.get("/doc-extraction-original")
def doc_extraction_original(request: Request):
    return templates.TemplateResponse("doc_extraction_original.html", {"request": request})


@router.get("/doc-extraction-optimized")
def doc_extraction_optimized(request: Request):
    return templates.TemplateResponse("doc_extraction_optimized.html", {"request": request})


@router.get("/prompt-optimization")
def prompt_optimization(request: Request):
    return templates.TemplateResponse("prompt_optimization.html", {"request": request})


@router.get("/table-parsing")
def table_parsing(request: Request):
    return templates.TemplateResponse("table_extraction.html", {"request": request})


# Keep the project root convenient while the UI itself is namespaced.
root_router = APIRouter()


@root_router.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/omniextract/", status_code=307)

