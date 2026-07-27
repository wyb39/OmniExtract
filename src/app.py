import os
from typing import Dict, Any

from pydantic import BaseModel
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from router import routerTest
from router import ui_router
import baseUtil


root_dir = os.path.abspath(baseUtil.get_root_path())
dist_dir = os.path.join(root_dir, "gui")
assets_dir = os.path.join(dist_dir, "assets")
print(f'dist_dir:{dist_dir}')

app = FastAPI(docs_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routerTest.router)
# Expose the same API under the prefix expected by the Jinja pages while
# retaining the unprefixed routes for existing clients.
app.include_router(routerTest.router, prefix="/omniextract")

app.mount("/omniextract/assets", StaticFiles(directory=assets_dir), name="omniextract-assets")
app.include_router(ui_router.root_router)
app.include_router(ui_router.router, prefix="/omniextract")
# The Dash application remains the default GUI entry point.  Do not mount it
# here: this FastAPI app is also used by the CLI service.
class BaseMap(BaseModel):
    data: Dict[str, Any]
