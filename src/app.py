import os
from typing import Dict, Any

from pydantic import BaseModel
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from router import routerTest
import baseUtil


dist_dir = os.path.join(baseUtil.get_root_path(), "gui")
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
# app.mount("/",StaticFiles(directory=dist_dir),name="static")
class BaseMap(BaseModel):
    data: Dict[str, Any]