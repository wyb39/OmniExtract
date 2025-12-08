from pydantic import BaseModel
from typing import Literal, Optional
from optimUtil import DspyField

class PathSettings(BaseModel):
    folder_path: str
    save_path: str
    convert_mode: Optional[Literal['byPart', 'wholeDoc']] = None
    file_type: Optional[Literal['PDF', 'scienceDirect', 'PMC', 'Arxiv']] = None

class BuildTrainSetParams(BaseModel):
    json_path: str
    dataset: str
    save_dir: str
    fields: list[DspyField]
    multiple: bool
    article_field: str
    article_parts: Optional[list[Literal["Title","Abstract","Introduction", "Method", "Result", "Discussion", "Conclusion"]]] = None

class TableExtractionParams(BaseModel):
    file_folder_path: str
    save_folder_path: str
    non_tabular_file_format: Optional[Literal['PDF', 'scienceDirect', 'PMC', 'Arxiv']] = None
    encoding: Optional[str] = "utf-8"
    verbose: Optional[bool] = False

class ExtractTableServiceParams(BaseModel):
    parsed_file_path: str
    save_folder_path: str
    outputFields: list[DspyField]
    classify_prompt: str
    extract_prompt: str
    extract_directly: Optional[bool] = False
    num_threads: Optional[int] = 6
    encoding: Optional[str] = "utf-8"