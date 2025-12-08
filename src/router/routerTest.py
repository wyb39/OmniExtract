from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import Literal
import os
import json
from model import (
    get_model_settings,
    ModelSettings,
)
from service import (
    optim,
    optim_custom,
    pred,
    file_to_md,
    md_to_json,
    file_to_json,
    parse_table_to_tsv,
    extract_table_service,
    build_optm_set,
)
from params import (
    PathSettings,
    TableExtractionParams,
    ExtractTableServiceParams,
    BuildTrainSetParams,
)
from optimUtil import OptimSettings
from evalUtil import PredictionSettings, PredTrainedSettings

router = APIRouter()


@router.get("/api/model_settings")
def get_model_settings_api(
    model_usage: Literal[
        "main", "visual", "prompt_generation", "judge", "coder"
    ] = "main",
):
    try:
        return get_model_settings(model_usage)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/modify_model")
def modify_model(data: ModelSettings):
    def save_current_model_settings(model_instance, data):
        model_instance.model_name = data.model_name
        model_instance.model_type = data.model_type
        model_instance.api_key = data.api_key
        model_instance.api_base = data.api_base
        model_instance.model_usage = data.model_usage
        model_instance.temperature = data.temperature
        model_instance.max_tokens = data.max_tokens
        model_instance.top_p = data.top_p
        model_instance.top_k = data.top_k
        model_instance.min_p = data.min_p
        model_instance.setting_status = True
        message = model_instance.save_model_settings()
        return message
        
    try:
        current_model_instance = get_model_settings(data.model_usage)
        save_message = save_current_model_settings(current_model_instance, data)
        logger.info(save_message)
        return {"message": save_message}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail="Invalid request: " + str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error : {e}")


@router.post("/api/optim")
async def optimapi(data: OptimSettings):
    try:
        logger.info(f"optim data:{data}")
        optim(data)
        return {"message": "optim completed"}
    except Exception as e:
        logger.info(f"Exception optim error:{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/optim/custom")
async def optimapi_custom(data: OptimSettings):
    try:
        logger.info(f"optim data:{data}")
        optim_custom(data)
        return {"message": "optim completed"}
    except Exception as e:
        logger.info(f"Exception optim error:{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/model_test_call")
def model_test_call(data: ModelSettings, prompt: str = "Hello"):
    try:
        result = data.test_call(prompt)
        return {"message": "model test_call completed", "result": result}
    except Exception as e:
        logger.info(f"Exception model_test_call error:{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/pred_optimized")
async def predapi(data: PredTrainedSettings):
    try:
        settings_path = os.path.join(data.load_dir, "optim_settings.json")
        with open(settings_path, "r") as f:
            item = json.load(f)
            prediction_settings = PredictionSettings.model_validate(item)
            print(prediction_settings)
        prediction_settings.save_dir = data.save_dir
        prediction_settings.dataset = data.dataset
        prediction_settings.judging = data.judging
        prompt_dir = os.path.join(data.load_dir, "optim_prompt.json")
        if not os.path.exists(prompt_dir):
            raise HTTPException(status_code=500, detail="prompt.json not found")
        pred(prediction_settings, prompt_dir=prompt_dir, output_file=data.output_file)
        return {"message": "prediction completed"}
    except Exception as e:
        logger.info(f"Exception pred error:{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/pred_original")
async def predapi_original(data: PredictionSettings):
    try:
        pred(data)
        return {"message": "prediction completed"}
    except Exception as e:
        logger.info(f"Exception pred error:{e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
def test():
    # logger.info(f'test:{data}')
    return {"message": "This is a test"}


@router.post("/api/file_to_md")
async def file_to_md_api(data: PathSettings):
    try:
        logger.info(f"file_to_md data: {data}")
        result = file_to_md(data.folder_path, data.save_path, data.file_type)
        return {"message": "file_to_md completed", "result": result}
    except Exception as e:
        logger.info(f"Exception file_to_md error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/md_to_json")
async def md_to_json_api(data: PathSettings):
    try:
        logger.info(f"md_to_json data: {data}")
        result = md_to_json(data.folder_path, data.save_path, data.convert_mode)
        return {"message": "md_to_json completed", "result": result}
    except Exception as e:
        logger.info(f"Exception md_to_json error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/file_to_json")
async def file_to_json_api(data: PathSettings):
    try:
        logger.info(f"file_to_json data: {data}")
        result = file_to_json(
            data.folder_path, data.save_path, data.file_type, data.convert_mode
        )
        return {"message": "file_to_json completed", "result": result}
    except Exception as e:
        logger.info(f"Exception file_to_json error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/parse_table_to_tsv")
async def parse_table_to_tsv_api(data: TableExtractionParams):
    try:
        logger.info(f"parse_table_to_tsv data: {data}")
        result = parse_table_to_tsv(
            file_folder_path=data.file_folder_path,
            save_folder_path=data.save_folder_path,
            non_tabular_file_format=data.non_tabular_file_format,
            encoding="utf-8",
            verbose=False,
        )
        return {"message": "parse_table_to_tsv completed", "result": result}
    except Exception as e:
        logger.info(f"Exception parse_table_to_tsv error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/extract_table_service")
async def extract_table_service_api(data: ExtractTableServiceParams):
    try:
        logger.info(f"extract_table_service data: {data}")
        # set default num_threads
        if data.num_threads is None:
            data.num_threads = 6
        # set default extract_directly
        if data.extract_directly is None:
            data.extract_directly = False
        result = extract_table_service(
            parsed_file_path=data.parsed_file_path,
            save_folder_path=data.save_folder_path,
            outputFields=data.outputFields,
            classify_prompt=data.classify_prompt,
            extract_prompt=data.extract_prompt,
            extract_directly=data.extract_directly,
            num_threads=data.num_threads,
            encoding="utf-8",
        )
        return {"message": "extract_table_service completed", "result": result}
    except Exception as e:
        logger.info(f"Exception extract_table_service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/build_optm_set")
async def build_optm_set_api(data: BuildTrainSetParams):
    try:
        logger.info(f"build_optm_set data: {data}")
        result = build_optm_set(
            json_path=data.json_path,
            dataset=data.dataset,
            save_dir=data.save_dir,
            fields=data.fields,
            multiple=data.multiple,
            article_field=data.article_field,
            article_parts=data.article_parts,
        )
        return {"message": "build_optm_set completed", "result": result}
    except Exception as e:
        logger.info(f"Exception build_optm_set error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
