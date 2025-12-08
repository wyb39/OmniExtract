import argparse
import os
import json
from loguru import logger
from model import (
    get_model_settings as get_model_settings_func,
    ModelSettings
)
from service import (
    optim,
    optim_custom,
    pred,
    file_to_md,
    md_to_json,
    file_to_json,
    build_optm_set,
    parse_table_to_tsv,
    extract_table_service

)
from params import (
    PathSettings,
    TableExtractionParams,
    ExtractTableServiceParams,
    BuildTrainSetParams,
)
from optimUtil import OptimSettings
from evalUtil import PredictionSettings
from .yamlParser import yaml_to_class


def get_model_settings(model_usage):
    try:
        return get_model_settings_func(model_usage)
    except Exception as e:
        logger.error(f"Error getting model settings: {e}")
        raise


def modify_model(data):
    def save_current_model_settings(model_instance, data):
        model_instance.model_name = data["model_name"]
        model_instance.model_type = data["model_type"]
        model_instance.api_key = data["api_key"]
        model_instance.api_base = data["api_base"]
        model_instance.model_usage = data["model_usage"]
        model_instance.temperature = data["temperature"]
        model_instance.max_tokens = data["max_tokens"]
        model_instance.top_p = data["top_p"]
        model_instance.top_k = data["top_k"]
        model_instance.min_p = data["min_p"]
        model_instance.setting_status = True
        message = model_instance.save_model_settings()
        return message

    try:
        current_model_instance = get_model_settings_func(data["model_usage"])
        save_message = save_current_model_settings(current_model_instance, data)
        logger.info(save_message)
        return {"message": save_message}
    except ValueError as ve:
        logger.error(f"Invalid request: {ve}")
        raise
    except Exception as e:
        logger.error(f"Internal Server Error: {e}")
        raise


def run_optim(data):
    try:
        logger.info(f"optim data: {data}")
        optim_settings = OptimSettings.model_validate(data)
        optim(optim_settings)
        logger.info("Optim completed")
        return {"message": "optim completed"}
    except Exception as e:
        logger.error(f"Exception optim error: {e}")
        raise


def run_optim_custom(data):
    try:
        logger.info(f"optim data: {data}")
        optim_settings = OptimSettings.model_validate(data)
        optim_custom(optim_settings)
        logger.info("Optim custom completed")
        return {"message": "optim completed"}
    except Exception as e:
        logger.error(f"Exception optim error: {e}")
        raise


def run_pred_optimized(data):
    try:
        settings_path = os.path.join(data["load_dir"], "optim_settings.json")
        with open(settings_path, "r") as f:
            item = json.load(f)
            prediction_settings = PredictionSettings.model_validate(item)
            logger.info(f"Prediction settings: {prediction_settings}")
        prediction_settings.save_dir = data["save_dir"]
        prediction_settings.dataset = data["dataset"]
        prediction_settings.judging = data["judging"]
        prompt_dir = os.path.join(data["load_dir"], "optim_prompt.json")
        if not os.path.exists(prompt_dir):
            raise FileNotFoundError("prompt.json not found")
        pred(
            prediction_settings, prompt_dir=prompt_dir, output_file=data["output_file"]
        )
        logger.info("Prediction completed")
        return {"message": "prediction completed"}
    except Exception as e:
        logger.error(f"Exception pred error: {e}")
        raise


def run_pred_original(data):
    try:
        prediction_settings = PredictionSettings.model_validate(data)
        pred(prediction_settings)
        logger.info("Prediction completed")
        return {"message": "prediction completed"}
    except Exception as e:
        logger.error(f"Exception pred error: {e}")
        raise


def run_model_test_call(data):
    try:
        prompt = data.get("prompt", "Hello")
        model_settings = ModelSettings.model_validate(data)
        result = model_settings.test_call(prompt)
        logger.info("model_test_call completed")
        return {"message": "model test_call completed", "result": result}
    except Exception as e:
        logger.error(f"Exception model_test_call error: {e}")
        raise


def run_file_to_md(data):
    try:
        logger.info(f"file_to_md data: {data}")
        path_settings = PathSettings.model_validate(data)
        result = file_to_md(
            path_settings.folder_path, path_settings.save_path, path_settings.file_type
        )
        logger.info("file_to_md completed")
        return {"message": "file_to_md completed", "result": result}
    except Exception as e:
        logger.error(f"Exception file_to_md error: {e}")
        raise


def run_md_to_json(data):
    try:
        logger.info(f"md_to_json data: {data}")
        path_settings = PathSettings.model_validate(data)
        result = md_to_json(
            path_settings.folder_path,
            path_settings.save_path,
            path_settings.convert_mode,
        )
        logger.info("md_to_json completed")
        return {"message": "md_to_json completed", "result": result}
    except Exception as e:
        logger.error(f"Exception md_to_json error: {e}")
        raise


def run_file_to_json(data):
    try:
        logger.info(f"file_to_json data: {data}")
        path_settings = PathSettings.model_validate(data)
        result = file_to_json(
            path_settings.folder_path,
            path_settings.save_path,
            path_settings.file_type,
            path_settings.convert_mode,
        )
        logger.info("file_to_json completed")
        return {"message": "file_to_json completed", "result": result}
    except Exception as e:
        logger.error(f"Exception file_to_json error: {e}")
        raise


def run_parse_table_to_tsv(data):
    try:
        logger.info(f"parse_table_to_tsv data: {data}")
        table_extraction_params = TableExtractionParams.model_validate(data)
        result = parse_table_to_tsv(
            file_folder_path=table_extraction_params.file_folder_path,
            save_folder_path=table_extraction_params.save_folder_path,
            non_tabular_file_format=table_extraction_params.non_tabular_file_format,
            encoding="utf-8",
            verbose=False,
        )
        logger.info("parse_table_to_tsv completed")
        return {"message": "parse_table_to_tsv completed", "result": result}
    except Exception as e:
        logger.error(f"Exception parse_table_to_tsv error: {e}")
        raise


def run_extract_table_service(data):
    try:
        logger.info(f"extract_table_service data: {data}")
        extract_table_service_params = ExtractTableServiceParams.model_validate(data)
        # set default num_threads
        if extract_table_service_params.num_threads is None:
            extract_table_service_params.num_threads = 6
        # set default extract_directly
        if extract_table_service_params.extract_directly is None:
            extract_table_service_params.extract_directly = False
        result = extract_table_service(
            parsed_file_path=extract_table_service_params.parsed_file_path,
            save_folder_path=extract_table_service_params.save_folder_path,
            outputFields=extract_table_service_params.outputFields,
            classify_prompt=extract_table_service_params.classify_prompt,
            extract_prompt=extract_table_service_params.extract_prompt,
            extract_directly=extract_table_service_params.extract_directly,
            num_threads=extract_table_service_params.num_threads,
            encoding="utf-8",
        )
        logger.info("extract_table_service completed")
        return {"message": "extract_table_service completed", "result": result}
    except Exception as e:
        logger.error(f"Exception extract_table_service error: {e}")
        raise


def run_build_optm_set(data):
    try:
        logger.info(f"build_optm_set data: {data}")
        build_optm_set_params = BuildTrainSetParams.model_validate(data)
        result = build_optm_set(
            json_path=build_optm_set_params.json_path,
            dataset=build_optm_set_params.dataset,
            save_dir=build_optm_set_params.save_dir,
            fields=build_optm_set_params.fields,
            multiple=build_optm_set_params.multiple,
            article_field=build_optm_set_params.article_field,
            article_parts=build_optm_set_params.article_parts,
        )
        logger.info("build_optm_set completed")
        return {"message": "build_optm_set completed", "result": result}
    except Exception as e:
        logger.error(f"Exception build_optm_set error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Command line interface for acurateLLM"
    )
    parser.add_argument("command", help="Command to execute")
    parser.add_argument(
        "yaml_file", help="Path to the YAML configuration file containing parameters"
    )
    args = parser.parse_args()

    # Read YAML file for parameters using yaml_to_class
    try:
        # We'll create a generic class to hold the parameters
        class Params:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        params = yaml_to_class(args.yaml_file, Params)
        data = vars(params)  # Convert to dictionary
        logger.info(f"Loaded parameters from {args.yaml_file}")
    except Exception as e:
        logger.error(f"Failed to read YAML file: {e}")
        raise

    # Execute the corresponding function based on the command
    try:
        command = args.command

        if command == "get_model_settings":
            result = get_model_settings(data.get("model_usage", "main"))
            logger.info(f"Model settings: {result}")
        elif command == "modify_model":
            result = modify_model(data)
        elif command == "optim":
            result = run_optim(data)
        elif command == "optim_custom":
            result = run_optim_custom(data)
        elif command == "pred_optimized":
            result = run_pred_optimized(data)
        elif command == "pred_original":
            result = run_pred_original(data)
        elif command == "model_test_call":
            result = run_model_test_call(data)
        elif command == "file_to_md":
            result = run_file_to_md(data)
        elif command == "md_to_json":
            result = run_md_to_json(data)
        elif command == "file_to_json":
            result = run_file_to_json(data)
        elif command == "parse_table_to_tsv":
            result = run_parse_table_to_tsv(data)
        elif command == "extract_table_service":
            result = run_extract_table_service(data)
        elif command == "build_optm_set":
            result = run_build_optm_set(data)
        else:
            raise ValueError(f"Unknown command: {command}")

        logger.info(f"Command {command} executed successfully")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Failed to execute command: {e}")
        raise
