import argparse
import os
import json
from loguru import logger
from src.model.model import (
    get_model_settings as get_model_settings_func,
    ModelSettings
)
from src.service.service import (
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
from src.model.params import (
    PathSettings,
    TableExtractionParams,
    ExtractTableServiceParams,
    BuildTrainSetParams,
)
from src.utils.optimUtil import OptimSettings
from src.utils.evalUtil import PredictionSettings
from src.workflow import workflow_service as workflow_svc
from .yamlParser import yaml_to_class
from src.common.error_handling import ReportedTaskError


def get_model_settings(model_usage):
    try:
        return get_model_settings_func(model_usage)
    except Exception as e:
        logger.error(f"Error getting model settings: {e}")
        raise


def modify_model(data):
    def save_current_model_settings(model_instance, data):
        validated = ModelSettings.model_validate(data)
        for field_name in validated.model_fields:
            if field_name in data:
                setattr(
                    model_instance,
                    field_name,
                    getattr(validated, field_name),
                )
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
        result = optim(optim_settings)
        logger.info("Optim completed")
        return result
    except Exception as e:
        logger.error(f"Exception optim error: {e}")
        raise


def run_optim_custom(data):
    try:
        logger.info(f"optim data: {data}")
        optim_settings = OptimSettings.model_validate(data)
        result = optim_custom(optim_settings)
        logger.info("Optim custom completed")
        return result
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
        result = pred(
            prediction_settings, prompt_dir=prompt_dir, output_file=data["output_file"]
        )
        logger.info("Prediction completed")
        return result
    except Exception as e:
        logger.error(f"Exception pred error: {e}")
        raise


def run_pred_original(data):
    try:
        prediction_settings = PredictionSettings.model_validate(data)
        result = pred(prediction_settings)
        logger.info("Prediction completed")
        return result
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
            path_settings.folder_path,
            path_settings.save_path,
            path_settings.file_type,
        )
        logger.info("file_to_md completed")
        return {
            "message": "file_to_md completed",
            "result": result,
            "processing_report": os.path.join(
                path_settings.save_path,
                "processing_report.json",
            ),
        }
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


def _require_input_path(data, key):
    path = data.get(key)
    if not path:
        raise ValueError(f"Missing required input file path: '{key}'")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def _resolve_workflow_base(data):
    """Return (workflow_id, base_path). Use data['base_path'] if provided, else create one."""
    base_path = data.get("base_path")
    if base_path:
        os.makedirs(base_path, exist_ok=True)
        workflow_id = os.path.basename(base_path.rstrip(os.sep)) or "cli_workflow"
        return workflow_id, base_path
    return workflow_svc.new_workspace()


def _workflow_summary(message, workflow_id, base_path, result, artifact_keys):
    summary = {
        "message": message,
        "workflow_id": workflow_id,
        "base_path": base_path,
        "status": result.get("status"),
        "processing_status": result.get("processing_status"),
        "processing_report": result.get("processing_report"),
    }
    for key in artifact_keys:
        value = result.get(key)
        if isinstance(value, str):
            summary[key] = value
    return summary


def run_workflow_doc_extraction(data):
    try:
        logger.info(f"workflow_doc_extraction data: {data}")
        zip_file_path = _require_input_path(data, "zip_file_path")
        workflow_id, base_path = _resolve_workflow_base(data)
        result = workflow_svc.run_workflow_doc_extraction(
            task_name=data.get("task_name") or workflow_id,
            contact_email=data.get("contact_email", ""),
            file_type=data.get("file_type", "PDF"),
            zip_file_path=zip_file_path,
            convert_mode=data.get("convert_mode", "byPart"),
            input_fields=data.get("inputFields", []),
            output_fields=data.get("outputFields", []),
            base_path=base_path,
            initial_prompt=data.get("initial_prompt", ""),
            judging_mode=data.get("judging_mode", "confidence"),
            threads=int(data.get("threads", 6)),
            multiple_entities=bool(data.get("multiple_entities", False)),
        )
        logger.info("workflow_doc_extraction completed")
        return _workflow_summary(
            "workflow_doc_extraction completed",
            workflow_id, base_path, result, ["result_zip"],
        )
    except Exception as e:
        logger.error(f"Exception workflow_doc_extraction error: {e}")
        raise


def run_workflow_table_extraction(data):
    try:
        logger.info(f"workflow_table_extraction data: {data}")
        zip_file_path = _require_input_path(data, "zip_file_path")
        workflow_id, base_path = _resolve_workflow_base(data)
        result = workflow_svc.run_workflow_table_extraction(
            task_name=data.get("task_name") or workflow_id,
            contact_email=data.get("contact_email", ""),
            file_type=data.get("file_type", "PDF"),
            zip_file_path=zip_file_path,
            output_fields=data.get("outputFields", []),
            base_path=base_path,
            classify_prompt=data.get("classify_prompt", ""),
            extract_prompt=data.get("extract_prompt", ""),
            threads=int(data.get("threads", 6)),
        )
        logger.info("workflow_table_extraction completed")
        return _workflow_summary(
            "workflow_table_extraction completed",
            workflow_id, base_path, result, ["format_tables_zip"],
        )
    except Exception as e:
        logger.error(f"Exception workflow_table_extraction error: {e}")
        raise


def run_workflow_prompt_optimization(data):
    try:
        logger.info(f"workflow_prompt_optimization data: {data}")
        zip_file_path = _require_input_path(data, "zip_file_path")
        dataset_file_path = _require_input_path(data, "dataset_file_path")
        workflow_id, base_path = _resolve_workflow_base(data)
        result = workflow_svc.run_workflow_prompt_optimization(
            task_name=data.get("task_name") or workflow_id,
            contact_email=data.get("contact_email", ""),
            file_type=data.get("file_type", "PDF"),
            zip_file_path=zip_file_path,
            dataset_file_path=dataset_file_path,
            convert_mode=data.get("convert_mode", "byPart"),
            input_fields=data.get("inputFields", []),
            output_fields=data.get("outputFields", []),
            base_path=base_path,
            initial_prompt=data.get("initial_prompt", ""),
            demos=int(data.get("demos", 1)),
            article_field=data.get("article_field", "article_field"),
            multiple_entities=bool(data.get("multiple_entities", False)),
        )
        logger.info("workflow_prompt_optimization completed")
        return _workflow_summary(
            "workflow_prompt_optimization completed",
            workflow_id, base_path, result, ["optimization_config_zip"],
        )
    except Exception as e:
        logger.error(f"Exception workflow_prompt_optimization error: {e}")
        raise


def run_workflow_doc_extraction_optimized(data):
    try:
        logger.info(f"workflow_doc_extraction_optimized data: {data}")
        zip_file_path = _require_input_path(data, "zip_file_path")
        config_zip_path = _require_input_path(data, "config_zip_path")
        workflow_id, base_path = _resolve_workflow_base(data)
        result = workflow_svc.run_workflow_doc_extraction_optimized(
            task_name=data.get("task_name") or workflow_id,
            contact_email=data.get("contact_email", ""),
            file_type=data.get("file_type", "PDF"),
            zip_file_path=zip_file_path,
            config_zip_path=config_zip_path,
            convert_mode=data.get("convert_mode", "byPart"),
            base_path=base_path,
            judging_mode=data.get("judging_mode", "confidence"),
            threads=int(data.get("threads", 6)),
        )
        logger.info("workflow_doc_extraction_optimized completed")
        return _workflow_summary(
            "workflow_doc_extraction_optimized completed",
            workflow_id, base_path, result, ["result_zip"],
        )
    except Exception as e:
        logger.error(f"Exception workflow_doc_extraction_optimized error: {e}")
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
        elif command == "workflow_doc_extraction":
            result = run_workflow_doc_extraction(data)
        elif command == "workflow_table_extraction":
            result = run_workflow_table_extraction(data)
        elif command == "workflow_prompt_optimization":
            result = run_workflow_prompt_optimization(data)
        elif command == "workflow_doc_extraction_optimized":
            result = run_workflow_doc_extraction_optimized(data)
        else:
            raise ValueError(f"Unknown command: {command}")

        logger.info(f"Command {command} executed successfully")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except ReportedTaskError as e:
        logger.error(f"Failed to execute command: {e}")
        print(
            json.dumps(
                {
                    "status": "failed",
                    "processing_report": e.report_path,
                    "issue": e.issue.to_dict(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Failed to execute command: {e}")
        raise
