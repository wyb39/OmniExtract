import dspy
import os
import json
import shutil
import tempfile
import pandas as pd
from loguru import logger
from typing import Literal
from custom_optimizer.miprov2_custom import MIPROv2_Custom
from optimUtil import (
    OptimSettings,
    wrapInput2Signature,
    loadCustomOptimData,
    createCustomMetric,
    saveOptimFiles,
    saveSettings,
    DspyField,
    checkImageSettings,
)
from evalUtil import (
    PredictionSettings,
    predCustomData,
    judgeFactory,
    custom_judge_metric,
    savePredictResult,
)
from model import (
    model_setting_instance,
    model_setting_instance_prompt,
    model_setting_instance_image,
)
from articleUtil import (
    parse_article_to_md,
    parse_xml_to_md,
    parse_pubmed_xml_to_md,
    parse_tex_to_md_batch,
)
from articleUtil import (
    split_md,
    convert_md_to_json,
    build_optm_set_from_document_and_extracted_information,
    build_optm_set_from_article_and_extracted_information,
    merge_json_files_to_dataset,
)
from tableExtractUtil import (
    classify_tables,
    generate_example_tables,
    generate_format_tables_with_extract4correct,
)
from error_handling import (
    ProcessingReport,
    REPORT_FILENAME,
    ReportedTaskError,
    map_exception,
    write_failure_and_wrap,
    write_report,
)
from processing_adapters import documents_to_json, process_table_documents
from token_usage import track_token_usage


def _validate_optimization_dataset(dataset):
    """DSPy needs at least two valid examples for a useful train/validation split."""

    try:
        size = len(dataset)
    except Exception as exc:
        raise ValueError(
            "Unable to determine the number of valid optimization records"
        ) from exc
    if size < 2:
        raise ValueError(
            "Optimization dataset must contain at least two valid records"
        )


def _reported_task(operation, save_dir, work, *, document_id="workflow"):
    """Run a task-fatal boundary and always leave a report beside its output."""

    workflow_id = os.path.basename(os.path.abspath(save_dir)) or operation
    with track_token_usage():
        try:
            result = work()
        except ReportedTaskError:
            raise
        except Exception as exc:
            raise write_failure_and_wrap(
                exc,
                save_dir,
                workflow_id=workflow_id,
                stage=operation,
                document_id=document_id,
            ) from exc

        report = ProcessingReport(workflow_id)
        report.success()
        report_path = write_report(report, os.path.join(save_dir, REPORT_FILENAME))
        if isinstance(result, dict):
            result["processing_report"] = report_path
            result["processing_status"] = report.processing_status
        return result


def _optim_impl(optim_settings: OptimSettings):
    save_dir = optim_settings.save_dir
    optimset = optim_settings.dataset
    if not os.path.exists(optimset):
        raise FileNotFoundError(f"File {optimset} does not exist.")
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError:
            raise OSError(f"Failed to create directory {save_dir}")

    # Configure log output to file
    log_file = os.path.join(save_dir, "optim.log")
    logger.add(log_file, rotation="10 MB", retention="10 days", level="INFO")

    logger.info("Saving prompt optimization settings...")
    saveSettings(optim_settings)
    logger.info("Creating prediction signature...")
    custom_signature = wrapInput2Signature(optim_settings)
    logger.info(f"{custom_signature}")
    predictor = dspy.Predict(custom_signature)
    logger.info("Loading optimization data...")
    list_dataset = loadCustomOptimData(optim_settings)
    _validate_optimization_dataset(list_dataset)
    llm = model_setting_instance.configure_model()
    dspy.configure(lm=llm)
    logger.info(llm)
    if model_setting_instance_prompt.setting_status:
        llm_prompt = model_setting_instance_prompt.configure_model()
    else:
        llm_prompt = llm
    logger.info("Building evaluation metric...")
    custom_metric = createCustomMetric(optim_settings)
    logger.info(custom_metric)
    # TODO customize the optimizer
    logger.info("Initializing optimizer...")
    optimizer = dspy.MIPROv2(
        custom_metric,
        prompt_model=llm_prompt,
        task_model=llm,
        auto=optim_settings.optim_burden,
        num_threads=optim_settings.threads,
    )
    optim = optimizer.compile(
        predictor,
        trainset=list_dataset,
        max_bootstrapped_demos=optim_settings.demos,
        max_labeled_demos=optim_settings.demos,
        view_data_batch_size=1,
        requires_permission_to_run=False,
    )
    logger.info(optim)
    saveOptimFiles(optim_settings, optim)
    return {"message": "optim success"}


def _optim_custom_impl(optim_settings: OptimSettings):
    save_dir = optim_settings.save_dir
    optimset = optim_settings.dataset
    if not os.path.exists(optimset):
        raise FileNotFoundError(f"File {optimset} does not exist.")
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError:
            raise OSError(f"Failed to create directory {save_dir}")

    # Configure log output to file
    log_file = os.path.join(save_dir, "optim.log")
    logger.add(log_file, rotation="10 MB", retention="10 days", level="INFO")
    logger.info(f"{optim_settings}")
    logger.info("Saving prompt optimization settings...")
    saveSettings(optim_settings)
    logger.info("Creating prediction signature...")
    custom_signature = wrapInput2Signature(optim_settings)
    logger.info(f"{custom_signature}")
    predictor = dspy.Predict(custom_signature)
    logger.info("Loading optimization data...")
    list_dataset = loadCustomOptimData(optim_settings)
    _validate_optimization_dataset(list_dataset)
    llm = model_setting_instance.configure_model()
    dspy.configure(lm=llm)
    if model_setting_instance_prompt.setting_status:
        llm_prompt = model_setting_instance_prompt.configure_model()
    else:
        llm_prompt = llm
    logger.info("Building evaluation metric...")
    custom_metric = createCustomMetric(optim_settings)
    # TODO customize the optimizer
    logger.info("Initializing optimizer...")
    optimizer = MIPROv2_Custom(
        custom_metric,
        prompt_model=llm_prompt,
        task_model=llm,
        auto=optim_settings.optim_burden,
        num_threads=optim_settings.threads,
    )
    optim = optimizer.compile(
        predictor,
        trainset=list_dataset,
        max_bootstrapped_demos=optim_settings.demos,
        max_labeled_demos=optim_settings.demos,
        view_data_batch_size=1,
        requires_permission_to_run=False,
    )
    logger.info(optim)
    saveOptimFiles(optim_settings, optim)
    return {"message": "optim success"}


def _optim_image_impl(optim_settings: OptimSettings):
    save_dir = optim_settings.save_dir
    optimset = optim_settings.dataset
    if not os.path.exists(optimset):
        raise FileNotFoundError(f"File {optimset} does not exist.")
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError:
            raise OSError(f"Failed to create directory {save_dir}")
    logger.info("Saving prompt optimization settings...")
    saveSettings(optim_settings)
    if not checkImageSettings(optim_settings):
        raise ValueError("Image settings are not configured correctly.")
    logger.info("Creating prediction signature...")
    custom_signature = wrapInput2Signature(optim_settings)
    logger.info(f"{custom_signature}")
    predictor = dspy.Predict(custom_signature)
    logger.info("Loading optimization data...")
    list_dataset = loadCustomOptimData(optim_settings)
    _validate_optimization_dataset(list_dataset)
    llm_image = model_setting_instance_image.configure_model()
    logger.info(llm_image)
    logger.info("Building evaluation metric...")
    custom_metric = createCustomMetric(optim_settings)
    logger.info(custom_metric)
    # TODO customize the optimizer
    logger.info("Initializing optimizer...")
    optimizer = dspy.MIPROv2(
        custom_metric,
        prompt_model=llm_image,
        task_model=llm_image,
        auto=optim_settings.optim_burden,
        num_threads=optim_settings.threads,
    )
    optim = optimizer.compile(
        predictor,
        trainset=list_dataset,
        max_bootstrapped_demos=optim_settings.demos,
        max_labeled_demos=optim_settings.demos,
        view_data_batch_size=1,
        requires_permission_to_run=False,
    )
    saveOptimFiles(optim_settings, optim)
    return {"message": "optim success"}


def optim(optim_settings: OptimSettings):
    """Run DSPy optimization with one actionable task-level failure report."""

    return _reported_task(
        "optimization",
        optim_settings.save_dir,
        lambda: _optim_impl(optim_settings),
        document_id="optimization",
    )


def optim_custom(optim_settings: OptimSettings):
    """Run custom optimization while retaining DSPy's detailed optim.log."""

    return _reported_task(
        "optimization",
        optim_settings.save_dir,
        lambda: _optim_custom_impl(optim_settings),
        document_id="optimization",
    )


def optim_image(optim_settings: OptimSettings):
    return _reported_task(
        "optimization",
        optim_settings.save_dir,
        lambda: _optim_image_impl(optim_settings),
        document_id="optimization",
    )


def _pred_impl(
    prediction_settings: PredictionSettings, prompt_dir="", output_file="result.json"
):
    llm = model_setting_instance.configure_model()
    dspy.configure(lm=llm)
    save_dir = prediction_settings.save_dir
    pred_set = prediction_settings.dataset
    if not os.path.exists(pred_set):
        raise FileNotFoundError(f"File {pred_set} does not exist.")
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError:
            raise OSError(f"Failed to create directory {save_dir}")

    # Configure log output to file
    log_file = os.path.join(save_dir, "prediction.log")
    logger.add(log_file, rotation="10 MB", retention="10 days", level="INFO")

    logger.info("Saving prompt prediction settings...")
    custom_signature = wrapInput2Signature(prediction_settings)
    predictor = dspy.Predict(custom_signature)
    logger.info("Loading predictor...")
    if prompt_dir != "":
        predictor.load(prompt_dir)
    logger.info(f"Predictor loaded: {predictor}")
    logger.info("Predicting...")
    df_result = predCustomData(
        prediction_settings,
        predictor,
        lm=llm,
        num_threads=prediction_settings.threads,
    )
    report = ProcessingReport.from_dict(
        df_result.attrs.get(
            "processing_report",
            {
                "workflow_id": os.path.basename(os.path.abspath(save_dir)),
                "processing_status": "success",
                "failed_documents": [],
            },
        )
    )
    logger.info("Judging...")
    judging = prediction_settings.judging
    if judging == "":
        savePredictResult(df_result, save_dir, output_file)
        logger.info(f"Results saved to {os.path.join(save_dir, output_file)}")
    else:
        judge = judgeFactory(prediction_settings, judging)
        logger.info("Evaluating with judge...")
        df_result_eval = custom_judge_metric(
            df_result,
            prediction_settings,
            judge,
            num_threads=prediction_settings.threads,
        )
        judgement_report = df_result_eval.attrs.get("processing_report")
        if isinstance(judgement_report, dict):
            report.merge(ProcessingReport.from_dict(judgement_report))
        savePredictResult(df_result_eval, save_dir, output_file)
        logger.info(f"Evaluated results saved to {os.path.join(save_dir, output_file)}")
    report_path = write_report(report, os.path.join(save_dir, REPORT_FILENAME))
    return {
        "message": "prediction completed",
        "result_file": os.path.join(save_dir, output_file),
        "processing_report": report_path,
        "processing_status": report.processing_status,
    }


def pred(
    prediction_settings: PredictionSettings, prompt_dir="", output_file="result.json"
):
    """Run prediction; row failures are reported without cancelling siblings."""

    with track_token_usage():
        try:
            return _pred_impl(prediction_settings, prompt_dir, output_file)
        except ReportedTaskError:
            raise
        except Exception as exc:
            raise write_failure_and_wrap(
                exc,
                prediction_settings.save_dir,
                workflow_id=os.path.basename(
                    os.path.abspath(prediction_settings.save_dir)
                )
                or "prediction",
                stage="prediction",
            ) from exc


def _md_to_json_impl(folder_path, save_path, convert_mode):
    """
    Convert markdown files to JSON format based on specified conversion mode

    Args:
        folder_path (str): Path to folder containing subfolders with markdown files
        save_path (str): Path to save the generated JSON files
        convert_mode (str): Conversion mode, which can be 'byPart' or 'wholeDoc'

    Returns:
        dict: Dictionary containing status and message

    Raises:
        ValueError: Raised when invalid paths or conversion mode is provided
    """

    # Validate conversion mode
    convert_mode = convert_mode.lower()
    if convert_mode not in ["bypart", "wholedoc"]:
        raise ValueError(
            f"Unsupported conversion mode: {convert_mode}\nSupported modes: byPart, wholeDoc"
        )

    # Call the corresponding conversion method
    if convert_mode == "bypart":
        split_md(folder_path, save_path)
        # Merge JSON files into a single dataset
        output_file = merge_json_files_to_dataset(save_path)
        return {
            "message": "Successfully converted markdown files to JSON by parts and merged into dataset",
            "dataset_file": output_file
        }
    else:
        convert_md_to_json(folder_path, save_path)
        # Merge JSON files into a single dataset
        output_file = merge_json_files_to_dataset(save_path)
        return {
            "message": "Successfully converted markdown files to JSON as whole documents and merged into dataset",
            "dataset_file": output_file
        }


def md_to_json(folder_path, save_path, convert_mode):
    """Convert Markdown and return the report path with the result."""

    return _reported_task(
        "json_convert",
        save_path,
        lambda: _md_to_json_impl(folder_path, save_path, convert_mode),
    )


def file_to_md(folder_path, save_path, file_type):
    """
    Convert files in the specified folder to Markdown format and save them

    Args:
        folder_path (str): Path to the folder containing source files
        save_path (str): Path to the destination folder for saved Markdown files
        file_type (str): File type, which can be one of PDF, scienceDirect, PMC, Arxiv

    Returns:
        list: List of generated markdown file paths

    Raises:
        ValueError: Raised when an invalid folder path is provided or the file type is not supported
    """

    # Preserve the long-standing list return type used by API clients.  The
    # richer file_to_json path uses the single-document adapter; here we infer
    # omissions from the batch parser and persist the same report contract.
    converters = {
        "pdf": parse_article_to_md,
        "sciencedirect": parse_xml_to_md,
        "pmc": parse_pubmed_xml_to_md,
        "arxiv": parse_tex_to_md_batch,
    }
    normalised = str(file_type).lower()
    if normalised not in converters:
        raise ValueError(
            f"Unsupported file type: {file_type}\n"
            "Supported types: PDF, scienceDirect, PMC, Arxiv"
        )
    try:
        markdown_files = converters[normalised](folder_path, save_path)
        report = ProcessingReport(
            os.path.basename(os.path.abspath(save_path)) or "file_to_md"
        )
        source_path = os.path.abspath(folder_path)
        suffix = ".pdf" if normalised == "pdf" else ".xml"
        if normalised == "arxiv":
            suffix = ".tex"
        sources = [
            path
            for root, _, files in os.walk(source_path)
            for name in files
            if name.lower().endswith(suffix)
            for path in [os.path.join(root, name)]
        ] if os.path.isdir(source_path) else []
        converted_stems = {
            os.path.splitext(os.path.basename(path))[0]
            for path in markdown_files
        }
        for source in sources:
            if os.path.splitext(os.path.basename(source))[0] in converted_stems:
                report.success()
            else:
                report.failure(
                    os.path.relpath(source, source_path).replace(os.sep, "/"),
                    map_exception(
                        RuntimeError("The parser did not generate Markdown output"),
                        "markdown_convert",
                    ),
                )
        if not sources:
            report.success(len(markdown_files) or 1)
        write_report(report, os.path.join(save_path, REPORT_FILENAME))
        return markdown_files
    except ReportedTaskError:
        raise
    except Exception as exc:
        raise write_failure_and_wrap(
            exc,
            save_path,
            workflow_id=os.path.basename(os.path.abspath(save_path))
            or "file_to_md",
            stage="document_parse",
        ) from exc


def file_to_json(
    folder_path,
    save_path,
    file_type,
    convert_mode,
):
    """
    Convert documents to JSON format through Markdown intermediate step

    Args:
        folder_path (str): Path to folder containing source files
        save_path (str): Path to save the generated JSON files
        file_type (str): File type, which can be one of PDF, scienceDirect, PMC, Arxiv
        convert_mode (str): Conversion mode for JSON output, which can be 'byPart' or 'wholeDoc'

    Returns:
        dict: Dictionary containing status and message

    Raises:
        ValueError: Raised when invalid paths, file type, or conversion mode is provided
    """

    try:
        json_files, report = documents_to_json(
            folder_path,
            save_path,
            file_type,
            convert_mode,
        )
        dataset_file = merge_json_files_to_dataset(save_path)
        report_path = write_report(
            report,
            os.path.join(save_path, REPORT_FILENAME),
        )
        return {
            "message": (
                f"Converted {len(json_files)} {file_type} document(s) "
                f"to JSON with {convert_mode} mode"
            ),
            "details": {
                "dataset_file": dataset_file,
                "json_files": json_files,
                "processing_report": report_path,
                "processing_status": report.processing_status,
            },
            "processing_report": report_path,
            "processing_status": report.processing_status,
        }
    except ReportedTaskError:
        raise
    except Exception as exc:
        raise write_failure_and_wrap(
            exc,
            save_path,
            workflow_id=os.path.basename(os.path.abspath(save_path))
            or "file_to_json",
            stage="document_parse",
        ) from exc


def _parse_table_to_tsv_impl(
    file_folder_path: str,
    save_folder_path: str,
    non_tabular_file_format: Literal["PDF", "scienceDirect", "PMC", "Arxiv"]
    | None = None,
    encoding: str = "utf-8",
    verbose: bool = False,
):
    """
    Parse tables from various file formats and save as TSV files

    Args:
        file_folder_path (str): Path to folder containing source files
        save_folder_path (str): Path to save output TSV files
        non_tabular_file_format (str): File format, can be None, 'PDF', 'scienceDirect', 'PMC', or 'Arxiv'
        encoding (str): File encoding, default is utf-8
        verbose (bool): Whether to print detailed information, default is False

    Returns:
        list: List of files that failed to process
    """
    from tableUtil import (
        parse_file_for_table_extraction_pdf,
        parse_file_for_table_extraction_tex,
        parse_file_for_table_extraction_pmc,
        parse_file_for_table_extraction_science_direct,
        extract_tsv_content_to_json,
    )

    err_files = []
    # Determine which parser to use based on file format
    if non_tabular_file_format is None or non_tabular_file_format.lower() == "pdf":
        # Use PDF parser
        err_files = parse_file_for_table_extraction_pdf(
            file_folder_path=file_folder_path,
            save_folder_path=save_folder_path,
            encoding=encoding,
        )
    elif non_tabular_file_format.lower() == "sciencedirect":
        # Use ScienceDirect parser
        err_files = parse_file_for_table_extraction_science_direct(
            file_folder_path=file_folder_path,
            save_folder_path=save_folder_path,
            encoding=encoding,
            verbose=verbose,
        )
    elif non_tabular_file_format.lower() == "pmc":
        # Use PMC parser
        err_files = parse_file_for_table_extraction_pmc(
            file_folder_path=file_folder_path,
            save_folder_path=save_folder_path,
            encoding=encoding,
            verbose=verbose,
        )
    elif non_tabular_file_format.lower() == "arxiv":
        # Use TeX parser for Arxiv
        err_files = parse_file_for_table_extraction_tex(
            file_folder_path=file_folder_path,
            save_folder_path=save_folder_path,
            encoding=encoding,
            verbose=verbose,
        )
    else:
        raise ValueError(
            f"Unsupported file format: {non_tabular_file_format}\nSupported formats: None, PDF, scienceDirect, PMC, Arxiv"
        )

    # Extract TSV content to JSON for all generated subdirectories
    if os.path.exists(save_folder_path):
        json_classify_path = os.path.join(save_folder_path, "json_classify")
        json_example_path = os.path.join(save_folder_path, "json_example")

        try:
            os.makedirs(json_classify_path, exist_ok=True)
            os.makedirs(json_example_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create json directories: {e}")

        for item in os.listdir(save_folder_path):
            item_path = os.path.join(save_folder_path, item)
            # Skip the output directories we just created or other non-directories
            if not os.path.isdir(item_path) or item in [
                "json_classify",
                "json_example",
            ]:
                continue

            try:
                extract_tsv_content_to_json(
                    item_path,
                    encoding=encoding,
                    processing_mode="classify",
                    save_path=json_classify_path,
                )
                extract_tsv_content_to_json(
                    item_path,
                    encoding=encoding,
                    processing_mode="example",
                    save_path=json_example_path,
                )
            except Exception as e:
                logger.error(f"Error extracting JSON for {item_path}: {e}")

    return err_files


def parse_table_to_tsv(
    file_folder_path: str,
    save_folder_path: str,
    non_tabular_file_format: Literal["PDF", "scienceDirect", "PMC", "Arxiv"]
    | None = None,
    encoding: str = "utf-8",
    verbose: bool = False,
):
    """Parse each document's tables and expose parser-returned file failures."""

    workflow_id = os.path.basename(os.path.abspath(save_folder_path)) or "tables"
    try:
        errors = _parse_table_to_tsv_impl(
            file_folder_path,
            save_folder_path,
            non_tabular_file_format,
            encoding,
            verbose,
        )
        report = ProcessingReport(workflow_id)
        error_names = set()
        for value in errors or []:
            if isinstance(value, (tuple, list)) and value:
                document_id = str(value[0])
                message = str(value[1]) if len(value) > 1 else "Table parsing failed"
            else:
                document_id = str(value)
                message = "Table parsing failed"
            error_names.add(os.path.basename(document_id))
            report.failure(
                document_id,
                map_exception(
                    RuntimeError(message),
                    "table_parse",
                ),
            )
        input_files = [
            item
            for item in os.listdir(file_folder_path)
            if os.path.isfile(os.path.join(file_folder_path, item))
        ]
        report.success(
            sum(os.path.basename(item) not in error_names for item in input_files)
        )
        report_path = write_report(
            report,
            os.path.join(save_folder_path, REPORT_FILENAME),
        )
        return {
            "message": "parse_table_to_tsv completed",
            "errors": errors or [],
            "processing_status": report.processing_status,
            "processing_report": report_path,
        }
    except ReportedTaskError:
        raise
    except Exception as exc:
        raise write_failure_and_wrap(
            exc,
            save_folder_path,
            workflow_id=workflow_id,
            stage="table_parse",
        ) from exc


def _build_optm_set_impl(
    json_path: str,
    dataset: str,
    save_dir: str,
    fields: list[DspyField],
    multiple: bool,
    article_field: str,
    article_parts: list[
        Literal["Title","Abstract","Introduction", "Method", "Result", "Discussion", "Conclusion"]
    ]
    | None,
):
    """
    Build optm dataset from article and extracted information.

    Args:
        json_path (str): Path to the JSON files
        dataset (str): Dataset name
        fields (list[DspyField]): List of DspyField objects
        multiple (bool): Whether to allow multiple values
        article_field (str): Article field name
        article_parts (list[Literal["Introduction", "Method", "Result", "Discussion", "Conclusion"]] | None): List of article parts

    Returns:
        The result of the called build_optm_set function (limited to first 5 items as examples)
    """
    # Create save directory if it doesn't exist
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create folder {save_dir}: {e}")
    
    if article_parts is None:
        result = build_optm_set_from_document_and_extracted_information(
            json_path=json_path,
            dataset=dataset,
            fields=fields,
            multiple=multiple,
            article_field=article_field,
        )
    else:
        result = build_optm_set_from_article_and_extracted_information(
            json_path=json_path,
            dataset=dataset,
            article_parts=article_parts,
            fields=fields,
            multiple=multiple,
            article_field=article_field,
        )
    
    # Save result to JSON file
    try:
        with open(os.path.join(save_dir, f"_optim_dataset.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise OSError(f"Failed to save JSON file {os.path.join(save_dir, f'_optim_dataset.json')}: {e}")
    
    # Return only first 5 items as examples (or all if less than 5)
    return result[:5] if len(result) > 5 else result


def build_optm_set(
    json_path: str,
    dataset: str,
    save_dir: str,
    fields: list[DspyField],
    multiple: bool,
    article_field: str,
    article_parts: list[
        Literal[
            "Title",
            "Abstract",
            "Introduction",
            "Method",
            "Result",
            "Discussion",
            "Conclusion",
        ]
    ]
    | None,
):
    return _reported_task(
        "optimization",
        save_dir,
        lambda: {
            "message": "Optimization dataset built",
            "preview": _build_optm_set_impl(
                json_path,
                dataset,
                save_dir,
                fields,
                multiple,
                article_field,
                article_parts,
            ),
            "dataset_file": os.path.join(save_dir, "_optim_dataset.json"),
        },
        document_id="optimization",
    )


def _extract_table_service_impl(
    parsed_file_path: str,
    save_folder_path: str,
    outputFields: list[DspyField],
    classify_prompt: str,
    extract_prompt: str,
    extract_directly: bool = False,
    num_threads: int = 6,
    encoding: str = "utf-8",
):
    """table extraction pipeline function"""
    llm = model_setting_instance.configure_model()
    dspy.configure(lm=llm)

    if not os.path.exists(save_folder_path):
        try:
            os.mkdir(save_folder_path)
        except OSError as e:
            raise OSError(f"Failed to create folder {save_folder_path}: {e}")

    parsed_file_path_temp = os.path.join(save_folder_path, "parsed_files_temp")
    try:
        os.mkdir(parsed_file_path_temp)
    except OSError as e:
        raise OSError(f"Failed to create folder {parsed_file_path_temp}: {e}")

    for root, dirs, files in os.walk(parsed_file_path):
        for file in files:
            full_path = os.path.join(root, file)
            if os.path.getsize(full_path) <= 0:
                os.remove(full_path)
            elif file.endswith(".tsv"):
                shutil.copy(full_path, parsed_file_path_temp)
            elif file.endswith(".meta"):
                shutil.copy(full_path, parsed_file_path_temp)

    # classify the tables
    try:
        classify_tables(
            parsed_file_path_temp,
            save_folder_path,
            classify_prompt,
            num_threads=num_threads,
            encoding=encoding,
        )
    except Exception as e:
        logger.error(f"Error in classify_tables: {e}")
        raise
    target_file_path = os.path.join(save_folder_path, "target_files")
    try:
        os.mkdir(target_file_path)
    except OSError as e:
        raise OSError(f"Failed to create folder {target_file_path}: {e}")
    classify_result = pd.read_excel(
        os.path.join(save_folder_path, "classification_result.xlsx")
    )
    target_result = classify_result[classify_result["is_target_table"]]
    for index, row in target_result.iterrows():
        current_source_path = os.path.join(parsed_file_path_temp, str(row["file_name"]))
        shutil.copy(current_source_path, target_file_path)

    # create examples for target files
    example_file_path = os.path.join(save_folder_path, "example_files")
    try:
        os.mkdir(example_file_path)
    except Exception as e:
        raise OSError(f"Failed to create folder {example_file_path}: {e}")
    try:
        generate_example_tables(
            target_file_path,
            example_file_path,
            extract_prompt,
            outputFields,
            num_threads,
            encoding,
        )
    except Exception as e:
        logger.error(f"Error in generate_example_tables: {e}")
        raise

    # generate format tables for target files
    format_table_path = os.path.join(save_folder_path, "format_tables")
    try:
        os.mkdir(format_table_path)
    except Exception as e:
        raise OSError(f"Failed to create folder {format_table_path}: {e}")
    # generate_format_tables(target_file_path, example_file_path, format_table_path,extract_prompt, outputFields, num_threads, encoding )
    try:
        generate_format_tables_with_extract4correct(
            target_file_path,
            example_file_path,
            format_table_path,
            extract_prompt,
            outputFields,
            num_threads,
            encoding,
            extract_directly=extract_directly,
        )
    except Exception as e:
        logger.error(f"Error in generate_format_tables_with_extract4correct: {e}")
        raise

    try:
        shutil.rmtree(parsed_file_path_temp)
    except Exception as e:
        logger.warning(
            f"Failed to remove temporary directory {parsed_file_path_temp}: {e}"
        )
    return {
        "message": "Table extraction completed successfully",
        "format_table_path": format_table_path,
    }


def extract_table_service(
    parsed_file_path: str,
    save_folder_path: str,
    outputFields: list[DspyField],
    classify_prompt: str,
    extract_prompt: str,
    extract_directly: bool = False,
    num_threads: int = 6,
    encoding: str = "utf-8",
):
    workflow_id = (
        os.path.basename(os.path.abspath(save_folder_path)) or "table_extraction"
    )
    with track_token_usage():
        try:
            results, report = process_table_documents(
                parsed_file_path,
                save_folder_path,
                lambda source, destination: _extract_table_service_impl(
                    str(source),
                    str(destination),
                    outputFields,
                    classify_prompt,
                    extract_prompt,
                    extract_directly,
                    num_threads,
                    encoding,
                ),
            )
            report_path = write_report(
                report,
                os.path.join(save_folder_path, REPORT_FILENAME),
            )
            return {
                "message": "Table extraction completed",
                "document_results": results,
                "processing_status": report.processing_status,
                "processing_report": report_path,
            }
        except ReportedTaskError:
            raise
        except Exception as exc:
            raise write_failure_and_wrap(
                exc,
                save_folder_path,
                workflow_id=workflow_id,
                stage="table_extract",
            ) from exc
