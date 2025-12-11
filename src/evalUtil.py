"""
Evaluation utility module
Provides model prediction, result evaluation, and saving functionality with unified exception handling
"""

from typing import List, Literal
from pathlib import Path

import pandas as pd
import dspy
from loguru import logger
from pydantic import BaseModel

from optimUtil import wrapOneDspyField, DspyField, create_output_model_class
from model import model_setting_instance, model_setting_instance_judge
from dspy.utils.parallelizer import ParallelExecutor


# class PredNovelSettings(OptimSettings):
#     judging: Literal["", "confidence", "score"] = "confidence"
#     output_file: str


class PredTrainedSettings(BaseModel):
    load_dir: str
    judging: Literal["", "confidence", "score"] = "confidence"
    dataset: str
    save_dir: str
    output_file: str = "result.json"


class PredictionSettings(BaseModel):
    """
    The settings for the novel prediction
    """

    inputFields: list[DspyField]
    outputFields: list[DspyField]
    initial_prompt: str = ""
    judging: Literal["", "confidence", "score"] = "confidence"
    dataset: str
    save_dir: str
    task: Literal["QA", "Extraction"] = "Extraction"
    threads: int = 6
    multiple: bool = False


def judgeFactory(
    prediction_settings: PredictionSettings,
    judging: Literal["", "confidence", "score"] = "confidence",
) -> dspy.Module:
    """Create evaluator based on settings

    Args:
        prediction_settings: Prediction settings
        judging: Evaluation mode

    Returns:
        dspy.Module: Evaluator module

    Raises:
        ValueError: When judging parameter is invalid
        RuntimeError: When evaluator creation fails
    """
    try:
        if judging == "confidence":
            return wrapConfidenceJudge(prediction_settings)
        elif judging == "score":
            logger.warning(
                "Score judge not implemented, falling back to confidence judge"
            )
            return wrapConfidenceJudge(prediction_settings)
        else:
            raise ValueError(f"Invalid judging mode: {judging}")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to create judge for mode '{judging}': {e}")
        raise RuntimeError(f"Judge creation failed: {e}") from e


def wrapConfidenceJudge(prediction_settings: PredictionSettings) -> dspy.ChainOfThought:
    """Wrap prediction settings into confidence evaluator

    Args:
        prediction_settings: Prediction settings

    Returns:
        dspy.ChainOfThought: Confidence evaluator

    Raises:
        ValueError: When input parameters are invalid
        RuntimeError: When evaluator creation fails
    """
    try:
        if not prediction_settings.inputFields:
            raise ValueError("inputFields cannot be empty")
        if not prediction_settings.outputFields:
            raise ValueError("outputFields cannot be empty")

        fields = {}
        for field in prediction_settings.inputFields:
            fields[field.name] = wrapOneDspyField(field, True)

        if prediction_settings.multiple:
            output_class = create_output_model_class(prediction_settings.outputFields)
            fields["extracted_information"] = (
                List[output_class],
                dspy.InputField(description="extracted_information"),
            )
        else:
            for field in prediction_settings.outputFields:
                fields[field.name] = wrapOneDspyField(field, True)

        if prediction_settings.task == "QA":
            fields["question"] = wrapOneDspyField(
                DspyField(
                    name="question",
                    field_type="str",
                    description=prediction_settings.initial_prompt,
                ),
                True,
            )

        fields["confidence"] = wrapOneDspyField(
            DspyField(
                name="confidence",
                field_type="float",
                range_min=0,
                range_max=1,
                description="confidence",
            ),
            False,
        )

        input_names = ", ".join(
            [f"[{item.name}]" for item in prediction_settings.inputFields]
        )
        output_names = ", ".join(
            [f"[{item.name}]" for item in prediction_settings.outputFields]
        )

        if prediction_settings.task == "QA":
            be = "are" if len(output_names) > 1 else "is"
            doc = f"As an objective and impartial judge, please judge if {output_names} {be} factually correct based on the {input_names}"
        else:
            doc = f"As an objective and impartial judge, please judge the accuracy of the {output_names} extracted from the {input_names}"

        judge_signature = dspy.make_signature(fields, doc, "judge_Signature")
        logger.debug(f"Created judge signature with fields: {list(fields.keys())}")
        return dspy.ChainOfThought(judge_signature)

    except ValueError as e:
        logger.error(f"Invalid parameters for confidence judge: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create confidence judge: {e}")
        raise RuntimeError(f"Confidence judge creation failed: {e}") from e


def custom_judge_metric(
    df: pd.DataFrame,
    prediction_settings: PredictionSettings,
    judge: dspy.Module,
    num_threads: int = 6,
) -> pd.DataFrame:
    """Evaluate dataset using evaluator

    Args:
        df: Input dataset
        prediction_settings: Prediction settings
        judge: Evaluator module
        num_threads: Number of parallel threads

    Returns:
        pd.DataFrame: Result dataset with confidence scores

    Raises:
        ValueError: When input parameters are invalid
        RuntimeError: When evaluation process fails
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    if num_threads <= 0:
        raise ValueError("num_threads must be positive")

    field_list = [item.name for item in prediction_settings.inputFields]
    if prediction_settings.multiple:
        field_list.append("extracted_information")
    else:
        field_list.extend([item.name for item in prediction_settings.outputFields])

    logger.info(f"Starting judgment with {len(df)} rows using {num_threads} threads")

    judge_dataset = []
    for index, row in df.iterrows():
        try:
            missing_fields = [f for f in field_list if f not in row or pd.isna(row[f])]
            if missing_fields:
                logger.warning(f"Row {index} missing fields: {missing_fields}")
                continue

            example = dspy.Example(
                **{field: row[field] for field in df.columns}
            ).with_inputs(*field_list)
            judge_dataset.append(example)
        except Exception as e:
            logger.error(f"Error preparing row {index} for judgment: {e}")
            continue

    if not judge_dataset:
        raise ValueError("No valid data to judge")

    def process_item(example):
        try:
            try:
                llm = (
                    model_setting_instance_judge.configure_model()
                    if getattr(model_setting_instance_judge, "setting_status", False)
                    else model_setting_instance.configure_model()
                )
            except Exception as e:
                logger.error(f"Failed to configure judge llm: {e}")
                llm = None
            return judge(lm=llm, **example.inputs())
        except Exception as e:
            logger.error(f"Error judging example: {e}")
            raise

    executor = ParallelExecutor(
        num_threads=num_threads,
        disable_progress_bar=True,
        max_errors=5,
        provide_traceback=False,
        compare_results=False,
    )

    try:
        results = executor.execute(process_item, judge_dataset)
        logger.info(f"Successfully judged {len(results)} items")
    except Exception as e:
        logger.warning(
            f"Parallel execution failed: {e}. Switching to sequential processing..."
        )
        results = []
        chunk_size = max(1, min(20, len(judge_dataset) // num_threads))

        for i in range(0, len(judge_dataset), chunk_size):
            chunk = judge_dataset[i : i + chunk_size]
            chunk_executor = ParallelExecutor(
                num_threads=min(num_threads, len(chunk)),
                disable_progress_bar=True,
                max_errors=5,
                provide_traceback=False,
                compare_results=False,
            )
            try:
                chunk_results = chunk_executor.execute(process_item, chunk)
                results.extend(chunk_results)
                logger.debug(
                    f"Processed chunk {i // chunk_size + 1}: {len(chunk_results)} items"
                )
            except Exception as chunk_error:
                logger.error(f"Chunk processing failed: {chunk_error}")
                # Fallback to individual processing
                for example in chunk:
                    try:
                        result = process_item(example)
                        results.append(result)
                    except Exception as item_error:
                        logger.error(f"Individual judgment failed: {item_error}")
                        results.append(None)

    list_out = []
    for index, (example, result) in enumerate(zip(judge_dataset, results)):
        try:
            result_dict = example.toDict()
            if result is not None:
                result_dict["confidence"] = getattr(result, "confidence", None)
            else:
                result_dict["confidence"] = None
            list_out.append(result_dict)
        except Exception as e:
            logger.error(f"Error processing result for row {index}: {e}")
            result_dict = example.toDict()
            result_dict["confidence"] = None
            list_out.append(result_dict)

    df_out = pd.DataFrame(list_out)
    logger.info(f"Judgment completed. Output shape: {df_out.shape}")
    return df_out


def predCustomData(
    prediction_settings: PredictionSettings,
    predictor: dspy.Module,
    lm,
    num_threads: int = 6,
) -> pd.DataFrame:
    """Use predictor for custom data prediction

    Args:
        prediction_settings: Prediction settings
        predictor: Predictor module
        lm: Language model
        num_threads: Number of parallel threads

    Returns:
        pd.DataFrame: Prediction result dataset

    Raises:
        FileNotFoundError: When dataset file does not exist
        ValueError: When dataset format is invalid or parameters are incorrect
        RuntimeError: When prediction process fails
    """
    dataset_path = prediction_settings.dataset
    logger.info(f"Loading dataset from {dataset_path}")

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not dataset_path.is_file():
        raise ValueError(f"Dataset path is not a file: {dataset_path}")

    try:
        suffix = dataset_path.suffix.lower()
        if suffix == ".json":
            df = pd.read_json(dataset_path)
        elif suffix == ".csv":
            df = pd.read_csv(dataset_path)
        elif suffix == ".tsv":
            df = pd.read_csv(dataset_path, sep="\t")
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        if df.empty:
            raise ValueError("Dataset is empty")

    except pd.errors.EmptyDataError:
        raise ValueError("Dataset file is empty or corrupted")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_path}: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}") from e

    input_field_list = [item.name for item in prediction_settings.inputFields]
    output_field_list = [item.name for item in prediction_settings.outputFields]

    if not input_field_list:
        raise ValueError("No input fields specified in prediction settings")
    if not output_field_list:
        raise ValueError("No output fields specified in prediction settings")

    # Validate field existence
    missing_fields = []
    for field in input_field_list:
        if field not in df.columns:
            missing_fields.append(field)

    if missing_fields:
        raise ValueError(f"Required fields not found in dataset: {missing_fields}")

    logger.info(f"Input fields: {input_field_list}")
    logger.info(f"Output fields: {output_field_list}")

    # Data quality check
    missing_data = df[input_field_list].isnull().sum()
    if missing_data.any():
        logger.warning(
            f"Missing values found: {missing_data[missing_data > 0].to_dict()}"
        )
        df = df.dropna(subset=input_field_list)
        logger.info(f"Filtered dataset shape after removing NaN: {df.shape}")

    if len(df) == 0:
        raise ValueError("No valid data to process after filtering")

    # Thread count validation
    if not isinstance(num_threads, int) or num_threads < 1:
        logger.warning(f"Invalid thread count {num_threads}, using 1")
        num_threads = 1

    # Parallel prediction
    executor = ParallelExecutor(
        num_threads=num_threads,
        disable_progress_bar=True,
        max_errors=5,
        provide_traceback=False,
        compare_results=False,
    )

    def predict_single_row(row):
        """Process single row prediction"""
        try:
            input_dict = {field: str(row[field]) for field in input_field_list}
            result = predictor(**input_dict)

            output_dict = {}
            for field in output_field_list:
                if prediction_settings.multiple:
                    items = getattr(result, "extracted_information", [])
                    output_dict["extracted_information"] = (
                        [item.dict() for item in items] if items else []
                    )
                else:
                    value = getattr(result, field, None)
                    output_dict[field] = str(value) if value is not None else None

            return output_dict

        except Exception as e:
            logger.warning(f"Prediction failed for row {row.name}: {e}")
            if prediction_settings.multiple:
                return {"extracted_information": []}
            else:
                return {field: None for field in output_field_list}

    try:
        results = executor.execute(
            predict_single_row, [row for _, row in df.iterrows()]
        )
    except Exception as e:
        logger.warning(
            f"Parallel execution failed: {e}, trying chunk-based parallel processing..."
        )

        # Chunk-based parallel processing fallback
        results = []
        chunk_size = max(1, min(20, len(df) // num_threads))

        for i in range(0, len(df), chunk_size):
            chunk = [row for _, row in df.iloc[i : i + chunk_size].iterrows()]
            chunk_executor = ParallelExecutor(
                num_threads=min(num_threads, len(chunk)),
                disable_progress_bar=True,
                max_errors=5,
                provide_traceback=False,
                compare_results=False,
            )
            try:
                chunk_results = chunk_executor.execute(predict_single_row, chunk)
                results.extend(chunk_results)
                logger.debug(
                    f"Processed chunk {i // chunk_size + 1}: {len(chunk_results)} items"
                )
            except Exception as chunk_error:
                logger.error(
                    f"Chunk processing failed: {chunk_error}, falling back to sequential processing"
                )
                # Sequential processing fallback
                for row in chunk:
                    try:
                        result = predict_single_row(row)
                        results.append(result)
                    except Exception as item_error:
                        logger.error(f"Individual prediction failed: {item_error}")
                        if prediction_settings.multiple:
                            results.append({"extracted_information": []})
                        else:
                            results.append({field: None for field in output_field_list})

    # Validate result completeness
    logger.info(results)
    if len(results) != len(df):
        raise RuntimeError(
            f"Result count mismatch: expected {len(df)}, got {len(results)}"
        )

    # Add prediction results
    if prediction_settings.multiple:
        df["extracted_information"] = [
            result.get("extracted_information", None) for result in results
        ]
    else:
        for field in output_field_list:
            df[f"{field}"] = [result.get(field, None) for result in results]

    logger.info(f"Prediction completed successfully for {len(df)} examples")
    return df


def savePredictResult(df: pd.DataFrame, save_dir: str, output_file: str) -> None:
    """Save prediction results to specified file

    Args:
        df: Prediction result DataFrame
        save_dir: Save directory path
        output_file: Output filename

    Returns:
        None

    Raises:
        ValueError: When input parameters are invalid
        RuntimeError: When saving process fails
    """
    if df.empty:
        raise ValueError("Cannot save empty DataFrame")

    if not output_file:
        raise ValueError("Output filename cannot be empty")

    save_path = Path(save_dir)
    file_path = save_path / output_file

    try:
        save_path.mkdir(parents=True, exist_ok=True)

        # Validate directory permissions
        if not save_path.is_dir():
            raise ValueError(f"Save path is not a directory: {save_path}")

        suffix = output_file.lower().split(".")[-1] if "." in output_file else ""
        if suffix == "json":
            df.to_json(file_path, orient="records")
        elif suffix == "csv":
            df.to_csv(file_path, index=False, encoding="utf-8")
        elif suffix == "tsv":
            df.to_csv(file_path, sep="\t", index=False, encoding="utf-8")
        elif suffix in ["xlsx", "xls"]:
            df.to_excel(file_path, index=False)
        elif suffix == "jsonl":
            df.to_json(file_path, orient="records", lines=True)
        else:
            raise ValueError("Invalid dataset format")

        logger.info(f"Prediction result saved successfully: {file_path}")

    except PermissionError:
        raise RuntimeError(f"Permission denied: Cannot write to {save_path}")
    except OSError as e:
        raise RuntimeError(f"File system error while saving: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to save prediction result: {e}") from e
