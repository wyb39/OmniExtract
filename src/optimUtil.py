import os
import dspy
import pandas as pd
from dspy import InputField, OutputField, make_signature
from loguru import logger
import shutil
from pydantic import BaseModel, Field, create_model
from dspy.adapters.image_utils import Image
from dspy.evaluate.metrics import answer_exact_match_str
from pathlib import Path
import types
from typing import Literal, List, Any
import textdistance as tds
import copy
import ast
from model import model_setting_instance, model_setting_instance_judge


class DspyField(BaseModel):
    name: str
    field_type: Literal[
        "str", "int", "float", "bool", "list", "literal", "list_literal", "image"
    ] = "str"
    range_min: float | None = None
    range_max: float | None = None
    literal_list: list[str] = []
    description: str = ""


class OptimSettings(BaseModel):
    """
    The settings for the prompt optimization
    """

    inputFields: list[DspyField]
    outputFields: list[DspyField]
    initial_prompt: str = ""
    dataset: str
    save_dir: str
    task: Literal["QA", "Extraction"] = "Extraction"
    optim_burden: Literal["light", "medium", "heavy"] = "medium"
    threads: int = 6
    demos: int = 0
    multiple: bool = False
    recall_prior: bool = False
    ai_evaluation: bool = False


"""
Create corresponding signatures for input and output
"""


def wrapInput2Signature(optim_settings):
    """
    Wraps the input and output fields into a signature that can be used by the model.
    """
    fields = dict()
    for field in optim_settings.inputFields:
        fields[field.name] = wrapOneDspyField(field, True)
    if optim_settings.multiple:
        assert optim_settings.task != "QA"
        output_class = create_output_model_class(optim_settings.outputFields)
        fields["extracted_information"] = (
            list[output_class],
            OutputField(description="extracted_information"),
        )
    else:
        for field in optim_settings.outputFields:
            fields[field.name] = wrapOneDspyField(field, False)
    try:
        return make_signature(fields, optim_settings.initial_prompt, "customSignature")
    except Exception as e:
        logger.error(f"{e.__class__.__name__}: {e}")
        raise


def wrapOneDspyField(field: DspyField, isInput: bool):
    """
    Wraps a DspyField into a dictionary that can be used by the dspy library.
    """
    try:
        field_dict = {}
        if field.description != "":
            field_dict["description"] = field.description
        num_map = {"int": int, "float": float}
        if field.field_type in num_map:
            if field.range_min is not None:
                field_dict["ge"] = num_map[field.field_type](field.range_min)
            if field.range_max is not None:
                field_dict["le"] = num_map[field.field_type](field.range_max)
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list[str],
            "literal": Literal[tuple(field.literal_list)],
            "list_literal": list[Literal[tuple(field.literal_list)]],
            "image": Image,
        }
        if isInput:
            return (type_map[field.field_type], InputField(**field_dict))
        else:
            return (type_map[field.field_type], OutputField(**field_dict))
    except Exception as e:
        logger.error(f"{e.__class__.__name__}: {e}")
        raise


def create_output_model_class(
    outputFields: List[DspyField], class_name="ExtractedInformation"
):
    """
    Dynamically create a Pydantic model class based on outputFields in train_settings
    """
    fields = {}
    for field in outputFields:
        field_type = field.field_type
        if field_type == "str":
            pydantic_type = str | None
        elif field_type == "int":
            pydantic_type = int | None
        elif field_type == "float":
            pydantic_type = float | None
        elif field_type == "bool":
            pydantic_type = bool | None
        elif field_type == "list":
            pydantic_type = List[str] | None
        elif field_type == "literal":
            pydantic_type = Literal[tuple(field.literal_list)]
        else:
            raise ValueError(f"Unsupported field type: {field_type}")

        field_kwargs: dict[str, Any] = {"default": None}
        if field.description:
            field_kwargs["description"] = field.description
        if field.range_min is not None:
            field_kwargs["ge"] = field.range_min
        if field.range_max is not None:
            field_kwargs["le"] = field.range_max

        fields[field.name] = (pydantic_type, Field(**field_kwargs))

    # Dynamically create the Pydantic model class
    OutputModel = create_model(class_name, **fields)
    logger.info(OutputModel.model_json_schema())
    return OutputModel


"""
Metric-related functions
"""


def compareStringsWithField(pred, exam, field):
    """
    Compare the matching degree between input and output strings
    """
    if pred[field.name] is None and exam[field.name] is None:
        return 1
    elif pred[field.name] is None or exam[field.name] is None:
        return 0
    else:
        list_candidate_scores = [
            int(answer_exact_match_str(pred[field.name], [exam[field.name]], frac=0.8)),
            tds.smith_waterman.normalized_similarity(
                pred[field.name], exam[field.name]
            ),
            tds.ratcliff_obershelp.normalized_similarity(
                pred[field.name], exam[field.name]
            ),
            tds.jaccard.normalized_similarity(pred[field.name], exam[field.name]),
            tds.lcsstr.normalized_similarity(pred[field.name], exam[field.name]),
        ]

        return max(list_candidate_scores) >= 0.8


def compareStrings(pred, exam):
    """
    Compare the matching degree between input and output strings
    """
    if pred is None and exam is None:
        return 1
    elif pred is None or exam is None:
        return 0
    else:
        list_candidate_scores = [
            int(answer_exact_match_str(pred, [exam], frac=0.8)),
            tds.smith_waterman.normalized_similarity(pred, exam),
            tds.ratcliff_obershelp.normalized_similarity(pred, exam),
            tds.jaccard.normalized_similarity(pred, exam),
            tds.lcsstr.normalized_similarity(pred, exam),
        ]
        return max(list_candidate_scores) >= 0.8


def jaccard_similarity(list1, list2):
    """
    Calculate the Jaccard similarity between two lists
    """
    list1 = copy.deepcopy(list1)
    list2 = copy.deepcopy(list2)
    if type(list1[0]) is str:
        list_intersection = []
        list_union = []
        for str1 in list1:
            for str2 in list2:
                if compareStrings(str1, str2):
                    list_intersection.append(str1)
                    list1.remove(str1)
                    list2.remove(str2)
                    break
        list_union.extend(list1)
        list_union.extend(list2)
        list_union.extend(list_intersection)
        similarity = len(list_intersection) / len(list_union)
    else:
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        similarity = intersection / union
    return similarity


def complex_recall_similarity(list1, list2):
    list1 = copy.deepcopy(list1)
    list2 = copy.deepcopy(list2)
    list_intersection = []
    item_threshold = 0.8 * pow(0.9, len(list(list1[0].dict().keys())) - 1)
    for item1 in list1:
        temp_score = -1
        current_item = None
        for item2 in list2:
            current_score = compareExtractedInformation(item1, item2)
            if current_score > temp_score:
                temp_score = current_score
                current_item = item2
        if temp_score > item_threshold:
            list_intersection.append(current_item)
            list1.remove(item1)
            list2.remove(current_item)
    return len(list_intersection)


def ai_evaluate(exam, pred, optim_settings):
    if optim_settings.recall_prior and optim_settings.multiple:
        doc_text = "As an objective and impartial judge, please judge the recall of the extracted_information compared to the reference_information. Note: the extracted_information is complex thus some differences compared to the reference could be tolerated."
    else:
        doc_text = "As an objective and impartial judge, please judge the accuracy of the extracted_information compared to the reference_information. Note: the extracted_information is complex thus some differences compared to the reference could be tolerated."
    # multiple choice
    if type(pred) is list:

        class _ai_evaluation_signature_1(dspy.Signature):
            __doc__ = doc_text
            extracted_information: list = dspy.InputField(
                description="The extracted_information curated by a large language model."
            )
            reference_information: list = dspy.InputField(
                description="The reference_information curated by a human."
            )
            score: float = dspy.OutputField(
                description="The judge's score from 0 to 1.", ge=0, le=1
            )

        judge = dspy.ChainOfThought(_ai_evaluation_signature_1)
        try:
            llm = (
                model_setting_instance_judge.configure_model()
                if getattr(model_setting_instance_judge, "setting_status", False)
                else model_setting_instance.configure_model()
            )
        except Exception as e:
            logger.error(f"Failed to configure judge llm: {e}")
            llm = None

        score = judge(
            extracted_information=pred,
            reference_information=exam,
            lm=llm,
        ).score
        return score
    else:

        class _ai_evaluation_signature(dspy.Signature):
            __doc__ = doc_text
            extracted_information: dict = dspy.InputField(
                description="The extracted_information curated by a large language model."
            )
            reference_information: dict = dspy.InputField(
                description="The reference_information curated by a human."
            )
            score: float = dspy.OutputField(
                description="The judge's score from 0 to 1.", ge=0, le=1
            )

        judge = dspy.ChainOfThought(_ai_evaluation_signature)
        try:
            llm = (
                model_setting_instance_judge.configure_model()
                if getattr(model_setting_instance_judge, "setting_status", False)
                else model_setting_instance.configure_model()
            )
        except Exception as e:
            logger.error(f"Failed to configure judge llm: {e}")
            llm = None

        score = judge(
            extracted_information=pred,
            reference_information=exam,
            lm=llm,
        ).score
        return score


def complex_jaccard_similarity(list1, list2):
    list1 = copy.deepcopy(list1)
    list2 = copy.deepcopy(list2)
    list_intersection = []
    list_union = []
    item_threshold = 0.8 * pow(0.9, len(list(list1[0].dict().keys())) - 1)
    for item1 in list1:
        temp_score = -1
        current_item = None
        for item2 in list2:
            current_score = compareExtractedInformation(item1, item2)
            if current_score > temp_score:
                temp_score = current_score
                current_item = item2
        if temp_score > item_threshold:
            list_intersection.append(current_item)
            list1.remove(item1)
            list2.remove(current_item)
    list_union.extend(list1)
    list_union.extend(list2)
    list_union.extend(list_intersection)
    similarity = len(list_intersection) / len(list_union)
    return similarity


def createCustomMetric(optim_settings: OptimSettings):
    """
    Create custom metric function
    :param optim_settings: settings for prediction
    :return: custom metric function
    """
    if optim_settings.multiple:

        def custom_metric(exam, pred, trace=None):
            score = 0
            exam_list = exam.extracted_information
            pred_list = pred.extracted_information
            if optim_settings.ai_evaluation:
                score = ai_evaluate(exam_list, pred_list, optim_settings)
            elif optim_settings.recall_prior:
                score = complex_recall_similarity(exam_list, pred_list)
            else:
                score = complex_jaccard_similarity(exam_list, pred_list)
            return score

    else:

        def custom_metric(exam, pred, trace=None):
            if optim_settings.ai_evaluation:
                exam_dict = {}
                pred_dict = {}
                for field in optim_settings.outputFields:
                    exam_dict[field.name] = exam[field.name]
                    pred_dict[field.name] = pred[field.name]
                return ai_evaluate(exam_dict, pred_dict, optim_settings)
            else:
                try:
                    score = 0
                    for field in optim_settings.outputFields:
                        if field.field_type == "str":
                            score += compareStringsWithField(pred, exam, field)
                        elif (
                            field.field_type == "int"
                            or field.field_type == "float"
                            or field.field_type == "bool"
                        ):
                            score += pred[field.name] == exam[field.name]
                        elif field.field_type == "literal":
                            score += (
                                pred[field.name].lower() == exam[field.name].lower()
                            )
                        elif field.field_type == "list":
                            if type(exam[field.name]) is str:
                                try:
                                    exam[field.name] = ast.literal_eval(
                                        exam[field.name]
                                    )
                                except (ValueError, SyntaxError):
                                    logger.warning(
                                        f"Failed to parse list from string: {exam[field.name]}"
                                    )
                                    exam[field.name] = []
                            score += jaccard_similarity(
                                pred[field.name], exam[field.name]
                            )
                            logger.info(
                                f"jaccard_similarity: {jaccard_similarity(pred[field.name], exam[field.name])}"
                            )
                    return score
                except Exception as e:
                    logger.error(f"{e.__class__.__name__}: {e}")
                    raise

    return custom_metric


def compareExtractedInformation(ele1, ele2):
    """
    Compare two Pydantic BaseModel instances and return a similarity score.
    :param model1: First Pydantic BaseModel instance
    :param model2: Second Pydantic BaseModel instance
    :return: similarity score
    """
    score = 0
    for field_name, value1 in ele1.dict().items():
        value2 = ele2.dict()[field_name]
        if isinstance(value1, list):
            score += jaccard_similarity(value1, value2)
        elif isinstance(value1, str):
            score += compareStrings(value1, value2)
        elif isinstance(value1, bool):
            score += value1 == value2
        elif isinstance(value1, int) or isinstance(value1, float):
            score += value1 == value2
    return score / len(list(ele2.dict().keys()))


# todo support multiple file types
def loadCustomOptimData(optim_settings):
    """
    Load custom data file
    :param optim_settings: settings for prediction
    :return: dspy example list
    """
    dataset = optim_settings.dataset
    logger.info(f"Loading dataset from {dataset}")
    # Check if file exists
    try:
        if not os.path.exists(dataset):
            logger.error(f"File {dataset} does not exist.")
            raise FileNotFoundError(f"File {dataset} does not exist.")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        raise
    try:
        if dataset.endswith(".json"):
            df = pd.read_json(dataset)
        elif dataset.endswith(".csv"):
            df = pd.read_csv(dataset)
        elif dataset.endswith(".tsv"):
            df = pd.read_csv(dataset, sep="\t")
        elif dataset.endswith(".xlsx"):
            df = pd.read_excel(dataset)
        else:
            logger.error(f"Error reading file {dataset}: Invalid dataset format")
            raise ValueError("Invalid dataset format")
    except Exception as e:
        logger.error(f"Error reading file {dataset}: {e}")
        raise

    if optim_settings.multiple:
        input_field_list = [item.name for item in optim_settings.inputFields]
        output_field_list = [item.name for item in optim_settings.outputFields]
        output_class = create_output_model_class(optim_settings.outputFields)

        def wrap_multiple_data(row, input_fields, output_fields, output_class):
            return wrapMultipleData(row, input_fields, output_fields, output_class)

        example_list = list(
            df.apply(
                wrap_multiple_data,
                args=(input_field_list, output_field_list, output_class),
                axis=1,
            )
        )
    else:
        input_field_list = [item.name for item in optim_settings.inputFields]
        output_field_list = [item.name for item in optim_settings.outputFields]

        def wrap_oneline_data(row, input_fields, output_fields):
            return wrapOnelineData(row, input_fields, output_fields)

        example_list = list(
            df.apply(
                wrap_oneline_data, args=(input_field_list, output_field_list), axis=1
            )
        )
    logger.info("Dataset loaded and processed successfully")
    return example_list


def process_field_value(field_name: str, field_value):
    """
    deal with the field value
    """
    if field_name == "image" and field_value is not None:
        return Image.from_file(field_value)
    return field_value


def wrapMultipleData(row, input_field_names, output_field_names, output_class):
    fields = dict()
    for field in input_field_names:
        fields[field] = process_field_value(field, row[field])
    fields["extracted_information"] = [
        output_class(**item) for item in row["extracted_information"]
    ]
    return dspy.Example(fields).with_inputs(*input_field_names)


def wrapOnelineData(row, input_field_names, output_field_names):
    fields = dict()
    for field in input_field_names:
        fields[field] = process_field_value(field, row[field])
    for field in output_field_names:
        fields[field] = row[field]
    return dspy.Example(fields).with_inputs(*input_field_names)


def saveSettings(optim_settings: OptimSettings):
    """
    Save the optim settings to a json file
    """
    save_dir = optim_settings.save_dir
    save_file = os.path.join(save_dir, "optim_settings.json")
    try:
        with open(save_file, "w", encoding="utf-8") as f:
            f.write(optim_settings.model_dump_json(indent=2))
    except Exception as e:
        logger.error(f"Failed to save settings: {e.__class__.__name__}: {e}")
        raise


def saveOptimFiles(optim_settings, optm, task="optim"):
    """
    Save optimized prompt files and dataset
    """
    dataset = optim_settings.dataset
    save_dir = optim_settings.save_dir

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    try:
        dataset_name = os.path.basename(dataset)
        target_dataset = os.path.join(save_dir, dataset_name)
        if Path(dataset).resolve() != Path(target_dataset).resolve():
            shutil.copy2(dataset, target_dataset)
            logger.info(f"Dataset copied to: {target_dataset}")
    except Exception as e:
        logger.error(f"Failed to copy dataset: {e.__class__.__name__}: {e}")
        raise

    try:
        optm_file = os.path.join(save_dir, f"{task}_prompt.json")
        optm.dump_state = types.MethodType(dump_state, optm)
        optm.save(optm_file)
        logger.info(f"Optimized prompt saved to: {optm_file}")
    except Exception as e:
        logger.error(f"Failed to save optimized prompt: {e.__class__.__name__}: {e}")
        raise


def checkImageSettings(optim_settings: OptimSettings) -> bool:
    """
    Check if the image settings are valid
    """
    flag = False
    for item in optim_settings.inputFields:
        if item.name == "image" and item.field_type == "image":
            flag = True
            break
    return flag


# Override dump_state for Predict
def dump_state(self, save_verbose=None):
    state_keys = ["lm", "traces", "train"]
    state = {k: getattr(self, k) for k in state_keys}

    state["demos"] = []
    for demo in self.demos:
        demo = demo.copy()
        for field in demo:
            # FIXME: Saving BaseModels as strings in examples doesn't matter because you never re-access as an object
            # It does matter for images
            if isinstance(demo[field], Image):
                demo[field] = demo[field].model_dump()
            elif isinstance(demo[field], BaseModel):
                demo[field] = demo[field].model_dump_json()
            elif isinstance(demo[field], list) and isinstance(
                demo[field][0], BaseModel
            ):
                demo[field] = [item.dict() for item in demo[field]]
        state["demos"].append(demo)

    state["signature"] = self.signature.dump_state()
    # `extended_signature` is a special field for `Predict`s like CoT.
    if hasattr(self, "extended_signature"):
        state["extended_signature"] = self.extended_signature.dump_state()
    return state
