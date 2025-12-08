import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    example_dir = Path(__file__).resolve().parent
    repo_root = example_dir.parent
    runs_original_dir = repo_root / "gui" / "runs" / "doc_extraction" / "original"
    runs_original_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_original_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = example_dir / "files_for_curation_json" / "_dataset.json"
    if not dataset.exists():
        alt = example_dir / "curated_breed_files_json" / "_dataset.json"
        dataset = alt if alt.exists() else dataset

    save_dir = example_dir / "breed_curation_original"
    save_dir.mkdir(parents=True, exist_ok=True)

    template_json_path = runs_original_dir / "run_20251205_030916" / "original_extraction_20251205_030916.json"

    input_fields = [
        {
            "name": "Document",
            "field_type": "str",
            "description": "Breed text from a breed standard file",
        }
    ]

    output_fields = [
        {
            "name": "size",
            "field_type": "str",
            "description": "The overall physical size of the dog breed, specified as small, medium, or large.",
        },
        {
            "name": "color",
            "field_type": "str",
            "description": "The general coat color of the breed, with possible variations separated by \"|\". For mixed colors, the primary color should be listed first, followed by secondary colors joined by \"&\" (e.g., \"black|brown&white\").",
        },
        {
            "name": "height_at_the_withers",
            "field_type": "str",
            "description": "The height of the dog measured from the ground to the highest point of the shoulders, preferably in inches or centimeters.",
        },
        {
            "name": "weight",
            "field_type": "str",
            "description": "The typical weight of the breed, ideally provided as a range in pounds or kilograms.",
        },
        {
            "name": "body_length",
            "field_type": "str",
            "description": "The length of the dog's body from shoulders to the base of the tail, preferably in inches or centimeters.",
        },
        {
            "name": "coat_length",
            "field_type": "str",
            "description": "The length of the breed's coat, categorized as short, medium, or long. If measurable, please specify in inches. For uneven coat lengths on different parts of the body, extract and note each measurement.",
        },
        {
            "name": "head_shape",
            "field_type": "str",
            "description": "The characteristic shape of the breed's head, such as round, wedge-shaped, or rectangular.",
        },
        {
            "name": "head_length",
            "field_type": "str",
            "description": "The measurable length of the head, from the nose tip to the back of the skull, preferably provided in inches. Avoid vague descriptions if measurements are unavailable.",
        },
        {
            "name": "head_size",
            "field_type": "str",
            "description": "The overall dimensions of the head, including relative proportions to the body size.",
        },
        {
            "name": "head_width",
            "field_type": "str",
            "description": "The width of the head measured across the broadest part, preferably in inches.",
        },
        {
            "name": "jaws_size",
            "field_type": "str",
            "description": "The size and development of the upper and lower jaws.",
        },
        {
            "name": "skull_shape",
            "field_type": "str",
            "description": "The shape of the skull, such as broad, narrow, or domed.",
        },
        {
            "name": "nose_shape",
            "field_type": "str",
            "description": "The shape of the nose, such as flat, pointed, or bulbous.",
        },
        {
            "name": "nose_color",
            "field_type": "str",
            "description": "The standard nose color for the breed, such as black, brown, or pink.",
        },
        {
            "name": "lip_color",
            "field_type": "str",
            "description": "The color of the lips, often matching the nose or coat.",
        },
        {
            "name": "cheek_shape",
            "field_type": "str",
            "description": "The contour of the cheeks, such as flat, pronounced, or rounded.",
        },
        {
            "name": "eye_size",
            "field_type": "str",
            "description": "The relative size of the eyes, preferably quantified with measurements if available.",
        },
        {
            "name": "eye_shape",
            "field_type": "str",
            "description": "The typical eye shape for the breed, such as almond, round, or oval.",
        },
        {
            "name": "eye_color",
            "field_type": "str",
            "description": "The standard eye color for the breed, including common variations.",
        },
        {
            "name": "eyelid_color",
            "field_type": "str",
            "description": "The color of the eyelids, which may contrast with or match the surrounding fur.",
        },
        {
            "name": "ear_size",
            "field_type": "str",
            "description": "The size of the ears, preferably quantified with measurements.",
        },
        {
            "name": "ear_shape",
            "field_type": "str",
            "description": "The typical ear shape, such as floppy, erect, or semi-erect.",
        },
        {
            "name": "ear_direction",
            "field_type": "str",
            "description": "The natural ear position, such as erect, drop, or upright. Note any descriptions of ear movement in different states (e.g., relaxed vs. alert).",
        },
        {
            "name": "feather",
            "field_type": "str",
            "description": "The presence of decorative or longer fur, with measurements in inches where possible.",
        },
        {
            "name": "ear_leather",
            "field_type": "str",
            "description": "The thickness or texture of the ear's skin.",
        },
        {
            "name": "neck_length",
            "field_type": "str",
            "description": "The length of the neck, preferably measured in inches.",
        },
        {
            "name": "back_length",
            "field_type": "str",
            "description": "The length of the back from shoulders to the hips, preferably in inches.",
        },
        {
            "name": "brisket_width",
            "field_type": "str",
            "description": "The width of the chest, preferably provided in inches or centimeters.",
        },
        {
            "name": "loin_length",
            "field_type": "str",
            "description": "The length of the loin area, preferably measured in inches.",
        },
        {
            "name": "muscle_strength",
            "field_type": "str",
            "description": "The muscular development of the breed, with quantitative descriptions if possible (e.g., shoulder circumference).",
        },
        {
            "name": "tail_length",
            "field_type": "str",
            "description": "The length of the tail, preferably measured in inches.",
        },
        {
            "name": "tail_direction",
            "field_type": "str",
            "description": "The direction of the tail, such as upward, downward, or horizontal. Please extract details of movement or resting positions if available.",
        },
        {
            "name": "tail_curve",
            "field_type": "str",
            "description": "The curvature of the tail, such as straight, sickle-shaped, or tightly curled.",
        },
        {
            "name": "tail_feather",
            "field_type": "str",
            "description": "The presence of long or decorative fur on the tail, preferably measured in inches.",
        },
        {
            "name": "shoulder_muscle",
            "field_type": "str",
            "description": "The muscular development in the shoulder area, with measurements or qualitative strength descriptions if possible.",
        },
        {
            "name": "feet_size",
            "field_type": "str",
            "description": "The size of the feet, preferably provided with length and width measurements.",
        },
        {
            "name": "feet_width",
            "field_type": "str",
            "description": "The width of the feet, measured across the broadest part in inches or centimeters.",
        },
        {
            "name": "feet_length",
            "field_type": "str",
            "description": "The length of the feet, measured from heel to toe in inches or centimeters.",
        },
        {
            "name": "personality",
            "field_type": "str",
            "description": "The typical temperament and behavioral traits of the breed, such as friendly, energetic, protective, or independent.",
        },
    ]

    initial_prompt = (
        "Given breed text from a breed standard file, Only extract breed trait value that reveals breed features. If you do not find the information, please give an empty string."
    )

    try:
        if template_json_path.exists():
            tmpl = json.loads(template_json_path.read_text(encoding="utf-8"))
            d = tmpl.get("data") or {}
            if isinstance(d.get("inputFields"), list):
                input_fields = d["inputFields"]
            if isinstance(d.get("outputFields"), list):
                output_fields = d["outputFields"]
            if isinstance(d.get("initial_prompt"), str):
                initial_prompt = d["initial_prompt"]
    except Exception:
        pass

    yml_path = run_dir / f"config_{ts}.yml"
    try:
        sys.path.append(str(repo_root))
        from gui.yml_generation.yml_doc_extraction import generate_novel_config_yaml

        generate_novel_config_yaml(
            dataset_path=str(dataset),
            save_dir=str(save_dir),
            input_fields=input_fields,
            output_fields=output_fields,
            initial_prompt=initial_prompt,
            judging="",
            task="Extraction",
            threads=6,
            multiple=True,
            output_path=str(yml_path),
        )
    except Exception:
        lines = []
        lines.append("# PredictionSettings YAML Configuration")
        lines.append(
            "# Configuration for dataset extraction from Nature Communications journal articles"
        )
        lines.append("# Copy this file and modify according to your needs")
        lines.append("# === INPUT FIELDS ===")
        lines.append("# Define what your model will receive as input")
        lines.append("inputFields:")
        for item in input_fields:
            lines.append(f'  - name: "{item.get("name", "")}"')
            lines.append(f'    field_type: "{item.get("field_type", "str")}"')
            if "description" in item:
                lines.append(f'    description: "{item["description"]}"')
        lines.append("# === OUTPUT FIELDS ===")
        lines.append("# Define what your model should extract or generate from the input")
        lines.append("outputFields:")
        for item in output_fields:
            lines.append(f'  - name: "{item.get("name", "")}"')
            lines.append(f'    field_type: "{item.get("field_type", "str")}"')
            if "description" in item:
                lines.append(f'    description: "{item["description"]}"')
        lines.append("# === PROMPT CONFIGURATION ===")
        lines.append("initial_prompt: |")
        for l in initial_prompt.split("\n"):
            lines.append(f"  {l}")
        lines.append("# === DATA AND SAVE CONFIGURATION ===")
        lines.append(f"dataset: {str(dataset)}")
        lines.append(f"save_dir: {str(save_dir)}")
        lines.append("# === EVALUATION SETTINGS ===")
        lines.append('judging: ""')
        lines.append("# === PROCESSING SETTINGS ===")
        lines.append('task: "Extraction"')
        lines.append("threads: 6")
        lines.append("multiple: true")
        lines.append("# === USAGE INSTRUCTIONS ===")
        lines.append("# 1. Replace dataset path with your actual dataset file path")
        lines.append("# 2. Modify save_dir to your desired output directory")
        lines.append("# 3. Update inputFields and outputFields to match your data structure")
        lines.append("# 4. Ensure your dataset has columns matching all field names")
        lines.append("# 5. Adjust threads based on your system's capabilities")
        yml_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = run_dir / f"original_extraction_{ts}.json"
    now_iso = datetime.now().isoformat()
    payload = {
        "name": "breed files curation(original)",
        "data": {
            "dataset": str(dataset),
            "save_dir": str(save_dir),
            "inputFields": input_fields,
            "outputFields": output_fields,
            "initial_prompt": initial_prompt,
            "judging": "",
            "task": "Extraction",
            "threads": 6,
            "multiple": True,
        },
        "status": "created",
        "created_time": now_iso,
        "modified_time": now_iso,
        "error": None,
        "result": {"message": "config generated"},
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(run_dir))
    print(str(yml_path))
    print(str(json_path))

    # Also generate doc_parsing runs matching existing examples
    doc_parsing_dir = repo_root / "gui" / "runs" / "doc_parsing"
    doc_parsing_dir.mkdir(parents=True, exist_ok=True)

    # Run A: latest timestamp -> files_for_curation -> files_for_curation_json
    now_a = datetime.now()
    run_a_ts = now_a.strftime("%Y%m%d_%H%M%S")
    run_a_dir = doc_parsing_dir / f"run_{run_a_ts}"
    run_a_dir.mkdir(parents=True, exist_ok=True)
    folder_a = example_dir / "files_for_curation"
    save_a = example_dir / "files_for_curation_json"
    save_a.mkdir(parents=True, exist_ok=True)
    yml_a = run_a_dir / f"file_to_json_config_{run_a_ts}.yml"
    json_a = run_a_dir / f"file_to_json_{run_a_ts}.json"

    try:
        from gui.yml_generation.yml_document_parsing import generate_document_parsing_yaml

        generate_document_parsing_yaml(
            folder_path=str(folder_a),
            save_path=str(save_a),
            file_type="PDF",
            convert_mode="wholeDoc",
            output_path=str(yml_a),
        )
    except Exception:
        yml_a.write_text(
            f"""# YAML Configuration for File to JSON Conversion
# Copy this file and modify according to your needs

# === FILE SOURCE CONFIGURATION ===
# Define the source folder containing files to be converted

folder_path: {str(folder_a)}  # Path to the folder containing source files

# === OUTPUT CONFIGURATION ===
# Define where the converted JSON files will be saved
save_path: {str(save_a)}  # Path to save JSON results

# === FILE TYPE CONFIGURATION ===
# Specify the type of files to be processed
# Optional values: 'PDF', 'scienceDirect', 'PMC', 'Arxiv'
file_type: "PDF"

# === CONVERSION MODE ===
# Choose the conversion strategy
# Optional values: 'byPart' (divide the document by article parts), 'wholeDoc' (convert entire document as one)
convert_mode: "wholeDoc"

# === USAGE INSTRUCTIONS ===
# 1. Replace folder_path with your actual source folder path
# 2. Modify save_path to your desired output directory
# 3. Update file_type according to your source file format
# 4. Choose appropriate convert_mode based on your requirements
# 5. The converted JSON files can be used for building datasets for prompt optimization or extraction tasks""",
            encoding="utf-8",
        )

    json_a_payload = {
        "name": "document parsing for breed files(for curation)",
        "data": {
            "folder_path": str(folder_a),
            "save_path": str(save_a),
            "file_type": "PDF",
            "convert_mode": "wholeDoc",
        },
        "status": "created",
        "created_time": now_a.isoformat(),
        "modified_time": now_a.isoformat(),
        "error": None,
        "result": {
            "message": "file_to_json config generated",
            "result": {
                "message": "Ready to convert files to JSON with wholeDoc mode",
                "details": {
                    "dataset_file": str(save_a / "_dataset.json"),
                },
            },
        },
    }
    json_a.write_text(json.dumps(json_a_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Run B: next latest timestamp -> curated_breed_files -> curated_breed_files_json
    now_b = now_a + timedelta(seconds=1)
    run_b_ts = now_b.strftime("%Y%m%d_%H%M%S")
    run_b_dir = doc_parsing_dir / f"run_{run_b_ts}"
    run_b_dir.mkdir(parents=True, exist_ok=True)
    folder_b = example_dir / "curated_breed_files"
    save_b = example_dir / "curated_breed_files_json"
    save_b.mkdir(parents=True, exist_ok=True)
    yml_b = run_b_dir / f"file_to_json_config_{run_b_ts}.yml"
    json_b = run_b_dir / f"file_to_json_{run_b_ts}.json"

    try:
        from gui.yml_generation.yml_document_parsing import generate_document_parsing_yaml

        generate_document_parsing_yaml(
            folder_path=str(folder_b),
            save_path=str(save_b),
            file_type="PDF",
            convert_mode="wholeDoc",
            output_path=str(yml_b),
        )
    except Exception:
        yml_b.write_text(
            f"""# YAML Configuration for File to JSON Conversion
# Copy this file and modify according to your needs

# === FILE SOURCE CONFIGURATION ===
# Define the source folder containing files to be converted

folder_path: {str(folder_b)}  # Path to the folder containing source files

# === OUTPUT CONFIGURATION ===
# Define where the converted JSON files will be saved
save_path: {str(save_b)}  # Path to save JSON results

# === FILE TYPE CONFIGURATION ===
# Specify the type of files to be processed
# Optional values: 'PDF', 'scienceDirect', 'PMC', 'Arxiv'
file_type: "PDF"

# === CONVERSION MODE ===
# Choose the conversion strategy
# Optional values: 'byPart' (divide the document by article parts), 'wholeDoc' (convert entire document as one)
convert_mode: "wholeDoc"

# === USAGE INSTRUCTIONS ===
# 1. Replace folder_path with your actual source folder path
# 2. Modify save_path to your desired output directory
# 3. Update file_type according to your source file format
# 4. Choose appropriate convert_mode based on your requirements
# 5. The converted JSON files can be used for building datasets for prompt optimization or extraction tasks""",
            encoding="utf-8",
        )

    json_b_payload = {
        "name": "document parsing for breed files(curated)",
        "data": {
            "folder_path": str(folder_b),
            "save_path": str(save_b),
            "file_type": "PDF",
            "convert_mode": "wholeDoc",
        },
        "status": "created",
        "created_time": now_b.isoformat(),
        "modified_time": now_b.isoformat(),
        "error": None,
        "result": {
            "message": "file_to_json config generated",
            "result": {
                "message": "Ready to convert files to JSON with wholeDoc mode",
                "details": {
                    "dataset_file": str(save_b / "_dataset.json"),
                },
            },
        },
    }
    json_b.write_text(json.dumps(json_b_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(run_a_dir))
    print(str(yml_a))
    print(str(json_a))
    print(str(run_b_dir))
    print(str(yml_b))
    print(str(json_b))

    # Generate optimized extraction run content under optm with specified timestamp
    optm_dir = repo_root / "gui" / "runs" / "doc_extraction" / "optm"
    optm_dir.mkdir(parents=True, exist_ok=True)
    optm_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    optm_run_dir = optm_dir / optm_ts
    optm_run_dir.mkdir(parents=True, exist_ok=True)

    load_dir = example_dir / "optimized_prompt"
    dataset_file = example_dir / "files_for_curation_json" / "_dataset.json"
    save_dir_optim = example_dir / "breed_curation_optimized"
    save_dir_optim.mkdir(parents=True, exist_ok=True)

    yml_optm = optm_run_dir / f"pred_optimized_config_{optm_ts}.yml"
    json_optm = optm_run_dir / f"pred_optimized_{optm_ts}.json"

    try:
        from gui.yml_generation.yml_doc_extraction import generate_optimized_config_yaml

        generate_optimized_config_yaml(
            load_dir=str(load_dir),
            dataset=str(dataset_file),
            save_dir=str(save_dir_optim),
            judging="",
            output_file="result.json",
            threads=6,
            output_path=str(yml_optm),
        )
    except Exception:
        lines = []
        lines.append("# YAML Configuration for Extraction tasks with optimization")
        lines.append("# Copy this file and modify according to your needs")
        lines.append("# === REQUIRED PATHS ===")
        lines.append("# Directory containing the optimized settings and prompts from previous optimization")
        lines.append(f"load_dir: {str(load_dir)}")
        lines.append("# Dataset to run predictions on")
        lines.append(f"dataset: {str(dataset_file)}")
        lines.append("# Directory where prediction results will be saved")
        lines.append(f"save_dir: {str(save_dir_optim)}")
        lines.append("# === EVALUATION SETTINGS ===")
        lines.append('# Evaluation mode: "confidence" (evaluate prediction confidence), "score" (evaluate prediction quality), or "" for no judgement')
        lines.append('judging: ""')
        lines.append("# === OUTPUT CONFIGURATION ===")
        lines.append("# Output file name for the prediction results")
        lines.append("output_file: result.json")
        yml_optm.write_text("\n".join(lines), encoding="utf-8")

    now_optm = datetime.now().isoformat()
    json_optm_payload = {
        "name": "breed files curation(optimized)",
        "data": {
            "load_dir": str(load_dir),
            "dataset": str(dataset_file),
            "save_dir": str(save_dir_optim),
            "judging": "",
            "threads": 6,
            "output_file": "result.json",
        },
        "status": "created",
        "created_time": now_optm,
        "modified_time": now_optm,
        "error": None,
        "result": {"message": "config generated"},
    }
    json_optm.write_text(
        json.dumps(json_optm_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(str(optm_run_dir))
    print(str(yml_optm))
    print(str(json_optm))

    # Generate prompt optimization run content with runtime timestamp
    prompt_runs_dir = repo_root / "gui" / "runs" / "prompt_optimization"
    prompt_runs_dir.mkdir(parents=True, exist_ok=True)
    prompt_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_run_dir = prompt_runs_dir / f"run_{prompt_ts}"
    prompt_run_dir.mkdir(parents=True, exist_ok=True)

    prompt_dataset = example_dir / "curated_dataset" / "_optim_dataset.json"
    prompt_save_dir = example_dir / "optimized_prompt"
    prompt_save_dir.mkdir(parents=True, exist_ok=True)

    yml_prompt_path = prompt_run_dir / f"optim_{prompt_ts}.yml"
    json_prompt_path = prompt_run_dir / f"optim_{prompt_ts}.json"

    try:
        from gui.yml_generation.yml_prompt_generation import (
            generate_prompt_optimization_yaml,
        )

        generate_prompt_optimization_yaml(
            dataset_path=str(prompt_dataset),
            save_dir=str(prompt_save_dir),
            experiment_name=f"optim_{prompt_ts}",
            input_fields=input_fields,
            output_fields=output_fields,
            initial_prompt=initial_prompt,
            task="Extraction",
            optim_burden="medium",
            threads=6,
            demos=1,
            multiple=False,
            ai_evaluation=True,
            recall_prior=False,
            output_path=str(yml_prompt_path),
        )
    except Exception:
        # Fallback YAML content
        lines = []
        lines.append("# Template Configuration for Optim Custom")
        lines.append("# Copy this file and modify according to your needs")
        lines.append("")
        lines.append("# === INPUT FIELDS ===")
        lines.append("# Define what data your model will receive")
        lines.append("inputFields:")
        for item in input_fields:
            lines.append(f'  - name: "{item.get("name", "")}"')
            lines.append(f'    field_type: "{item.get("field_type", "str")}"')
            if "description" in item:
                lines.append(f'    description: "{item["description"]}"')
        lines.append("")
        lines.append("# === OUTPUT FIELDS ===")
        lines.append("# Define what your model should extract or generate")
        lines.append("outputFields:")
        for item in output_fields:
            lines.append(f'  - name: "{item.get("name", "")}"')
            lines.append(f'    field_type: "{item.get("field_type", "str")}"')
            if "description" in item:
                lines.append(f'    description: "{item["description"]}"')
        lines.append("")
        lines.append("# === PROMPT CONFIGURATION ===")
        lines.append("initial_prompt: |")
        for l in initial_prompt.split("\n"):
            lines.append(f"  {l}")
        lines.append("")
        lines.append("# === DATA AND SAVE CONFIGURATION ===")
        lines.append(f"dataset: {str(prompt_dataset)}")
        lines.append(f"save_dir: {str(prompt_save_dir)}")
        lines.append("")
        lines.append("# === OPTIMIZATION SETTINGS ===")
        lines.append("task: Extraction")
        lines.append("optim_burden: medium")
        lines.append("threads: 6")
        lines.append("demos: 1")
        lines.append("")
        lines.append("# === EXTRACTION MODE ===")
        lines.append("multiple: False")
        lines.append("recall_prior: False")
        lines.append("ai_evaluation: True")
        lines.append("")
        lines.append("# === USAGE INSTRUCTIONS ===")
        lines.append("# 1. Replace all placeholder paths with your actual file paths")
        lines.append("# 2. Modify inputFields and outputFields to match your data structure")
        lines.append("# 3. Update initial_prompt with your specific task instructions")
        lines.append("# 4. Ensure your dataset CSV has columns matching all field names")
        lines.append("# 5. Run with: python cli_handler.py optim_custom path/to/this/config.yml")
        yml_prompt_path.write_text("\n".join(lines), encoding="utf-8")

    now_prompt_iso = datetime.now().isoformat()
    json_prompt_payload = {
        "name": "prompt optimization for breed",
        "data": {
            "inputFields": input_fields,
            "outputFields": output_fields,
            "initial_prompt": initial_prompt,
            "dataset": str(prompt_dataset),
            "save_dir": str(prompt_save_dir),
            "task": "Extraction",
            "optim_burden": "medium",
            "threads": 6,
            "demos": 1,
            "multiple": False,
            "ai_evaluation": True,
        },
        "status": "created",
        "created_time": now_prompt_iso,
        "modified_time": now_prompt_iso,
        "error": None,
        "result": {"message": "optim config generated"},
    }
    json_prompt_path.write_text(
        json.dumps(json_prompt_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(str(prompt_run_dir))
    print(str(yml_prompt_path))
    print(str(json_prompt_path))

    build_runs_dir = repo_root / "gui" / "runs" / "build_optim_dataset"
    build_runs_dir.mkdir(parents=True, exist_ok=True)
    build_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    build_run_dir = build_runs_dir / f"run_{build_ts}"
    build_run_dir.mkdir(parents=True, exist_ok=True)

    json_dir = example_dir / "curated_breed_files_json"
    curated_dataset_path = example_dir / "breed_curated_data.json"
    save_dir_build = example_dir / "curated_dataset"
    save_dir_build.mkdir(parents=True, exist_ok=True)

    yml_build = build_run_dir / f"build_optm_set_config_{build_ts}.yml"
    json_build = build_run_dir / f"build_dataset_{build_ts}.json"

    try:
        sys.path.append(str(repo_root))
        from gui.yml_generation.yml_build_optm_dataset import (
            generate_build_optm_dataset_yml_from_dash_callback,
        )

        yml_content = generate_build_optm_dataset_yml_from_dash_callback(
            json_path=str(json_dir),
            curated_dataset_path=str(curated_dataset_path),
            fields_data=output_fields,
            multiple_entities=False,
            article_field="breed_file",
            article_parts=[],
            save_directory=str(save_dir_build),
        )
        yml_build.write_text(yml_content, encoding="utf-8")
    except Exception:
        lines = []
        lines.append("# Template Configuration for Build Optm Set")
        lines.append("# This config is used for the /api/build_optm_set endpoint")
        lines.append("")
        lines.append(f"json_path: {str(json_dir)}")
        lines.append(f"dataset: {str(curated_dataset_path)}")
        lines.append("fields:")
        for item in output_fields:
            lines.append(f"- name: {item.get('name', '')}")
            lines.append(f"  field_type: {item.get('field_type', 'str')}")
            desc = item.get("description")
            if isinstance(desc, str) and desc:
                lines.append(f"  description: {desc}")
        lines.append("multiple: false")
        lines.append("article_field: breed_file")
        lines.append(f"save_dir: {str(save_dir_build)}")
        yml_build.write_text("\n".join(lines), encoding="utf-8")

    now_build = datetime.now().isoformat()
    json_build_payload = {
        "name": "build optimization dataset for breed",
        "data": {
            "json_path": str(json_dir),
            "dataset": str(curated_dataset_path),
            "save_dir": str(save_dir_build),
            "fields": output_fields,
            "multiple": False,
            "article_field": "breed_file",
            "article_parts": None,
        },
        "status": "created",
        "created_time": now_build,
        "modified_time": now_build,
        "error": None,
        "result": {
            "message": "build_optm_set config generated",
            "result": {
                "message": "Ready to build optimization dataset",
                "details": {
                    "target_file": str(save_dir_build / "_optim_dataset.json"),
                },
            },
        },
    }
    json_build.write_text(
        json.dumps(json_build_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(str(build_run_dir))
    print(str(yml_build))
    print(str(json_build))

    # Generate table parsing run content
    table_parsing_dir = repo_root / "gui" / "runs" / "table_parsing"
    table_parsing_dir.mkdir(parents=True, exist_ok=True)
    tp_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tp_run_dir = table_parsing_dir / f"run_{tp_ts}"
    tp_run_dir.mkdir(parents=True, exist_ok=True)

    tp_folder = example_dir / "ewas_sup_tables"
    tp_save = example_dir / "ewas_sup_tables_parsed"
    tp_save.mkdir(parents=True, exist_ok=True)
    tp_yml = tp_run_dir / f"parse_table_to_tsv_config_{tp_ts}.yml"
    try:
        sys.path.append(str(repo_root))
        from gui.yml_generation.yml_table_parsing import generate_table_parsing_yaml
        generate_table_parsing_yaml(
            file_folder_path=str(tp_folder),
            save_folder_path=str(tp_save),
            non_tabular_file_format="PDF",
            output_path=str(tp_yml),
        )
    except Exception:
        lines = []
        lines.append("# parse_table_to_tsv Configuration File")
        lines.append("# This file is used to configure parameters for table parsing functionality")
        lines.append("")
        lines.append(f"file_folder_path: {str(tp_folder)}")
        lines.append(f"save_folder_path: {str(tp_save)}")
        lines.append("non_tabular_file_format: PDF")
        lines.append("")
        lines.append("# Usage Instructions:")
        lines.append("# 1. Set file_folder_path to the folder path containing source files with tables to parse")
        lines.append("# 2. Set save_folder_path to the folder path where parsed TSV files will be saved")
        lines.append("# 3. Set non_tabular_file_format parameter based on source file type, optional values are 'PDF', 'scienceDirect', 'PMC', 'Arxiv'")
        lines.append("# 4. If file encoding is not utf-8, you can modify the encoding parameter")
        lines.append("# 5. If you need to see detailed processing information, you can set verbose to true")
        tp_yml.write_text("\n".join(lines), encoding="utf-8")

    tp_json = tp_run_dir / f"parse_table_to_tsv_{tp_ts}.json"
    now_tp = datetime.now().isoformat()
    tp_payload = {
        "name": "EWAS table parsing",
        "data": {
            "file_folder_path": str(tp_folder),
            "save_folder_path": str(tp_save),
            "non_tabular_file_format": "PDF",
        },
        "status": "created",
        "created_time": now_tp,
        "modified_time": now_tp,
        "error": None,
        "result": {"message": "parse_table_to_tsv config generated", "result": []},
    }
    tp_json.write_text(json.dumps(tp_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(tp_run_dir))
    print(str(tp_yml))
    print(str(tp_json))

    # Generate table extraction run content
    table_extraction_dir = repo_root / "gui" / "runs" / "table_extraction"
    table_extraction_dir.mkdir(parents=True, exist_ok=True)
    te_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    te_run_dir = table_extraction_dir / f"run_{te_ts}"
    te_run_dir.mkdir(parents=True, exist_ok=True)

    te_parsed = example_dir / "ewas_sup_tables_parsed"
    te_save = example_dir / "ewas_sup_tables_extract_result"
    te_save.mkdir(parents=True, exist_ok=True)

    te_output_fields = [
        {"name": "probe_id", "field_type": "str", "description": "Unique identifier for the probe for DNA methylation used in the study,such as 'cgxxxxxxxx'"},
        {"name": "p_value", "field_type": "str", "description": "P-value associated with the statistical test conducted in the study"},
        {"name": "adjusted_p_value", "field_type": "str", "description": "Adjusted P-value associated with the statistical test conducted in the study"},
        {"name": "delta_beta", "field_type": "str", "description": "Change in DNA methylation level between the case and control groups."},
        {"name": "estimate", "field_type": "str", "description": "Estimate value of the probe from ewas study"},
        {"name": "case_beta", "field_type": "str", "description": "Mean DNA methylation level in the case group, represented by Beta values, ranging from 0 to 1"},
        {"name": "control_beta", "field_type": "str", "description": "Mean DNA methylation level in the control group, represented by Beta values, ranging from 0 to 1"},
    ]

    te_classify_prompt = (
        "Given the table and meta information extracted from a biological article, the categories of the target table are:\n"
        "1. table about DNA differential methylation analysis.\n"
        "2. table about Epigenome-Wide Association Study(EWAS).\n"
        "2. table about DNA methylation probes and Mendelian Randomization analysis.\n\n"
        "Please classify whether the table belongs to the target table categories."
    )
    te_extract_prompt = (
        "Parse the provided tab-delimited table containing methylation probe data and generate a JSON output with the following structure. For each row (probe), extract:\n"
        "probe_id (Unique identifier for the probe for DNA methylation used in the study,such as 'cg08540958')\n"
        "p_value (P-value associated with the statistical test conducted in the study)\n"
        "adjusted_p_value (Adjusted P-value associated with the statistical test conducted in the study)\n"
        "delta_beta (Change in DNA methylation level between the case and control groups.)\n"
        "estimate (Estimate value of the probe from ewas study)\n"
        "case_beta (Mean DNA methylation level in the case group, represented by Beta values, ranging from 0 to 1)\n"
        "control_beta. (Mean DNA methylation level in the control group, represented by Beta values, ranging from 0 to 1)\n"
        "Follow these rules:\n"
        "Map column names exactly as specified above.\n"
        "Represent missing/empty values as ''.\n"
        "Ensure numerical values retain precision (do not round).\n"
        "Ensure extracting target information for every probe."
    )

    te_yml = te_run_dir / f"extract_table_service_config_{te_ts}.yml"
    try:
        sys.path.append(str(repo_root))
        from gui.yml_generation.yml_table_parsing import generate_extract_table_yaml
        generate_extract_table_yaml(
            parsed_file_path=str(te_parsed),
            save_folder_path=str(te_save),
            output_fields=te_output_fields,
            classify_prompt=te_classify_prompt,
            extract_prompt=te_extract_prompt,
            num_threads=6,
            encoding="utf-8",
            output_path=str(te_yml),
        )
    except Exception:
        lines = []
        lines.append("# extract_table_service Configuration File")
        lines.append("# This file is used to configure parameters for table extraction service functionality")
        lines.append("")
        lines.append(f"parsed_file_path: {str(te_parsed)}")
        lines.append(f"save_folder_path: {str(te_save)}")
        lines.append("outputFields:")
        for item in te_output_fields:
            lines.append(f"  - name: {item['name']}")
            lines.append(f"    field_type: {item['field_type']}")
            lines.append(f"    description: {item['description']}")
        lines.append("classify_prompt: |")
        for l in te_classify_prompt.split("\n"):
            lines.append(f"  {l}")
        lines.append("extract_prompt: |")
        for l in te_extract_prompt.split("\n"):
            lines.append(f"  {l}")
        lines.append("num_threads: 6")
        lines.append("encoding: utf-8")
        lines.append("")
        lines.append("# Usage Instructions:")
        lines.append("# 1. Set parsed_file_path to the folder path containing parsed files")
        lines.append("# 2. Set save_folder_path to the folder path where extracted tables will be saved")
        lines.append("# 3. Configure outputFields with the list of fields you want to extract")
        lines.append("# 4. Set appropriate prompts for classification and extraction")
        lines.append("# 5. If you want to extract directly without classification, set extract_directly to true")
        lines.append("# 6. Adjust num_threads according to your system capabilities")
        lines.append("# 7. If file encoding is not utf-8, you can modify the encoding parameter")
        te_yml.write_text("\n".join(lines), encoding="utf-8")

    te_json = te_run_dir / f"extract_table_service_{te_ts}.json"
    now_te = datetime.now().isoformat()
    te_payload = {
        "name": "EWAS table extraction",
        "data": {
            "parsed_file_path": str(te_parsed),
            "save_folder_path": str(te_save),
            "outputFields": te_output_fields,
            "classify_prompt": te_classify_prompt,
            "extract_prompt": te_extract_prompt,
            "extract_directly": False,
            "num_threads": 6,
        },
        "status": "created",
        "created_time": now_te,
        "modified_time": now_te,
        "error": None,
        "result": {
            "message": "extract_table_service config generated",
            "result": {
                "message": "Ready to extract tables",
                "format_table_path": str(te_save / "format_tables"),
            },
        },
    }
    te_json.write_text(json.dumps(te_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(te_run_dir))
    print(str(te_yml))
    print(str(te_json))


if __name__ == "__main__":
    main()
