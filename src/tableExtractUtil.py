import dspy
import sys
import subprocess
import os
import shutil
import pandas as pd
from loguru import logger
from model import get_model_settings
from dspy.utils.parallelizer import ParallelExecutor
from optimUtil import create_output_model_class, DspyField
from typing import List
import chardet


def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]
        return encoding if encoding is not None else "utf-8"


class tableCodeSignature(dspy.Signature):
    __doc__ = """You are a Python programming expert, capable of implementing various requests through Python coding. My request is: write a piece of Python code based on pandas to transform the source table into the demo table format."""
    source_table: str = dspy.InputField(description="Part of the source table content.")
    demo_table: str = dspy.InputField(description="The demo table.")
    source_table_path: str = dspy.InputField(
        description="The path of the source table."
    )
    output_table_path: str = dspy.InputField(
        description="The path of the output table."
    )
    code: str = dspy.OutputField(description="The Python code.")


def generate_table_code(
    source_table, demo_table, source_table_path, output_table_path, script_path
):
    """
    Generate Python code to transform the source table into the demo table format.
    """
    table_code_generator = dspy.Predict(tableCodeSignature)
    code = table_code_generator(
        source_table=source_table,
        demo_table=demo_table,
        source_table_path=source_table_path,
        output_table_path=output_table_path,
    )
    with open(script_path, "w") as f:
        f.write(code["code"])


def write_code_to_file(code, script_path):
    with open(script_path, "w") as f:
        f.write(code)


def print_coder_model_settings():
    try:
        # 获取coder模型设置
        coder_settings = get_model_settings("coder")
        print("Coder Model Settings:")
        print(f"Model Name: {coder_settings.model_name}")
        print(f"Model Type: {coder_settings.model_type}")
        print(f"API Base: {coder_settings.api_base}")
        print(f"Temperature: {coder_settings.temperature}")
        print(f"Max Tokens: {coder_settings.max_tokens}")
        print(f"Top P: {coder_settings.top_p}")
        print(f"Top K: {coder_settings.top_k}")
        print(f"Min P: {coder_settings.min_p}")
        return coder_settings
    except Exception as e:
        logger.error(f"Error getting coder model settings: {e}")
        return None


def run_py(script_path):
    try:
        result = subprocess.run(
            [sys.executable, script_path], capture_output=True, text=True
        )

        if result.stderr:
            logger.info(f"Error executing script {script_path}:{result.stderr}")
            return {"message": f"Error executing script {script_path}:{result.stderr}"}
        else:
            return {"message": "success"}

    except subprocess.CalledProcessError as e:
        logger.info(f"Error executing script {script_path}:{e}")
        # CalledProcessError doesn't have stderr attribute, using result.stderr or e.output
        error_message = e.output if e.output else str(e)
        return {"message": f"Error executing script {script_path}:{error_message}"}
    except Exception as e:
        logger.info(f"Error executing script {script_path}:{e}")
        return {"message": f"Error executing script {script_path}:{e}"}


def read_file(file_path, start_line, end_line):
    encoding = detect_encoding(file_path)
    with open(file_path, "r", encoding=encoding) as file:
        lines = file.readlines()[start_line:end_line]
    return "\n".join(lines)


class tableReactSignature(dspy.Signature):
    __doc__ = """As a professional data analyst, you are tasked with transforming the data in `source_table` to exactly match the structure, data types, and formatting of `demo_table`. The transformed table should be saved into `output_table_path`. Please place the generated code files in the same folder as `output_table_path`, too. Please ensure the transformed output maintains consistency with `demo_table` in all aspects, including column names, data organization, and any applied transformations.After the conversion is completed, please help me read the first few lines of the transformed table and compare them with the demo table to verify its correctness."""
    source_table: str = dspy.InputField(description="Part of the source table content.")
    demo_table: str = dspy.InputField(description="The demo table content.")
    source_table_path: str = dspy.InputField(
        description="The path of the source table."
    )
    output_table_path: str = dspy.InputField(
        description="The path of the output table."
    )
    result: bool = dspy.OutputField(
        description="Whether the table is transformed successfully."
    )


def classify_tables(
    file_folder_path: str,
    save_folder_path: str,
    prompt: str,
    num_threads: int = 6,
    encoding: str = "utf-8",
):
    if not os.path.exists(file_folder_path):
        raise OSError(f"Folder {file_folder_path} does not exist.")

    if not os.path.exists(save_folder_path):
        try:
            os.mkdir(save_folder_path)
        except OSError as e:
            raise OSError(f"Failed to create folder {save_folder_path}: {e}")

    tsv_files = [file for file in os.listdir(file_folder_path) if file.endswith(".tsv")]
    meta_files = [
        file for file in os.listdir(file_folder_path) if file.endswith(".meta")
    ]

    class TableClassifier(dspy.Signature):
        __doc__ = prompt
        table_content: str = dspy.InputField(description="The content of the table.")
        table_meta: str = dspy.InputField(
            description="The meta information of the table."
        )
        is_target_table: bool = dspy.OutputField(
            description="Whether the table is the target table."
        )
        classification_reason: str = dspy.OutputField(
            description="The reason for the classification."
        )

    tableClassifier = dspy.ChainOfThought(TableClassifier)
    table_dataset = []
    # generate dataset for predict
    try:
        for file in tsv_files:
            table_content = ""
            encoding = detect_encoding(os.path.join(file_folder_path, file))
            file_df = pd.read_csv(
                os.path.join(file_folder_path, file),
                sep="\t",
                encoding=encoding,
                header=None,
            )
            if len(file_df) > 20:
                for index, row in file_df[0:10].iterrows():
                    row_values = [
                        str(value) if not pd.isnull(value) else " "
                        for value in row.values
                    ]
                    table_content += "\t".join(row_values) + "\n"
                for index, row in file_df[-10:-1].iterrows():
                    row_values = [
                        str(value) if not pd.isnull(value) else " "
                        for value in row.values
                    ]
                    table_content += "\t".join(row_values) + "\n"
            else:
                for index, row in file_df.iterrows():
                    row_values = [
                        str(value) if not pd.isnull(value) else " "
                        for value in row.values
                    ]
                    table_content += "\t".join(row_values) + "\n"
            table_meta = ""
            meta_file_name = ".".join(file.split(".")[:-1]) + ".meta"
            if meta_file_name in meta_files:
                with open(
                    os.path.join(file_folder_path, meta_file_name),
                    "r",
                    encoding="utf-8",
                ) as f:
                    meta_list = f.readlines()
                    if len(meta_list) > 0:
                        table_meta = meta_list[0]
            table_dataset.append(
                dspy.Example(
                    table_content=table_content, table_meta=table_meta
                ).with_inputs("table_content", "table_meta")
            )

        def process_item(example):
            return tableClassifier(**example.inputs())

        executor = ParallelExecutor(
            num_threads=num_threads,
            disable_progress_bar=True,
            max_errors=5,
            provide_traceback=False,
            compare_results=False,
        )
        results = executor.execute(process_item, table_dataset)
        assert len(results) == len(tsv_files)
        list_result = [result.is_target_table for result in results]
        list_reason = [result.classification_reason for result in results]
        df_output = pd.DataFrame(
            data={
                "file_name": tsv_files,
                "is_target_table": list_result,
                "classification_reason": list_reason,
            }
        )
        df_output.to_excel(
            os.path.join(save_folder_path, "classification_result.xlsx"), index=False
        )
    except Exception as e:
        logger.info(e)


def generate_example_tables(
    file_folder_path: str,
    save_folder_path: str,
    prompt: str,
    outputFields: List[DspyField],
    num_threads: int = 6,
    encoding: str = "utf-8",
):
    if not os.path.exists(file_folder_path):
        raise OSError(f"Folder {file_folder_path} does not exist.")

    if not os.path.exists(save_folder_path):
        try:
            os.mkdir(save_folder_path)
        except OSError as e:
            raise OSError(f"Failed to create folder {save_folder_path}: {e}")

    tsv_files = [file for file in os.listdir(file_folder_path) if file.endswith(".tsv")]
    meta_files = [
        file for file in os.listdir(file_folder_path) if file.endswith(".meta")
    ]

    fields = dict()
    fields["table_content"] = (
        str,
        dspy.InputField(description="Table content for extraction"),
    )
    fields["table_meta"] = (
        str,
        dspy.InputField(description="Meta information for extraction"),
    )
    example_table_class = create_output_model_class(outputFields=outputFields)
    fields["extracted_information"] = (
        list[example_table_class],
        dspy.OutputField(description="Extracted information"),
    )
    tableExampleSignature = dspy.make_signature(fields, prompt, "tableExampleSignature")
    tableExamplePredictor = dspy.Predict(tableExampleSignature)

    table_dataset = []
    try:
        for file in tsv_files:
            table_content = ""
            encoding = detect_encoding(os.path.join(file_folder_path, file))
            file_df = pd.read_csv(
                os.path.join(file_folder_path, file),
                sep="\t",
                encoding=encoding,
                header=None,
            )
            if len(file_df) > 10:
                for index, row in file_df[0:10].iterrows():
                    row_values = [
                        str(value) if not pd.isnull(value) else " "
                        for value in row.values
                    ]
                    table_content += "\t".join(row_values) + "\n"
            else:
                for index, row in file_df.iterrows():
                    row_values = [
                        str(value) if not pd.isnull(value) else " "
                        for value in row.values
                    ]
                    table_content += "\t".join(row_values) + "\n"
            table_meta = ""
            meta_file_name = ".".join(file.split(".")[:-1]) + ".meta"
            if meta_file_name in meta_files:
                with open(
                    os.path.join(file_folder_path, meta_file_name),
                    "r",
                    encoding="utf-8",
                ) as f:
                    meta_list = f.readlines()
                    if len(meta_list) > 0:
                        table_meta = meta_list[0]
            table_dataset.append(
                dspy.Example(
                    table_content=table_content, table_meta=table_meta
                ).with_inputs("table_content", "table_meta")
            )

        def process_item(example):
            return tableExamplePredictor(**example.inputs())

        executor = ParallelExecutor(
            num_threads=num_threads,
            disable_progress_bar=True,
            max_errors=5,
            provide_traceback=False,
            compare_results=False,
        )
        results = executor.execute(process_item, table_dataset)
        assert len(results) == len(tsv_files)
        extracted_files = dict()
        for index, file_name in enumerate(tsv_files):
            extracted_info = results[index]["extracted_information"]
            output_file_path = os.path.join(
                save_folder_path, file_name.replace(".tsv", "_example.tsv")
            )
            if extracted_info is not None and len(extracted_info) > 0:
                logger.info(extracted_info)
                df_out = pd.DataFrame(
                    [item.dict() for item in extracted_info if item is not None]
                )
                df_out.to_csv(output_file_path, sep="\t", index=False)
                extracted_files[file_name] = True
            else:
                extracted_files[file_name] = False
        df_info = pd.DataFrame(
            extracted_files.items(), columns=["file_name", "extracted"]
        )
        df_info.to_excel(
            os.path.join(save_folder_path, "extracted_info.xlsx"), index=False
        )
        return True
    except Exception as e:
        logger.error(f"Error in generate_example_tables: {e}")
        raise e


def generate_format_tables(
    file_folder_path,
    demo_folder_path,
    save_folder_path,
    num_threads: int = 6,
    encoding: str = "utf-8",
):
    if not os.path.exists(file_folder_path):
        raise OSError(f"Folder {file_folder_path} does not exist.")
    if not os.path.exists(demo_folder_path):
        raise OSError(f"Folder {demo_folder_path} does not exist.")

    assert os.path.abspath(file_folder_path) != os.path.abspath(demo_folder_path), (
        "file_folder_path and demo_folder_path cannot be the same folder."
    )

    if not os.path.exists(save_folder_path):
        try:
            os.mkdir(save_folder_path)
        except OSError as e:
            raise OSError(f"Failed to create folder {save_folder_path}: {e}")

    write_code_to_file_tool = dspy.react.Tool(
        write_code_to_file,
        desc="Write the generated code to a file.",
        args={"script_path": str, "code": str},
    )
    run_py_tool = dspy.react.Tool(
        run_py, desc="Run the generated python script.", args={"script_path": str}
    )
    read_file_tool = dspy.react.Tool(
        read_file,
        desc="Read the content of a file.",
        args={"file_path": str, "start_line": int, "end_line": int},
    )
    tableReactor = dspy.ReAct(
        tableReactSignature,
        tools=[write_code_to_file_tool, run_py_tool, read_file_tool],
        max_iters=10,
    )
    tsv_with_demo = []
    table_dataset = []
    tsv_files = [file for file in os.listdir(file_folder_path) if file.endswith(".tsv")]
    demo_tsv_files = [
        file for file in os.listdir(demo_folder_path) if file.endswith(".tsv")
    ]
    try:
        for tsv_file in tsv_files:
            tsv_file_path = os.path.join(file_folder_path, tsv_file)
            demo_file_name = tsv_file.replace(".tsv", "_example.tsv")
            if demo_file_name in demo_tsv_files:
                tsv_with_demo.append(tsv_file)
                demo_file_path = os.path.join(demo_folder_path, demo_file_name)
                table_content = ""
                demo_content = ""
                encoding = detect_encoding(tsv_file_path)
                file_df = pd.read_csv(
                    tsv_file_path, sep="\t", encoding=encoding, header=None
                )
                if len(file_df) > 100:
                    for index, row in file_df[0:50].iterrows():
                        row_values = [
                            str(value) if not pd.isnull(value) else " "
                            for value in row.values
                        ]
                        table_content += "\t".join(row_values) + "\n"
                    for index, row in file_df[-50:-1].iterrows():
                        row_values = [
                            str(value) if not pd.isnull(value) else " "
                            for value in row.values
                        ]
                        table_content += "\t".join(row_values) + "\n"
                else:
                    for index, row in file_df.iterrows():
                        row_values = [
                            str(value) if not pd.isnull(value) else " "
                            for value in row.values
                        ]
                        table_content += "\t".join(row_values) + "\n"
                demo_df = pd.read_csv(
                    demo_file_path, sep="\t", header=None, encoding=encoding
                )
                for index, row in demo_df.iterrows():
                    row_values = [
                        str(value) if not pd.isnull(value) else " "
                        for value in row.values
                    ]
                    demo_content += "\t".join(row_values) + "\n"
                if len(table_content) > 30000:
                    table_content = ""
                    for index, row in file_df[0:10].iterrows():
                        row_values = [
                            str(value) if not pd.isnull(value) else " "
                            for value in row.values
                        ]
                        table_content += "\t".join(row_values) + "\n"
                    for index, row in file_df[-10:-1].iterrows():
                        row_values = [
                            str(value) if not pd.isnull(value) else " "
                            for value in row.values
                        ]
                        table_content += "\t".join(row_values) + "\n"
                source_table_path = tsv_file_path
                out_table_path = os.path.join(
                    save_folder_path, os.path.basename(tsv_file_path)
                )
                table_dataset.append(
                    dspy.Example(
                        source_table=table_content,
                        demo_table=demo_content,
                        source_table_path=source_table_path,
                        output_table_path=out_table_path,
                    ).with_inputs(
                        "source_table",
                        "demo_table",
                        "source_table_path",
                        "output_table_path",
                    )
                )

        def process_item(example):
            return tableReactor(**example.inputs())

        executor = ParallelExecutor(
            num_threads=num_threads,
            disable_progress_bar=True,
            max_errors=5,
            provide_traceback=False,
            compare_results=False,
        )

        results = executor.execute(process_item, table_dataset)
    except Exception as e:
        logger.error(f"Exception generate format tables error:{e}")
    assert len(results) == len(tsv_with_demo)

    for idx, result in enumerate(results):
        if not result["result"]:
            logger.error("Exception generate format tables error: no result generated")

    # Clean up .py files in save_folder_path
    for file_name in os.listdir(save_folder_path):
        if not file_name.endswith(".tsv"):
            file_path = os.path.join(save_folder_path, file_name)
            try:
                os.remove(file_path)
                logger.info(f"Deleted {file_path}")
            except OSError as e:
                logger.error(f"Error deleting file {file_path}: {e}")


# Extract tables directly if error occurs when excuting the script
def extract_table_directly(
    file_folder_path: str,
    save_folder_path: str,
    prompt: str,
    outputFields: list,
    num_threads: int = 6,
    encoding: str = "utf-8",
):
    if not os.path.exists(file_folder_path):
        raise OSError(f"Folder {file_folder_path} does not exist.")

    if not os.path.exists(save_folder_path):
        try:
            os.mkdir(save_folder_path)
        except OSError as e:
            raise OSError(f"Failed to create folder {save_folder_path}: {e}")

    tsv_files = [file for file in os.listdir(file_folder_path) if file.endswith(".tsv")]
    fields = dict()
    fields["table_content"] = (
        str,
        dspy.InputField(description="Table content for extraction"),
    )
    extract_table_class = create_output_model_class(outputFields=outputFields)
    fields["extracted_information"] = (
        list[extract_table_class],
        dspy.OutputField(description="Extracted information"),
    )
    tableExtractSignature = dspy.make_signature(fields, prompt, "tableExtractSignature")

    main_model_settings = get_model_settings("main")
    if main_model_settings.max_tokens < 16000:
        main_model_settings.max_tokens = 16000

    tableExtractPredictor = dspy.Predict(
        tableExtractSignature, model=main_model_settings
    )
    table_dataset = []
    try:
        for file in tsv_files:
            table_content = ""
            encoding = detect_encoding(os.path.join(file_folder_path, file))
            file_df = pd.read_csv(
                os.path.join(file_folder_path, file),
                sep="\t",
                encoding=encoding,
                header=None,
            )
            if len(file_df) > 10:
                for index, row in file_df.iterrows():
                    row_values = [
                        str(value) if not pd.isnull(value) else " "
                        for value in row.values
                    ]
                    table_content += "\t".join(row_values) + "\n"
            else:
                for index, row in file_df.iterrows():
                    row_values = [
                        str(value) if not pd.isnull(value) else " "
                        for value in row.values
                    ]
                    table_content += "\t".join(row_values) + "\n"
            table_dataset.append(
                dspy.Example(table_content=table_content).with_inputs("table_content")
            )

        def process_item(example):
            return tableExtractPredictor(**example.inputs(), lm=main_model_settings)

        executor = ParallelExecutor(
            num_threads=num_threads,
            disable_progress_bar=True,
            max_errors=5,
            provide_traceback=False,
            compare_results=False,
        )
        results = executor.execute(process_item, table_dataset)
        assert len(results) == len(tsv_files)
        extracted_files = dict()
        for index, file_name in enumerate(tsv_files):
            extracted_info = results[index]["extracted_information"]
            output_file_path = os.path.join(save_folder_path, file_name)
            if extracted_info is not None and len(extracted_info) > 0:
                logger.info(extracted_info)
                df_out = pd.DataFrame(
                    [item.dict() for item in extracted_info if item is not None]
                )
                df_out.to_csv(output_file_path, sep="\t", index=False)
                extracted_files[file_name] = True
            else:
                extracted_files[file_name] = False
    except Exception as e:
        logger.info(e)


def generate_format_tables_with_extract4correct(
    file_folder_path,
    demo_folder_path,
    save_folder_path,
    extract_prompt,
    outputFields,
    num_threads: int = 6,
    encoding: str = "utf-8",
    extract_directly: bool = False,
):
    if not os.path.exists(file_folder_path):
        raise OSError(f"Folder {file_folder_path} does not exist.")
    if not os.path.exists(demo_folder_path):
        raise OSError(f"Folder {demo_folder_path} does not exist.")

    assert os.path.abspath(file_folder_path) != os.path.abspath(demo_folder_path), (
        "file_folder_path and demo_folder_path cannot be the same folder."
    )

    if not os.path.exists(save_folder_path):
        try:
            os.mkdir(save_folder_path)
        except OSError as e:
            raise OSError(f"Failed to create folder {save_folder_path}: {e}")

    # table_code_generator = dspy.ChainOfThought(tableCodeSignature)
    write_code_to_file_tool = dspy.react.Tool(
        write_code_to_file,
        desc="Write the generated code to a file.",
        args={"script_path": str, "code": str},
    )
    run_py_tool = dspy.react.Tool(
        run_py, desc="Run the generated python script.", args={"script_path": str}
    )
    read_file_tool = dspy.react.Tool(
        read_file,
        desc="Read the content of a file.",
        args={"file_path": str, "start_line": int, "end_line": int},
    )
    tableReactor = dspy.ReAct(
        tableReactSignature,
        tools=[write_code_to_file_tool, run_py_tool, read_file_tool],
        max_iters=10,
    )
    # print(tableReactor.tools['write_code_to_file'].desc)
    # print(tableReactor.tools["write_code_to_file"].args)
    tsv_with_demo = []
    table_dataset = []
    tsv_files = [file for file in os.listdir(file_folder_path) if file.endswith(".tsv")]
    demo_tsv_files = [
        file for file in os.listdir(demo_folder_path) if file.endswith(".tsv")
    ]

    try:
        for tsv_file in tsv_files:
            tsv_file_path = os.path.join(file_folder_path, tsv_file)
            demo_file_name = tsv_file.replace(".tsv", "_example.tsv")
            if demo_file_name in demo_tsv_files:
                tsv_with_demo.append(tsv_file)
                demo_file_path = os.path.join(demo_folder_path, demo_file_name)
                table_content = ""
                demo_content = ""
                encoding = detect_encoding(tsv_file_path)
                file_df = pd.read_csv(
                    tsv_file_path, sep="\t", encoding=encoding, header=None
                )
                if len(file_df) > 100:
                    for index, row in file_df[0:50].iterrows():
                        row_values = [
                            str(value) if not pd.isnull(value) else " "
                            for value in row.values
                        ]
                        table_content += "\t".join(row_values) + "\n"
                    for index, row in file_df[-50:-1].iterrows():
                        row_values = [
                            str(value) if not pd.isnull(value) else " "
                            for value in row.values
                        ]
                        table_content += "\t".join(row_values) + "\n"
                else:
                    for index, row in file_df.iterrows():
                        row_values = [
                            str(value) if not pd.isnull(value) else " "
                            for value in row.values
                        ]
                        table_content += "\t".join(row_values) + "\n"
                demo_df = pd.read_csv(
                    demo_file_path, sep="\t", header=None, encoding=encoding
                )
                for index, row in demo_df.iterrows():
                    row_values = [
                        str(value) if not pd.isnull(value) else " "
                        for value in row.values
                    ]
                    demo_content += "\t".join(row_values) + "\n"
                if len(table_content) > 30000:
                    table_content = ""
                    for index, row in file_df[0:10].iterrows():
                        row_values = [
                            str(value) if not pd.isnull(value) else " "
                            for value in row.values
                        ]
                        table_content += "\t".join(row_values) + "\n"
                    for index, row in file_df[-10:-1].iterrows():
                        row_values = [
                            str(value) if not pd.isnull(value) else " "
                            for value in row.values
                        ]
                        table_content += "\t".join(row_values) + "\n"
                source_table_path = tsv_file_path
                out_table_path = os.path.join(
                    save_folder_path, os.path.basename(tsv_file_path)
                )
                table_dataset.append(
                    dspy.Example(
                        source_table=table_content,
                        demo_table=demo_content,
                        source_table_path=source_table_path,
                        output_table_path=out_table_path,
                    ).with_inputs(
                        "source_table",
                        "demo_table",
                        "source_table_path",
                        "output_table_path",
                    )
                )

        if not table_dataset:
            logger.info("No matching demo files found")
            return

        def process_item(example):
            return tableReactor(**example.inputs())

        executor = ParallelExecutor(
            num_threads=num_threads,
            disable_progress_bar=True,
            max_errors=5,
            provide_traceback=False,
            compare_results=False,
        )
    except Exception as e:
        logger.error(f"Exception in generate_format_tables_with_extract4correct: {e}")

    results = executor.execute(process_item, table_dataset)

    # Clean up .py files in save_folder_path
    for file_name in os.listdir(save_folder_path):
        if not file_name.endswith(".tsv"):
            file_path = os.path.join(save_folder_path, file_name)
            try:
                os.remove(file_path)
                logger.info(f"Deleted {file_path}")
            except OSError as e:
                logger.error(f"Error deleting file {file_path}: {e}")
    assert len(results) == len(tsv_with_demo)

    for idx, result in enumerate(results):
        if not result["result"]:
            if extract_directly:
                _handle_extraction_failure(
                    file_folder_path,
                    save_folder_path,
                    tsv_with_demo[idx],
                    extract_prompt,
                    outputFields,
                    num_threads,
                    encoding,
                )

        # code = result["code"]
        # code = code.replace("```python", "").replace("```","").replace("{source_table_path}",table_dataset[idx]["source_table_path"]).replace("{output_table_path}",table_dataset[idx]["output_table_path"]).strip()
        # if code is not None:
        #     code_file_name = tsv_with_demo[idx].replace(".tsv",".py")
        #     code_file_path = os.path.join(save_folder_path, code_file_name)
        #     with open(code_file_path, "w", encoding="utf-8") as f:
        #         f.write(code)
        #     try:
        #         message = run_py(code_file_path)["message"]
        #         if message != "success" and extract_directly:
        #             logger.info(f"Script execution failed: {message}, trying direct extraction")
        #             _handle_extraction_failure(file_folder_path, save_folder_path, tsv_with_demo[idx], extract_prompt, outputFields, num_threads, encoding)
        #     except Exception as e:
        #         logger.info(f"Exception during script execution: {e}")
        #         if extract_directly:
        #             _handle_extraction_failure(file_folder_path, save_folder_path, tsv_with_demo[idx], extract_prompt, outputFields, num_threads, encoding)


def _handle_extraction_failure(
    file_folder_path,
    save_folder_path,
    err_tsv_file,
    extract_prompt,
    outputFields,
    num_threads,
    encoding,
):
    """extract err table directly"""
    try:
        encoding = detect_encoding(os.path.join(file_folder_path, err_tsv_file))
        df_err = pd.read_csv(
            os.path.join(file_folder_path, err_tsv_file), sep="\t", encoding=encoding
        )
        if len(df_err) <= 200:
            logger.info("Trying to extract information directly from the table")
            temp_extract_file = os.path.join(save_folder_path, "extract_temp")
            if os.path.exists(temp_extract_file):
                shutil.rmtree(temp_extract_file)
            os.mkdir(temp_extract_file)
            shutil.copy(
                os.path.join(file_folder_path, err_tsv_file),
                os.path.join(temp_extract_file, err_tsv_file),
            )
            extract_table_directly(
                temp_extract_file,
                save_folder_path,
                extract_prompt,
                outputFields,
                num_threads,
                encoding,
            )
            shutil.rmtree(temp_extract_file)
    except Exception as e:
        logger.error(f"Error in extraction failure handler: {e}")
