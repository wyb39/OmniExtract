# OmniExtract
OmniExtract 是一个基于 [DSPy](https://github.com/stanfordnlp/dspy) 的 LLM 自动抽取工具，专门用于文献与文档中的信息抽取任务。它利用提示词优化工程技术，基于精心整理的数据提升抽取性能，并提供多种文件格式解析工具，支持从原始文档（如 PDF 或 XML）和表格文件中批量抽取多属性实体。

OmniExtract 的视频教程请访问 https://www.bilibili.com/video/BV12QywBhE1p。

## 如何从多个文件中抽取多属性实体

您可以按照以下步骤从多个文件中抽取多属性实体：

1. **配置大语言模型**
   设置用于抽取的大语言模型的连接与参数。包括指定模型端点、API 密钥以及模型相关的具体配置。

2. **解析待抽取的文件**
   使用内置的文件解析工具处理您的源文档。OmniExtract 支持多种格式，包括 PDF、XML 和表格文件。

3. **基于现有数据优化提示词（可选）**
   利用 OmniExtract 的提示词优化工程技术，使用整理好的数据来优化您的抽取提示词，这可以显著提升抽取的准确性和一致性。

4. **从文档或表格中抽取信息**
   执行抽取流程，从解析后的文档中获取多属性实体。抽取的数据将按照您指定的输出格式进行结构化组织。

## 核心特性
- 多文档批量处理
- 支持多种文件格式
- 通过提示词优化提升抽取性能
- 多属性实体的结构化输出

## 快速开始

### 环境要求
- Python 3.10
- Git

### 下载代码
```bash
cd OmniExtract
```

### 创建虚拟环境
创建虚拟环境以隔离项目依赖：

#### Windows 系统：
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux 系统：
```bash
python3 -m venv venv
source venv/bin/activate
```

### 安装依赖
使用 pip 安装所需的 Python 包。建议先单独安装 torch，以便选择与您的系统匹配的版本：

1. 先安装与您的系统匹配的 torch 版本。请从 https://pytorch.org/get-started/locally/
   中选择适合您平台的安装命令，例如：
   ```bash
   pip install torch
   ```
2. 移除 `requirements.txt` 中的 `torch` 依赖（步骤 1 中已经安装过 torch，
   再次安装会覆盖您选择的版本）。
3. 安装其余依赖：
   ```bash
   pip install -r requirements.txt
   ```

**请注意：您需要根据您的系统安装正确版本的 torch，并且必须先安装 torch，
再运行 `pip install -r requirements.txt`。**

### 启动 Web UI 服务
安装依赖后，启动 Web UI 服务：
```bash
python src/main.py
```
请确保您的虚拟环境已激活。
然后在浏览器中打开 http://127.0.0.1:9000/omniextract/ 使用该工具。
它与现有的 CLI 服务使用相同的 API 处理程序，并服务于 `ui_jinja/templates`
中迁移的模板。

Jinja 页面在同一前缀下提交四个后台工作流：
`run_workflow_doc_extraction`、`run_workflow_doc_extraction_optimized`、
`run_workflow_prompt_optimization` 和 `run_workflow_table_extraction`。
每个请求都会返回一个 `workflow_id` 和一个带令牌的 `workflow_url`。打开该 URL
即可查看工作流状态，并在准备就绪后下载结果。页面每五分钟自动刷新一次，
因此无需浏览器轮询代码。使用返回的令牌调用
`/omniextract/api/workflow/{workflow_id}/status` 即可访问 JSON 状态端点。
结果文件仅通过状态页面上显示的带令牌的工件链接提供；本地文件系统路径
永远不会暴露。

联系邮箱是可选的。如果省略，工作流仍可通过返回的状态 URL 完全使用，
并且不会尝试发送邮件通知。

邮件通知是可选的。在启动服务前配置 `OMNI_EXTRACT_SMTP_SENDER` 和
`OMNI_EXTRACT_SMTP_PASSWORD`（以及可选的 SMTP 服务器变量）以启用它们。

旧版的 Dash GUI 仍然可用。使用 `python -m gui.app` 启动，然后在浏览器中
打开 http://127.0.0.1:8050/ 使用该工具。

对于 Linux 服务器部署，请设置一个密码来加密模型的 `api_key` 并向外部公开服务。
GUI 默认已监听 `0.0.0.0`，因此只需设置加密密钥并启动即可：
```bash
OMNI_EXTRACT_ENCRYPTION_KEY=YOUR_PASSWORD python -m gui.app
```

### 模型运行时设置

`python src/main.py` 的服务启动参数中提供两个独立的 DSPy 响应缓存开关：

- `--cache-for-optimization` / `--no-cache-for-optimization`（默认：启用）
  适用于 `optim`、`optim_custom`、图像提示词优化及其基于模型的评估指标。
- `--cache-for-other` / `--no-cache-for-other`（默认：禁用）
  适用于预测、判定、解析、表格抽取和模型连接测试。

这些开关控制 DSPy 的本地精确响应缓存。它们不会启用或禁用提供方的
提示词缓存。由 DSPy 响应缓存提供服务的调用不会计入处理报告中的
`model_calls`、`input_tokens` 和 `output_tokens`。

### 处理错误报告

CLI 命令和四个 Jinja 后台工作流会创建 `processing_report.json`。失败文档会
与同一批次中的其他文档隔离，因此工作流可以以 `processing_status: "partial"`
结束，同时仍然返回所有可用的结果。

该报告包含工作流结果、受影响的文档以及聚合的提供方上报的模型 token 用量：

```json
{
  "workflow_id": "20260730_182514_567020_43c4ac02",
  "processing_status": "partial",
  "failed_documents": [
    {
      "document_id": "broken.pdf",
      "issues": [
        {
          "stage": "markdown_convert",
          "code": "MARKDOWN_GENERATION_FAILED",
          "message": "The parser did not generate Markdown output",
          "action": "Check the source document or try another supported parser.",
          "retryable": false
        }
      ]
    }
  ],
  "token_usage": {
    "model_calls": 3,
    "input_tokens": 12480,
    "output_tokens": 936,
    "cached_input_tokens": 8192,
    "cache_creation_input_tokens": null
  }
}
```

`token_usage` 字段聚合提供方上报的 token 用量；当提供方未返回足够信息时，
两个缓存字段为 `null`。

支持的错误码按类别分组如下：

**输入/源文件/解析问题**

| 错误码 | 含义 / 建议操作 |
|---|---|
| `SOURCE_NOT_FOUND` | 输入文件不存在。请检查并重新上传。 |
| `SOURCE_INVALID` | 输入文件无效。请检查文件格式和编码。 |
| `FILE_ACCESS_DENIED` | 文件或目录访问被拒绝。请检查权限后重试。 |
| `DOCUMENT_PARSE_FAILED` | 文档解析失败。请检查格式并重新上传。 |
| `MARKDOWN_GENERATION_FAILED` | Markdown 转换失败。请检查源文档或换用其他解析器。 |
| `JSON_CONVERSION_FAILED` | JSON 转换失败。请检查生成的 Markdown 与章节结构。 |
| `TABLE_PARSE_FAILED` | 表格解析失败。请检查文档/表格格式并重试。 |

**模型 / 提供方问题**

| 错误码 | 含义 / 建议操作 |
|---|---|
| `MODEL_TIMEOUT`（可重试） | 模型请求超时。请检查模型服务并减少线程数。 |
| `MODEL_RATE_LIMITED`（可重试） | 触发限流。请稍候重试，或减少线程数。 |
| `MODEL_UNAVAILABLE`（可重试） | 模型服务不可用。请检查 API 地址和提供方状态后重试。 |
| `MODEL_AUTH_FAILED` | 认证失败。请检查所配置的模型凭据。 |

**预测 / 抽取问题**

| 错误码 | 含义 / 建议操作 |
|---|---|
| `PREDICTION_FAILED` | 预测失败。请检查输入字段和模型服务后重试。 |
| `JUDGEMENT_FAILED` | 判定失败。请检查判定模型设置；结果可能仍可使用。 |
| `TABLE_EXTRACTION_FAILED` | 表格抽取失败。请检查表格提示词和模型服务后重试。 |

**提示词优化问题**

| 错误码 | 含义 / 建议操作 |
|---|---|
| `OPTIMIZATION_FAILED` | 提示词优化失败。请检查数据集和 `optim.log` 后重试。 |
| `OPTIM_DATASET_NOT_FOUND` | 优化数据集缺失。请将 `dataset` 设置为已存在的 JSON、CSV、TSV 或 XLSX 文件。 |
| `OPTIM_DATASET_EMPTY` | 优化数据集为空。请至少添加两条完整记录。 |
| `OPTIM_DATASET_TOO_SMALL` | 优化数据集过小。请提供至少两条完整、有效的记录。 |

**输出问题**

| 错误码 | 含义 / 建议操作 |
|---|---|
| `OUTPUT_WRITE_FAILED` | 无法写入输出。请检查权限、可用空间和文件锁。 |

**兜底错误码**

| 错误码 | 含义 / 建议操作 |
|---|---|
| `TASK_FAILED` | 未分类的任务失败。请检查任务输入和应用日志后重试。 |

`retryable: true` 的问题通常可通过重试解决。详细的解析器/模型回溯信息
保留在应用程序日志中。对于 CLI 运行，完成时打印的 JSON 包含
`processing_report`；对于 Jinja 工作流，状态页面会列出受影响的文档并将
报告作为可下载的工件暴露，该报告也会包含在返回的结果 ZIP 中。

`process/` 下的运行时输出、本地的 `error_handling_experiment/` 工作区
以及 `settings/model_settings_*.json` 被 Git 忽略。模型设置可能包含凭据，
必须在每个部署环境中本地配置。

### 使用命令行与配置文件
您可以通过命令行界面开始使用 OmniExtract。有关详细的配置说明，请参阅 src/yml 目录中的 README 文件。

## 重要说明

> **PDF 输入使用基于 Marker 的解析器。**
> Marker 是 main 分支上 PDF 的生产后端。后端名称通过
> `src/parsing/articleUtil.py` 中的 `PDF_PARSER_BACKEND`（默认 `"marker"`）固定，
> 该配置有意保持为内部选项，不会改变现有的 CLI/API/YAML 输入参数。
> 请确保遵守 Marker 的使用要求和许可条款。
> 详细信息请参阅 Marker 的官方文档：
> https://github.com/datalab-to/marker
> https://github.com/datalab-to/marker/blob/master/README.md
