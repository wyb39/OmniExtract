# PDF 到 Markdown 解析流程实施计划

## 1. 目标

基于 `pypdf` 和 `pdfplumber` 构建面向 LLM 的 PDF 到 Markdown 解析流程。

输出应尽量接近 `src/articleUtil.py` 当前支持的 Markdown 结构，并满足：

- 保留文档标题和章节层级。
- 保留可识别的粗体语义。
- 恢复正文段落、列表和多栏阅读顺序。
- 去除重复页眉、页脚和页码噪声。
- 正确解析规则表格、无框表格和复杂表格。
- 对复杂表格使用 HTML，避免 Markdown pipe table 造成结构丢失。
- 将版面坐标和置信度保存在调试数据中，不污染交付给 LLM 的 Markdown。

## 2. 输出契约

### 2.1 Markdown 层级

- 文档标题使用 `#`。
- Abstract、Introduction、Method、Result、Discussion、Conclusion、References 等主章节使用 `##`。
- 子章节依次使用 `###`、`####`。
- 相邻自然段之间保留一个空行。
- 粗体使用 `**text**`。
- 项目符号和编号列表使用标准 Markdown 列表。
- 表格标题紧邻对应表格。

### 2.2 表格格式

- 简单矩形表格且不存在合并单元格时，输出 GFM pipe table。
- 存在多行表头、合并单元格、跨列或复杂嵌套内容时，输出 HTML `<table>`。
- 表格单元格中的换行统一为 `<br>`。
- 不猜测缺失的单元格内容；无法可靠恢复时保留空值并记录低置信度。

### 2.3 非目标

- 第一阶段不实现 OCR。
- 对纯扫描 PDF 返回明确的 `ocr_required` 状态。
- 第一阶段不重建复杂公式；先保留可提取文本和原始字符顺序。

## 3. 精简架构

```text
pdf_parsing_experiment/
├── README.md
├── PLAN.md
├── pdf_to_markdown.py
├── markdown_renderer.py
└── tests/
    ├── expected/
    ├── output/
    ├── evaluate_corpus.py
    └── test_pdf_to_markdown.py
```

核心实现只保留两个文件：

- `pdf_to_markdown.py`：公开 API 和 CLI、`pypdf` 元数据/文本回退、`pdfplumber`
  字符与表格抽取、阅读顺序、页眉页脚过滤、标题和段落识别。
- `markdown_renderer.py`：将轻量中间结果渲染为 Markdown，负责粗体、标题、列表、
  GFM 表格和 HTML 复杂表格。

测试 PDF 保留在现有 `test_pdfs/`，测试代码、期望结果和生成结果单列，不计入核心文件。
在实验结论稳定前不再拆分 `extractors`、`layout`、`tables` 等子包；只有迁移到 `src`
且出现明确复用边界时再考虑拆分。

## 4. 轻量中间模型

先将两个解析器的结果归一化，再生成 Markdown：

```text
Document
└── Page
    └── Block（text / heading / list / table）
```

模型直接定义在 `pdf_to_markdown.py` 中，只保留渲染和调试实际需要的字段：

- `Document`：元数据、页面列表、解析状态和警告。
- `Page`：页码、宽高和已排序的块。
- `Block`：类型、文本或表格数据、边界框、字号、粗体比例、标题层级和置信度。

字符、单词和行仅作为 `pdf_to_markdown.py` 内部临时结构使用，不建立公共类层级。
调试 JSON 直接由上述对象序列化，避免维护第二套模型。

## 5. 分阶段实施

### 阶段 A：依赖与基准集

1. 将 `pdfplumber` 固定为与现有 Marker/Surya 依赖兼容的版本。
2. 保留 Marker 输出作为对照基线。
3. 准备至少 10 类 PDF：
   - 单栏论文
   - 双栏论文
   - 中英文混排
   - 多层级标题
   - 规则线表格
   - 无框表格
   - 合并单元格表格
   - 跨页表格
   - 脚注和引用密集文档
   - 扫描 PDF

### 阶段 B：底层抽取

1. 使用 `pypdf` 提取：
   - 文档元数据
   - 页面数
   - 目录和书签
   - 加密状态
   - 基础文本兜底
2. 使用 `pdfplumber` 提取：
   - 字符和单词
   - 字体名和字号
   - 字符边界框
   - 线条、矩形和图片区域
   - 候选表格和单元格
3. 归一化为 `Document -> Page -> Block`，并可选输出调试 JSON。

### 阶段 C：版面和阅读顺序

1. 对重复页眉、页脚和页码进行跨页统计并删除。
2. 根据水平间隔和文本块边界识别单栏、多栏和跨栏区域。
3. 先按栏排序，再按纵坐标排序，恢复阅读顺序。
4. 在单文件内部将字符聚合为临时行，再输出 Block。
5. 处理连字符断词、软换行、重复字符和异常空格。

### 阶段 D：标题、段落和样式

1. 统计全文正文字号和字体分布。
2. 综合以下信号识别标题：
   - 相对字号
   - 粗体字体
   - 上下留白
   - 文本长度
   - 章节编号
   - PDF 目录
3. 根据字体名中的 `Bold`、`Semibold`、`Black`、`Heavy` 等标记识别粗体。
4. 合并连续同样式 Span，避免生成碎片化 Markdown。
5. 识别项目符号、编号列表、脚注和参考文献条目。

### 阶段 E：表格解析

1. 先检测表格区域，并从正文抽取区域中排除，避免内容重复。
2. 依次尝试：
   - `lines` 策略解析规则表格
   - `lines_strict` 策略排除装饰矩形
   - `text` 策略解析无框表格
3. 根据行列稳定性、空单元格比例和文字覆盖率选择最优结果。
4. 修复跨行文本、表头、多行单元格和跨页表格。
5. 简单表格输出 Markdown，复杂表格输出 HTML。
6. 使用 `debug_tablefinder` 生成可视化调试图。

### 阶段 F：Markdown 渲染

1. 由 `markdown_renderer.py` 依据轻量中间模型统一生成 Markdown。
2. 转义会破坏 Markdown 的特殊字符。
3. 保留标题、粗体、列表、表格标题和图注。
4. 输出：
   - `{name}.md`
   - `{name}.debug.json`
   - 可选的页面和表格调试图

### 阶段 G：验证与迁移

1. 与 Marker 输出进行结构对照。
2. 使用页面 PNG 对照阅读顺序、表格边界和内容完整性。
3. 在 `tests/` 中建立语料评估、快照测试和单元测试。
4. 稳定后将模块迁移到 `src`。
5. 在 `articleUtil.py` 中增加新的解析后端，但保留 Marker 回退路径。

## 6. 验收标准

- 机器生成 PDF 的正文无大段缺失或重复。
- 单栏和双栏文档阅读顺序正确。
- 标题层级与人工判断基本一致。
- 可识别的粗体文本保留为 Markdown 粗体。
- 重复页眉、页脚和页码不进入正文。
- 基准表格不存在整体错列、重复提取或正文混入。
- 复杂表格使用 HTML 后能够保留表头和合并单元格结构。
- 输出能够直接被现有 `ParsedMarkdown` 和 `convert_md_to_json` 使用。
- 扫描 PDF 不产生误导性空 Markdown，而是返回明确的 OCR 状态。

## 7. 当前执行顺序

1. 降级并固定兼容版本的 `pdfplumber`。
2. 在 `pdf_to_markdown.py` 中完成轻量模型、双引擎抽取和单栏版面重建。
3. 在同一文件中增加多栏、页眉页脚、标题和粗体识别。
4. 在 `markdown_renderer.py` 中完成正文和表格输出。
5. 使用 `tests/evaluate_corpus.py` 对 `test_pdfs/` 全量运行并记录问题。
6. 完善复杂表格与回归测试。
7. 验证稳定后迁移到 `src`，届时再按真实复用边界决定是否拆分。

## 8. 第一版实现状态

已完成：

- 两文件核心实现：`pdf_to_markdown.py` 与 `markdown_renderer.py`。
- `pdfplumber` 字符坐标、字体、粗体和表格抽取。
- `pypdf` 无文本页回退与元数据补充。
- 轻量 `Document -> Page -> Block` 模型及调试 JSON。
- 双栏阅读顺序、跨栏区段、重复页眉页脚、段落和断词处理。
- H1 文档标题、H2/H3 章节、粗体、列表、GFM/HTML 表格渲染。
- 扫描文档 `ocr_required` 状态。
- 独立单元测试和全语料评估脚本。

验证记录：

- 单元测试 4 项全部通过。
- `test_pdfs/` 的 31 份文件各取前 5 页完成兼容测试，0 个解析异常。
- 最新代码对 31 份文件各取前 2 页再次回归，0 个解析异常；30 份为 `ok`，
  `DAA.pdf` 为预期的 `ocr_required`。
- 已将代表性双栏论文首页渲染为 PNG，对照修正 Abstract、Keywords、章节与双栏正文顺序。

进入 `src` 前仍需完成：

- 扩充表格真值与跨页表格用例。
- 抑制没有可靠边界的矢量图坐标轴标签。
- 与 Marker 输出做可量化结构对比。
- 为公式和 OCR 设计明确的后续处理接口。
