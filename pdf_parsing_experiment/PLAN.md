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

## 3. 建议架构

```text
pdf_parsing_experiment/
├── README.md
├── PLAN.md
├── models.py
├── pipeline.py
├── renderer.py
├── extractors/
│   ├── pypdf_extractor.py
│   └── pdfplumber_extractor.py
├── layout/
│   ├── reading_order.py
│   ├── paragraph_builder.py
│   ├── heading_detector.py
│   └── header_footer.py
├── tables/
│   ├── detector.py
│   ├── normalizer.py
│   └── renderer.py
└── tests/
    ├── fixtures/
    ├── expected/
    └── test_pipeline.py
```

## 4. 中间文档模型

先将两个解析器的结果归一化，再生成 Markdown：

```text
Document
└── Page
    └── Block
        └── Line
            └── Span
```

核心字段包括：

- `Document`：标题、作者、元数据、目录、页面列表和解析状态。
- `Page`：页码、页面宽高、文本块、表格、图片区域。
- `Block`：类型、边界框、阅读顺序、置信度。
- `Line`：基线、字号、行距、文本。
- `Span`：文本、字体、字号、粗体、斜体、边界框。
- `Table`：标题、边界框、行列数、单元格和合并关系。

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
3. 输出可序列化的中间文档 JSON，方便调试。

### 阶段 C：版面和阅读顺序

1. 对重复页眉、页脚和页码进行跨页统计并删除。
2. 根据水平间隔和文本块边界识别单栏、多栏和跨栏区域。
3. 先按栏排序，再按纵坐标排序，恢复阅读顺序。
4. 将字符聚合为 Span、Line、Block。
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

1. 依据中间模型统一生成 Markdown。
2. 转义会破坏 Markdown 的特殊字符。
3. 保留标题、粗体、列表、表格标题和图注。
4. 输出：
   - `{name}.md`
   - `{name}.debug.json`
   - 可选的页面和表格调试图

### 阶段 G：验证与迁移

1. 与 Marker 输出进行结构对照。
2. 使用页面 PNG 对照阅读顺序、表格边界和内容完整性。
3. 建立快照测试和单元测试。
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
2. 实现中间文档模型。
3. 完成单页字符、单词和基础表格抽取。
4. 完成单栏版面重建。
5. 增加多栏、标题和粗体识别。
6. 完善复杂表格。
7. 建立基准测试并迁移到 `src`。
