# PDF 到 Markdown 解析流程实施计划

## 当前架构决策（2026-07-26）

以下决策覆盖本文后续章节中关于旧版 native-first 和表格融合策略的历史描述：

1. `auto` / `hybrid` 必须先完成整篇 PDF 所有页面的 PP-DocLayoutV2，
   然后才开始原生字符提取和区域填充。
2. PDF 页面渲染及原生字符坐标统一由 `pypdfium2` 提供；hybrid 和完整
   OpenDoc PDF 路径均不得调用 PyMuPDF。
3. PP-DocLayoutV2 检测为 `table` 的区域默认交给 UniRec，不再调用
   pdfplumber 表格检测。仅当机器生成 PDF 的超大表格存在稳定原生网格且预计超过
   UniRec token 容量时，使用 PDFium 字符网格容量兜底。
4. `pdfplumber` 仅保留在显式 `native` 对照后端，不参与 `auto` /
   `hybrid` 主流程。
5. `auto` 的失败回退使用同样由 PDFium 提供图像的完整 OpenDoc，不再
   回退到 native-first。
6. Debug JSON 必须记录 `renderer=pypdfium2`、
   `native_text_provider=pypdfium2`、`table_provider=unirec`，以及逐区域
   UniRec 耗时、token 数、停止原因、旋转和容量兜底状态。

## 1. 目标

基于 `pypdf`、`pdfplumber` 和 OpenDoc 构建面向 LLM 的 PDF 到 Markdown 解析流程。
统一采用 `layout-first hybrid`：所有页面先经过 PP-DocLayoutV2，原生 PDF
优先将字符层映射到 OpenDoc 版面区域，仅对表格、公式、扫描页和原生字符覆盖不足的
区域调用 UniRec。

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

### 2.3 OCR 边界

- 所有页面默认运行 OpenDoc Layout，统一获得区域类别、边界框和阅读顺序。
- 原生 PDF 优先使用 pdfplumber/PyMuPDF 字符层填充版面区域，避免 OCR 改写字符和单位。
- 表格、显示公式、原生字符缺失或覆盖率过低的区域，按块调用 UniRec。
- 扫描 PDF 按 Layout 区域调用 UniRec，不要求整页作为单一 OCR 图像。
- 支持显式强制使用 OpenDoc 解析整份文档。
- OpenDoc Layout 失败时回退到 native，并记录明确警告；UniRec 失败时保留原生文本或
  `ocr_required` 状态和失败原因。
- 不因局部字体编码异常而默认 OCR 整份机器生成 PDF。
- 不对拥有可靠字符层的普通正文执行 OCR。

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
  字符与表格抽取、OpenDoc Layout、字符区域映射、选择性 UniRec、失败回退，以及
  轻量中间模型。
- `markdown_renderer.py`：将轻量中间结果渲染为 Markdown，负责粗体、标题、列表、
  GFM 表格和 HTML 复杂表格。

测试 PDF 保留在现有 `test_pdfs/`，测试代码、期望结果和生成结果单列，不计入核心文件。
在实验结论稳定前不再拆分 `extractors`、`layout`、`tables` 等子包；只有迁移到 `src`
且出现明确复用边界时再考虑拆分。

## 4. 轻量中间模型

先将原生提取与 OpenDoc 后端的结果归一化，再生成 Markdown：

```text
Document
└── Page
    └── Block（text / heading / list / table / raw_markdown）
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
3. 使用 PP-DocLayoutV2 为所有页面生成统一版面区域。
4. 将 PDF 字符坐标归一化后映射到 OpenDoc 区域，保留字符、字体和粗体信息。
5. 归一化为 `Document -> Page -> Block`，并可选输出调试 JSON。
6. 仅为需要识别的区域懒加载 UniRec；保留 OpenDoc 标签、边界框、置信度和计时。

### 阶段 C：版面和阅读顺序

1. 以 OpenDoc Layout 标签过滤页眉、页脚、页码、图片和图表噪声。
2. 以 Layout 输出顺序统一处理单栏、多栏和跨栏区域。
3. 根据渲染倍率、页面旋转和 CropBox 将像素坐标转换为 PDF 坐标。
4. 每个字符只分配给一个最具体的 Layout 区域，避免区域重叠造成重复。
5. 对未被 Layout 覆盖的原生字符使用 native 规则补充，并记录覆盖率。
6. 处理连字符断词、软换行、重复字符和异常空格。

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

1. 同时使用 PP-DocLayoutV2 与 pdfplumber 产生表格候选。
2. Layout 表格与 native 表格重叠时优先使用原生字符表格，避免 OCR 数值错误。
3. Layout 检出但 native 无可靠结果时，仅对该表格区域调用 UniRec。
4. Layout 漏检但 native 候选具有稳定行列、短单元格和足够数值密度时保留 native 表格。
5. 对疑似双栏正文、参考文献或图示误判的 native 表格执行更严格拒绝规则。
6. native 表格依次尝试：
   - `lines` 策略解析规则表格
   - `lines_strict` 策略排除装饰矩形
   - `text` 策略解析无框表格
7. 根据行列稳定性、空单元格比例和文字覆盖率选择最优结果。
8. 修复跨行文本、表头、多行单元格和跨页表格。
9. 简单表格输出 Markdown，复杂表格或 UniRec 表格输出 HTML。

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
- 扫描 PDF 通过 OpenDoc 输出 Markdown；失败时返回明确的 OCR 状态。
- 机器生成 PDF 的普通正文不得因启用 hybrid 而被 OCR 改写单位、拼写或引用编号。
- 常驻 CPU 模式下，Layout 阶段目标不超过 0.5 秒/页（当前 30 页基准为 0.336 秒/页）。
- Debug JSON 能区分 `layout-native`、`layout-unirec` 和 native fallback 来源。

## 7. 当前执行顺序

1. 保留 `native` 和完整 `opendoc` 作为对照后端。
2. 新增 `hybrid` 后端，并使 `auto` 优先执行 layout-first hybrid。
3. 独立初始化 PP-DocLayoutV2，按页面输出布局区域和阅读顺序。
4. 将 pdfplumber 字符、字体和粗体映射到布局区域。
5. 对表格、公式和低覆盖区域懒加载 UniRec；扫描页按块识别。
6. 融合 Layout 与 native 表格候选，抑制双栏正文/参考文献误判。
7. 使用 `tests/evaluate_corpus.py` 对 5 篇审阅集和完整语料回归速度与质量。
8. 验证稳定后迁移到 `src`，届时再按真实复用边界决定是否拆分。

## 8. 第一版实现状态

已完成：

- 两文件核心实现：`pdf_to_markdown.py` 与 `markdown_renderer.py`。
- `pdfplumber` 字符坐标、字体、粗体和表格抽取。
- `pypdf` 无文本页回退与元数据补充。
- 轻量 `Document -> Page -> Block` 模型及调试 JSON。
- 双栏阅读顺序、跨栏区段、重复页眉页脚、段落和断词处理。
- H1 文档标题、H2/H3 章节、粗体、列表、GFM/HTML 表格渲染。
- `native`、`hybrid`、`opendoc` 和 `auto` 四种解析模式。
- OpenDoc 扫描文档解析、Markdown 适配和自动 OCR 回退。
- 独立单元测试和全语料评估脚本。
- 5 篇、30 页 PP-DocLayoutV2 CPU 基准：纯 Layout 10.074 秒，平均 0.336 秒/页。

已完成的 layout-first 第一版：

- `hybrid` / `auto` 的 OpenDoc Layout 统一入口。
- 原生字符到 Layout 区域的坐标映射、来源和覆盖率调试信息。
- 表格/公式/扫描区域的选择性 UniRec。
- Layout 与 native 表格候选融合。
- LayoutDetector 与选择性 UniRec 在同一进程内按配置缓存，批量文件不重复加载模型。
- 5 篇、30 页 hybrid 回归全部成功，总解析计时 53.4 秒；完整 OpenDoc 对照约
  1175.0 秒。
- hybrid 正确保留第 1、3、4 篇真实表格，并抑制第 2 篇图示和第 5 篇双栏参考文献
  的错误表格化。
- `DAA.pdf` 扫描首页通过 10 个 Layout 区域选择性 OCR，状态为 `ok`。

验证记录：

- 单元测试 8 项全部通过。
- `test_pdfs/` 的 31 份文件各取首页完成 hybrid 兼容回归，31/31 为 `ok`，
  无 Layout 回退。
- `test_pdfs/` 的 31 份文件各取前 5 页完成兼容测试，0 个解析异常。
- native 基线对 31 份文件各取前 2 页回归，0 个解析异常；30 份为 `ok`，
  `DAA.pdf` 为预期的 `ocr_required`。
- 已将代表性双栏论文首页渲染为 PNG，对照修正 Abstract、Keywords、章节与双栏正文顺序。
- OpenDoc CPU 模式成功解析 `DAA.pdf` 扫描页，自动回退后状态为 `ok`，
  保留中英文标题、正文和章节层级。
- 适配 openocr-python 0.1.5 在特定宽高比下生成 799 像素输入的问题，统一修正为
  PP-DocLayoutV2 要求的 800×800 张量。

进入 `src` 前仍需完成：

- 扩充表格真值与跨页表格用例。
- 抑制没有可靠边界的矢量图坐标轴标签。
- 对原生字体映射产生的 `(cid:*)`、连字符和数学符号错误增加局部字符级融合，避免在
  原生错误与整块 OCR 错误之间二选一。
- 与 Marker 输出做可量化结构对比。
- 扩充 OpenDoc 表格、公式和多页扫描文档回归样本。
