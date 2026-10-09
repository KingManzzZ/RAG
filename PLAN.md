# 个人财富规划 RAG 检索模块 — 设计与实施计划书

> 版本：v0.2（待审阅）
> 日期：2026-10-09
> 状态：**待用户确认后执行**
> v0.1 → v0.2 修订：① 切分策略重写为**源码级模仿 RAGFlow**（明确「模仿对象 → 具体逻辑 → 我方适配」的对应关系与全部参数默认值，见 4.2）；② 嵌入/重排模型切换为 **Qwen3 最新套件**（Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B）；③ 混合检索升级为 **Qdrant 原生 prefetch + RRF 融合**。

---

## 1. 项目背景与目标

为「个人财富规划 Agent 助手」构建 RAG（检索增强生成）模块。本模块负责将专业金融知识（研报、教材、法规、笔记等）摄入为可检索的知识库，并在用户以自然语言（中文）提问时，返回**高相关、可溯源**的证据片段。

**本阶段目标（仅检索）**

- 完成 文档摄入 → 解析 → 切分 → 向量化 → 入库 → 混合检索 → 重排 的完整链路；
- 对外提供 Python 库 + CLI 两种调用方式；
- 为后续「LLM 答案编排」阶段预留干净的接口（本模块只输出排序后的证据块与来源信息，不做答案生成）。

**明确不做（后续阶段）**

- 答案生成 / 多轮对话编排 / Agent 工具调用；
- 智能体规划、金融计算工具等。

---

## 2. 需求与边界（已确认）

| 项目 | 确认结果 |
| --- | --- |
| 面向用户 | 中文用户，金融领域自然语言提问 |
| 知识源格式 | 纯文本版 PDF、Word/PPT/Excel、网页/HTML/Markdown（**不含扫描版 PDF**） |
| 数据规模 | 小规模：几十篇文档（数千 ~ 数万切块） |
| 构建路线 | 自建轻量管线（组合最强开源组件，**源码级模仿**其核心设计） |
| 交付形态 | Python 库 + CLI（无 HTTP 服务） |
| 本地模型 | **Qwen3 套件**：Qwen3-Embedding-0.6B（嵌入）+ Qwen3-Reranker-0.6B（重排） |

---

## 3. 目标运行环境与约束

**部署机（非本机）**：RTX 5060 Ti 8GB（Blackwell 架构）、16GB DDR5-6000 内存、R7 9700 CPU。

| 约束 | 影响与对策 |
| --- | --- |
| 8GB 显存 | 大模型全部走 API；本地仅嵌入 + 重排模型（FP16 合计约 2.5GB，余量充足） |
| Blackwell 架构（sm_120） | 需要 PyTorch 2.8+ / CUDA 12.8（cu128）运行库；部署时需确认 NVIDIA 驱动版本 ≥ 570 |
| 16GB 内存 | 解析（Docling）与入库为批处理任务，按批控制并发即可 |
| 小模型本地运行 | Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B 均在 8GB 显存内运行；代码做设备自适应（CUDA 优先，CPU 兜底） |

开发及部署机环境确认：
- 开发机（本机）：RTX 4060 8GB、8GB内存
- 部署机（非本机）：RTX 5060 Ti 8GB、16GB DDR5-6000 内存、R7 9700 CPU。

---

## 4. 核心技术选型（调研结论）

### 4.1 文档解析：Docling（主）+ PyMuPDF（轻量快速通道）

| 方案 | 优势 | 劣势 | 结论 |
| --- | --- | --- | --- |
| **Docling**（IBM，MIT） | **统一覆盖 PDF/Word/PPT/Excel/HTML**，输出结构化 Markdown + 版面信息（标题层级、表格结构），CPU 可跑 | 处理速度一般 | ✅ 主解析器 |
| PyMuPDF4LLM | 极轻量、速度快 | 仅 PDF，标题/表格结构识别弱 | ✅ 备选快速通道 |
| MinerU 2.x | PDF 解析最强（公式/表格/多栏） | 依赖重型模型下载、仅 PDF、小规模场景性价比低 | ⏸ 暂不引入，留作备选 |

**理由**：本阶段数据为纯文本版 PDF + Office + 网页，且规模小。Docling 一个工具统一多格式、输出带标题层级的 Markdown，为后续「结构感知切分」提供最重要的输入（标题树）。MinerU 的强项（扫描件、公式）在本场景用不上。

### 4.2 切分策略（核心设计）— 源码级模仿 RAGFlow 并适配

#### 调研结论

- **RAGFlow（DeepDoc + template-based chunking）**：中文生态最完整、生产验证最充分的「结构优先」切分体系——按文档类型提供 10+ 种切分模板（naive / title / table / laws / book / paper / qa / manual / presentation / one），核心思想是**先还原文档结构再切分**，而非盲目按字数切；
- **Chonkie（新一代轻量切块库）**：提供 Recursive / Semantic / Late / Neural 等 9 种策略，可作为内部工具与增强项；
- **Chroma 研究**：递归切分 400 tokens 即达 85–90% 召回；语义切分仅提升至 91–92%，但需嵌入每个句子，成本高；
- **Late chunking（JinaAI，arXiv 2409.04701）**：长文档 +6.5 nDCG@10，但嵌入开销 2–3 倍，对短文档无增益。

**选型决策**：以 **RAGFlow 的具体实现为模仿蓝本**（中文文档类型覆盖最全、参数经过生产验证、与我们的规模匹配），逐项适配到本管线；Chonkie 作为可选增强库；语义切分 / late chunking **不默认启用**（成本/收益比不足，评估集验证后再定）。

#### 模仿对象 → 具体逻辑 → 我方适配（源码级对应表）

| 模仿对象（RAGFlow 源码） | 核心逻辑 | 我方适配 |
| --- | --- | --- |
| `rag/flow/chunker/title_chunker/`（TitleChunker / GroupTitleChunker） | 按标题层级切分文档，保留层级；块超 `chunk_token_cap` 时用句子边界检测再切分，必要时**硬切分**（hard split）；可配 `levels`、`chunk_token_cap` | 「结构切分层」主逻辑：以 Docling 输出的 Markdown 标题树为骨架；层级范围 H1–H4，cap=512 tokens |
| `rag/app/naive.py::naive_merge()`（→ `rag/nlp/__init__.py`） | `split-then-merge`：按分隔符 `"\n!?。；！？"`（含中文标点）切分 → `tiktoken(cl100k_base)` 计数 → 贪心合并至 `chunk_token_num`（默认 **512**） | 「节内切分层」原样采用该算法；分隔符、512 token 上限、tiktoken 计数方式全部对齐 |
| `internal/ingestion/component/chunker/token.go`（TokenChunker，Go 版） | 嵌套块携带 **"mom"（父块）与 "text"（子块）** 两级字段；支持 `overlapped_percent` 重叠；超限单元不强行切、截断下移 | **父子块结构直接对齐其 mom/text 设计**：子块（text）作检索单元，父块（mom=所属节全文）存元数据供上下文扩展；重叠默认 0，可配置对齐 |
| `rag/app/table.py` | 行级切分表格；检测多行表头（合并单元格）；逐行转文本块 | 表格作为**原子块**；超限表格按行切分且**重复表头**、标记「续表」与行范围；行内容转「表头: 值」文本供嵌入 |
| `rag/app/laws.py` | 用 `Node` 树表达章节/条款层级，展平为块列表时**把层级上下文写入块元数据** | 法规/条款类金融文档（如《商业银行理财业务监督管理办法》）的层级保留，写入 `heading_path` 元数据 |
| 位置标签 `@@page\tleft\tright\ttop\tbottom##` | 块携带文档坐标，支持溯源与预览裁图 | 简化为**页码 + 来源文件**元数据（满足溯源，不做裁图） |

**同时吸收的其他开源思想（轻量适配）**：

- **面包屑注入**（替代 late chunking 的上下文保留，成本低）：每个子块文本前置标题路径，如「资产配置 > 战略与战术配置 > 再平衡策略」，使嵌入自带上下文；
- **金融实体保护**（我方新增）：切分边界不落在数字、百分比、金额、法规编号中间（如「4.5%」「第 24 条」「2024 年」不可拆开）；
- **不做的部分**：RAGFlow 的模板可视化人工干预、知识图谱、图片裁切——超出本阶段范围。

#### 切分流程（六步）与参数默认值

```
标准 Markdown（含标题层级）
 ① 结构切分：标题树 → 以最小完整「节」为单元（模仿 TitleChunker）
 ② 节内切分：split-then-merge，贪心合并至 512 tokens（模仿 naive_merge）
 ③ 超限兜底：单句超上限 → 句子边界再切分 → 硬切分（模仿 TitleChunker hard split）
 ④ 表格处理：原子块；超限按行 + 表头重复 + 续表标记（模仿 table.py）
 ⑤ 父子块组装：mom（节全文）/ text（子块）两级结构（对齐 TokenChunker）
 ⑥ 元数据组装：面包屑注入 + 页码溯源 + 金融实体保护
```

| 参数 | 默认值 | 对齐来源 / 依据 |
| --- | --- | --- |
| 子块上限 | **512 tokens**（tiktoken cl100k_base） | RAGFlow `chunk_token_num` 默认值 |
| 标题层级范围 | H1–H4 | RAGFlow TitleChunker `levels` |
| 分隔符 | `\n ! ? 。 ； ！ ？` | RAGFlow naive 默认 `delimiter` |
| 块间重叠 | **0**（可配置开启） | 对齐 naive 默认；重叠能力对齐 `overlapped_percent` |
| 合并策略 | 贪心合并（split-then-merge） | `naive_merge()` |
| 硬切分 | 仅当单句 > 上限时触发 | TitleChunker 行为 |
| 表格 | 原子块；超限按行 + 表头重复 | `table.py` 行级策略 |
| 面包屑 | 开启（标题路径前缀） | 我方适配（late chunking 轻量替代） |

**为什么子块用 512 tokens**：与 RAGFlow 生产默认一致、便于对照；位于 Chroma 研究验证的高召回区间（约 400 tokens 附近达峰）；Qwen3-Embedding 支持 32K 上下文，无需为模型上限让步。

### 4.3 嵌入模型：Qwen3-Embedding-0.6B（本地推理）

**选型决策**：采用 Qwen3 Embedding 套件。2026 年 MTEB 多语言榜单上 Qwen3-Embedding-8B 位居第一（取代 BGE-M3 登顶），0.6B 是 8GB 显存场景下同门最佳尺寸，且与重排模型同源、格式协同（Apache 2.0 许可）。

**规格与部署要点**：

| 项目 | 说明 |
| --- | --- |
| 规格 | 0.6B 参数 / 28 层 / 32K 上下文 / **1024 维**稠密向量 / 支持 MRL 截维（可降为 512 省存储）/ instruction-aware |
| 加载方式 | `sentence-transformers` 官方支持直接加载；推荐启用 `flash_attention_2` 加速省显存；last-token pooling 需 `padding_side="left"` |
| 查询格式（关键） | 查询侧使用指令前缀：`Instruct: {任务描述}\nQuery: {query}`（任务描述定制为金融检索场景）；**文档侧不加前缀** |
| 显存占用 | FP16 约 1.2GB |
| 备选 | vLLM / TEI 服务化（本阶段单机直载即可，不需引入） |

### 4.4 混合检索与融合：Qwen3 稠密 + jieba-BM25 稀疏 + Qdrant 原生 RRF

- **稠密路**：Qwen3-Embedding-0.6B（1024 维，余弦）——语义匹配；
- **稀疏路**：`jieba` 精确模式分词 → 自建词表（term → index 持久化映射）→ 词频写入 Qdrant **稀疏向量并启用 IDF 修正**（近似 BM25 打分，金融术语/数字/法规名称依赖此路）；
  - 备选 A：`bm25s` 独立索引（jieba 分词），应用层 RRF；
  - 备选 B：Qdrant 官方 `Qdrant/bm25` 稀疏模型（上手最快，但中文分词效果一般，不推荐）；
- **融合**：Qdrant Query API **原生 RRF**——一次请求完成「dense top-30 + sparse top-30 双路召回 → RRF 融合 → top-20」，无需应用层手写融合逻辑；
- **过滤**：payload 过滤（文档类型 / 来源 / 时间），与召回同请求完成。

### 4.5 重排：Qwen3-Reranker-0.6B（本地推理）

| 项目 | 说明 |
| --- | --- |
| 原理（关键） | decoder-only **判别式**重排：**不是 generate**；以 yes/no 两个 token 的 logit 差值作为相关性得分（官方 chat template 拼接 system + query + document） |
| 部署 | `transformers` 本地推理，FP16 约 1.2GB；与嵌入模型可同时驻留（合计 ~2.5GB < 8GB） |
| 流程 | 融合后的 top-20 → 逐条打分重排 → 取 top-k（默认 8） |
| 备选升级 | 4B/8B 版质量更高但超 8GB 显存；本阶段锁定 0.6B，评估后可考虑量化版 |

### 4.6 向量库：Qdrant（本地嵌入模式）

| 候选 | 结论 |
| --- | --- |
| **Qdrant**（Apache-2.0） | ✅ 首选：单机性能最强（Rust/HNSW，p95 毫秒级）；**原生 dense + sparse 命名向量、稀疏 IDF 修正、Query API prefetch + RRF 融合**（省去应用层融合与第二套索引）；payload 过滤；`QdrantClient(path=...)` 本地嵌入模式**零部署**；数据可无缝迁移到 Docker/服务模式（后续升级零代码改动） |
| LanceDB | 备选：纯嵌入式，但混合检索生态略弱 |
| Milvus | 不合适：K8s 级运维，小规模过重 |
| Chroma | 不合适：混合检索与过滤能力弱 |

### 4.7 选型一览表

| 环节 | 选型 | 协议 | 运行位置 |
| --- | --- | --- | --- |
| 文档解析 | Docling（备选 PyMuPDF4LLM） | MIT | CPU |
| 切分 | 自研（源码级模仿 RAGFlow：结构切分 + naive_merge + mom/text 父子块） | — | CPU |
| 稠密嵌入 | Qwen3-Embedding-0.6B | Apache 2.0 | 本地 GPU/CPU |
| 稀疏检索 | jieba + BM25 稀疏向量（Qdrant IDF） | MIT | CPU |
| 向量库 | Qdrant 本地模式 | Apache-2.0 | CPU（磁盘） |
| 融合 | Qdrant 原生 RRF（prefetch ×2） | — | — |
| 重排 | Qwen3-Reranker-0.6B | Apache 2.0 | 本地 GPU/CPU |
| CLI/框架 | Typer + Pydantic + Rich | — | — |

---

## 5. 系统架构

### 5.1 数据流

```
【摄入管线】（离线批处理）
原始文档 ──► 解析器(Docling) ──► 标准化 Markdown+结构元数据
        ──► 切分(六步策略：结构切分→naive_merge→父子块→表格→面包屑)
        ──► Qwen3-Embedding 批量编码(dense) + jieba 词表更新(sparse)
        ──► Qdrant 入库(本地 db/，dense+sparse 双命名向量)

【检索管线】（在线）
中文查询 ──► Qwen3 查询编码(指令前缀) + jieba 分词
        ──► Qdrant 单请求双路召回(dense top-30 + sparse top-30)
        ──► 原生 RRF 融合 → top-20 ──► Qwen3-Reranker 精排 → Top-K 证据块
        ──► 返回：文本 + 标题面包屑 + 来源文件 + 页码 + 父块(mom)引用
```

### 5.2 目录结构

```
RAG/
├── PLAN.md                      # 本计划书
├── pyproject.toml               # 依赖与打包（uv/pip 可装）
├── config.yaml                  # 全部可调参数（切分、检索、设备）
├── data/                        # 原始知识文档（用户放入）
├── db/                          # Qdrant 本地数据目录
├── src/wealth_rag/
│   ├── config.py                # 配置加载(pydantic)
│   ├── models.py                # Chunk / SearchResult 数据模型
│   ├── ingestion/
│   │   ├── parser.py            # Docling 解析 → 标准 Markdown
│   │   ├── chunker.py           # 六步切分策略（核心，对齐 4.2）
│   │   └── pipeline.py          # 摄入总流程编排
│   ├── embedding/
│   │   └── encoder.py           # Qwen3-Embedding 封装(设备自适应、指令前缀)
│   ├── sparse/
│   │   └── bm25.py              # jieba 分词 + 词表持久化 + 稀疏向量构造
│   ├── store/
│   │   └── qdrant_store.py      # 建库/写入/检索(prefetch+RRF)/统计
│   ├── retrieval/
│   │   ├── hybrid.py            # 双路召回编排（含过滤）
│   │   └── rerank.py            # Qwen3-Reranker 封装(yes/no logit 打分)
│   └── cli.py                   # Typer CLI 入口
└── eval/
    ├── queries.jsonl            # 评估集（待标注）
    └── run_eval.py              # 评估脚本
```

### 5.3 Chunk 数据模型（示意）

```python
class Chunk:
    id: str                    # 稳定哈希：doc_id + 位置
    text: str                  # 含面包屑前缀的检索文本（= text 子块）
    raw_text: str              # 不含面包屑的原文（返回给 LLM 用）
    doc_id: str; doc_title: str
    source_path: str; page: int | None
    heading_path: list[str]    # ["资产配置", "再平衡策略"]
    chunk_type: str            # text | table | table_continued
    parent_id: str | None      # mom 块引用（所属节）
    parent_text: str | None    # mom 块全文（上下文扩展用）
```

---

## 6. 模块设计与实现要点

### 6.1 摄入管线

1. **解析**：按扩展名路由（.pdf/.docx/.pptx/.xlsx → Docling；.md/.html → 直读/转换），统一产出标准 Markdown + 标题结构；
2. **切分**：执行 4.2 六步策略，输出 mom/text 两级 Chunk 列表；
3. **编码**：Qwen3-Embedding 批量编码（文档侧无指令前缀，batch 可控），产出 1024 维稠密向量；jieba 分词更新词表并生成稀疏向量；
4. **入库**：Qdrant 本地模式写入（dense+sparse 双命名向量；同名文档默认 upsert 覆盖；`--rebuild` 全量重建并重建 jieba 词表）。

### 6.2 检索管线

1. 查询处理：加指令前缀（金融检索任务描述）+ jieba 分词；
2. Qdrant 单请求双路召回：dense top-30 + sparse top-30 → 原生 RRF → top-20；
3. Qwen3-Reranker 逐条打分 → top-k（默认 8）；
4. 结果组装（含溯源字段，格式对齐后续 LLM 编排所需的 evidence 结构）。

### 6.3 CLI 设计（Typer）

```bash
rag ingest ./data                # 摄入目录（增量）
rag ingest ./data --rebuild      # 全量重建
rag search "什么是再平衡策略？"    # 检索（默认混合+重排）
rag search "..." --top-k 5 --no-rerank
rag eval                         # 运行评估集输出指标
rag info                         # 知识库统计（文档数/块数/索引状态）
```

---

## 7. 评估方案（小规模、轻量）

- **评估集**：30~50 条金融领域真实风格问题，人工标注「应命中的文档与块」（可半自动：先用系统召回、人工修正）；
- **指标**：Recall@5/10、MRR、nDCG@10（脚本自算，不引入重型评估框架）;
- **关键 A/B 实验**（验证核心设计）：
  1. 结构感知切分 + RAGFlow 式合并 **vs** 朴素递归 512 切分；
  2. 有无面包屑注入；
  3. 有无 Qwen3-Reranker 精排；
- 评估结果作为「参数微调」与「是否启用 Chonkie 增强（语义/late chunking）」的决策依据。

---

## 8. 里程碑规划

| 阶段 | 内容 | 验收标准 |
| --- | --- | --- |
| M1 | 项目骨架 + 解析管线 | 样本文档（PDF/Word/Excel/MD 各 ≥1）解析为标准 Markdown，标题层级正确 |
| M2 | 切分模块（核心） | 按 4.2 六步策略输出 mom/text Chunk 列表，人工抽查 3 篇文档切分质量合格 |
| M3 | 编码 + Qdrant 入库 | `rag ingest` 全流程跑通；`rag info` 统计正确 |
| M4 | 混合检索 + 重排 | `rag search` 返回带溯源的结果，典型问题命中合理 |
| M5 | CLI 完善 + 评估 | 评估集指标产出，3 组 A/B 实验结论明确 |
| M6 | 部署适配 | 目标机（5060 Ti）环境安装文档 + 模型下载脚本 + 设备自适应验证 |

---

## 9. 风险与开放问题（需你确认）

| # | 事项 | 说明 | 建议 |
| --- | --- | --- | --- |
| 1 | 解析器最终质量 | Docling 对个别金融 PDF 表格/多栏识别可能有瑕疵 | M1 用你的**真实样例文档**验证；不佳则启用 PyMuPDF 快速通道或局部引入 MinerU |
| 2 | 部署机驱动/环境 | Blackwell 显卡需 CUDA 12.8+（PyTorch cu128）与驱动 ≥ 570；`flash-attention-2` 在 sm_120 上若安装困难，改用 PyTorch SDPA（功能不变） | 部署前确认目标机驱动版本 |
| 3 | 模型下载 | Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B（合计约 2.5GB） | 建议 ModelScope 源下载（国内稳定），附下载脚本 |
| 4 | 开发机 GPU | 本机若无 NVIDIA GPU，开发期 CPU 推理跑通即可 | 代码设备自适应，无需改动 |
| 5 | 项目/包命名 | 暂定 `wealth_rag` | 你可指定其他名称 |
| 6 | 评估集来源 | 需要你的领域问题样例（或由我先草拟、你修正） | M5 前提供 |

**请你审阅本计划书，特别是：**
1. 4.2 的「模仿对象 → 逻辑 → 适配」对应表与全部参数默认值是否符合预期；
2. 4.3–4.5 Qwen3 套件（Embedding/Reranker 0.6B）与 jieba-BM25 稀疏方案；
3. 第 9 节开放问题的取舍。

---

## 10. 参考来源

**切分策略（核心）**

- [RAGFlow 切分方法详解（DeepWiki：naive/title/table/laws/book 等源码级说明）](https://deepwiki.com/infiniflow/ragflow/6.2-task-execution-and-processing-system)
- [RAGFlow 数据集配置与内置解析（General/逐模板切分规则）](https://ragflow.io/docs/dataset_configuration)
- [RAGFlow 文档切片方式学习总结（CSDN）](https://blog.csdn.net/m0_71240768/article/details/162344469)
- [Chonkie 官方文档（Recursive/Semantic/Late/Neural 等 9 种切块器）](https://docs.chonkie.ai/common/open-source)
- [Chonkie DeepWiki（LateChunker/NeuralChunker 实现对比）](https://deepwiki.com/chonkie-inc/chonkie)
- [Advanced RAG Chunking Techniques 2026: Late Chunking, Semantic, Parent-Child](https://futureagi.com/blog/advanced-chunking-techniques-for-rag/)
- [RAG Chunking Strategies: The 2026 Benchmark Guide](https://www.premai.io/blog/rag-chunking-strategies-the-2026-benchmark-guide/)
- [Best Chunking Strategies for RAG 2026（Chroma 召回数据）](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [Late Chunking 效果分析（NFCorpus +6.5 nDCG@10）](https://ianas.fr/blog/2026/04/15/chunking-optimal-rag/)

**模型选型**

- [Qwen3 Embedding 官方博客（0.6B/4B/8B 规格、MRL、instruction-aware）](https://qwen.ai/blog?id=qwen3-embedding)
- [Qwen3 Embedding 中文博客（qwenlm）](https://qwenlm.github.io/zh/blog/qwen3-embedding/)
- [Qwen3-Embedding-0.6B 模型卡（sentence-transformers 用法，ModelScope）](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B/summary)
- [Qwen3-Embedding 家族评测（MTEB 多语言第一，2026）](https://theaibench.ai/models/qwen3-embedding/)
- [Qwen3-Reranker 部署避坑（logits yes/no 打分，CSDN）](https://mcp.csdn.net/6a7fdbdc10ee7a33f29b2f38.html)
- [Qwen3-Reranker 本地部署实测（RTX 4060 显存 ~1.2GB）](https://adg.csdn.net/6a76b93310ee7a33f298398a.html)
- [SGLang Rerank 模型支持（Qwen3-Reranker yes/no logprob 机制）](https://sgl-project.github.io/supported_models/retrieval_ranking/rerank_models.html)

**向量库与混合检索**

- [Qdrant 混合检索官方文档（prefetch + RRF 原生融合）](https://qdrant.tech/documentation/search/text-search/hybrid-search/)
- [Qdrant 中文文档：混合与多阶段查询](https://qdrant.org.cn/documentation/concepts/hybrid-queries/)
- [BM25 + RRF + Cross-Encoder 重排实战（Qdrant 原生混合能力）](https://aiworkflowlab.dev/article/how-to-build-hybrid-search-rag-bm25-rrf-fusion-cross-encoder-reranking)
- [Vector Database Showdown 2026: Qdrant vs LanceDB vs Milvus vs Chroma](https://www.web3aiblog.com/blog/vector-database-showdown-pinecone-weaviate-qdrant-lancedb-chroma-may-2026)
- [Open source vector databases 对比 2026](https://botmonster.com/ai/open-source-vector-databases-qdrant-milvus-weaviate/)

**解析与环境**

- [Best Open Source RAG Frameworks 2026（RAGFlow 复杂文档解析定位）](https://ossalt.com/blog/best-open-source-rag-frameworks-2026)
- [MinerU（备选解析器）](https://pypi.org/project/mineru/2.6.6/)
- [PaddleOCR-VL 0.9B 文档解析 SOTA（备选参考）](https://arxiv.org/html/2510.14528v3/)
- [RTX 5060 Ti 本地模型能力与显存参考](https://quelllm.fr/guide/llm-rtx-5060-ti)