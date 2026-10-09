# 技术总结：个人财富规划 RAG 检索模块

> 日期：2026-10-09
> 对应计划书：[PLAN.md](./PLAN.md)
> 范围：文档摄入（解析 → 切分 → 编码 → 入库）+ 检索（混合召回 → 融合 → 重排）；不含答案编排。

---

## 1. 整体技术架构

### 1.1 定位与约束

本模块是「个人财富规划 Agent 助手」的知识检索底座：把专业金融知识（研报、教材、法规、笔记）转化为可检索的证据库，
面向中文用户提供高相关、可溯源的证据片段，供后续 LLM 编排阶段拼接回答。

关键约束（决定了全部选型）：

| 约束 | 具体条件 |
| --- | --- |
| 硬件 | 部署机：RTX 5060 Ti 8GB / 16GB 内存 / R7 9700（Blackwell，sm_120）；开发机：RTX 4060 8GB |
| 模型分工 | 大模型走 API；小模型（嵌入、重排）本地运行，须装进 8GB 显存 |
| 语言与领域 | 中文为主，金融领域（术语、数字、法规编号密集） |
| 数据规模 | 小规模：几十篇文档（数千 ~ 数万切块） |
| 交付形态 | Python 库 + CLI，便于后续 Agent 直接集成 |

### 1.2 分层架构

```
┌──────────────────────────── 接入层 ────────────────────────────┐
│  rag CLI（Typer）：ingest / search / eval / info              │
├──────────────────────────── 编排层 ────────────────────────────┤
│  摄入：pipeline（解析→切分→编码→入库）                          │
│  检索：Retriever（查询编码→双路召回→RRF 融合→重排→证据组装）      │
├──────────────────────────── 能力层 ────────────────────────────┤
│  解析 DocumentParser（Docling）   切分 MarkdownChunker（六步）   │
│  编码 Qwen3Embedder               稀疏 jieba + BM25/稀疏向量     │
│  重排 Qwen3Reranker（yes/no logit）│ 计数 TokenCounter（tiktoken）│
├──────────────────────────── 存储层 ────────────────────────────┤
│  Qdrant 本地模式（dense + sparse 命名向量 / payload 过滤）        │
└────────────────────────────────────────────────────────────────┘
```

### 1.3 数据流与数据模型

**摄入管线（离线）**：

```
原始文档 → Docling 解析为「标准 Markdown（含标题层级/页码标记）」
        → 六步切分（结构切分→split-then-merge→硬切分兜底→表格行切→父子块→元数据）
        → Qwen3-Embedding 批量编码（稠密）+ jieba 词频（稀疏）
        → Qdrant 入库
```

**检索管线（在线）**：

```
中文查询 → Qwen3 查询编码（带指令前缀）+ jieba 分词
        → 稠密 top-30 + 稀疏 top-30 双路召回
        → RRF 融合 top-20 → Qwen3-Reranker 精排 → Top-K 证据块
        → 返回：文本 + 标题面包屑 + 来源 + 页码 + 父块(mom)上下文
```

**核心数据模型**：

- `Chunk`：**mom/text 两级结构**——`text` 子块（≤512 tokens，参与嵌入与检索），
  `mom` 父块（所属「节」全文，存入子块元数据供上下文扩展，不单独入库）；
- `SearchResult`：证据块 + 溯源字段（`citation` 可直接拼接给 LLM）。

### 1.4 目录结构

```
RAG/
├── PLAN.md / TECH_SUMMARY.md
├── config.yaml                  # 全部可调参数（切分/检索/设备/模型）
├── pyproject.toml               # 依赖与 rag 命令入口
├── data/                        # 知识文档（samples/ 为测试样例）
├── db/                          # Qdrant 本地库 + 稀疏词表
├── models/                      # 本地模型（Qwen3-Embedding / Qwen3-Reranker）
└── src/wealth_rag/
    ├── config.py  models.py  evaluation.py  cli.py
    ├── ingestion/  (parser · chunker · tokenizer · pipeline)
    ├── embedding/  (encoder)
    ├── sparse/     (bm25)
    ├── store/      (qdrant_store)
    └── retrieval/  (hybrid · rerank)
```

---

## 2. 技术要点：候选方案、选用方案与理由

### 要点总览

| # | 技术要点 | 候选方案 | 选用方案 | 核心理由 |
| --- | --- | --- | --- | --- |
| 1 | 文档解析 | Docling / MinerU / PyMuPDF4LLM / RAGFlow DeepDoc | **Docling**（备选 PyMuPDF） | 多格式统一、输出结构化 Markdown、无需 OCR、CPU 可跑 |
| 2 | 切分策略 | 固定长度 / 递归 / 语义切分 / late chunking / RAGFlow 模板化 | **源码级模仿 RAGFlow 六步策略** | 结构优先最契合金融文档；参数经生产验证；成本可控 |
| 3 | 嵌入模型 | API embedding / BGE-M3 / Qwen3-Embedding | **Qwen3-Embedding-0.6B（本地）** | MTEB 多语言最强系列；0.6B 适配 8GB 显存 |
| 4 | 稀疏检索 | ES/BM25 全家桶 / BGE-M3 sparse / Qdrant/bm25 / jieba+自建 | **jieba 分词 + BM25**（两种落地方式） | 中文分词可控；金融术语/数字强依赖关键词路 |
| 5 | 融合 | 加权归一化 / RRF | **RRF** | 无参数调优、稳定；可原生或应用层实现 |
| 6 | 重排 | 不重排 / bge-reranker-v2-m3 / Qwen3-Reranker | **Qwen3-Reranker-0.6B（本地）** | 与嵌入同源；判别式打分；显存友好 |
| 7 | 向量库 | Milvus / Qdrant / LanceDB / Chroma / FAISS | **Qdrant 本地模式** | 零部署、payload 过滤、稀疏向量能力、可平滑升级服务化 |
| 8 | 运行设备 | 纯 CPU / CUDA / API | **设备自适应（CUDA 优先，CPU 兜底）+ FP16** | 8GB 显存余量充足；双机环境差异无感 |
| 9 | 切分计数粒度 | 字符数 / tiktoken tokens | **tiktoken cl100k_base，512 tokens** | 与 RAGFlow 默认对齐，便于参照调参 |
| 10 | 中文金融适配 | 通用切分 / 领域增强 | **面包屑注入 + 金融实体保护 + 表格文本化** | 提升中文召回与上下文完整性 |
| 11 | 工程形态 | HTTP 服务 / 库+CLI | **Python 库 + Typer CLI** | 后续 Agent 直接 import，无部署负担 |
| 12 | 幂等与增量 | 全量重建 / 文档级 upsert | **稳定 doc_id + 文档级删重建 + `--rebuild`** | 小规模下简单可靠，避免状态漂移 |

### 要点 1：文档解析 —— 为什么是 Docling

- **候选**：RAGFlow DeepDoc（最强但绑定整体系统）、MinerU（PDF 解析最强但依赖重型模型下载、仅 PDF）、
  PyMuPDF4LLM（极轻但标题/表格结构识别弱）、Docling（IBM，MIT 许可）。
- **选用**：Docling 为主解析器，PyMuPDF4LLM 留作快速通道备选。
- **理由**：本阶段数据为**纯文本版** PDF + Office + 网页，且规模小——MinerU 的强项（扫描件、公式、多栏）
  用不上，其模型依赖反而成为负担；Docling 一个工具统一覆盖全部格式，
  并且输出**带标题层级的 Markdown**，这正是后续「结构感知切分」的必需输入。

### 要点 2：切分策略（核心）—— 为什么模仿 RAGFlow

- **候选**：固定/递归长度切分（通用但破坏语义边界）、语义切分（召回 91–92% 但需嵌入每个句子，成本高）、
  late chunking（长文档 +6.5 nDCG 但嵌入开销 2–3 倍、对短文档无增益）、
  **RAGFlow 模板化/结构感知切分**。
- **选用**：源码级模仿 RAGFlow 的具体实现并适配，落地为六步策略：
  1. **结构切分**（对齐 `title_chunker`）：以 Markdown 标题树为骨架，按「节」切分；
  2. **节内切分**（对齐 `naive_merge`）：按中文标点 `\n!?。；！？` 切分 → tiktoken 计数 → 贪心合并至 512 tokens；
  3. **超限兜底**：单句超限时按句子边界再切分、必要时硬切分；
  4. **表格处理**（对齐 `table.py`）：表格为原子块，超限按行切分且**重复表头 + 续表标记**；
  5. **父子块**（对齐 TokenChunker `mom/text`）：子块检索、节全文（mom）入元数据供上下文扩展；
  6. **元数据组装**：面包屑注入 + 页码溯源 + 金融实体保护。
- **理由**：
  - RAGFlow 是中文生态中生产验证最充分的切分体系，其参数（512 tokens、中文标点分隔符、标题层级）
    可以直接继承，**免去了从零调参**；
  - 「结构优先」对金融文档（法规、研报、教材）最契合：章节/条款本身即语义边界，
    固定长度切分会在「第 X 条」「4.5%」这类关键实体中间断开；
  - 语义切分与 late chunking 的成本/收益比在本场景不成立（数据非超长文档流），
    列为评估集验证后的可选增强。

### 要点 3：嵌入模型 —— 为什么是 Qwen3-Embedding-0.6B

- **候选**：调用 API 嵌入（效果好但有网络依赖、按量计费、隐私顾虑）、BGE-M3（568M，成熟但已被超越）、
  Qwen3-Embedding 0.6B / 4B / 8B（2025 发布，MTEB 多语言榜单第一系列）。
- **选用**：Qwen3-Embedding-0.6B 本地推理。
- **理由**：与用户「小模型本地运行」的分工一致；0.6B 权重 FP16 仅约 1.2GB，
  在 8GB 显存内与重排模型共存无压力；支持 32K 上下文与 MRL 降维、instruction-aware 查询前缀；
  8B 版本质量更高但超出显存，4B 亦逼近上限，0.6B 是显存约束下的最优尺寸。

### 要点 4：稀疏检索 —— 为什么是 jieba + BM25

- **候选**：Elasticsearch 全文检索（重，需独立服务）、BGE-M3 稀疏输出（换模型）、
  Qdrant 官方 `Qdrant/bm25` 稀疏模型（上手快但**中文分词效果一般**）、jieba + 自建 BM25。
- **选用**：jieba 精确模式分词 + BM25 打分，提供两种落地方式：
  - **方式 A（默认候选）**：词频构造稀疏向量写入 Qdrant，服务端 **IDF 修正**（近似 BM25），
    与稠密路一起用 Qdrant 原生 `prefetch` + `RRF` 单请求完成；
  - **方式 B（兜底）**：`rank_bm25` 本地索引 + 应用层 RRF（不依赖 Qdrant 稀疏能力）。
- **理由**：金融场景中「年化收益率」「第 24 条」「12000 元」这类关键词/数字是强命中信号，
  稠密向量对数字与专用术语不敏感，必须保留关键词路；中文分词用 jieba 可控、可停用词治理，
  而官方 bm25 模型对中文分词不友好。两种落地方式并存使实现不受存储层能力限制。

### 要点 5：融合 —— 为什么是 RRF

- **候选**：分数加权归一化（需调参且两路分数量纲不同）、**RRF（倒数排名融合）**。
- **选用**：RRF（k=60），优先 Qdrant 原生实现（省一次往返），兜底为应用层实现。
- **理由**：RRF 只依赖排名不依赖分数分布，**无需调参、跨模型稳定**，是混合检索的事实标准；
  实现代价极小（融合逻辑约 10 行），且天然适配「召回数量不同」的两路结果。

### 要点 6：重排 —— 为什么是 Qwen3-Reranker-0.6B

- **候选**：不重排（召回质量不足）、bge-reranker-v2-m3（成熟）、Qwen3-Reranker 0.6B/4B/8B。
- **选用**：Qwen3-Reranker-0.6B 本地推理，对融合后 top-20 精排取 top-k。
- **理由**：与嵌入模型同源同代，打分风格协同；实现上它是 **decoder-only 判别式重排**——
  取「yes/no」两个 token 的末位 logits 做 softmax，**不是文本生成**，单条打分约毫秒级，
  FP16 约 1.2GB 显存；4B/8B 超显存或逼近上限，留作后续量化评估项。

### 要点 7：向量库 —— 为什么是 Qdrant 本地模式

- **候选**：Milvus（K8s 级运维，过重）、LanceDB（嵌入式但混合检索生态弱）、
  Chroma（混合检索与过滤能力弱）、FAISS（无元数据过滤/无升级路径）、**Qdrant**。
- **选用**：`QdrantClient(path=...)` 本地嵌入模式，零部署；命名向量承载 dense（+sparse）。
- **理由**：本地模式无需任何服务进程（契合「个人机器上的小项目」），
  数据文件可原样迁移到 Qdrant 服务模式（后续升级零代码改动）；
  payload 过滤（按 doc_id/类型）与稀疏向量能力内建；Rust/HNSW 单机性能为同梯队最强。

### 要点 8：运行设备与显存预算

- **候选**：全部上云/API、全本地大模型、本地只放小模型。
- **选用**：设备自适应（`auto` → CUDA 优先、CPU 兜底）+ FP16；本地仅嵌入与重排（合计约 2.5GB 显存）。
- **理由**：8GB 显存装不下高质大模型，答案编排交给 API 是既定分工；
  嵌入/重排小模型本地化则保证检索环节**离线可用、零边际成本、数据不出本机**；
  开发机（4060）与部署机（5060 Ti，Blackwell sm_120）架构不同但都需要 CUDA 12.8+/PyTorch 2.8+，
  FP16 + 设备自适应代码使两机行为一致（部署细节见计划书第 9 节风险项）。

### 要点 9 ~ 12（工程性要点）

- **计数粒度**：按 tiktoken token 而非字符计数，与 RAGFlow 的 512-token 默认直接对齐，调参有参照系；
- **领域增强**：面包屑（标题路径）注入嵌入文本——低成本的 late-chunking 替代；金融实体保护避免
  「第 24 | 条」「4.5 | %」被切到两块；表格逐行文本化为「表头: 值」供嵌入与关键词命中；
- **交付形态**：库 + CLI（Typer），Agent 阶段直接 `from wealth_rag import ...`，或经 CLI 批处理；
- **幂等入库**：`doc_id` 由相对路径稳定派生、`chunk_id` 由 `doc_id + 序号` 稳定派生（UUID 格式匹配 Qdrant 约束），
  文档级「先删后插」保证重复摄入与文档更新不产生脏数据；`--rebuild` 为全量重建开关。

---

## 3. 验证与实测结果

> 验证环境（开发机）：RTX 4060 Laptop 8GB / Python 3.12 / torch 2.9.1+cu128 / transformers 5.19 / docling 2.135 / qdrant-client 1.19.1。
> 验证日期：2026-10-09。

### 3.1 各环节验证结论

| 环节 | 验证内容 | 结果 |
| --- | --- | --- |
| 切分器 | 样例 Markdown（含长节 / 表格 / 多级标题）切分 | ✅ 6 块；长节自动产生 mom/text 父子块（注入父块全文）；表格独立成块并「表头: 值」文本化；面包屑全部注入；512-token 上限精确（tiktoken cl100k_base 可用） |
| 解析 | md / docx / xlsx / html / pdf 五种格式 | ✅ 前四种经 Docling 输出带标题层级的标准 Markdown；PDF 走 PyMuPDF 快速通道（含页码标记），Docling 失败时自动回退 |
| 存储 | Qdrant 本地模式：稀疏向量 + IDF 修正 + 原生 prefetch/RRF | ✅ 全部支持，单请求完成双路召回与融合；文档级删除、统计、全量遍历均正常 |
| 编码 | Qwen3-Embedding-0.6B 本地 GPU（FP16） | ✅ 1024 维归一化向量；语义排序正确（「再平衡」查询命中对应文档 0.83） |
| 重排 | Qwen3-Reranker-0.6B yes/no 打分 | ✅ 相关块 0.79 / 0.89，无关块 0.003 ~ 0.008，区分度明确 |
| 端到端 | 5 篇样例文档全量重建摄入 | ✅ 5/5 文件、14 块；耗时 25.9s（含模型加载）；检索含冷启动 ~15s，热查询秒级 |
| 评估 | 6 条样例查询，Recall@10 / MRR / nDCG@10 | ✅ 含重排与不含重排均为 1.000（样例集区分度有限，重排增益需真实评估集验证） |

### 3.2 实施中发现的问题与处理（重要经验）

1. **transformers v5 参数更名**：`torch_dtype` → `dtype`，代码已做版本自适应（`dtype_kwargs`）；
2. **Qwen3 chat template 不适配 Reranker 用法（关键坑）**：transformers v5 内置模板会注入 ` thinking` 脚手架并改写内容，
   导致 yes/no 打分失真（修复前相关/无关统一约 0.004；改用官方**手动拼接格式**后恢复正常区分度），代码已固化官方格式并注释原因；
3. **PyPI 默认安装的 torch 为 CPU 版**（`2.14.1+cpu`），须显式安装 CUDA 构建；
   实测阿里云 pytorch 镜像不稳定（~115KB/s 且连接中断），改用官方源（~1.9MB/s），最终锁定 **2.9.1+cu128**（同时满足 4060 与部署机 Blackwell）；
4. **Docling PDF 版面模型在本机网络不可达**（HF xethub 超时；`docling-tools.exe` 被系统应用管控策略拦截），
   已实现 `pdf_engine: auto`（Docling 优先、失败自动回退 PyMuPDF），并保留部署机网络可用时直接使用 Docling 的能力；
5. **Qdrant 点 ID 约束**：必须为 UUID/整数格式，chunk_id 已由 md5 稳定派生为 UUID 字符串；
6. **Qdrant 本地模式退出噪音**：客户端在解释器退出阶段清理报错，已通过 `atexit` 提前显式关闭解决。

### 3.3 版本与交付物

- 依赖版本：`torch 2.9.1+cu128` / `transformers 5.19.0` / `docling 2.135.0` / `qdrant-client 1.19.1` / `tiktoken 0.14.0` / `jieba 0.42.1`；
- 本地模型：`models/Qwen3-Embedding-0.6B`（1.15GB）、`models/Qwen3-Reranker-0.6B`（1.15GB），ModelScope 下载约 32MB/s；
- 交付清单：`src/wealth_rag/`（10 个模块）、`config.yaml`、`rag` 命令（ingest / search / eval / info）、`eval/` 评估脚本与格式示例、`DEPLOY.md` + `scripts/download_models.ps1`（部署适配）。

---

## 4. 已知限制与后续演进

1. **语义切分 / late chunking**：当前不启用，待真实评估集验证收益后决定（Chonkie 可作为增强库引入）；
2. **多模态**：图表理解（研报图表、扫描件）不在本阶段；如需，可在解析层引入视觉模型；
3. **知识图谱**：实体关系抽取（LightRAG 路线）留作后续升级；
4. **服务化**：如 Agent 与检索跨进程部署，可把本地 Qdrant 切到服务模式或加一层薄 FastAPI；
5. **评估集**：当前为小规模人工评估方案，指标仅文档级命中；后续可扩展至块级标注与自动化回归。