# 部署指南（目标机：RTX 5060 Ti 8GB / 16GB 内存 / R7 9700）

> 适用：把本 RAG 模块部署到目标机（Windows）。开发机已验证流程见 TECH_SUMMARY.md。

## 0. 前置要求

| 项目 | 要求 | 检查命令 |
| --- | --- | --- |
| NVIDIA 驱动 | ≥ 570（Blackwell sm_120 需要） | `nvidia-smi` |
| Python | 3.12（推荐 conda 环境） | `python --version` |
| 磁盘 | ≥ 10GB 可用（模型 ~2.5GB + 依赖） | — |

## 1. 创建环境与安装依赖

```powershell
# 1) 创建独立环境
conda create -y -n wealth_rag python=3.12

# 2) 安装 CUDA 版 PyTorch（cu128，支持 5060 Ti 的 sm_120）
#    已实测版本：torch 2.9.1+cu128
D:\miniconda\envs\wealth_rag\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3) 安装项目依赖（进入项目目录后执行）
cd <项目目录>
D:\miniconda\envs\wealth_rag\python.exe -m pip install -e ".[dev,fast]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 备注：`dev` 提供 modelscope（模型下载）；`fast` 提供 PyMuPDF 快速通道（PDF 解析备选）。
> 如阿里云 pytorch 镜像下载不稳定，直接用上面的官方源（实测 ~1.9MB/s）。

## 2. 下载本地模型（ModelScope 源，国内快）

```powershell
.\scripts\download_models.ps1
```

或手动执行（等效）：

```powershell
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir=r'.\models\Qwen3-Embedding-0.6B')"
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-Reranker-0.6B', local_dir=r'.\models\Qwen3-Reranker-0.6B')"
```

模型放置在项目根目录 `models/` 下时会被自动识别（见 `config.yaml → paths.models_dir`）。

## 3. 配置检查（config.yaml）

```yaml
device: auto                 # 自动检测 CUDA（5060 Ti 会启用 GPU）
paths:
  data_dir: ./data           # 放入你的真实知识文档
  db_dir: ./db               # 向量库目录（随项目走，可直接拷贝迁移）
parsing:
  pdf_engine: auto           # Docling 优先；若目标机无法访问 HuggingFace，自动回退 PyMuPDF
```

## 4. 部署验证清单

```powershell
# ① GPU 可用性（应输出 5060 Ti 与 True）
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ② 摄入样例文档（验证全链路；随后可删除 data/samples）
python -m wealth_rag.cli ingest data/samples --rebuild
python -m wealth_rag.cli info                    # 应显示 设备=cuda、5 文档、14 块

# ③ 检索冒烟
python -m wealth_rag.cli search "什么是再平衡策略？"
```

## 5. 已知问题与对策

| 问题 | 表现 | 对策 |
| --- | --- | --- |
| Docling PDF 版面模型下载失败 | 解析 PDF 报网络错误 | 已内置自动回退 PyMuPDF；或设置 `$env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET="1"` 后重试；或直接 `pdf_engine: pymupdf` |
| tiktoken 首次使用需下载词表 | 首次切分需联网 | 词表缓存在 `cache/` 内，之后离线可用（无网时自动退化为启发式计数） |
| flash-attention 安装困难 | 编译失败 | 不需要——默认使用 PyTorch SDPA，功能不受影响 |
| 显存余量 | 双模型驻留 | Qwen3-Embedding + Reranker FP16 合计约 2.5GB，8GB 显存有余量（当前按需分时加载） |