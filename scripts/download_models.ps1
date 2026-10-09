# 从 ModelScope 下载本地小模型（Qwen3-Embedding + Qwen3-Reranker）
# 用法：在项目根目录执行  .\scripts\download_models.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$py = "python"
if ($env:CONDA_PREFIX -and (Test-Path "$env:CONDA_PREFIX\python.exe")) {
    $py = "$env:CONDA_PREFIX\python.exe"
}
Write-Host "使用解释器: $py"

& $py -m pip show modelscope *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "安装 modelscope ..."
    & $py -m pip install modelscope -q -i https://pypi.tuna.tsinghua.edu.cn/simple
}

Write-Host "下载 Qwen3-Embedding-0.6B ..."
& $py -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir=r'$root\models\Qwen3-Embedding-0.6B')"

Write-Host "下载 Qwen3-Reranker-0.6B ..."
& $py -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-Reranker-0.6B', local_dir=r'$root\models\Qwen3-Reranker-0.6B')"

Write-Host "完成：模型已保存到 $root\models"