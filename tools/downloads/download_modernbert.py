
# 文件迁入子目录后，从项目根目录定位数据和依赖。
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _project_sys.path:
    _project_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, configure_legacy_workdir
configure_legacy_workdir()

from huggingface_hub import snapshot_download

model_name = "tasksource/ModernBERT-large-nli"

local_path = r"D:\BaiduNetdiskDownload\PretrainedModel\models\tasksource--ModernBERT-large-nli"

snapshot_download(
    repo_id=model_name,
    local_dir=local_path
)

print("下载完成：", local_path)