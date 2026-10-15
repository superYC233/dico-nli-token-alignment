
# 文件迁入子目录后，从项目根目录定位数据和依赖。
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _project_sys.path:
    _project_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, configure_legacy_workdir
configure_legacy_workdir()

from transformers import AutoTokenizer, AutoModel

# DeBERTa模型名称
model_name = "microsoft/deberta-v3-base"

# 指定模型下载缓存位置
cache_path = r"D:\BaiduNetdiskDownload\PretrainedModel\models"

# 加载分词器
my_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_path
)

# 加载DeBERTa模型
my_model = AutoModel.from_pretrained(
    model_name,
    cache_dir=cache_path
)