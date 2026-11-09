"""项目路径：历史脚本迁入子目录后，仍从项目根目录读取数据和结果。"""
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent


def get_model_cache_dir():
    """使用 DICO_MODEL_CACHE 指定缓存；未指定时使用项目缓存或 HF 默认缓存。"""
    configured = os.environ.get("DICO_MODEL_CACHE")
    if configured:
        return str(Path(configured).expanduser())
    local_cache = PROJECT_ROOT / "model_cache"
    return str(local_cache) if local_cache.is_dir() else None


def configure_legacy_workdir():
    """兼容原脚本中的 ./data、./model、./results 相对路径。"""
    os.chdir(PROJECT_ROOT)

