"""从ModelScope下载Qwen3-1.7B；模型目录可按本机情况修改。"""
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download(
    model_id='Qwen/Qwen3-1.7B',
    local_dir='D:/LLM/model/Qwen3-1.7B'
)

print('模型下载完成：')
print(model_dir)
