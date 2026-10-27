import os

# 国内网络慢时取消下面这行的注释，走镜像
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id='MoritzLaurer/deberta-v3-base-zeroshot-v2.0',
    local_dir=r'D:\BaiduNetdiskDownload\PretrainedModel\models\deberta-v3-base-zeroshot-v2.0',
    ignore_patterns=['*.h5', '*.msgpack', '*.onnx', '*.ot'],
)

print('下载完成：', local_dir)