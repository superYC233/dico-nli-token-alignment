"""仅在没有本地 WordNet 时运行；训练过程不会下载资源。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.deberta_kim_gate.wordnet_resource import ensure_wordnet, resource_info


if __name__ == '__main__':
    ensure_wordnet(allow_download=True)
    print(resource_info())
