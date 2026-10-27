import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.deberta_wordnet.wordnet_resource import ensure_wordnet, resource_info


if __name__ == '__main__':
    ensure_wordnet(allow_download=True)
    print('WordNet 已就绪：', resource_info())
