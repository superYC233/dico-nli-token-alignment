"""下载公开的分领域源标注，用于核查语境；不加入训练集。"""

import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / 'sources'
BASE = 'https://raw.githubusercontent.com/Jonapa/assessing_logical_coherence/main/data/'


def main():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for split in ['train', 'test']:
        for domain in ['headlines', 'images']:
            name = f'PhrasIS.{split}.{domain}.all.txt'
            path = SOURCE_DIR / name
            if not path.exists():
                print('下载公开源文件：', name, flush=True)
                with urllib.request.urlopen(BASE + name, timeout=45) as response:
                    data = response.read()
                if not data or b'\t' not in data or b'<html' in data[:100].lower():
                    raise ValueError('源文件格式不正确：' + name)
                path.write_bytes(data)
            records.append({'file': name, 'url': BASE + name,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (SOURCE_DIR / 'manifest.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
