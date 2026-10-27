
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


script_dir = QWEN_DIR
project_root = script_dir.parent.parent

train_path = (
    project_root
    / 'data/train/dico_nli_train_track1_participant_labeled.csv'
)

source_dir = project_root / 'data/phrasis_sources'
output_dir = script_dir / 'outputs/sixclass_data'

source_urls = {
    'PhrasIS.train.all.txt': (
        'https://raw.githubusercontent.com/Jonapa/'
        'assessing_logical_coherence/main/data/PhrasIS.train.all.txt'
    ),
    'PhrasIS.test.all.txt': (
        'https://raw.githubusercontent.com/Jonapa/'
        'assessing_logical_coherence/main/data/PhrasIS.test.all.txt'
    )
}


def text_pair_key(text1, text2):
    # 仅整理空白，保留大小写和标点。
    text1 = ' '.join(text1.split())
    text2 = ' '.join(text2.split())

    # 只用于匹配 N 的对称关系。
    # 正反方向使用同一个键。
    return tuple(sorted((text1, text2)))


def get_source_files():
    source_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    source_info = []

    for filename, url in source_urls.items():
        path = source_dir / filename

        if not path.exists():
            print('下载源文件：', filename)
            request = Request(
                url,
                headers={'User-Agent': 'DiCo-NLI'}
            )

            try:
                with urlopen(request, timeout=30) as response:
                    data = response.read()
            except Exception as error:
                raise RuntimeError(
                    f'下载失败：{url}\n'
                    f'也可以手动下载，并保存为：{path}'
                ) from error

            path.write_bytes(data)

        else:
            print('使用已有源文件：', filename)

        data = path.read_bytes()

        if not data:
            raise ValueError(f'源文件为空：{path}')

        paths.append(path)

        # 保存来源和文件指纹，方便之后追溯。
        source_info.append({
            'filename': filename,
            'url': url,
            'sha256': hashlib.sha256(data).hexdigest()
        })

    return paths, source_info


def build_source_index(paths):
    index = defaultdict(list)

    for path in paths:
        lines = path.read_text(
            encoding='utf-8-sig'
        ).splitlines()

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            # 源文件没有表头，使用制表符分隔。
            # 第 2 列是关系，第 3、4 列是两个短语。
            columns = line.split('\t')

            if len(columns) < 4:
                raise ValueError(
                    f'源文件格式错误：'
                    f'{path.name}，第 {line_number} 行'
                )

            relation = columns[1].strip().upper()

            # 只检索 N 的三个细分类别。
            if relation not in ('SIMI', 'REL', 'OPPO'):
                continue

            key = text_pair_key(
                columns[2],
                columns[3]
            )

            index[key].append({
                'relation': relation,
                'filename': path.name,
                'line_number': line_number
            })

    return index


def attach_relations(train_df, source_index):
    result = train_df.copy()

    # 原 label 列保留，另外新增两列。
    result['n_relation'] = ''
    result['six_label'] = ''

    positive_labels = {
        'FORWARD_ENTAILMENT': 'F',
        'BACKWARD_ENTAILMENT': 'B',
        'EQUIVALENCE': 'E'
    }

    audit_records = []

    for index, row in result.iterrows():
        label = row['label']

        if label in positive_labels:
            result.at[index, 'six_label'] = positive_labels[label]
            continue

        if label != 'NEGATIVE_OTHER':
            raise ValueError(f'未知任务标签：{label}')

        key = text_pair_key(
            row['text1'],
            row['text2']
        )

        matches = source_index.get(key, [])

        relations = sorted({
            item['relation']
            for item in matches
        })

        if len(relations) == 1:
            status = 'matched'
            relation = relations[0]

            result.at[index, 'n_relation'] = relation
            result.at[index, 'six_label'] = relation

        else:
            status = (
                'unmatched'
                if not relations
                else 'ambiguous'
            )

        audit_records.append({
            'instance_id': row['instance_id'],
            'pair_id': row['pair_id'],
            'text1': row['text1'],
            'text2': row['text2'],
            'status': status,
            'candidate_relations': '|'.join(relations),
            'matched_sources': '|'.join(
                f"{item['filename']}:{item['line_number']}"
                for item in matches
            )
        })

    return result, pd.DataFrame(audit_records)


if __name__ == '__main__':
    paths, source_info = get_source_files()
    source_index = build_source_index(paths)

    # 只读取本任务官方 Train。
    train_df = pd.read_csv(
        train_path,
        dtype=str,
        keep_default_na=False
    )

    result, audit = attach_relations(
        train_df,
        source_index
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 先保存匹配审计表。
    audit_path = output_dir / 'train_n_label_audit.csv'

    audit.to_csv(
        audit_path,
        index=False,
        encoding='utf-8-sig'
    )

    # 有未匹配或类别冲突时停止，不猜标签。
    unresolved = audit[audit['status'] != 'matched']

    if not unresolved.empty:
        print(unresolved.to_string(index=False))

        raise ValueError(
            f'有 {len(unresolved)} 条 N 无法唯一确定类别。'
            f'请检查审计表：{audit_path}'
        )

    # 原始官方 CSV 不覆盖，另存一个带细分类别的版本。
    output_path = output_dir / 'train_with_relations.csv'

    result.to_csv(
        output_path,
        index=False,
        encoding='utf-8-sig'
    )

    manifest = {
        'official_train': str(train_path),
        'official_train_sha256': hashlib.sha256(
            train_path.read_bytes()
        ).hexdigest(),
        'train_rows': len(result),
        'n_rows': len(audit),
        'n_relation_counts': result.loc[
            result['label'] == 'NEGATIVE_OTHER',
            'n_relation'
        ].value_counts().to_dict(),
        'sources': source_info
    }

    (output_dir / 'source_manifest.json').write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2
        ),
        encoding='utf-8'
    )

    print('\n===== 六类数据准备完成 =====')
    print('训练样本数：', len(result))
    print('已匹配 N 样本数：', len(audit))

    print('\n六类标签数量：')
    print(result['six_label'].value_counts())

    print('\n训练数据：', output_path)
    print('匹配审计表：', audit_path)