"""右键运行：生成无示例 / 随机 2 例 / 检索 2 例的 Prompt。"""
import json
import sys

import numpy as np
import pandas as pd

import config as cfg
from retrieval_utils import (
    read_frame, clean_candidate_pool, eligible_indices, pick_examples,
    cached_embeddings, render_prompt, preparation_spec,
)

sys.path.insert(0, str(cfg.QWEN_DIR))
from data_utils import build_prompt, tokenizer


def prepare():
    if cfg.NUM_EXAMPLES < 1:
        raise ValueError('NUM_EXAMPLES 必须大于零。')
    train = read_frame(cfg.TRAIN_FILE)
    dev = read_frame(cfg.DEV_FILE)
    # 检索部分只接收 ID 和文本，不接收 Dev 金标。
    queries = dev[['instance_id', 'pair_id', 'text1', 'text2']].copy()
    template = build_prompt('{text1}', '{text2}')
    spec = preparation_spec(train, queries, template)
    manifest_path = cfg.PREPARED_DIR / 'manifest.json'
    required_files = ['prompts.csv', 'demonstrations.csv', 'candidate_pool.csv',
                      'excluded_candidates.csv', 'query_exclusions.csv', 'prompt_preview.txt']
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['spec'] != spec:
            raise ValueError('此 RUN_NAME 已有不同配置的示例，请在 config.py 换一个 RUN_NAME。')
        if all((cfg.PREPARED_DIR / name).exists() for name in required_files):
            print('示例已准备好，配置一致，跳过重复准备：', cfg.PREPARED_DIR)
            return

    cfg.PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    pool, excluded = clean_candidate_pool(train)
    print('原始 Train：', len(train), '；清理后候选行：', len(pool))
    print('排除冲突或重复行：', len(excluded), '；Dev：', len(queries))
    # 相同短语只编码一次；仅对文本编码，不带标签或任务定义。
    texts = list(dict.fromkeys([*pool['text1'], *pool['text2'],
                               *queries['text1'], *queries['text2']]))
    vectors = cached_embeddings(texts, cfg.PREPARED_DIR)
    lookup = {text: i for i, text in enumerate(texts)}
    train1 = vectors[[lookup[t] for t in pool['text1']]]
    train2 = vectors[[lookup[t] for t in pool['text2']]]
    query1 = vectors[[lookup[t] for t in queries['text1']]]
    query2 = vectors[[lookup[t] for t in queries['text2']]]
    # 一次矩阵乘法算全部文本对的分数，约 660×3000，不需要向量数据库。
    all_scores = 0.5 * (query1 @ train1.T + query2 @ train2.T)

    prompts, demonstrations, exclusions = [], [], []
    for number, query in enumerate(queries.to_dict('records')):
        scores = all_scores[number]
        eligible = eligible_indices(pool, query)
        exclusions.append({'instance_id': query['instance_id'],
                           'excluded_candidate_rows': len(pool) - len(eligible)})
        for method in cfg.METHODS:
            indices = [] if method == 'baseline' else pick_examples(
                pool, eligible, scores, cfg.NUM_EXAMPLES, method, cfg.SEED + number
            )
            examples = pool.iloc[indices].to_dict('records')
            prompt = render_prompt(template, query, examples)
            token_count = len(tokenizer.encode(prompt))
            if token_count > cfg.MAX_PROMPT_TOKENS:
                raise ValueError(f"{method}/{query['instance_id']} 有 {token_count} tokens，超过限制。")
            prompts.append({'method': method, **query, 'prompt': prompt,
                            'prompt_tokens': token_count})
            for rank, index in enumerate(indices, 1):
                source = pool.iloc[index]
                demonstrations.append({
                    'method': method, 'instance_id': query['instance_id'], 'rank': rank,
                    'source_instance_id': source['instance_id'], 'source_pair_id': source['pair_id'],
                    'source_text1': source['text1'], 'source_text2': source['text2'],
                    'source_label': source['label'], 'source_short_label': cfg.LABEL_MAP[source['label']],
                    'similarity': float(scores[index]),
                })

    prompt_frame = pd.DataFrame(prompts)
    prompt_frame.to_csv(cfg.PREPARED_DIR / 'prompts.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(demonstrations).to_csv(cfg.PREPARED_DIR / 'demonstrations.csv', index=False, encoding='utf-8-sig')
    pool.to_csv(cfg.PREPARED_DIR / 'candidate_pool.csv', index=False, encoding='utf-8-sig')
    excluded.to_csv(cfg.PREPARED_DIR / 'excluded_candidates.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(exclusions).to_csv(cfg.PREPARED_DIR / 'query_exclusions.csv', index=False, encoding='utf-8-sig')
    preview = []
    for method in cfg.METHODS:
        first = prompt_frame[prompt_frame['method'] == method].iloc[0]
        preview.append(f"===== {method} / {first['instance_id']} =====\n{first['prompt']}")
    (cfg.PREPARED_DIR / 'prompt_preview.txt').write_text('\n\n'.join(preview), encoding='utf-8')
    manifest = {'spec': spec, 'original_train_rows': len(train), 'candidate_rows': len(pool),
                'excluded_rows': len(excluded), 'dev_rows': len(queries),
                'mean_prompt_tokens': prompt_frame.groupby('method')['prompt_tokens'].mean().to_dict()}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('准备完成：', cfg.PREPARED_DIR)
    print('可先查看 prompt_preview.txt 和 demonstrations.csv，确认例子是否合适。')


if __name__ == '__main__':
    prepare()

