"""只分析0.935493版本已有预测，不加载模型、不训练、不生成训练文本。"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_manual_aug.prepare_manual_data import REFERENCE_DIR, FULL


OUTPUT_DIR = Path(__file__).resolve().parent / 'reference_analysis'
REVIEW_NOTES = {
    'dico_phrasis_0000034': '新闻实体与职务片段可能依赖原新闻上下文；不据此编造同实体训练样本。',
    'dico_phrasis_0000198': '常见词汇包含关系候选：chocolate与candy的范围差别；适合用不同词汇做包含关系练习。',
    'dico_phrasis_0000375': '陈述与说话的动作粒度不同；片段缺少宾语和语境。',
    'dico_phrasis_0000432': '疑似标签或语境问题：black top和necklace本身不能保证big necklace；保留官方标签但不直接仿写。',
    'dico_phrasis_0000497': '跑道与机场的地点联系不等于短语的严格包含；可能依赖源语境。',
    'dico_phrasis_0000499': '外观与对象称呼混合，语法片段不完整；需要人工核验。',
    'dico_phrasis_0000616': '对象类型、属性和of片段混合；不能只归因于词汇知识不足。',
    'dico_phrasis_0000729': '在人类指称下是定义改写的候选；适合补不含Dev词汇的等价改写。',
    'dico_phrasis_0000817': 'rocks next to water并不必然是海岸；金标E可能依赖上下文。',
    'dico_phrasis_0000948': 'crackdown与violence并非在所有语境等价；不把该金标直接推广为一般规则。',
    'dico_phrasis_0000979': 'after与to的关系角色不同；源对齐或语境可能影响金标E。',
    'dico_phrasis_0001103': 'ginger与orange的颜色近似及指称约定；不把近似颜色一律标E。',
    'dico_phrasis_0001156': '草地属性与lawn地点指称可能依赖场景；不视作无条件同义词。',
    'dico_phrasis_0001456': '具体运动与playing的解释需要事件语境；不是纯字符串方向问题。',
    'dico_phrasis_0001551': '疑似方向或标注问题：按通常集合解释Saudi Women更窄；不据此翻转任务定义。',
    'dico_phrasis_0001567': '事故事件与死亡统计是新闻语境关系；片段信息不足。',
    'dico_phrasis_0001781': 'train修饰station的范围限制候选；可以用不相关的新词补相同逻辑。',
    'dico_phrasis_0001940': '进餐与吃东西的事件范围区别；不能仅按词数判断等价。',
    'dico_phrasis_0002242': 'Opens有多义性；脱离对象时不能一律等价于开始营业。',
    'dico_phrasis_0002463': '名词片段与动作表达相关但不保证等价；可能涉及标注口径。',
    'dico_phrasis_0002492': '等级类别与最高级比较不是同一个属性；避免把强烈相似当等价。',
    'dico_phrasis_0002616': '目标与到达/接触的介词含义有差别，但单独片段需要语境。',
    'dico_phrasis_0002738': '事件状态之间不能互推：被定罪并不保证入狱；适合补独立事件状态对照。',
    'dico_phrasis_0002900': '通常senator属于lawmaker；金标N与字面包含可能冲突，不能把它当一般词义规则。',
    'dico_phrasis_0002933': '地域修饰、立法机构称谓与MP范围可能依赖标注口径；需要核验。',
}


def error_family(gold, prediction):
    if gold == 'N':
        return 'N_to_E' if prediction == 'E' else 'N_to_FB'
    if gold == 'E':
        return 'E_to_FB'
    if prediction == 'E':
        return 'FB_to_E'
    return 'F_B_reversal'


def main():
    path = REFERENCE_DIR / 'epoch_3/predictions_with_scores.csv'
    frame = pd.read_csv(path, keep_default_na=False)
    short = {value: key for key, value in FULL.items()}
    frame['gold'] = frame['gold_label'].map(short)
    errors = frame[frame['gold'] != frame['pair_prediction']].copy()
    errors['error_family'] = [error_family(gold, prediction)
                              for gold, prediction in zip(errors['gold'], errors['pair_prediction'])]
    errors['assistant_review_note'] = errors['pair_id'].map(REVIEW_NOTES).fillna('待复核')
    errors['review_status'] = 'assistant_analysis; not independently adjudicated'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUTPUT_DIR / 'all_errors.csv', index=False, encoding='utf-8-sig')
    groups = errors.drop_duplicates('pair_id')
    groups.to_csv(OUTPUT_DIR / 'error_groups_for_review.csv', index=False, encoding='utf-8-sig')
    matrix = pd.crosstab(frame['gold'], frame['pair_prediction']).reindex(
        index=['F', 'B', 'E', 'N'], columns=['F', 'B', 'E', 'N'], fill_value=0,
    )
    matrix.to_csv(OUTPUT_DIR / 'confusion_matrix.csv', encoding='utf-8-sig')
    counts = errors.groupby('error_family').agg(rows=('instance_id', 'size'), groups=('pair_id', 'nunique'))
    counts.to_csv(OUTPUT_DIR / 'error_counts.csv', encoding='utf-8-sig')
    lines = [
        '# 0.935493版本错误复查', '',
        f'参考：{path.as_posix()}', '',
        '同一checkpoint：独立W-F1=0.480876，pair-aware W-F1=0.935493。',
        f'结构后错误{len(errors)}行，涉及{errors.pair_id.nunique()}组。', '',
        '| 错误 | 行数 | 组数 |', '|---|---:|---:|',
    ]
    for family, row in counts.iterrows():
        lines.append(f'| {family} | {row.rows} | {row.groups} |')
    lines += [
        '', '## 本次增强针对什么', '',
        '主要错误池是E与F/B之间的混淆：32行、16组。不是所有错误都能归因于模型不懂词义。',
        '新增训练草稿使用新词汇练习：属性限制、严格上下位、等价表达，以及共享词但属性或对象不同的N。',
        '没有将Dev中的错误文本改标签后放入Train，也没有替换几个Dev词就保留原句训练。', '',
        '## 标签与语境问题', '',
        'Saudi Women/Saudis的方向、black top与big necklace的关系、senator/lawmaker的N标签等，',
        '按字面集合解释存在疑问。其他若干新闻片段及介词关系需要原上下文。',
        '以上均是复核线索，不是已确认的官方标注错误；本次不修改任何官方标签。',
        '不能承诺用一般语义更一致的人工样本就能修复所有官方Dev错误。', '',
        '## 开发边界', '',
        'Dev错误分析参与了增强方向选择，因此本次是Dev指导的开发实验。',
        '新增样本即使与Dev词汇零交集，也不能恢复一个从未使用过的测试集。',
        '生成/过滤脚本只把Dev文本用作拒绝黑名单；分析文件不进入模型或训练输入。',
        '新增标签为助手手工设计草稿，尚无独立人工标注者审核。', '',
        '完整25组及逐组复核意见见error_groups_for_review.csv。',
    ]
    (OUTPUT_DIR / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest = {'reference_predictions': str(path.resolve()),
                'reference_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUTPUT_DIR / 'sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(counts.to_string())
    print('错误报告：', OUTPUT_DIR / 'REPORT.md')


if __name__ == '__main__':
    main()
