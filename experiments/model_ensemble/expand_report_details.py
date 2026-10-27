"""展开各组方案的模型、数据、训练和权重说明；不运行真实模型。"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUTPUT = Path(__file__).resolve().parent / 'outputs/run01_all_existing'
OFFICIAL = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
BASE3290 = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged'
FULL = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']

# 名称、数据编号、结构、训练方式；参数值另外从每次运行保存的配置读取。
DESCRIPTIONS = {
    'M007': ('原始DeBERTa-base＋对齐＋词汇', 'official', 'microsoft原始预训练底座，没有MNLI/ANLI任务微调；自建CLS四分类头＋Token Alignment＋20维Lexical修正分支。', '全参数、两阶段：先Row CE，再0.5×三组合Pair CE。'),
    'M008': ('DeBERTa-base双向联合头LoRA，seed7', 'relation', '共享DeBERTa分别编码两个输入方向，再用联合四分类头；本版没有原Token Alignment分支，也没有Lexical。编码器用LoRA适配。', '每条原始行计算四分类CE，没有单独Pair CE；LoRA r16/alpha32/dropout0.1，作用于query_proj/value_proj。'),
    'M009': ('DeBERTa-base双向对齐联合头', 'relation', '两方向都提取原CLS和Token Alignment特征，再送联合四分类头；没有Lexical，不是三个组合分数相加。', '全参数四分类CE，没有独立Pair CE阶段；双向编码和批次组织均与原两阶段版不同。'),
    'M014': ('DeBERTa-base＋词汇＋E改写88', 'closure', '原CLS四分类头＋Token Alignment＋Lexical；closure88是历史文件名。', '全参数两阶段；注意文件的data_augmentation=false字段与实际3130行不一致，实际增加了88行，本文按CSV记录。'),
    'M017': ('CLS、对齐、词汇特征拼接版', 'official', '归一化CLS、token比较特征及词汇特征拼接，共1312维，经联合分类器输出四类；不是三个logit分支直接相加。', '全参数两阶段；Lexical有单独学习率，Pair后按结构F1选模。'),
    'M018': ('冻结0.938底座，只训练N门', 'relation', '以已经按Dev选出的0.938权重为冻结底座，保留F/B/E条件判断，只新增并训练N二分类头；没有Lexical。', '修复阶段仅训练n_head，BCE判断N/非N；主干和既有F/B/E输出冻结，没有Pair CE。p(N)=sigmoid(g)，其余概率为(1-p(N))×softmax(F/B/E)。'),
    'M027': ('原0.9306词汇版，seed42', 'official', 'DeBERTa NLI-base＋原CLS四分类头＋Token Alignment＋20维Lexical修正分支。', '全参数两阶段；Row N权重1.5，Pair EE权重1.25。'),
    'M028': ('词汇版，类别全部等权', 'official', '与原词汇版相同的CLS＋Alignment＋Lexical结构。', '全参数两阶段；Row权重[1,1,1,1]、Pair权重[1,1,1]，与0.9306版的权重设置不同。'),
    'M030': ('词汇版，N行权重1.8', 'official', '与原词汇版相同的CLS＋Alignment＋Lexical结构。', '全参数两阶段；Row的N权重改为1.8，Pair仍为[1,1,1.25]。'),
    'M032': ('词汇版，Pair阶段参数锚定', 'official', '保留原CLS＋Alignment＋Lexical。', '先完整Row CE并保存该阶段参数作为锚；Pair更新后按近端公式拉回Row锚点，anchor_strength=1000，之后再评估。'),
    'M036': ('词汇版＋scope200＋关系48', 'relation', '0.938的数据与原CLS/Alignment结构，再加20维Lexical；全参数训练。', '先Row再Pair，与历史0.938使用同一3290行训练快照；未另加局部语义头。'),
    'M044': ('词汇混合训练＋R-Drop', 'official', 'DeBERTa NLI-base＋Alignment＋Lexical。', '每批同时计算Row和Pair；同一批做两次dropout前向，平均CE并加四类对称KL，R-Drop权重1.0。'),
    'M045': ('词汇混合训练＋OOF难例加权', 'official', 'DeBERTa NLI-base＋Alignment＋Lexical；不增加文本行。', '用Train内部OOF预测选每类难例，hard_fraction=0.25、倍率2；按类归一化权重，只修改Row CE样本权重，Pair CE不改。'),
    'M046': ('词汇版，Row CE＋反向KL', 'official', 'DeBERTa NLI-base＋Alignment＋Lexical，不使用NLI pooler。', '混合组批；保留四类Row CE，用0.5×四类正反对称KL替代Pair CE。只在真实可逆组对齐F/B，没有另造N反向监督。'),
    'M047': ('词汇版＋首批100组证据监督', 'official', 'DeBERTa NLI-base＋Alignment＋Lexical，新增训练期token证据分类头。', '混合Row CE＋0.5 Pair CE＋0.1 token CE。首批100组标注覆盖175行，仅supported部分参与证据监督；不是把reason文字拼进输入，也不是生成reason。'),
    'M048': ('词汇版＋全Train证据监督', 'official', '与M047相同的主分类结构和训练期证据头。', '混合Row CE＋0.5 Pair CE＋0.1 token CE；3042行都有理由审阅标注，1756行supported用于证据监督，其他行仍保留原四分类监督。'),
    'M050': ('无Lexical的原结构对照，seed7', 'official', 'DeBERTa NLI-base＋原CLS四分类头＋Token Alignment，没有Lexical。', '全参数两阶段；用于换seed检验的无词汇对照，不是seed42的0.9248权重。'),
    'M051': ('词汇版换seed7', 'official', 'DeBERTa NLI-base＋原CLS四分类头＋Token Alignment＋Lexical。', '全参数两阶段，seed7；属于可靠性实验，不是新的预训练模型。'),
    'M054': ('scope200＋早期人工80', 'manual80', 'DeBERTa NLI-base＋CLS＋Token Alignment；没有Lexical。', '全参数两阶段；在3242行scope底座上再加早期80行人工草案。'),
    'M056': ('scope200＋动作事件200', 'event200', 'DeBERTa NLI-base＋CLS＋Token Alignment；没有Lexical。', '全参数两阶段；事件/动作200是额外训练数据，不是模型分支。'),
    'M059': ('原0.938版：scope200＋关系48', 'relation', 'DeBERTa NLI-base＋原CLS四分类头＋Token Alignment；没有Lexical、没有LoRA。', '全参数两阶段，先完整Row CE，再0.5 Pair CE；这就是提供0.938结构分的原模型。'),
    'M060': ('0.938同数据，Row/Pair混合训练', 'relation', '与M059相同的CLS＋Alignment，没有Lexical。', '每批按真实配对组和N单例混合，单次前向中算Row CE＋0.5 Pair CE；改变了组批、更新步数和前向暴露，不只是改shuffle顺序。'),
    'M064': ('0.938数据＋人工局部对照128', 'local_manual', 'DeBERTa NLI-base＋原CLS＋Alignment；没有Lexical。', '全参数两阶段；从NLI预训练底座重新开始，不是继续训练已选中的0.938权重。'),
    'M065': ('0.938数据＋复制Train128', 'local_copy', '与M064相同的CLS＋Alignment，没有Lexical。', '全参数两阶段；这是M064的复制对照，实际加入的是官方Train副本，目的在匹配新增行数、类别和Pair预算。'),
    'M071': ('DeBERTa LoRA r16/dropout0.1，seed42', 'relation', 'DeBERTa NLI-base＋CLS＋Alignment；没有Lexical。冻结大部分编码器参数，训练LoRA和任务头；这不是Qwen的r16版本。', '混合Row CE＋0.5 Pair CE；LoRA r16/alpha32/dropout0.1，target=query_proj/value_proj，seed42；没有梯度投影。'),
    'M072': ('DeBERTa LoRA r16/dropout0，seed7', 'relation', '与M071同类的DeBERTa LoRA＋CLS＋Alignment，没有Lexical。', '混合Row CE＋0.5 Pair CE；r16/alpha32/dropout0，seed7。保存配置的comparison_note沿用了旧文案，本文以lora_dropout实际数值0为准。'),
    'M075': ('0.938数据＋语义对照64，两阶段', 'semantic64', '原DeBERTa CLS＋Alignment；没有Lexical、没有LoRA。', '全参数两阶段；开启动态padding和梯度检查点，因而训练轨迹也与历史固定padding的0.938有差别。'),
    'M080': ('0.938数据＋仅Row阶段SAM', 'relation', '原DeBERTa CLS＋Alignment，没有Lexical、没有LoRA。', '先全部Row-SAM，rho=0.05，再普通0.5 Pair CE；只有Row阶段做参数扰动并二次前后向。'),
    'M081': ('0.938数据＋Row/Pair双阶段SAM', 'relation', '与M080相同的原CLS＋Alignment，没有Lexical。', 'Row和Pair阶段都使用SAM，二者rho均0.05；仍是先Row再Pair，不是混合训练。'),
    'M082': ('DeBERTa-large＋对齐＋词汇混合训练', 'official', 'NLI-large底座，hidden_size1024、24层；自建CLS四分类头＋Alignment＋Lexical。', '全参数混合组批，Row CE＋0.5 Pair CE；使用bf16 autocast和梯度检查点，按独立F1选模。'),
    'M083': ('DeBERTa-xsmall＋对齐＋词汇混合训练', 'official', 'cross-encoder/nli-deberta-v3-xsmall，hidden_size384、12层；CLS＋Alignment＋Lexical。', '全参数混合Row CE＋0.5 Pair CE，按独立F1选模。'),
    'M084': ('原0.9355版：仅加scope200', 'scope', 'DeBERTa NLI-base＋原CLS四分类头＋Token Alignment；没有Lexical。', '全参数两阶段；与官方3042行原版相比，训练文本新增scope200。'),
    'M087': ('冻结底6层，只训练Row CE', 'official', 'DeBERTa NLI-base＋原CLS四分类头＋Token Alignment，没有Lexical。冻结embedding和底部6层，顶部6层及任务头可训练。', '只用四类Row CE，Pair权重为0，不做Pair CE。按组组织批次仅为匹配训练安排，训练行每轮经过一次，无增强、无过采样。'),
    'M093': ('ModernBERT-base＋原对齐配方', 'official', 'tasksource/ModernBERT-base-nli替换DeBERTa，仍用原CLS四分类头＋Token Alignment；不使用原NLI三分类头，没有Lexical。', '全参数两阶段，SDPA后端，float32；本报告冻结使用盘点时保存的epoch2分数，不自动跟随之后的新运行。'),
    'M094': ('Qwen2.5-3B-Instruct，C＋reason', 'qwenC', '生成式Qwen，LoRA r16/alpha32/dropout0.05、all-linear；没有DeBERTa Token Alignment和Lexical。', '标签SFT＋0.2解释SFT，两个独立聊天Prompt、分别按监督token归一化；不使用Pair CE，预测只问标签，不提供Dev解释。'),
}


def file_link(label, path):
    return f'[{label}]({Path(path).as_posix()})'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def datasets():
    definitions = {
        'official': ('官方Train', OFFICIAL, '3042', '仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。'),
        'scope': ('官方Train＋scope200', ROOT / 'experiments/deberta_original_aug/outputs/run01_original_scope/training_data.csv', '3042＋200＝3242', '200行范围/修饰与结构对照：属性限制、同义结构改写、中心对象替换、并列范围和顺序等。是历史人工模板草案，不是官方额外金标。'),
        'relation': ('官方Train＋scope200＋关系48', BASE3290 / 'training_data.csv', '3042＋200＋48＝3290', '在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。'),
        'closure': ('官方Train＋E改写88', ROOT / 'experiments/deberta_closure/closure88_train.csv', '3042＋88＝3130', '直接核对CSV：原3042行保留，新增88行全部标E，组成44个双向组。例如文件将play/plays、is holding/holding标E。closure只是旧文件名；当前未找到完整生成脚本，不能仅凭名字断言88行全由经过核验的Train传递闭包推导。'),
        'manual80': ('scope底座＋早期人工80', ROOT / 'experiments/deberta_manual_aug/outputs/run01_original_manual/training_data.csv', '3042＋200＋80＝3322', '80行早期人工对照，涉及修饰限制、上下位、词汇/拼写等价、属性独立、中心词替换、事件状态等；不是后来的relation48。'),
        'event200': ('scope底座＋动作事件200', ROOT / 'experiments/deberta_manual_aug/outputs/run03_event_200/training_data.csv', '3042＋200＋200＝3442', '额外200行为20个动作/事件家族，每家族10行。包括动作具体化、同义改写、事件阶段/状态差异；这200行F40、B40、E80、N40。'),
        'local_manual': ('3290底座＋人工局部对照128', ROOT / 'experiments/deberta_manual_aug/outputs/run11_local128_manual/training_data.csv', '3042＋200＋48＋128＝3418', '额外128行F/B/E/N各32。16个训练家族，类型为名称/品牌边界、用途与地点、路径与行为者、角色限制。另有64行诊断数据不进入训练或选模。'),
        'local_copy': ('3290底座＋复制官方Train128', ROOT / 'experiments/deberta_manual_aug/outputs/run12_local128_copy/training_data.csv', '3042＋200＋48＋复制128＝3418', '复制128行已有官方Train样本，F/B/E/N各32；不是把人工local128加进去，也不代表新增128条独立语义知识。copy_control_sources.csv给出了原始ID和新副本ID。'),
        'semantic64': ('3290底座＋语义对照64', ROOT / 'experiments/deberta_manual_aug/outputs/run20_semantic64_staged_seed42/training_data.csv', '3042＋200＋48＋64＝3354', '额外64行F/B/E/N各16，分food_activity、aircraft_scope、modifier_scope、event_roles四个家族，各16行；官方Train词汇重组成新对照，涉及论元、上下位、修饰范围、事件角色及状态。'),
        'qwenC': ('Qwen C标签数据＋解释辅助任务', ROOT / 'experiments/qwen25_3b_reason/outputs/run01_c_reason02/C/prepared/train_C_with_reasons.csv', '3042＋筛选后的反向N475＝3517', '3517行都训练标签；其中2040行训练解释，包括1756行官方来源和284行反向N来源。解释是额外目标，不是再追加2040行标签文本，也没有把Dev理由输入模型。'),
    }
    result = {}
    for key, (title, path, formula, description) in definitions.items():
        frame = pd.read_csv(path, keep_default_na=False)
        counts = {short: int(frame.label.eq(full).sum()) for short, full in zip('FBEN', FULL)}
        result[key] = {'name': title, 'path': str(path), 'formula': formula, 'description': description,
                       'rows': len(frame), 'class_counts': counts, 'sha256': sha(path)}
    return result


def config_folder(prediction):
    folder = Path(prediction).parent
    while folder != ROOT:
        if (folder / 'run_config.json').is_file():
            return folder
        folder = folder.parent
    raise FileNotFoundError(prediction)


def member_details(member, data):
    key = member['id']
    title, dataset_key, architecture, training = DESCRIPTIONS[key]
    folder = config_folder(member['prediction_file'])
    path = folder / 'run_config.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    epoch_match = re.search(r'epoch_(\d+)', member['prediction_file'])
    metadata, metadata_file = {}, None
    for candidate in [folder / 'best_metrics.json', *[Path(p).parent / 'best_metrics.json' for p in member['checkpoint_candidates_existing']]]:
        if candidate.is_file():
            metadata, metadata_file = json.loads(candidate.read_text(encoding='utf-8')), candidate
            break
    epoch = int(epoch_match.group(1)) if epoch_match else metadata.get('best_epoch', metadata.get('epoch', '未记录'))
    epoch = member.get('selected_epoch', epoch)
    training = training.replace('，按独立F1选模', '')
    note = []
    if key == 'M014':
        note.append('新增88行全为E；旧data_augmentation=false不能据此解释为无增强。新88行与Dev归一化无序整对精确交集为0；这不证明来源独立或与Dev词汇完全隔离。')
    if key == 'M065':
        note.append('这是复制对照，不是人工128版。副本来自Train，不能把人工样本的Dev词汇隔离结论套到这些副本上。')
    if key == 'M072':
        note.append('保存配置中comparison_note有沿用旧实验的文字；实际lora.lora_dropout=0.0且seed=7。')
    if key == 'M094':
        note.append('融合选用训练结束回载best_adapter后的缓存：独立0.838121、结构0.902142；没有使用训练期epoch4较高的0.843581/0.905397。此前回调batch16与结束重评batch8不同，代码后来已统一。')
    if key in ['M047', 'M048']:
        note.append('reason在这里指Train理由标注转换成token证据监督；与Qwen生成完整解释的SFT不同。标注由助手制作，未独立人工复核。')
    if dataset_key in ['scope', 'relation', 'manual80', 'event200', 'local_manual', 'semantic64']:
        note.append('新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。')
    if member.get('checkpoint_note'):
        note.append('当前采用第22轮的历史预测CSV；未确认保留对应轮次的模型权重。原best_model.pth对应旧选模，不能当成第22轮权重；实际运行需要重新训练并按pair-aware选最佳。')
    return {
        **member, 'readable_name': title, 'dataset_key': dataset_key,
        'architecture_detail': architecture, 'training_detail': training,
        'selected_epoch': epoch, 'seed': cfg.get('seed'),
        'run_config_file': str(path), 'run_config_sha256': sha(path),
        'metadata_file': str(metadata_file) if metadata_file else None,
        'run_config': cfg, 'dataset': data[dataset_key], 'notes': note,
    }


def hyperparameters(detail):
    cfg = detail['run_config']
    if detail['id'] == 'M094':
        t = cfg['training']
        lora = cfg['lora']
        return f"lr={t['learning_rate']}；微批次{t['per_device_train_batch_size']}×累积{t['gradient_accumulation_steps']}；计划{t['num_train_epochs']}轮；bf16；r={lora['r']}、alpha={lora['lora_alpha']}、dropout={lora['lora_dropout']}；reason权重={cfg['reason_weight']}。"
    parts = [f"主学习率={cfg.get('learning_rate', '未记录')}", f"计划{cfg.get('epochs', '未记录')}轮", f"max_length={cfg.get('max_length', '未记录')}"]
    if cfg.get('lexical_learning_rate') is not None:
        parts.append(f"词汇分支学习率={cfg['lexical_learning_rate']}")
    if 'row_batch_size' in cfg:
        parts.append(f"Row/Pair batch={cfg['row_batch_size']}/{cfg.get('pair_batch_size', '无')}")
    elif 'group_batch_size' in cfg:
        parts.append(f"group batch={cfg['group_batch_size']}（组数）")
    elif 'unit_batch_size' in cfg:
        parts.append(f"文本对单元batch={cfg['unit_batch_size']}")
    elif 'train_batch_size' in cfg:
        parts.append(f"训练batch={cfg['train_batch_size']}")
    if 'row_weights' in cfg:
        parts.append(f"Row权重F/B/E/N={cfg['row_weights']}")
    if 'pair_weights' in cfg:
        parts.append(f"Pair权重FB/BF/EE={cfg['pair_weights']}")
    if 'consistency_weight' in cfg:
        parts.append(f"反向KL权重={cfg['consistency_weight']}")
    return '；'.join(parts) + '。'


def include_official_scheme(recipes, selected):
    """将严格官方数据搜索的C08作为S16纳入同一报告，原15组保持不变。"""
    source = OUTPUT.parent / 'run02_official_or_copy/recipes.json'
    if 'S16' in recipes or not source.is_file():
        return recipes, selected
    import numpy as np
    from evaluation_functions import score_files
    from experiments.deberta_lexical_mixed_pair.analyze_fusion import decode, metrics, log_softmax

    strict = json.loads(source.read_text(encoding='utf-8'))['C08']
    row = strict['recipe'].copy()
    code = 'S16'
    title = '仅官方3042行：分层融合，最高结构分'
    dev = pd.read_csv(ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv', keep_default_na=False).reset_index(drop=True)
    gold = dev.label.map(dict(zip(FULL, range(4)))).to_numpy()
    pairs = np.array([group.index.to_numpy() for _, group in dev.groupby('pair_id', sort=False) if len(group) == 2])
    frozen_members = json.loads((OUTPUT / 'members.json').read_text(encoding='utf-8'))
    ids = {member['id']: index for index, member in enumerate(frozen_members)}
    frozen = np.load(OUTPUT / 'aligned_scores.npz')['log_probability']
    experts = row['members'].split('+')
    q = np.mean([np.exp(log_softmax(frozen[ids[key], :, :3])) for key in experts], axis=0)
    g = np.exp(frozen[ids[row['gate']], :, 3])
    logits = np.log(np.maximum(np.column_stack([(1 - g[:, None]) * q, g]), np.finfo(float).tiny))
    table = dev.copy()
    for i, label in enumerate('FBEN'):
        table[f'logit_{label}'] = logits[:, i]
    records = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        prediction = decode(logits, pairs, method)
        values = metrics(gold, prediction, pairs)
        expected = row['independent_f1'] if method == 'independent' else row['pair_f1']
        assert abs(values['weighted_f1'] - expected) < 1e-12
        table[column] = np.array(list('FBEN'))[prediction]
        folder = OUTPUT / 'official_checks' / code / method
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'predictions.csv'
        pd.DataFrame({'instance_id': dev.instance_id, 'label': np.array(FULL)[prediction]}).to_csv(path, index=False)
        score = score_files(ROOT / 'data/dev/dico_nli_dev_track1_reference.csv', path, output_dir=folder)
        assert abs(score.weighted_f1 - expected) < 1e-12
        records.append({'method': method, **values})
    before = decode(frozen[ids['M059']], pairs, 'pair_aware')
    after = decode(logits, pairs, 'pair_aware')
    table.to_csv(OUTPUT / f'{code}_predictions.csv', index=False, encoding='utf-8-sig')
    changed = table.loc[before != after].copy()
    changed['baseline_prediction'] = np.array(list('FBEN'))[before[before != after]]
    changed.to_csv(OUTPUT / f'{code}_changes_vs_0938.csv', index=False, encoding='utf-8-sig')
    row.update({'code': code, 'title': title,
                'member_names': ' | '.join(member['name'] for member in strict['members']),
                'fixed_vs_0938': int(((before != gold) & (after == gold)).sum()),
                'broken_vs_0938': int(((before == gold) & (after != gold)).sum())})
    recipes[code] = {'title': title, 'mode': row['mode'], 'direction_members': experts,
                     'gate': row['gate'], 'members': strict['members'], 'same_scores_metrics': records,
                     'forced_singleton_N': False, 'gold_used_for_selection': True,
                     'gold_used_for_per_row_prediction_routing': False,
                     'search_source': str(source), 'source_recipe': 'C08',
                     'training_data_policy': 'All six members use only official3042 and original labels; no added texts or evidence annotations.'}
    selected = pd.concat([selected.reset_index(), pd.DataFrame([row])], ignore_index=True).set_index('code')
    selected.to_csv(OUTPUT / 'recommended_combinations.csv', encoding='utf-8-sig')
    (OUTPUT / 'selected_recipes.json').write_text(json.dumps(recipes, ensure_ascii=False, indent=2), encoding='utf-8')
    return recipes, selected


def expand_report():
    recipes = json.loads((OUTPUT / 'selected_recipes.json').read_text(encoding='utf-8'))
    selected = pd.read_csv(OUTPUT / 'recommended_combinations.csv', keep_default_na=False).set_index('code')
    recipes, selected = include_official_scheme(recipes, selected)
    members = {m['id']: m for recipe in recipes.values() for m in recipe['members']}
    assert set(members).issubset(DESCRIPTIONS)
    data = datasets()
    # 同一个成员在不同方案中可以采用不同选模规则；明细必须按方案保存。
    scheme_details = {code: {m['id']: member_details(m, data) for m in recipe['members']}
                      for code, recipe in recipes.items()}
    details = {key: member_details(member, data) for key, member in sorted(members.items())}
    # 原始35成员事实快照保留，用于复算历史基线。
    if not (OUTPUT / 'member_details.json').exists():
        (OUTPUT / 'member_details.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTPUT / 'member_details_by_scheme.json').write_text(json.dumps(scheme_details, ensure_ascii=False, indent=2), encoding='utf-8')
    backup = OUTPUT / 'RESULTS_before_details.md'
    if not backup.exists():
        shutil.copy2(OUTPUT / 'RESULTS.md', backup)

    lines = [f'# 现有模型多模型预测比较：{len(recipes)}组方案的完整说明', '',
             '整理日期：2026-10-26。这里把历史目录简称展开为实际模型、训练数据、训练方式和当前选模规则。成绩是保留的历史逐轮预测CSV的离线复算，不是本次重新训练后的成绩。', '',
             '**当前规则：** S03、S07、S08、S13、S14统一pair-aware选模后结构分下降，保留原选模；其余11组所有成员统一按Dev pair-aware F1选最佳，同分取最早轮。成员名单、数据和融合公式不变。该规则按这份Dev比较结果确定，属于开发集选择。',
             '**训练入口：** 目前仅S01、S11、S12、S14、S16有独立PY入口。S01、S11、S12、S16使用 `run02_pair_select`；S14保留原 `run01_retrain` 及混合选模。运行前检查outputs中的运行记录和实际模型保存目录：已有权重且数据、配方和最佳轮次匹配，就直接复用，支持跨RUN_NAME，不因改名而重复训练；只有CSV或配置而没有实际权重的成员才需训练。原选模指标不同、但保存轮次也恰好是当前指标最佳轮次时，可以直接复用。已缓存完整N头模型时也跳过它的前置底座训练。修改对下次启动生效，不改变已经运行中的进程。缺失模型的新权重保存到 `D:/model/SEAMEVAL/ensemble_selected/<RUN_NAME>/<成员编号>/`；实际使用的模型路径会写入本次结果清单。',
             '**权重与分数区别：** 下文历史权重路径不保证当前仍存在，也不能仅凭路径证明它对应当前轮次；尤其S12的M018改用第22轮预测，对应权重未确认保留，需要重新训练。',
             f"完整对照：{file_link('16组原选模与全pair-aware选模比较', OUTPUT / 'selection_comparison/RESULTS.md')}。当前推荐表和逐组明细已更新；原始97模型搜索表、旧成员快照及选模比较保留历史口径。", '',
             '## 先说明“一个模型”和“一组方案”', '',
             '一个成员是某次训练选定的一份权重，例如“DeBERTa＋对齐＋词汇、seed7”。一组方案让若干成员各自预测同一个Dev，再合并分数。成员可以共用同一个预训练骨干，只是数据、训练方式或随机种子不同。',
             '**各成员的训练数据是分别使用的，不是把下表所有行数加起来，再共同训练一个新模型。**',
             f'本报告的97份分数与同一官方660行Dev按instance_id对齐并去重；{len(recipes)}组涉及{len(members)}个不同成员。S16来自严格限定官方Train或复制样本的补充搜索，实际六个成员均只用官方3042行。所有方案均不强制单例判N，不按每行金标选择模型。',
             '独立F1表示不使用文件分组解码；它不保证每个成员内部只做一次编码，例如双向联合头内部会编码两个方向。', '',
             f'## {len(recipes)}组结果总表', '',
             '| 编号 | 融合方案 | 当前选模 | 实际模型数 | independent F1 | pair-aware F1 | N正确/106 | FBE正确/554 |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for code in recipes:
        row = selected.loc[code]
        policy = '全部pair-aware' if recipes[code].get('selection_policy') == 'pair_aware' else '保留原规则'
        lines.append(f"| [{code}](#{code.lower()}) | {row.title} | {policy} | {row.model_count} | {row.independent_f1:.6f} | {row.pair_f1:.6f} | {int(row.pair_N)} | {int(row.pair_FBE)} |")

    lines += ['', '## 三种融合方法', '',
              '普通平均：每个成员先对F/B/E/N四类做softmax，再等权平均，p=Σp_m/K。几何平均：对四类log-prob等权平均，再归一化，等价于归一化的概率几何均值。',
              '分层融合：指定一个模型提供g=p(N)；方向成员各自仅在F/B/E内归一化，等权平均得到q。最终p(N)=g，p(F/B/E)=(1-g)q。N门也计入实际模型数；若N门同时作为方向成员，只算一个实际模型。',
              '所有方法都先得到一份固定融合分数：independent取四类最大值；pair-aware对真实两行组比较FB/BF/EE，单例保留该融合模型的四类判断。因此每行表格的两项F1来自同一方案，不是拼两个不同方案的最高分。', '',
              '## 数据包对照：数字分别是什么意思', '',
              '下表是**训练文本行数**，不是独立源句数量，也不是所有成员数据的并集大小。类别顺序为F/B/E/N。', '',
              '| 数据编号 | 数据包 | 行数构成 | 实际总行数 | 类别数量F/B/E/N |',
              '| --- | --- | --- | ---: | --- |']
    for key, item in data.items():
        lines.append(f"| `{key}` | {item['name']} | {item['formula']} | {item['rows']} | {' / '.join(str(item['class_counts'][x]) for x in 'FBEN')} |")
    for key, item in data.items():
        lines += ['', f"### 数据包 `{key}`：{item['name']}", '', item['description'], '',
                  f"实际训练CSV：{file_link(Path(item['path']).name, item['path'])}"]
    lines += ['', '### 容易混淆的几个名字', '',
              '- `closure88`：实际是官方3042＋88行E标签改写，不应只看文件夹名就解释成已证实的传递闭包算法。新增88行与Dev的归一化无序文本对精确交集为0，但这不是来源独立性证明。',
              '- `local128_manual`：增加128行新造的局部对照；`local128_copy`：增加128行官方Train副本，两者都从3290行底座出发。',
              '- `event200`：从3242行scope底座加200行动作/事件例子，总计3442，不包含relation48。',
              '- `semantic64`：从3290行底座加64行语义对照，总计3354；64行来自四个家族，仍采用先Row后Pair训练。',
              '- “词汇版”指手工20维Lexical特征分支，不是WordNet词表，也不是额外NLI预训练。',
              '- DeBERTa的reason/evidence版训练token证据位置；Qwen的reason版训练生成解释，两者不是同一种辅助任务。', '',
              '## 逐组解释与完整成员', '']

    for code, recipe in recipes.items():
        row = selected.loc[code]
        current_details = scheme_details[code]
        lines += [f'<a id="{code.lower()}"></a>', f'### {code}：{recipe["title"]}', '',
                  f"本组共{int(row.model_count)}个实际模型。独立F1={row.independent_f1:.6f}，pair-aware F1={row.pair_f1:.6f}。N对{int(row.pair_N)}/106，配对F/B/E对{int(row.pair_FBE)}/554。", '']
        policy = '所有成员统一按Dev pair-aware F1选最佳，同分取最早轮。' if recipe.get('selection_policy') == 'pair_aware' else '保留各成员原选模规则，因为统一pair-aware会降低本组融合后的结构F1。'
        lines += ['**本组当前选模：** ' + policy, '']
        if code == 'S12':
            lines += ['本组原成绩为独立0.870621、结构0.941924。统一pair-aware后改用M018第22轮，独立降为0.863688、结构升为0.945094，因此当前不再同时达到独立≥0.87与结构≥0.94；标题已相应更名。', '']
        if recipe['mode'] in ['probability', 'log_probability']:
            action = '四类概率普通平均' if recipe['mode'] == 'probability' else '四类log-prob几何平均'
            lines.append(f"操作：下面{len(recipe['members'])}个成员各预测一次，做{action}；每个成员权重均为1/{len(recipe['members'])}。")
        else:
            gate = current_details[recipe['gate']]
            lines.append(f"操作：由{recipe['gate']}“{gate['readable_name']}”提供N概率；方向成员为{'、'.join(recipe['direction_members'])}，各自F/B/E条件概率按1/{len(recipe['direction_members'])}等权平均。N门不是先给硬标签，也不是单例统一判N。")
        if code == 'S16':
            lines += ['', '**本组的六个成员全部只用官方3042行训练文本和原始标签，没有scope、relation、closure、人工样本或反向N，也没有reason/evidence额外标注；不需要复制数据。**']
        lines += ['', '本组成员和各自使用的数据：', '',
                  '| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |',
                  '| --- | --- | --- | --- | ---: | ---: | ---: | --- |']
        for member in recipe['members']:
            d = current_details[member['id']]
            role = '四类共同投票'
            if recipe['mode'].startswith('hier_'):
                role = 'N概率门' if d['id'] == recipe['gate'] else 'F/B/E方向专家'
                if d['id'] == recipe['gate'] and d['id'] in recipe['direction_members']:
                    role = 'N门＋方向专家'
            lines.append(f"| {d['id']} | {d['readable_name']} | {role} | {d['dataset']['formula']} | {d['dataset']['rows']} | {d['seed']} | {d['selected_epoch']} | {d['independent_f1']:.6f} / {d['pair_f1']:.6f} |")
        for member in recipe['members']:
            d = current_details[member['id']]
            lines += ['', f"#### {d['id']}：{d['readable_name']}", '',
                      f"- **基础模型：** `{d['model_name']}`。",
                      f"- **结构：** {d['architecture_detail']}",
                      f"- **训练文本：** {d['dataset']['formula']}。{d['dataset']['description']}",
                      f"- **训练方式：** {d['training_detail']}",
                      f"- **主要参数：** seed={d['seed']}；{hyperparameters(d)}",
                      f"- **本次采用版本：** 第{d['selected_epoch']}轮；单模型独立F1={d['independent_f1']:.6f}、结构F1={d['pair_f1']:.6f}。",
                      f"- **本组当前选模规则：** `{d.get('selection', '未记录')}`。",
                      f"- **历史训练入口的选模规则：** `{d['run_config'].get('selection', d['run_config'].get('select_by', d.get('original_selection', '未记录')))}`；本组以当前规则为准。",
                      f"- **配置依据：** {file_link('该次运行保存的run_config.json', d['run_config_file'])}。",
                      f"- **训练数据文件：** {file_link(Path(d['dataset']['path']).name, d['dataset']['path'])}。",
                      f"- **本次参与融合的预测：** {file_link('predictions_with_scores.csv', d['prediction_file'])}。"]
            for weight in d['checkpoint_candidates_existing']:
                lines.append(f"- **历史权重路径（未确认当前仍存在）：** {file_link(Path(weight).name, weight)}。")
            for weight in d.get('historical_checkpoint_paths', []):
                lines.append(f"- **旧选模权重路径（不代表当前第22轮）：** {file_link(Path(weight).name, weight)}。")
            if code in ['S01', 'S11', 'S12', 'S14', 'S16']:
                run_name = 'run01_retrain' if code == 'S14' else 'run02_pair_select'
                filename = 'best_adapter/adapter_model.safetensors' if d['id'] == 'M094' else 'best_model.pth'
                planned = f"D:/model/SEAMEVAL/ensemble_selected/{run_name}/{d['id']}/{filename}"
                lines.append(f"- **本方案重训后保存位置：** `{planned}`；不是声称该文件已经训练生成。")
            for note in d['notes']:
                lines.append(f'- **需保留的说明：** {note}')
        lines += ['', f"本组结构预测相对原0.938：修好{int(row.fixed_vs_0938)}行、破坏{int(row.broken_vs_0938)}行。",
                  f"本组融合分数和逐行预测：{file_link(code + '_predictions.csv', OUTPUT / (code + '_predictions.csv'))}。", '']

    lines += ['## 怎样阅读结果，避免误解', '',
              '1. 97份成员分数不是97种预训练架构。绝大多数共用同一个NLI-base骨干，只是在数据、seed、分支或训练目标上有差别。',
              '2. S03是在本次Dev搜索里同时超过独立0.87和结构0.94的普通平均方案；S01优先结构分；S02优先独立分。它们不是同一个方案的三个不同选模时刻。',
              '3. S14虽然结构分最高，但独立分仍低，不能当成独立判断已达到0.965。',
              '4. 标题“官方3042行”只表示没有新增训练文本，并不表示没有额外证据标注、OOF加权或外部NLI初始化。S06和S13都含辅助证据标注模型。',
              '5. scope、relation、event、local和semantic增强有历史Dev引导开发记录。新样本与Dev的字符串交集为0，不等于模板、语义和研究过程完全独立；隐藏Test未提供，不能声称已证明与Test无交集。',
              '6. 这些高分是在重复使用的Dev上做大量组合搜索后选出的，存在选择偏差。1—3成员穷举，较大方案采用beam扩展，不保证全局最优。',
              '7. 本次没有加载真实模型重跑。历史权重路径不保证当前存在或对应所选轮次；可复核的是历史CSV的融合分数及官方评分。LoRA成员还需要对应基础模型和自定义任务头，不能把best_model.pth一律当完整独立模型直接加载。',
              '8. 单模型成绩一般不代表对集成没用；本次某些弱成员参与高分组合，只证明它们在这份Dev上的分数有互补，不能推断其单独训练方法普遍优越。', '',
              '## 文件索引', '',
              f"- {file_link(str(len(recipes)) + '组组合汇总', OUTPUT / 'recommended_combinations.csv')}",
              f"- {file_link('固定成员、原始分数来源与融合方式', OUTPUT / 'selected_recipes.json')}",
              f"- {file_link('当前逐方案成员明细和选模轮次', OUTPUT / 'member_details_by_scheme.json')}",
              f"- {file_link('原始35成员事实快照（历史基线）', OUTPUT / 'member_details.json')}",
              f"- {file_link('所有已计算组合', OUTPUT / 'all_combinations.csv')}",
              f"- {file_link('97份分数的来源清单', OUTPUT / 'members.csv')}",
              f"- {file_link('按成员数量比较', OUTPUT / 'best_per_model_count.csv')}",
              f"- {file_link('扩充前的原报告备份', backup)}", '']
    (OUTPUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return {'schemes': len(recipes), 'unique_members': len(details), 'datasets': len(data),
            'lines': len(lines), 'report': str(OUTPUT / 'RESULTS.md')}


if __name__ == '__main__':
    print(json.dumps(expand_report(), ensure_ascii=False, indent=2))
