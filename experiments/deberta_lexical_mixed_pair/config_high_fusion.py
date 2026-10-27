from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_NAME = 'fusion02_high_pair_models'
OUTPUT_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']

# 每个成员只用历史最佳checkpoint对应的预测，不重新选择epoch。
MEMBERS = {
    'relation_0938': {
        'prediction': PROJECT_ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged/epoch_2/predictions_with_scores.csv',
        'checkpoint': Path('D:/model/SEAMEVAL/deberta_manual_aug/run06_relation48_staged/best_model.pth'),
        'training_data': 'official 3042 + scope 200 + relation 48 = 3290; historical manual augmentation; no Lexical',
    },
    'scope_0935': {
        'prediction': PROJECT_ROOT / 'experiments/deberta_original_aug/outputs/run01_original_scope/epoch_3/predictions_with_scores.csv',
        'checkpoint': Path('D:/model/SEAMEVAL/deberta_original_aug/run01_original_scope/best_model.pth'),
        'training_data': 'official 3042 + scope 200 = 3242; historical manual augmentation; no Lexical',
    },
    'lexical_0930': {
        'prediction': PROJECT_ROOT / 'experiments/deberta_lexical/outputs/run02_branch_lr/lexical_seed_42/epoch_3/predictions_with_scores.csv',
        'checkpoint': PROJECT_ROOT / 'model/deberta_lexical/run02_branch_lr/lexical_seed_42/best_model.pth',
        'training_data': 'official 3042; Token Alignment + Lexical',
    },
    'original_0924': {
        'prediction': PROJECT_ROOT / 'experiments/qwen_lora/outputs/teacher_scores/run01/dev_teacher_scores.csv',
        'checkpoint': PROJECT_ROOT / 'model/deberta_nli_base_token_alignment_official_full_control_seed_42/epoch_7.pth',
        'training_data': 'official 3042; original Token Alignment; cached teacher export; checkpoint missing at configured path',
    },
}

# 只比较三个等权概率平均方案；不扫描权重、温度，不强制单例为N。
RECIPES = {
    'prob_top2': {'mode': 'probability', 'members': ['relation_0938', 'scope_0935']},
    'prob_high3': {'mode': 'probability', 'members': ['relation_0938', 'scope_0935', 'lexical_0930']},
    'prob_high4': {'mode': 'probability', 'members': ['relation_0938', 'scope_0935', 'lexical_0930', 'original_0924']},
}

BASELINE = 'relation_0938'
BOOTSTRAP_RECIPES = list(RECIPES)
BOOTSTRAP_ROUNDS = 5000
SEED = 42
