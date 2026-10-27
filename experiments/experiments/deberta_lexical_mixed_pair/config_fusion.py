from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_NAME = 'fusion01_existing_predictions'
OUTPUT_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']

# 只使用实际保存checkpoint对应的预测，不使用原mixed ep2等“仅存预测”的轮次。
MEMBERS = {
    'lexical_kl': {
        'prediction': PROJECT_ROOT / 'experiments/deberta_lexical_mixed_pair/outputs/run07_reversal_kl_seed42/epoch_2/predictions_with_scores.csv',
        'checkpoint': Path('D:/model/SEAMEVAL/deberta_lexical_mixed_pair/run07_reversal_kl_seed42/best_model.pth'),
        'training_data': 'official Train 3042',
    },
    'native_kl': {
        'prediction': PROJECT_ROOT / 'experiments/deberta_reversal_kl/outputs/run01_native_ce_kl_seed42/step_935/predictions_with_scores.csv',
        'checkpoint': Path('D:/model/SEAMEVAL/deberta_reversal_kl/run01_native_ce_kl_seed42/best_model.pth'),
        'training_data': 'official Train 3042',
    },
    'pooler': {
        'prediction': PROJECT_ROOT / 'experiments/deberta_lexical_pooler/outputs/run01_nli_pooler_seed42/epoch_10/predictions_with_scores.csv',
        'checkpoint': Path('D:/model/SEAMEVAL/deberta_lexical_pooler/run01_nli_pooler_seed42/best_model.pth'),
        'training_data': 'official Train 3042',
    },
    'oof': {
        'prediction': PROJECT_ROOT / 'experiments/deberta_lexical_mixed_pair/outputs/run06_oof_hard/epoch_7/predictions_with_scores.csv',
        'checkpoint': Path('D:/model/SEAMEVAL/deberta_lexical_mixed_pair/run06_oof_hard/best_model.pth'),
        'training_data': 'official Train 3042; OOF hard-example weighting',
    },
    'staged_lexical': {
        'prediction': PROJECT_ROOT / 'experiments/deberta_lexical/outputs/run02_branch_lr/lexical_seed_42/epoch_3/predictions_with_scores.csv',
        'checkpoint': PROJECT_ROOT / 'model/deberta_lexical/run02_branch_lr/lexical_seed_42/best_model.pth',
        'training_data': 'official Train 3042',
    },
    'qwen_r16': {
        'prediction': PROJECT_ROOT / 'experiments/qwen_lora/outputs/qwen3_0p6b_abc/run04_r16/C/predictions_with_scores.csv',
        'checkpoint': Path('D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_abc/run04_r16/C/best_adapter'),
        'training_data': 'Train C 3517 = official 3042 + historical reversed-N augmentation 475',
    },
}

# 固定等权方案。比较这些方案本身已经使用Dev，不宣称独立验证或复现作者最优集成。
RECIPES = {
    'prob_two': {'mode': 'probability', 'members': ['lexical_kl', 'native_kl']},
    'log_two': {'mode': 'log_probability', 'members': ['lexical_kl', 'native_kl']},
    'prob_clean3': {'mode': 'probability', 'members': ['lexical_kl', 'native_kl', 'pooler']},
    'log_clean3': {'mode': 'log_probability', 'members': ['lexical_kl', 'native_kl', 'pooler']},
    'prob_clean4': {'mode': 'probability', 'members': ['lexical_kl', 'native_kl', 'pooler', 'oof']},
    'log_clean4': {'mode': 'log_probability', 'members': ['lexical_kl', 'native_kl', 'pooler', 'oof']},
    'prob_diverse3': {'mode': 'probability', 'members': ['lexical_kl', 'native_kl', 'qwen_r16']},
    'log_diverse3': {'mode': 'log_probability', 'members': ['lexical_kl', 'native_kl', 'qwen_r16']},
    'hier_clean': {'mode': 'hierarchical',
                   'gates': ['lexical_kl', 'native_kl', 'pooler'],
                   'directions': ['lexical_kl', 'native_kl', 'pooler']},
    'hier_direction3': {'mode': 'hierarchical',
                        'gates': ['lexical_kl', 'native_kl', 'pooler'],
                        'directions': ['lexical_kl', 'staged_lexical', 'qwen_r16']},
    'hier_direction_kl': {'mode': 'hierarchical',
                          'gates': ['lexical_kl', 'native_kl', 'pooler'],
                          'directions': ['lexical_kl']},
    'hier_gate_diverse': {'mode': 'hierarchical',
                         'gates': ['lexical_kl', 'native_kl', 'qwen_r16'],
                         'directions': ['lexical_kl', 'staged_lexical', 'qwen_r16']},
    'hier_direction2': {'mode': 'hierarchical',
                        'gates': ['lexical_kl', 'native_kl', 'pooler'],
                        'directions': ['lexical_kl', 'staged_lexical']},
}

BASELINE = 'lexical_kl'
BOOTSTRAP_RECIPES = ['prob_diverse3', 'log_diverse3', 'hier_direction3', 'hier_direction2']
BOOTSTRAP_ROUNDS = 5000
SEED = 42
