from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN_NAME = 'run02_binding'
DATA_DIR = HERE / 'prepared' / RUN_NAME
OUTPUT_DIR = HERE / 'outputs' / RUN_NAME

# 本次只诊断已训练好的纯 Row CE 最佳模型，不启动训练。
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_row_ce_frozen/run01_freeze6')
AUDIT_DIR = ROOT / 'experiments/deberta_generalization_audit/outputs/row_ce_frozen_epoch8'
TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'

# 保持已有模型的推理协议。
BATCH_SIZE = 16
MAX_LENGTH = 128
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']

# 每个来源必须是 Train 里的 F 行。选择过程没有按新探针的模型答案筛样本。
# family、Train instance_id 后缀、中心词、修饰语、替换对象、属性展开、人物标记。
FAMILIES = [
    ('dog', '0000100__en-en__flipped', 'dog', 'white', 'cat', 'is white', False),
    ('boy', '0000142__en-en__flipped', 'boy', 'young', 'bird', 'is young', True),
    ('camera', '0000105__en-en__flipped', 'camera', 'large', 'table', 'is large', False),
    ('baby', '0000113__en-en__flipped', 'baby', 'small', 'camera', 'is small', True),
    ('girl', '0001512__en-en__original', 'girl', 'young', 'cat', 'is young', True),
    ('frisbee', '0001545__en-en__original', 'Frisbee', 'green', 'ball', 'is green', False),
    ('woman', '0000431__en-en__flipped', 'woman', 'young', 'man', 'is young', True),
    ('bicycle', '0000455__en-en__flipped', 'bicycle', 'pink', 'bus', 'is pink', False),
    ('bus', '0000456__en-en__flipped', 'bus', 'green', 'car', 'is green', False),
    ('cat', '0001809__en-en__original', 'cat', 'gray', 'dog', 'is gray', False),
    ('train', '0000472__en-en__flipped', 'train', 'blue', 'car', 'is blue', False),
    ('car', '0000482__en-en__flipped', 'car', 'silver', 'train', 'is silver', False),
    ('bird', '0000512__en-en__flipped', 'bird', 'yellow', 'fish', 'is yellow', False),
    ('man', '0001462__en-en__original', 'man', 'shirtless', 'woman', 'is shirtless', True),
    ('table', '0000558__en-en__flipped', 'table', 'wooden', 'chair', 'is wooden', False),
    ('motorcycle', '0000564__en-en__flipped', 'motorcycle', 'BMW', 'car', 'is made by BMW', False),
    ('stick', '0000592__en-en__flipped', 'stick', 'wooden', 'table', 'is wooden', False),
    ('helmet', '0001812__en-en__original', 'helmet', 'bike', 'glove', 'is designed for cycling', False),
    ('house', '0001936__en-en__original', 'house', 'gray', 'boat', 'is gray', False),
    ('sheep', '0001810__en-en__original', 'sheep', 'white', 'dog', 'is white', False),
]
