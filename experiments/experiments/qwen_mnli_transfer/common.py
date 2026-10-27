import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

from experiments.qwen_mnli_transfer import config as cfg
from experiments.generative_nli import load_tokenizer


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )


def check_empty_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'目录已有结果，避免覆盖，请更改 config.py 的 RUN_NAME：{path}')


def check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('训练需要 CUDA；本脚本不会自动改为 CPU 长时间训练。')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('当前 GPU 不支持 bf16，请先检查 GPU / PyTorch 环境。')


def attach_adapter(base_model, adapter_path=None):
    if adapter_path is None:
        # 第一阶段才新建 LoRA。第二阶段必须接着使用已训练的这个 LoRA。
        lora_config = LoraConfig(
            r=cfg.LORA_R, lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT, target_modules='all-linear',
            bias='none', task_type='CAUSAL_LM',
        )
        return get_peft_model(base_model, lora_config)

    adapter_config = PeftConfig.from_pretrained(str(adapter_path), local_files_only=True)
    if (adapter_config.r != cfg.LORA_R or adapter_config.lora_alpha != cfg.LORA_ALPHA
            or adapter_config.lora_dropout != cfg.LORA_DROPOUT):
        raise ValueError('MNLI Adapter 的 r / alpha / dropout 与当前配置不一致。')
    # 默认是推理模式；这里必须设置 True，否则学生第二阶段无法更新 LoRA。
    return PeftModel.from_pretrained(
        base_model, str(adapter_path), is_trainable=True, local_files_only=True,
    )


def load_training_model(adapter_path=None):
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.MODEL_PATH), dtype='auto', local_files_only=True,
    )
    model.config.use_cache = False
    model = attach_adapter(model, adapter_path)
    model.print_trainable_parameters()
    return model


def check_mnli_adapter():
    manifest_path = cfg.MNLI_ADAPTER_DIR / 'transfer_config.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('请先运行 train_mnli.py，完成后才能继续 train_dico.py。')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['stage'] != 'mnli':
        raise ValueError('所选目录不是第一阶段的 MNLI Adapter。')
    if Path(manifest['model_path']).resolve() != cfg.MODEL_PATH.resolve():
        raise ValueError('两个阶段必须使用同一份基础模型，请检查 MODEL_PATH。')
    weight_path = cfg.MNLI_ADAPTER_DIR / 'adapter_model.safetensors'
    if not weight_path.is_file() or file_hash(weight_path) != manifest['adapter_sha256']:
        raise ValueError('MNLI Adapter 权重缺失或与清单不一致。')
    return manifest


def save_adapter(trainer, tokenizer, path, metadata):
    path = Path(path)
    trainer.save_model(str(path))
    tokenizer.save_pretrained(str(path))
    metadata = dict(metadata)
    metadata['model_path'] = str(cfg.MODEL_PATH)
    metadata['adapter_sha256'] = file_hash(path / 'adapter_model.safetensors')
    write_json(path / 'transfer_config.json', metadata)


def check_supervision(trainer, raw_dataset, tokenizer):
    # SFTTrainer 预处理后再逐条核查：Prompt 全部屏蔽，答案与 EOS 都受到监督。
    prepared = trainer.train_dataset
    if len(prepared) != len(raw_dataset):
        raise ValueError('SFTTrainer 丢弃了样本，检查长度和答案位置。')
    for raw, encoded in zip(raw_dataset, prepared):
        prompt_ids = tokenizer(raw['prompt'])['input_ids']
        full_ids = tokenizer(raw['prompt'] + raw['completion'] + tokenizer.eos_token)['input_ids']
        expected_labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if encoded['input_ids'] != full_ids or encoded['labels'] != expected_labels:
            raise ValueError('实际 SFT 输入或 completion mask 与预期不一致。')
    print('全部训练样本的 Prompt、答案及 EOS 监督位置检查通过。')
