import sys
sys.path.append('./')
import argparse
from loguru import logger
import os
from os.path import join
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from sklearn import metrics
from component.collator import PretrainCollator, SFTDataCollator
from component.argument import CustomizedArguments
from component.template import template_dict
from component.dataset import (
    UnifiedSFTDataset,
    ChatGLM2SFTDataset,
    ChatGLM3SFTDataset,
    UnifiedDPODataset
)
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    Trainer,
    AddedToken
)
from modeling_classifier import BaichuanForSequenceClassification
from datasets import load_dataset, concatenate_datasets
import datasets
from itertools import chain
from tqdm import tqdm
import json
from trl import DPOTrainer, get_kbit_device_map
import torch.nn as nn


def setup_everything():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_args_file", type=str, default='train_args/pretrain/full/bloom-1b1-pretrain-full.json', help="")
    parser.add_argument("--train_args_file", type=str, default='train_args/sft/qlora/qwen-7b-sft-qlora.json', help="")
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    # 加载训练配置文件
    with open(train_args_file, "r") as f:
        train_args = json.load(f)
    # 保存训练参数到输出目录
    with open(join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)
    # 设置随机种子
    set_seed(training_args.seed)

    # check some setting
    assert args.task_type in ['pretrain', 'sft', 'dpo'], "task_type should be in ['pretrain', 'sft', 'dpo']"
    assert args.train_mode in ['full', 'lora', 'qlora'], "task_type should be in ['full', 'lora', 'qlora']"
    assert sum([training_args.fp16, training_args.bf16]) == 1, "only one of fp16 and bf16 can be True"

    return args, training_args

def load_tokenizer(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')
    return tokenizer


def load_model(args, training_args):
    """
    加载模型
    """
    assert training_args.bf16 or training_args.fp16, 'bf16 or fp16 should be True'
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    logger.info(f'Train model with {args.train_mode}')

    # init model kwargs
    # todo add flash attention
    # attn_implementation = None
    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    if args.train_mode == 'qlora':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None
    model_kwargs = dict(
        trust_remote_code=True,
        # attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = BaichuanForSequenceClassification.from_pretrained(args.model_name_or_path, **model_kwargs)

    # moe模型，需要考虑负载均衡的loss
    if 'output_router_logits' in model.config.to_dict():
        logger.info('set output_router_logits as True')
        model.config.output_router_logits = True
    # QLoRA: casts all the non int8 modules to full precision (fp32) for stability
    if args.train_mode == 'qlora' and args.task_type in ['pretrain', 'sft']:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    # LoRA: Enables the gradients for the input embeddings
    if args.train_mode == 'lora' and args.task_type in ['pretrain', 'sft']:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # init peft_config
    if args.train_mode == 'full':
        peft_config = None
    else:
        # 找到所有需要插入adapter的全连接层
        #target_modules = find_all_linear_names(model, args.train_mode)
        target_modules = ['W_pack']
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            #modules_to_save=['classifier'],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS", #CAUSAL_LM
        )

    # init peft model
    if args.train_mode in ['lora', 'qlora'] and args.task_type in ['pretrain', 'sft']:
        model = get_peft_model(model, peft_config)
        print(model)
        logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return {
        'model': model,
        'peft_config': peft_config
    }


def load_sft_dataset(args, tokenizer):
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[args.template_name]
    dataset_cls = UnifiedSFTDataset
    if 'chatglm2' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM2SFTDataset')
        dataset_cls = ChatGLM2SFTDataset
    elif 'chatglm3' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM3SFTDataset')
        dataset_cls = ChatGLM3SFTDataset
    else:
        logger.info('Loading data with UnifiedSFTDataset')
        
    train_dataset = dataset_cls(args.train_file, tokenizer, args.max_seq_length, template)
    eval_dataset = None
    if args.eval_file is not None:
        eval_dataset = dataset_cls(args.eval_file, tokenizer, args.max_seq_length, template)
    return train_dataset, eval_dataset

def compute_metrics(p):
    # p包含模型预测的结果和标签
    predictions, labels = p
    # 将预测结果转换为整数，因为sklearn期望的是整数标签
    predictions = predictions.argmax(-1)
    # 计算分类报告
    precision = metrics.precision_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions)
    # 返回你想要记录的指标
    return {
        "precision": precision,
        "f1-score": f1,
        "recall": recall
        # 可以添加更多指标...
    }


def init_components(args, training_args):
    """
    初始化各个组件
    """
    training_args.ddp_find_unused_parameters = False
    logger.info('Initializing components...')

    # 加载tokenizer
    tokenizer = load_tokenizer(args)
    # 加载model
    components = load_model(args, training_args)
    model = components['model']
    peft_config = components['peft_config']

    logger.info('Train model with sft task')
    train_dataset, eval_dataset = load_sft_dataset(args, tokenizer)
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer


def main():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = init_components(args, training_args)
    # 开始训练
    logger.info("*** starting training ***")
    train_result = trainer.train()
    # 保存最好的checkpoint
    final_save_path = join(training_args.output_dir)
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
