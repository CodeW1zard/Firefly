from transformers import AutoTokenizer, AutoConfig, AddedToken, HfArgumentParser, BitsAndBytesConfig
import torch
from loguru import logger
import copy
import json
import time
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
from component.collator import EvalDataCollator
from component.template import template_dict
from component.dataset import UnifiedSFTDataset
from modeling_classifier import BaichuanForSequenceClassification
from component.argument import InferenceArguments
from sklearn import metrics
from peft import PeftModel

def build_prompt_chatglm3(tokenizer, query, history, system=None):
    history.append({"role": 'user', 'message': query})
    # system
    input_ids = tokenizer.get_prefix_tokens() + \
                [tokenizer.get_command(f"<|system|>")] + \
                tokenizer.encode(system, add_special_tokens=False)
    # convs
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            tokens = [tokenizer.get_command(f"<|user|>")] + \
                     tokenizer.encode(message, add_special_tokens=False) + \
                     [tokenizer.get_command(f"<|assistant|>")]
        else:
            tokens = tokenizer.encode(message, add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids += tokens

    return input_ids

def build_prompt(tokenizer, template, query, history, system=None):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system

    if template_name == 'chatglm2':
        prompt = tokenizer.build_prompt(query, history)
        input_ids = tokenizer.encode(prompt)
    elif template_name == 'chatglm3':
        input_ids = build_prompt_chatglm3(tokenizer, query, history, system)
    else:
        history.append({"role": 'user', 'message': query})
        input_ids = []

        # setting system information
        if system_format is not None:
            # system信息不为空
            if system is not None:
                system_text = system_format.format(content=system)
                input_ids = tokenizer.encode(system_text, add_special_tokens=False)
        # concat conversation
        for item in history:
            role, message = item['role'], item['message']
            if role == 'user':
                message = user_format.format(content=message, stop_token=tokenizer.eos_token)
            else:
                message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
            tokens = tokenizer.encode(message, add_special_tokens=False)
            input_ids += tokens
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


def load_tokenizer(model_name_or_path):
    # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
        # llama不支持fast
        # use_fast=False if config.model_type == 'llama' else True
    )

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    return tokenizer

def load_dataset(args, tokenizer):
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
        
    eval_dataset = None
    if args.eval_file is not None:
        eval_dataset = dataset_cls(args.eval_file, tokenizer, args.max_seq_length, template)
    return eval_dataset

def load_model(model_name_or_path, load_in_4bit=False, adapter_name_or_path=None):
    # 是否使用4bit量化进行推理
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None

    # 加载base model
    model = BaichuanForSequenceClassification.from_pretrained(
        model_name_or_path,
#             load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto',
        quantization_config=quantization_config
    )
    #model.save_pretrained(model_name_or_path+'_classifier')

    # 加载adapter
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)
    logger.info(model)

    return model


def main():
    # 使用合并后的模型进行推理
    # model_name_or_path = 'Qwen/Qwen-7B-Chat'
    # template_name = 'qwen'
    #  adapter_name_or_path = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--args_file", type=str, default='train_args/sft/qlora/qwen-7b-sft-qlora.json', help="")
    args = parser.parse_args()
    parser = HfArgumentParser((InferenceArguments))
    args = parser.parse_json_file(json_file=args.args_file)
    args = list(args)[0]
    template = template_dict[args.template_name]
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = True
    tokenizer = load_tokenizer(args.model_name_or_path)
    eval_dataset = load_dataset(args, tokenizer)
    data_collator = EvalDataCollator(tokenizer, args.max_seq_length)

    
    for step in range(100, 601, 100):
        torch.cuda.empty_cache()
        adapter_name_or_path = os.path.join(args.adapter_name_or_path, 'checkpoint-%d' % step)
        save_file = os.path.join(adapter_name_or_path, 'eval.json')
        # 加载模型
        logger.info(f'Loading model from: {args.model_name_or_path}')
        logger.info(f'adapter_name_or_path: {adapter_name_or_path}')
#         try:
        model = load_model(
            args.model_name_or_path,
            load_in_4bit=load_in_4bit,
            adapter_name_or_path=adapter_name_or_path
        ).eval()
        st = model.base_model.classifier.state_dict()
        
       
        st = time.time()
        dataloader = DataLoader(eval_dataset, 
           batch_size=args.per_device_eval_batch_size, 
           collate_fn=data_collator
           )
        losses = []
        probs = []
        labels =[]
        wf = open(save_file, 'w')
        cnt = 0
        for marks, batch in dataloader:
            logger.info(f'{adapter_name_or_path} {cnt}')
            cnt += 1
            for k, v in batch.items():
                batch[k] = v.to(model.device)
            with torch.no_grad():
                out = model(**batch)
#             print(marks, out)
            logits = out.logits
#             losses.append(out.loss.detach().cpu().numpy().tolist())
            logits = logits.detach().cpu().float()
            prob = torch.nn.functional.softmax(logits, dim=1)[:, 1].numpy()
            probs.append(prob)
            labels.append(batch['labels'].cpu().detach().numpy())
            for i, mark in enumerate(marks):
                mark['score'] = '%.4f' % prob[i]
                mark = json.dumps(mark, ensure_ascii=False)
                wf.write(mark +'\n')
        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
#         losses = np.array(losses)
        logger.info(metrics.classification_report(labels, probs>0.5))
#         logger.info(f'loss=%.4f' % losses.mean())
        wf.close()
#         except Exception as e:
#             logger.error(e)
#             continue

if __name__ == '__main__':
    main()

