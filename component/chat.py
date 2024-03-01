from transformers import AutoTokenizer, AutoConfig, AddedToken
from transformers.generation.utils import GenerationConfig
import torch
from loguru import logger
import copy
import json
import time
import os

from utils import ModelUtils
from template import template_dict


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


def main():
    # 使用合并后的模型进行推理
    # model_name_or_path = 'Qwen/Qwen-7B-Chat'
    # template_name = 'qwen'
    #  adapter_name_or_path = None

    model_name_or_path = '/DATA/jupyter/share/LLM_NBS/Baichuan2-13B-Chat'
    template_name = 'baichuan2'
    file = './data/v1_prompt_label/test.json'
    file = '/DATA/jupyter/personal/ChatGLM3/finetune_demo/data/v1_prompt_label/test.json'
    
    template = template_dict[template_name]
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = True
    for step in range(300, 901, 100):
        torch.cuda.empty_cache()
        adapter_name_or_path = './output/baichuan2_13b_b1_acc16_epoch3_rk64_a16_lr2e4/checkpoint-%d' % step
        save_file = './data/v1_prompt_label/13b-checkpoint-%d.json' % step
        # 加载模型
        logger.info(f'Loading model from: {model_name_or_path}')
        logger.info(f'adapter_name_or_path: {adapter_name_or_path}')
#         try:
        model = ModelUtils.load_model(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            adapter_name_or_path=adapter_name_or_path
        ).eval()
        tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
        generation_config = GenerationConfig.from_pretrained('/DATA/jupyter/personal/Firefly/output/')
        generation_config.eos_token_id = tokenizer.eos_token_id
        logger.info(generation_config)
        wf = open(save_file, 'w')
        st = time.time()
        with open(file, 'r') as rf:
            for cnt, line in enumerate(rf):
                if not line:
                    continue
                
                sample = json.loads(line.strip('\n'))
                prompt = [sample['conversations'][0]]
                response = model.chat(tokenizer, prompt, generation_config=generation_config)
                sample['ans'] = response
                logger.info(f'{cnt} {response}')
                wf.write(json.dumps(sample, ensure_ascii=False) + '\n')
        wf.close()
#         except Exception as e:
#             logger.error(e)
#             continue

if __name__ == '__main__':
    main()

