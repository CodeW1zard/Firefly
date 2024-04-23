from typing import Any, Dict, List
import torch
from loguru import logger


class SFTDataCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出batch中的最大长度
        marks, batch = list(zip(*batch))
        texts = [x['text'] for x in batch]
        labels = [x.get('label') for x in batch]
        labels = torch.tensor(labels).long() if len(labels)>0 and labels[0] is not None else None
        encode = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding='longest', truncation=True, max_length=self.max_seq_length)
        encode['prompt_lengths'] = (encode.input_ids != self.pad_token_id).sum(axis=1)
        encode['labels'] = labels if len(labels)>0 and labels[0] is not None else None
        return encode

class EvalDataCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出batch中的最大长度
        
        marks, batch = list(zip(*batch))
        texts = [x['text'] for x in batch]
        labels = [x.get('label') for x in batch]
        labels = torch.tensor(labels).long() if len(labels)>0 and labels[0] is not None else None
        encode = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding='longest', truncation=True, max_length=self.max_seq_length)
        encode['prompt_lengths'] = (encode.input_ids != self.pad_token_id).sum(axis=1)
        encode['labels'] = labels if len(labels)>0 and labels[0] is not None else None
        return marks, encode
    
class PretrainCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [x['input_ids'] for x in batch if x['input_ids'] is not None]
        # 找出batch中的最大长度
        lengths = [len(x) for x in batch]
        # 取出batch中的最大长度，如果超过max_seq_length，则取max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, labels_batch = [], [], []
        for x in batch:
            input_ids = x
            attention_mask = [1] * len(input_ids)

            padding_len = batch_max_len - len(input_ids)
            # padding
            labels = input_ids + [-100] * padding_len
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_mask_batch.append(attention_mask)

        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels_batch
        }
        return inputs

