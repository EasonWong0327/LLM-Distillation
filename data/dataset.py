"""
数据集定义模块，包含用于蒸馏的数据集类
"""

import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class DistillationDataset(Dataset):
    """
    用于知识蒸馏的数据集类
    支持多种格式的指令微调数据处理
    """

    def __init__(self, dataset, tokenizer, max_length, domain_label=None):
        """
        初始化蒸馏数据集

        Args:
            dataset: 原始数据集
            tokenizer: 分词器
            max_length: 最大序列长度
            domain_label: 领域标签（可选，用于区分不同领域数据）
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = min(max_length, tokenizer.model_max_length)
        self.domain_label = domain_label
        
        # 设置特殊token
        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if not hasattr(tokenizer, 'unk_token') or tokenizer.unk_token is None:
            tokenizer.unk_token = "[UNK]"
            tokenizer.unk_token_id = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            # 获取文本
            sample = self.dataset[idx]
            prompt = next((sample[k] for k in ['ask', 'question', 'instruction', 'prompt'] if k in sample), None)
            response = next((sample[k] for k in ['answer', 'output', 'response'] if k in sample), None)
            
            if not prompt or not response or not isinstance(prompt, str) or not isinstance(response, str):
                return self._get_default_sample()
            
            # 编码
            full_text = f"{prompt}\n{response}"
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 处理token ID
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            
            # 处理超出词表范围的token
            if input_ids.max() >= self.tokenizer.vocab_size:
                input_ids = torch.where(
                    (input_ids >= self.tokenizer.vocab_size) & (input_ids != self.tokenizer.pad_token_id),
                    self.tokenizer.unk_token_id,
                    input_ids
                )
            
            # 创建标签
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            sample = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
            if self.domain_label is not None:
                sample["domain_label"] = torch.tensor(self.domain_label, dtype=torch.long)
            
            return sample
            
        except Exception as e:
            logger.error(f"处理样本 {idx} 时出错: {str(e)}")
            return self._get_default_sample()

    def _get_default_sample(self):
        """返回默认样本"""
        default_input = torch.zeros(self.max_length, dtype=torch.long)
        default_mask = torch.zeros(self.max_length, dtype=torch.long)
        default_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        
        sample = {
            "input_ids": default_input,
            "attention_mask": default_mask,
            "labels": default_labels
        }
        
        if self.domain_label is not None:
            sample["domain_label"] = torch.tensor(self.domain_label, dtype=torch.long)
            
        return sample