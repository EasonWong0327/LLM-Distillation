"""
数据管理器模块，负责加载、处理和组织训练所需的各类数据集
csv格式
"""

import os
import glob
import random
import logging
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, ConcatDataset, WeightedRandomSampler

from data.dataset import DistillationDataset
from data.data_utils import create_mock_medical_dataset

logger = logging.getLogger(__name__)


class DataManager:
    """
    数据管理器：处理多阶段蒸馏的数据加载、混合和批处理
    """

    def __init__(self, config, tokenizer):
        """
        初始化数据管理器

        Args:
            config: 配置对象，包含数据相关参数
            tokenizer: 分词器
        """
        self.config = config
        self.tokenizer = tokenizer
        self.general_dataset = None
        self.medical_dataset = None
        self.eval_datasets = {}

    def load_datasets(self):
        """加载所有数据集"""
        self.general_dataset = self._load_dataset(self.config.general_csv_path, "通用知识", 0)
        self.medical_dataset = self._load_dataset(self.config.medical_csv_path, "医疗领域", 1)
        self._prepare_eval_datasets()
        return self

    def _load_dataset(self, csv_path, dataset_name, domain_label):
        """加载单个数据集"""
        if not csv_path:
            logger.error(f"未提供{dataset_name}数据集路径")
            return None

        try:
            # 读取CSV文件
            if os.path.isdir(csv_path):
                csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
                if not csv_files:
                    return None
                data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            else:
                data = pd.read_csv(csv_path)

            # 统一列名
            column_mapping = {
                'ask': ['ask', 'question', 'instruction', 'prompt'],
                'answer': ['answer', 'output', 'response']
            }
            
            for target, alternatives in column_mapping.items():
                for alt in alternatives:
                    if alt in data.columns:
                        data = data.rename(columns={alt: target})
                        break

            # 数据清理
            data = data.dropna(subset=['ask', 'answer'])
            data = data[
                (data['ask'].str.len() > 0) & 
                (data['answer'].str.len() > 0) &
                (data['ask'].str.len() <= self.config.max_seq_length) &
                (data['answer'].str.len() <= self.config.max_seq_length)
            ]

            if len(data) == 0:
                return None

            # 创建数据集
            dataset_dict = {col: data[col].tolist() for col in data.columns}
            dataset = Dataset.from_dict(dataset_dict)

            # 拆分训练集和验证集
            train_indices, val_indices = train_test_split(
                range(len(dataset)),
                test_size=0.1,
                random_state=self.config.seed
            )

            # 创建训练集
            train_dataset = DistillationDataset(
                Subset(dataset, train_indices),
                self.tokenizer,
                self.config.max_seq_length,
                domain_label=domain_label
            )

            # 创建验证集
            val_dataset = DistillationDataset(
                Subset(dataset, val_indices),
                self.tokenizer,
                self.config.max_seq_length,
                domain_label=domain_label
            )

            # 保存验证集
            self.eval_datasets[dataset_name] = val_dataset

            logger.info(f"加载了 {len(data)} 条{dataset_name}样本")
            return train_dataset

        except Exception as e:
            logger.error(f"加载{dataset_name}数据集失败: {str(e)}")
            return None

    def _prepare_eval_datasets(self):
        """准备评估数据集"""
        if not hasattr(self.config, 'eval_csv_path') or not self.config.eval_csv_path:
            return

        try:
            data = pd.read_csv(self.config.eval_csv_path)
            dataset = Dataset.from_dict({col: data[col].tolist() for col in data.columns})
            
            self.eval_datasets["eval"] = DistillationDataset(
                dataset,
                self.tokenizer,
                self.config.max_seq_length,
                domain_label=1
            )
        except Exception as e:
            logger.error(f"加载评估数据集失败: {str(e)}")

    def get_stage_datasets(self, stage):
        """获取训练阶段的数据集"""
        if stage == 1:  # 基础蒸馏
            return self.general_dataset, {"general": self.eval_datasets.get("general")}
        elif stage == 2:  # 领域自适应
            combined_dataset = self._create_mixed_dataset(
                self.config.stage2.get("medical_data_ratio_start", 0.5)
            )
            return combined_dataset, self.eval_datasets
        elif stage == 3:  # 领域微调
            if self.config.stage3.get("use_only_medical_data", True):
                return self.medical_dataset, {"medical": self.eval_datasets.get("medical")}
            return self._create_mixed_dataset(0.9), self.eval_datasets
        raise ValueError(f"无效的训练阶段: {stage}")

    def _create_mixed_dataset(self, medical_ratio):
        """创建混合数据集"""
        if not self.general_dataset or not self.medical_dataset:
            return None

        if medical_ratio <= 0:
            return self.general_dataset
        if medical_ratio >= 1:
            return self.medical_dataset

        return ConcatDataset([
            self.general_dataset,
            self.medical_dataset
        ])

    def get_eval_guided_dataset(self, eval_results):
        """
        基于评估结果创建引导采样的数据集（适用于阶段3）

        Args:
            eval_results: 评估结果，包含每个样本的表现

        Returns:
            采样后的数据集
        """
        if not self.config.stage3["use_eval_guided_sampling"]:
            return self.medical_dataset

        # 这里实现基于评估结果的采样策略
        # 以样本的错误率为权重进行采样
        weights = [max(0.1, 1.0 - score) for score in eval_results["sample_scores"]]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        logger.info("启用评估驱动的数据采样")
        return self.medical_dataset, sampler

    def get_dataloader(self, dataset, batch_size, shuffle=True, sampler=None):
        """
        创建数据加载器

        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱数据
            sampler: 采样器（可选）

        Returns:
            数据加载器
        """
        if sampler is not None:
            shuffle = False  # 使用采样器时不能同时打乱

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=True,
            num_workers=2
        )


