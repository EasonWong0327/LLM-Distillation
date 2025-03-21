#!/usr/bin/env python
"""
Qwen多阶段知识蒸馏主执行脚本
从Qwen-72B (teacher)到Qwen-7B-instruct (student)的三阶段知识蒸馏
- 阶段1: 基础蒸馏（学习通用知识）
- 阶段2: 领域自适应蒸馏（逐步增加医疗数据比例，启用混合损失）
- 阶段3: 特定领域微调（聚焦医疗领域性能优化）
"""

import os
import argparse
import logging
import torch

from config import get_config
from modeling.distiller import QwenDistiller
from data.data_manager import DataManager

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("distillation.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen 7B multi-step distiller")

    # 模型配置
    parser.add_argument("--teacher_model", type=str, default=None)
    parser.add_argument("--student_model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # 数据配置        A high quality CSV dataset
    parser.add_argument("--general_csv_path", type=str, default=None)
    parser.add_argument("--medical_csv_path", type=str, default=None)
    parser.add_argument("--eval_csv_path", type=str, default=None)

    # 训练配置
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_lora", action="store_true")

    # 阶段控制
    parser.add_argument("--start_stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--end_stage", type=int, default=3, choices=[1, 2, 3])
    # 起始检查点路径 暂时用不着
    parser.add_argument("--checkpoint_path", type=str, default=None)

    # 其他配置
    # 是否使用Weights & Biases进行实验跟踪
    parser.add_argument("--use_wandb", action="store_true")

    return parser.parse_args()


def update_config_with_args(config, args):
    """使用命令行参数更新配置"""
    if args.teacher_model:
        config.teacher_model_name_or_path = args.teacher_model
    if args.student_model:
        config.student_model_name_or_path = args.student_model
    if args.output_dir:
        config.output_dir = args.output_dir


    if args.general_csv_path:
        config.general_csv_path = args.general_csv_path
    if args.medical_csv_path:
        config.medical_csv_path = args.medical_csv_path
    if args.eval_csv_path:
        config.eval_csv_path = args.eval_csv_path

    if args.seed is not None:
        config.seed = args.seed
    if args.fp16:
        config.fp16 = True
    if args.use_lora:
        config.use_lora = True

    if args.use_wandb:
        config.use_wandb = True

    return config



def main():
    """主函数"""
    args = parse_args()

    config = get_config()
    config = update_config_with_args(config, args)

    config.fp16 = False
    logger.info("已禁用混合精度训练以提高稳定性")

    os.makedirs(config.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.error("没有GPU玩个O")
        exit()

    distiller = QwenDistiller(config)

    tokenizer = distiller.tokenizer

    data_manager = DataManager(config, tokenizer).load_datasets()

    if args.start_stage <= 1 <= args.end_stage:
        # 阶段1: 基础蒸馏
        logger.info("=" * 50)
        logger.info("开始阶段1: 基础蒸馏")
        stage1_checkpoint = distiller.train_stage(1, data_manager)
        prev_checkpoint = stage1_checkpoint
    else:
        prev_checkpoint = args.checkpoint_path

    if args.start_stage <= 2 <= args.end_stage:
        # 阶段2: 领域自适应蒸馏
        logger.info("=" * 50)
        logger.info("开始阶段2: 领域自适应蒸馏")
        stage2_checkpoint = distiller.train_stage(2, data_manager, prev_checkpoint=prev_checkpoint)
        prev_checkpoint = stage2_checkpoint

    if args.start_stage <= 3 <= args.end_stage:
        # 阶段3: 领域微调
        logger.info("=" * 50)
        logger.info("开始阶段3: 领域微调")
        stage3_checkpoint = distiller.train_stage(3, data_manager, prev_checkpoint=prev_checkpoint)
        final_checkpoint = stage3_checkpoint
    else:
        final_checkpoint = prev_checkpoint

    logger.info(f"蒸馏完成")
    logger.info(f"最终模型路径: {final_checkpoint}")


if __name__ == "__main__":
    main()

'''
启动命令：

python run_distillation.py \
    --teacher_model "/mnt/workspace/Qwen2.5-1.5B-Instruct" \
    --student_model "/mnt/workspace/Qwen2.5-0.5B-Instruct" \
    --output_dir "/mnt/workspace/health/output" \
    --general_csv_path  "/mnt/workspace/health/data/general_data" \
    --medical_csv_path "/mnt/workspace/health/data/health_data" \
    --eval_csv_path "/mnt/workspace/health/data/eval_data/eval_test.csv" \
    --seed 42 \
    --fp16 \
    --use_lora \
    --start_stage 1 \
    --end_stage 3 \
    --use_wandb

python run_distillation.py \
    --teacher_model "/mnt/workspace/Qwen2.5-1.5B-Instruct" \
    --student_model "/mnt/workspace/Qwen2.5-0.5B-Instruct" \
    --output_dir "/mnt/workspace/health/output" \
    --general_csv_path "/mnt/workspace/health/data/general_data" \
    --medical_csv_path "/mnt/workspace/health/data/health_data" \
    --eval_csv_path "/mnt/workspace/health/data/eval_data/eval_test.csv" \
    --seed 42 \
    --use_lora \
    --start_stage 1 \
    --end_stage 3

'''