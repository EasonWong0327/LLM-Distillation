"""
配置类，定义多阶段知识蒸馏的所有相关参数

前置知识：大模型超参基础知识、lora、能知道蒸馏和微调的区别和流程
"""


class DistillationConfig:
    def __init__(self):
        # 模型配置
        self.teacher_model_name_or_path = "Qwen/Qwen-72B"  # 教师模型路径
        self.student_model_name_or_path = "Qwen/Qwen-7B-instruct"  # 学生模型路径
        self.output_dir = "./qwen_distilled_model"  # 输出路径

        # 训练配置 - 全局
        self.evaluation_strategy = "steps"
        self.logging_steps = 100
        self.save_steps = 1000
        self.eval_steps = 500
        self.save_total_limit = 5
        self.gradient_accumulation_steps = 16
        self.max_grad_norm = 1.0
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.max_seq_length = 512
        self.fp16 = False
        self.bf16 = False  # 如果你的GPU支持bf16，可以启用
        self.seed = 42
        self.use_wandb = True

        # LoRA配置
        self.use_lora = True
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        # 数据集配置
        self.general_dataset_name = "tatsu-lab/alpaca"  # 通用知识数据集
        self.medical_dataset_name = "medical_instructions"  # 假设的医疗指令数据集
        self.medical_dataset_path = "./medical_data"  # 医疗数据集本地路径
        self.eval_csv_path = None  # 这里默认设为空

        # 阶段1: 基础蒸馏配置
        self.stage1 = {
            "name": "base_distillation",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,  # 减小批次大小
            "per_device_eval_batch_size": 2,
            "learning_rate": 1e-5,  # 较低的学习率
            "temperature": 1.0,  # 较低的温度
            "alpha_kl": 0.3,  # 降低KL散度权重
            "alpha_ce": 0.7,  # 增加交叉熵权重
            "use_intermediate_distillation": False,
        }

        # 阶段2: 领域自适应蒸馏配置
        self.stage2 = {
            "name": "domain_adaptation",
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "learning_rate": 3e-5,
            "temperature": 2.0,  # 中等温度
            "alpha_kl": 0.7,  # KL散度损失权重
            "alpha_ce": 0.3,  # 交叉熵损失权重
            "use_intermediate_distillation": True,  # 启用中间层蒸馏
            "feature_distillation_weight": 0.2,  # 特征蒸馏权重
            "attention_distillation_weight": 0.2,  # 注意力蒸馏权重
            "medical_data_ratio_start": 0.3,  # 起始医疗数据比例
            "medical_data_ratio_end": 0.7,  # 结束医疗数据比例
            "data_ratio_schedule": "linear_increase",  # 医疗数据比例增长策略
            # add：梯度平衡策略
            "use_gradient_balancing": True,
            # add：对比学习
            "use_contrastive_learning": True,
            "contrastive_loss_weight": 0.1,
        }

        # 阶段3: 微调配置
        self.stage3 = {
            "name": "fine_tuning",
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "learning_rate": 1e-5,
            "temperature": 1.0,  # 低温度，更聚焦于硬标签
            "alpha_kl": 0.3,  # KL散度损失权重低
            "alpha_ce": 0.7,  # 交叉熵损失权重高
            "use_intermediate_distillation": False,  # 不使用中间层蒸馏
            "use_only_medical_data": True,  # 仅使用医疗数据
            # 新增：特定领域评估驱动采样
            "use_eval_guided_sampling": True,
        }

        # 医疗评估配置
        self.medical_evaluation = {
            "use_external_eval": True,  # 使用外部医疗评估
            "medical_qa_dataset": "medical_qa_benchmark",  # 医疗问答基准
            "medical_metrics": ["accuracy", "precision", "recall", "f1"],  # 评估指标
        }


def get_config():
    """获取默认配置"""
    return DistillationConfig()