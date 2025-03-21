"""
知识蒸馏器核心实现
管理教师模型和学生模型，实现多阶段知识蒸馏流程

0302 add : 没资源跑72B，就改成最小的了
0321 add :增加大量注释
"""

import os
import logging
import torch
import numpy as np
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from tqdm import tqdm
import wandb

from modeling.losses import DistillationLoss, GradientBalancingLoss

logger = logging.getLogger(__name__)


class QwenDistiller:
    """
    Qwen模型知识蒸馏器
    支持三阶段知识蒸馏：基础蒸馏、领域自适应蒸馏和领域微调
    """

    def __init__(self, config):
        """
        初始化蒸馏器

        Args:
            config: 配置对象
        """
        self.config = config
        self.set_seed(config.seed)

        self.device = torch.device("cuda")
        print('!!!!!!!!!!!!',self.device)
        logger.info(f"使用设备: {self.device}")

        # 初始化日志跟踪
        self.initialize_tracking()

        # 加载tokenizer
        logger.info("正在加载tokenizer...")
        self.tokenizer = self.load_tokenizer()

        print(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
        print(f"Tokenizer special tokens: {self.tokenizer.special_tokens_map}")
        print(f"Tokenizer model max length: {self.tokenizer.model_max_length}")

        # 加载教师模型
        logger.info("正在加载教师模型...")
        self.teacher_model = self.load_teacher_model()

        # 学生模型将在各阶段初始化
        self.student_model = None

        # 特征投影层（用于中间层蒸馏时处理维度不匹配）
        self.feature_projections = {}

        # 保存配置
        os.makedirs(config.output_dir, exist_ok=True)

    def set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        set_seed(seed)

    def initialize_tracking(self):
        """初始化实验跟踪"""
        if self.config.use_wandb:
            wandb.init(project="qwen_multi_stage_distillation", config=vars(self.config))

    def load_tokenizer(self):
        """加载tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_model_name_or_path,  # 学生和教师使用相同的tokenizer
            use_fast=True,
            trust_remote_code=True,
        )
        # 确保有填充token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def load_teacher_model(self):
        """加载教师模型"""
        teacher_model = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model_name_or_path,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            trust_remote_code=True,
            device_map="auto",  # 自动分配到可用设备
            cache_dir="./model_cache",
            output_hidden_states=True,
            output_attentions=True if "stage2" in self.config.__dict__ and
                                      self.config.stage2.get("attention_distillation_weight", 0) > 0 else None
        )
        teacher_model = teacher_model.to(self.device)
        # 设为评估模式，不更新参数
        teacher_model.eval()
        return teacher_model

    def load_student_model(self, stage, from_checkpoint=None):
        """
        加载学生模型

        Args:
            stage: 训练阶段
            from_checkpoint: 检查点路径（如果有）

        Returns:
            学生模型
        """
        # 获取当前阶段配置
        stage_config = getattr(self.config, f"stage{stage}")

        if from_checkpoint and os.path.exists(from_checkpoint):
            logger.info(f"从检查点加载学生模型: {from_checkpoint}")
            if self.config.use_lora:
                # 加载基础模型
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.student_model_name_or_path,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    trust_remote_code=True,
                    cache_dir="./model_cache",
                    output_hidden_states=True,
                    output_attentions=True if stage == 2 and
                                              stage_config.get("attention_distillation_weight", 0) > 0 else None
                )
                # 加载LoRA权重
                student_model = PeftModel.from_pretrained(base_model, from_checkpoint)
            else:
                # 直接加载完整模型
                student_model = AutoModelForCausalLM.from_pretrained(
                    from_checkpoint,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    trust_remote_code=True,
                    output_hidden_states=True,
                    output_attentions=True if stage == 2 and
                                              stage_config.get("attention_distillation_weight", 0) > 0 else None
                )
        else:
            logger.info(f"从预训练权重加载学生模型")
            student_model = AutoModelForCausalLM.from_pretrained(
                self.config.student_model_name_or_path,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                trust_remote_code=True,
                cache_dir="./model_cache",
                output_hidden_states=True,
                output_attentions=True if stage == 2 and
                                          stage_config.get("attention_distillation_weight", 0) > 0 else None
            )

            # 应用LoRA配置
            if self.config.use_lora:
                logger.info("应用LoRA配置")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.lora_target_modules,
                )
                student_model = get_peft_model(student_model, peft_config)

                if from_checkpoint:
                    logger.info(f"加载LoRA检查点: {from_checkpoint}")
                    student_model.load_state_dict(torch.load(from_checkpoint, map_location=self.device))

        # 初始化中间层蒸馏的特征投影
        if stage == 2 and stage_config.get("use_intermediate_distillation", False):
            self._init_feature_projections(student_model)

        #if not self.config.use_lora:  # LoRA已经在内部处理设备分配
        student_model = student_model.to(self.device)

        # 设为训练模式
        student_model.train()

        return student_model

    def _init_feature_projections(self, student_model):
        """
        初始化特征投影层
        用于中间层蒸馏时处理学生模型和教师模型的隐藏层维度差异

        Args:
            student_model: 学生模型
        """
        # 如果不需要特征蒸馏，跳过
        if not self.config.stage2.get("feature_distillation_weight", 0) > 0:
            return

        # 检查是否已经初始化
        if self.feature_projections:
            return

        logger.info("初始化特征投影层...")

        # 获取学生和教师模型的隐藏层大小
        try:
            # 准备一个简单的输入来获取模型隐藏层尺寸
            sample_input = torch.ones((1, 10), dtype=torch.long).to(self.device)

            with torch.no_grad():
                # 获取学生模型隐藏层尺寸
                student_outputs = student_model(sample_input, output_hidden_states=True)
                student_hidden_states = student_outputs.hidden_states

                # 获取教师模型隐藏层尺寸
                teacher_outputs = self.teacher_model(sample_input, output_hidden_states=True)
                teacher_hidden_states = teacher_outputs.hidden_states

            # 为每一层创建投影层（如果尺寸不同）
            for i, (s_hidden, t_hidden) in enumerate(zip(student_hidden_states, teacher_hidden_states)):
                s_dim = s_hidden.size(-1)
                t_dim = t_hidden.size(-1)

                if s_dim != t_dim:
                    # 创建从学生维度到教师维度的线性投影
                    projection = torch.nn.Linear(s_dim, t_dim, bias=False).to(self.device)
                    # 使用正交初始化
                    torch.nn.init.orthogonal_(projection.weight)
                    self.feature_projections[i] = projection
                    logger.info(f"创建第{i}层特征投影: {s_dim} -> {t_dim}")

        except Exception as e:
            logger.warning(f"初始化特征投影失败: {str(e)}")
            # 出错时清空投影层字典
            self.feature_projections = {}

    def setup_optimizer_and_scheduler(self, student_model, num_training_steps, stage):
        """
        设置优化器和学习率调度器

        Args:
            student_model: 学生模型
            num_training_steps: 总训练步数
            stage: 当前训练阶段

        Returns:
            optimizer: 优化器
            scheduler: 学习率调度器
        """
        # 获取当前阶段配置
        stage_config = getattr(self.config, f"stage{stage}")

        # 设置优化器
        optimizer = AdamW(
            student_model.parameters(),
            lr=stage_config["learning_rate"],
            weight_decay=self.config.weight_decay,
        )

        # 设置学习率调度器
        warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, scheduler

    def save_checkpoint(self, student_model, filepath):
        """
        保存模型检查点

        Args:
            student_model: 学生模型
            filepath: 保存路径
        """
        logger.info(f"保存模型检查点: {filepath}")
        os.makedirs(filepath, exist_ok=True)

        # 根据是否使用LoRA选择不同的保存方式
        if self.config.use_lora:
            student_model.save_pretrained(filepath)
        else:
            student_model.save_pretrained(filepath)

        # 保存tokenizer
        self.tokenizer.save_pretrained(filepath)

    def train_step(self, batch, stage):
        """单步训练"""
        try:
            # 确保输入数据格式正确
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            
            # 添加批次维度（如果需要）
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            # 检查数据有效性
            if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
                logger.warning("检测到无效的输入数据")
                return torch.tensor(0.0, device=self.device)
                
            # 获取教师模型输出
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True
                )
            
            # 获取学生模型输出
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                labels=labels
            )
            
            # 计算损失
            stage_config = getattr(self.config, f"stage{stage}")
            loss = self.compute_loss(
                student_outputs,
                teacher_outputs,
                labels,
                stage_config
            )
            
            return loss
            
        except Exception as e:
            logger.error(f"训练步骤出错: {str(e)}")
            return torch.tensor(0.0, device=self.device)
            
    def compute_loss(self, student_outputs, teacher_outputs, labels, config):
        """计算损失"""
        losses = {}
        
        # 交叉熵损失
        if hasattr(student_outputs, "loss"):
            losses["ce_loss"] = student_outputs.loss
        
        # KL散度损失
        if config.get("kl_loss_weight", 0) > 0:
            kl_loss = self.compute_kl_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                config.get("temperature", 1.0)
            )
            losses["kl_loss"] = kl_loss * config["kl_loss_weight"]
        
        # 特征损失
        if config.get("feature_loss_weight", 0) > 0:
            feature_loss = self.compute_feature_loss(
                student_outputs.hidden_states,
                teacher_outputs.hidden_states
            )
            losses["feature_loss"] = feature_loss * config["feature_loss_weight"]
        
        # 注意力损失
        if config.get("attention_loss_weight", 0) > 0:
            attention_loss = self.compute_attention_loss(
                student_outputs.attentions,
                teacher_outputs.attentions
            )
            losses["attention_loss"] = attention_loss * config["attention_loss_weight"]
        
        # 计算总损失
        total_loss = sum(losses.values())
        
        return total_loss

    def evaluate(self, student_model, eval_dataloader, loss_fn, stage):
        """
        评估模型

        Args:
            student_model: 学生模型
            eval_dataloader: 评估数据加载器
            loss_fn: 损失函数
            stage: 当前训练阶段

        Returns:
            eval_results: 评估结果字典
        """
        logger.info("开始评估...")
        student_model.eval()
        total_loss = 0
        loss_components = {}

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # 移至设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # 获取输入
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch.get("labels")
                domain_labels = batch.get("domain_label")

                input_ids = input_ids.to(self.device)  # 确保 input_ids 在 GPU
                attention_mask = attention_mask.to(self.device)

                # 学生模型前向传播
                student_outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                    output_attentions=stage == 2 and self.config.stage2.get("attention_distillation_weight", 0) > 0
                )

                # 教师模型前向传播
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                    output_attentions=stage == 2 and self.config.stage2.get("attention_distillation_weight", 0) > 0
                )

                # 计算蒸馏损失
                loss, loss_dict = loss_fn(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    labels=labels,
                    domain_labels=domain_labels
                )

                # 累加损失
                total_loss += loss.item()

                # 累加各组成部分
                for k, v in loss_dict.items():
                    if k not in loss_components:
                        loss_components[k] = 0
                    loss_components[k] += v

                # 收集预测和标签用于计算指标
                if labels is not None:
                    preds = student_outputs.logits.argmax(dim=-1)
                    # 过滤掉填充部分 (-100)
                    valid_indices = labels != -100
                    filtered_preds = preds[valid_indices]
                    filtered_labels = labels[valid_indices]
                    all_preds.extend(filtered_preds.cpu().numpy())
                    all_labels.extend(filtered_labels.cpu().numpy())

        # 计算平均损失
        num_batches = len(eval_dataloader)
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}

        # 计算其他指标
        eval_results = {
            "loss": avg_loss,
            **avg_components
        }

        # 如果有足够的预测和标签，计算精度
        if all_preds and all_labels:
            accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
            eval_results["accuracy"] = accuracy

        logger.info(f"评估结果: {eval_results}")
        return eval_results

    def generate_samples(self, model, prompts, max_length=512):
        """使用模型生成文本样本"""
        model.eval()
        results = []

        for prompt in prompts:
            try:
                # 编码输入
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # 使用更保守的生成参数
                with torch.no_grad():
                    output_ids = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=max_length,
                        do_sample=False,  # 使用贪婪解码提高稳定性
                        num_beams=1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                # 解码输出
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                results.append(output_text)
            except Exception as e:
                logger.warning(f"生成样本时出错: {str(e)}")
                results.append(f"[生成失败: {str(e)}]")

        return results

    def train_stage(self, stage, data_manager, prev_checkpoint=None):
        """
        执行一个阶段的训练

        Args:
            stage: 阶段编号 (1, 2, 或 3)
            data_manager: 数据管理器
            prev_checkpoint: 上一阶段的检查点路径

        Returns:
            checkpoint_path: 当前阶段最佳检查点路径
        """
        logger.info(f"开始阶段{stage}训练: {getattr(self.config, f'stage{stage}')['name']}")

        # 获取当前阶段配置
        stage_config = getattr(self.config, f"stage{stage}")

        # 初始化数据集
        train_dataset, eval_datasets = data_manager.get_stage_datasets(stage)

        # 创建数据加载器
        train_dataloader = data_manager.get_dataloader(
            train_dataset,
            batch_size=stage_config["per_device_train_batch_size"],
            shuffle=True
        )

        eval_dataloaders = {
            domain: data_manager.get_dataloader(
                dataset,
                batch_size=stage_config["per_device_eval_batch_size"],
                shuffle=False
            )
            for domain, dataset in eval_datasets.items()
        }

        # 加载学生模型
        self.student_model = self.load_student_model(stage, from_checkpoint=prev_checkpoint)

        # 初始化损失函数
        loss_fn = DistillationLoss(self.config, stage)

        # 计算训练步数
        num_epochs = stage_config["num_train_epochs"]
        total_steps = len(train_dataloader) * num_epochs

        # 设置优化器和学习率调度器
        optimizer, scheduler = self.setup_optimizer_and_scheduler(
            self.student_model, total_steps, stage
        )

        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler() if self.config.fp16 else None

        # 跟踪最佳模型
        best_eval_loss = float("inf")
        best_checkpoint_path = None

        # 训练循环
        global_step = 0
        optimizer.zero_grad()  # 确保初始梯度为零

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # 对于阶段2，更新医疗数据比例
            if stage == 2:
                train_dataset = data_manager.update_stage2_data_ratio(epoch, num_epochs)
                train_dataloader = data_manager.get_dataloader(
                    train_dataset,
                    batch_size=stage_config["per_device_train_batch_size"],
                    shuffle=True
                )

            # 训练一个epoch
            self.student_model.train()
            epoch_loss = 0
            loss_components = {}

            # 进度条
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")


            for step, batch in enumerate(progress_bar):
                # 执行训练步骤（只执行前向和反向传播）
                loss_dict = self.train_step(
                    batch,
                    stage
                )


                # 累加损失
                epoch_loss += loss_dict

                # 累加各组成部分
                for k, v in loss_dict.items():
                    if k not in loss_components:
                        loss_components[k] = 0
                    loss_components[k] += v

                # 更新进度条
                progress_bar.set_postfix({
                    "loss": loss_dict,
                    "lr": optimizer.param_groups[0]["lr"]
                })
                # 梯度累积
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # 检查梯度是否有NaN/Inf
                    valid_gradients = True

                    # 如果使用混合精度训练，先解缩放梯度
                    if scaler:
                        scaler.unscale_(optimizer)

                    # 检查梯度
                    for param in self.student_model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                valid_gradients = False
                                logger.warning("检测到无效梯度，跳过更新步骤")
                                break

                    if valid_gradients:
                        # 梯度裁剪（已解缩放）
                        torch.nn.utils.clip_grad_norm_(
                            self.student_model.parameters(),
                            self.config.max_grad_norm
                        )

                        # 更新参数
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        # 更新学习率
                        scheduler.step()

                    # 清零梯度
                    optimizer.zero_grad()

                    global_step += 1

                    # 记录日志
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        avg_components = {k: v / (step + 1) for k, v in loss_components.items()}

                        # 记录到wandb
                        if self.config.use_wandb:
                            wandb.log({
                                f"train/loss": avg_loss,
                                f"train/lr": optimizer.param_groups[0]["lr"],
                                **{f"train/{k}": v for k, v in avg_components.items()},
                                "epoch": epoch,
                                "global_step": global_step
                            })

                    # 保存检查点
                    if global_step % self.config.save_steps == 0:
                        checkpoint_path = os.path.join(
                            self.config.output_dir,
                            f"stage{stage}_checkpoint-{global_step}"
                        )
                        self.save_checkpoint(self.student_model, checkpoint_path)

                    # 评估
                    if global_step % self.config.eval_steps == 0:
                        eval_results = {}

                        # 对每个评估数据集进行评估
                        for domain, eval_dataloader in eval_dataloaders.items():
                            domain_results = self.evaluate(
                                self.student_model,
                                eval_dataloader,
                                loss_fn,
                                stage
                            )

                            # 添加领域前缀
                            domain_eval_results = {f"{domain}/{k}": v for k, v in domain_results.items()}
                            eval_results.update(domain_eval_results)

                        # 计算平均损失（如果有多个数据集）
                        if len(eval_dataloaders) > 1:
                            avg_eval_loss = sum(r["loss"] for r in eval_results.values()) / len(eval_dataloaders)
                        else:
                            # 使用唯一数据集的损失
                            domain = list(eval_results.keys())[0]
                            avg_eval_loss = eval_results[f"{domain}/loss"]

                        # 记录到wandb
                        if self.config.use_wandb:
                            wandb.log({
                                **eval_results,
                                "eval/avg_loss": avg_eval_loss,
                                "epoch": epoch,
                                "global_step": global_step
                            })

                        # 保存最佳模型
                        if avg_eval_loss < best_eval_loss:
                            best_eval_loss = avg_eval_loss
                            best_checkpoint_path = os.path.join(
                                self.config.output_dir,
                                f"stage{stage}_best"
                            )
                            self.save_checkpoint(self.student_model, best_checkpoint_path)
                            logger.info(f"保存最佳模型，评估损失: {best_eval_loss:.4f}")

            # Epoch结束，进行一次评估
            eval_results = {}

            for domain, eval_dataloader in eval_dataloaders.items():
                domain_results = self.evaluate(
                    self.student_model,
                    eval_dataloader,
                    loss_fn,
                    stage
                )

                # 添加领域前缀
                domain_eval_results = {f"{domain}/{k}": v for k, v in domain_results.items()}
                eval_results.update(domain_eval_results)

            # 计算平均损失
            if len(eval_dataloaders) > 1:
                avg_eval_loss = sum(eval_results[f"{domain}/loss"] for domain in eval_dataloaders.keys()) / len(
                    eval_dataloaders)
            else:
                # 使用唯一数据集的损失
                domain = list(eval_dataloaders.keys())[0]
                avg_eval_loss = eval_results[f"{domain}/loss"]

            # 记录到wandb
            if self.config.use_wandb:
                wandb.log({
                    **eval_results,
                    "eval/avg_loss": avg_eval_loss,
                    "epoch": epoch + 1,
                    "global_step": global_step
                })

            # 保存每个epoch的模型
            epoch_checkpoint_path = os.path.join(
                self.config.output_dir,
                f"stage{stage}_epoch-{epoch + 1}"
            )
            self.save_checkpoint(self.student_model, epoch_checkpoint_path)

            # 更新最佳模型
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                best_checkpoint_path = os.path.join(
                    self.config.output_dir,
                    f"stage{stage}_best"
                )
                self.save_checkpoint(self.student_model, best_checkpoint_path)
                logger.info(f"更新最佳模型，评估损失: {best_eval_loss:.4f}")

            # 生成一些示例（用于监控生成质量）
            if stage == 3 or epoch == num_epochs - 1:
                self.generate_quality_check(stage, epoch)

        # 训练结束，保存最终模型
        final_checkpoint_path = os.path.join(
            self.config.output_dir,
            f"stage{stage}_final"
        )
        self.save_checkpoint(self.student_model, final_checkpoint_path)

        # 返回最佳检查点路径
        return best_checkpoint_path or final_checkpoint_path

    def generate_quality_check(self, stage, epoch):
        """
        生成一些示例来检查质量

        Args:
            stage: 当前训练阶段
            epoch: 当前epoch
        """
        logger.info(f"生成质量检查样本 (阶段{stage}, Epoch {epoch})")

        # 设置测试提示
        prompts = [
            "Human: 请解释一下高血压的常见症状和治疗方法\n\nAssistant: ",
            "Human: 什么是2型糖尿病？有哪些风险因素？\n\nAssistant: ",
            "Human: 如何预防心血管疾病？\n\nAssistant: ",
            "Human: 抗生素耐药性是什么意思？为什么会出现这种情况？\n\nAssistant: ",
            "Human: 给一个30岁的人提供一些常见的健康检查建议\n\nAssistant: "
        ]

        # 使用学生模型生成
        student_outputs = self.generate_samples(self.student_model, prompts)

        # 使用教师模型生成（作为参考）
        with torch.no_grad():
            teacher_outputs = self.generate_samples(self.teacher_model, prompts)

        # 输出到日志
        for i, (prompt, student_out, teacher_out) in enumerate(zip(prompts, student_outputs, teacher_outputs)):
            logger.info(f"样本 {i + 1}:")
            logger.info(f"提示: {prompt}")
            logger.info(f"学生模型: {student_out[:500]}...")  # 截断以避免日志过长
            logger.info(f"教师模型: {teacher_out[:500]}...")
            logger.info("-" * 50)

        # 记录到wandb
        if self.config.use_wandb:
            # 格式化输出以便在wandb中显示
            table_data = []
            for i, (prompt, student_out, teacher_out) in enumerate(zip(prompts, student_outputs, teacher_outputs)):
                table_data.append([prompt, student_out, teacher_out])

            generation_table = wandb.Table(
                columns=["Prompt", "Student Model", "Teacher Model"],
                data=table_data
            )

            wandb.log({
                f"generations/stage{stage}_epoch{epoch}": generation_table,
                "epoch": epoch,
                "stage": stage
            })

    def run_multi_stage_distillation(self, data_manager):
        """
        运行多阶段知识蒸馏流程

        Args:
            data_manager: 数据管理器

        Returns:
            final_model_path: 最终模型路径
        """
        logger.info("开始多阶段知识蒸馏流程")

        # 阶段1: 基础蒸馏
        logger.info("=" * 50)
        logger.info("开始阶段1: 基础蒸馏")
        stage1_checkpoint = self.train_stage(1, data_manager)

        # 阶段2: 领域自适应蒸馏
        logger.info("=" * 50)
        logger.info("开始阶段2: 领域自适应蒸馏")
        stage2_checkpoint = self.train_stage(2, data_manager, prev_checkpoint=stage1_checkpoint)

        # 阶段3: 领域微调
        logger.info("=" * 50)
        logger.info("开始阶段3: 领域微调")
        stage3_checkpoint = self.train_stage(3, data_manager, prev_checkpoint=stage2_checkpoint)

        # 汇报最终模型路径
        logger.info(f"多阶段知识蒸馏完成")
        logger.info(f"最终模型路径: {stage3_checkpoint}")

        # 最终评估
        if self.config.medical_evaluation["use_external_eval"]:
            self.final_evaluation(stage3_checkpoint, data_manager)

        return stage3_checkpoint

    def final_evaluation(self, model_path, data_manager):
        """
        对最终模型进行全面评估

        Args:
            model_path: 模型路径
            data_manager: 数据管理器
        """
        logger.info("开始最终评估...")

        # 加载最终模型
        final_model = self.load_student_model(3, from_checkpoint=model_path)
        final_model.eval()

        # 使用所有评估数据集
        all_results = {}

        for domain, dataset in data_manager.eval_datasets.items():
            logger.info(f"评估领域: {domain}")

            dataloader = data_manager.get_dataloader(
                dataset,
                batch_size=self.config.stage3["per_device_eval_batch_size"],
                shuffle=False
            )

            # 使用阶段3的损失函数配置
            loss_fn = DistillationLoss(self.config, 3)

            # 评估
            domain_results = self.evaluate(final_model, dataloader, loss_fn, 3)

            # 添加到总结果
            all_results[domain] = domain_results

        # 计算平均结果
        avg_results = {}
        for metric in ["loss", "accuracy"]:
            values = [results.get(metric, 0) for results in all_results.values()]
            if values:
                avg_results[f"avg_{metric}"] = sum(values) / len(values)

        # 记录最终结果
        logger.info("最终评估结果:")
        for domain, results in all_results.items():
            logger.info(f"{domain}: {results}")
        logger.info(f"平均: {avg_results}")

        # 记录到wandb
        if self.config.use_wandb:
            flattened_results = {}
            for domain, results in all_results.items():
                for k, v in results.items():
                    flattened_results[f"final_eval/{domain}/{k}"] = v

            for k, v in avg_results.items():
                flattened_results[f"final_eval/{k}"] = v

            wandb.log(flattened_results)

        print(f"Sample text: {full_text[:100]}")  # 打印前100个字符
        print(f"Sample text encoding: {full_text.encode('utf-8')[:100]}")  # 打印UTF-8编码