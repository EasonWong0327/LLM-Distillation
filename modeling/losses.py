"""
损失函数模块，包含知识蒸馏所需的各种损失计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss:
    """
    知识蒸馏损失的组合
    支持KL散度损失、交叉熵损失、特征蒸馏损失等
    """

    def __init__(self, config, stage):
        """
        初始化蒸馏损失

        Args:
            config: 配置对象
            stage: 当前训练阶段 (1, 2, 或 3)
        """
        self.config = config
        self.stage = stage

        # 获取当前阶段的配置
        if stage == 1:
            self.stage_config = config.stage1
        elif stage == 2:
            self.stage_config = config.stage2
        elif stage == 3:
            self.stage_config = config.stage3
        else:
            raise ValueError(f"无效的训练阶段: {stage}")

        # 损失权重
        self.alpha_kl = self.stage_config["alpha_kl"]
        self.alpha_ce = self.stage_config["alpha_ce"]
        self.temperature = self.stage_config["temperature"]

        # 是否使用中间层蒸馏
        self.use_intermediate_distillation = self.stage_config.get("use_intermediate_distillation", False)

        # 如果使用中间层蒸馏，获取相关权重
        if self.use_intermediate_distillation:
            self.feature_distillation_weight = self.stage_config.get("feature_distillation_weight", 0.0)
            self.attention_distillation_weight = self.stage_config.get("attention_distillation_weight", 0.0)

        # 是否使用对比学习（阶段2）
        self.use_contrastive_learning = stage == 2 and self.stage_config.get("use_contrastive_learning", False)
        if self.use_contrastive_learning:
            self.contrastive_loss_weight = self.stage_config.get("contrastive_loss_weight", 0.1)
            self.contrastive_loss = ContrastiveLoss(temperature=0.5)

    def compute_kl_loss(self, student_logits, teacher_logits):
        """计算KL散度损失"""
        temperature = min(self.temperature, 1.5)  # 限制最大温度

        # 确保输入张量需要梯度
        if not student_logits.requires_grad:
            student_logits.requires_grad_(True)

        # 应用温度缩放
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # 添加epsilon防止零概率
        epsilon = 1e-5
        teacher_probs = teacher_probs * (1 - epsilon) + epsilon / teacher_probs.size(-1)

        # 计算KL散度
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
            log_target=False,
        )

        # 限制KL损失值的范围
        kl_loss = torch.clamp(kl_loss * (temperature ** 2), max=10.0)

        return kl_loss

    def compute_feature_distillation_loss(self, student_features, teacher_features):
        """
        计算特征蒸馏损失

        Args:
            student_features: 学生模型的特征表示
            teacher_features: 教师模型的特征表示

        Returns:
            特征蒸馏损失
        """
        # 确保学生特征需要梯度
        for feature in student_features:
            if not feature.requires_grad:
                feature.requires_grad_(True)

        # 使用MSE损失计算特征表示间的差异
        mse_loss = nn.MSELoss()
        loss = 0.0

        # 选择部分关键层
        total_layers = len(student_features)
        selected_layers = list(range(0, total_layers, total_layers // 4))  # 选择4个关键层

        for layer_idx in selected_layers:
            if layer_idx < len(student_features) and layer_idx < len(teacher_features):
                s_feat = student_features[layer_idx]
                t_feat = teacher_features[layer_idx]

                # 检查维度是否匹配
                if s_feat.size(-1) != t_feat.size(-1):
                    # 如果维度不匹配，使用平均池化调整序列维度
                    s_feat = torch.mean(s_feat, dim=1)  # [batch_size, hidden_dim]
                    t_feat = torch.mean(t_feat, dim=1)  # [batch_size, hidden_dim]

                    # 创建线性投影层（如果不存在）
                    if not hasattr(self, 'feature_projections'):
                        self.feature_projections = {}
                    
                    if layer_idx not in self.feature_projections:
                        # 创建从学生维度到教师维度的线性投影
                        self.feature_projections[layer_idx] = nn.Linear(
                            s_feat.size(-1),
                            t_feat.size(-1),
                            bias=False
                        ).to(s_feat.device)
                        # 使用正交初始化
                        nn.init.orthogonal_(self.feature_projections[layer_idx].weight)

                    # 应用投影
                    s_feat = self.feature_projections[layer_idx](s_feat)

                # 确保特征在同一设备上
                s_feat = s_feat.to(t_feat.device)

                # 计算损失
                layer_loss = mse_loss(s_feat, t_feat)
                loss += layer_loss

        return loss / len(selected_layers)

    def compute_attention_distillation_loss(self, student_attentions, teacher_attentions):
        """
        计算注意力蒸馏损失

        Args:
            student_attentions: 学生模型的注意力权重
            teacher_attentions: 教师模型的注意力权重

        Returns:
            注意力蒸馏损失
        """
        loss = 0.0
        n_layers = 0

        # 同样选择部分关键层
        selected_layers = [-1, -3, -6, -9]

        for idx in selected_layers:
            if abs(idx) < len(student_attentions) and abs(idx) < len(teacher_attentions):
                s_att = student_attentions[idx]
                t_att = teacher_attentions[idx]

                # 对于注意力权重，通常使用KL散度来度量差异
                s_att = torch.where(s_att <= 0, torch.ones_like(s_att) * 1e-8, s_att)
                t_att = torch.where(t_att <= 0, torch.ones_like(t_att) * 1e-8, t_att)

                # 由于多头注意力机制，可能需要处理多头
                # 这里简化为平均所有头的注意力
                if len(s_att.shape) > 3:  # [batch, heads, seq, seq]
                    s_att = torch.mean(s_att, dim=1)  # [batch, seq, seq]
                    t_att = torch.mean(t_att, dim=1)

                # 计算KL散度
                loss += F.kl_div(
                    F.log_softmax(s_att, dim=-1),
                    F.softmax(t_att, dim=-1),
                    reduction="batchmean",
                )
                n_layers += 1

        return loss / max(n_layers, 1)

    def compute_contrastive_loss(self, student_embeddings, teacher_embeddings, domain_labels=None):
        """
        计算对比学习损失

        Args:
            student_embeddings: 学生模型的嵌入表示
            teacher_embeddings: 教师模型的嵌入表示
            domain_labels: 领域标签

        Returns:
            对比学习损失
        """
        if not self.use_contrastive_learning:
            return 0.0

        return self.contrastive_loss(student_embeddings, teacher_embeddings, domain_labels)

    def __call__(self, student_outputs, teacher_outputs, labels=None, domain_labels=None):
        """
        计算总蒸馏损失

        Args:
            student_outputs: 学生模型的输出
            teacher_outputs: 教师模型的输出
            labels: 标签
            domain_labels: 领域标签

        Returns:
            总损失, 损失字典 (包含各组成部分)
        """
        loss_dict = {}

        # 1. 语言模型交叉熵损失
        if labels is not None and self.alpha_ce > 0:
            ce_loss = student_outputs.loss  # 学生模型自身的交叉熵损失
            loss_dict["ce_loss"] = ce_loss.item()
        else:
            ce_loss = 0.0
            loss_dict["ce_loss"] = 0.0

        # 2. KL散度损失
        if self.alpha_kl > 0:
            student_logits = student_outputs.logits
            with torch.no_grad():
                teacher_logits = teacher_outputs.logits

            kl_loss = self.compute_kl_loss(student_logits, teacher_logits)
            loss_dict["kl_loss"] = kl_loss.item()
        else:
            kl_loss = 0.0
            loss_dict["kl_loss"] = 0.0

        # 3. 特征蒸馏损失
        if self.use_intermediate_distillation and self.feature_distillation_weight > 0:
            student_features = student_outputs.hidden_states
            teacher_features = teacher_outputs.hidden_states

            feature_loss = self.compute_feature_distillation_loss(student_features, teacher_features)
            loss_dict["feature_loss"] = feature_loss.item()
        else:
            feature_loss = 0.0
            loss_dict["feature_loss"] = 0.0

        # 4. 注意力蒸馏损失
        if self.use_intermediate_distillation and self.attention_distillation_weight > 0:
            if hasattr(student_outputs, "attentions") and student_outputs.attentions is not None:
                student_attentions = student_outputs.attentions
                teacher_attentions = teacher_outputs.attentions

                attention_loss = self.compute_attention_distillation_loss(
                    student_attentions, teacher_attentions
                )
                loss_dict["attention_loss"] = attention_loss.item()
            else:
                attention_loss = 0.0
                loss_dict["attention_loss"] = 0.0
        else:
            attention_loss = 0.0
            loss_dict["attention_loss"] = 0.0

        # 5. 对比学习损失（适用于阶段2）
        if self.use_contrastive_learning:
            # 使用最后一层隐藏状态作为嵌入表示
            student_emb = student_outputs.hidden_states[-1][:, 0]  # [CLS]嵌入
            teacher_emb = teacher_outputs.hidden_states[-1][:, 0]

            contrastive_loss = self.compute_contrastive_loss(
                student_emb, teacher_emb, domain_labels
            )
            loss_dict["contrastive_loss"] = contrastive_loss.item()
        else:
            contrastive_loss = 0.0
            loss_dict["contrastive_loss"] = 0.0

        # 计算总损失
        total_loss = (
                self.alpha_ce * ce_loss +
                self.alpha_kl * kl_loss
        )

        # 添加特征蒸馏损失
        if self.use_intermediate_distillation:
            total_loss += (
                    self.feature_distillation_weight * feature_loss +
                    self.attention_distillation_weight * attention_loss
            )

        # 添加对比学习损失
        if self.use_contrastive_learning:
            total_loss += self.contrastive_loss_weight * contrastive_loss

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict


class ContrastiveLoss(nn.Module):
    """
    对比学习损失
    用于学习不同领域的数据表示
    """

    def __init__(self, temperature=0.5):
        """
        初始化对比学习损失

        Args:
            temperature: 温度参数，控制对比学习的敏感度
        """
        super().__init__()
        self.temperature = temperature
        self.projection = None  # 动态创建的投影层

    def forward(self, student_embeddings, teacher_embeddings, domain_labels=None):
        """
        计算对比学习损失

        Args:
            student_embeddings: 学生模型的嵌入表示 [batch_size, embed_dim]
            teacher_embeddings: 教师模型的嵌入表示 [batch_size, embed_dim]
            domain_labels: 领域标签 [batch_size]

        Returns:
            对比学习损失
        """
        batch_size = student_embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=student_embeddings.device)

        # 检查维度是否匹配
        if student_embeddings.size(-1) != teacher_embeddings.size(-1):
            # 如果维度不匹配，创建投影层
            if self.projection is None:
                self.projection = nn.Linear(
                    student_embeddings.size(-1),
                    teacher_embeddings.size(-1),
                    bias=False
                ).to(student_embeddings.device)
                # 使用正交初始化
                nn.init.orthogonal_(self.projection.weight)
            
            # 应用投影
            student_embeddings = self.projection(student_embeddings)

        # 标准化嵌入
        student_embeddings = F.normalize(student_embeddings, dim=1)
        teacher_embeddings = F.normalize(teacher_embeddings, dim=1)

        # 计算余弦相似度矩阵
        sim_matrix = torch.matmul(student_embeddings, teacher_embeddings.t()) / self.temperature

        # 创建标签：对角线为正例
        labels = torch.arange(batch_size, device=sim_matrix.device)

        # 如果有领域标签，可以更精细地定义正负例
        if domain_labels is not None:
            # 创建领域相似度矩阵：相同领域为1，不同领域为0
            domain_sim = domain_labels.unsqueeze(1) == domain_labels.unsqueeze(0)
            # 将对角线（自身）设为0，因为对比学习中不应使用自身作为正例
            domain_sim.fill_diagonal_(0)

            # 根据领域相似度矩阵调整损失计算
            pos_mask = domain_sim.float()
            neg_mask = (~domain_sim).float()

            # 计算正例损失和负例损失
            pos_loss = -torch.log(torch.exp(sim_matrix) * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
            neg_loss = torch.log(torch.sum(torch.exp(sim_matrix) * neg_mask, dim=1) + 1e-8)

            # 合并损失
            loss = (pos_loss + neg_loss).mean()
        else:
            # 标准的InfoNCE损失
            loss = F.cross_entropy(sim_matrix, labels)

        return loss


class GradientBalancingLoss:
    """
    梯度平衡损失
    用于阶段2，确保不同领域的梯度贡献平衡
    """

    def __init__(self, config):
        """
        初始化梯度平衡损失

        Args:
            config: 配置对象
        """
        self.config = config

    def __call__(self, loss, domain_labels):
        """
        应用梯度平衡

        Args:
            loss: 原始损失
            domain_labels: 领域标签

        Returns:
            平衡后的损失
        """
        if not self.config.stage2.get("use_gradient_balancing", False):
            return loss

        # 分离不同领域的损失
        if domain_labels is None:
            return loss

        # 获取每个样本的损失
        per_sample_losses = loss.view(-1)

        # 按领域分组
        general_indices = (domain_labels == 0).nonzero(as_tuple=True)[0]
        medical_indices = (domain_labels == 1).nonzero(as_tuple=True)[0]

        if len(general_indices) == 0 or len(medical_indices) == 0:
            return loss

        # 计算每个领域的平均损失
        general_loss = per_sample_losses[general_indices].mean()
        medical_loss = per_sample_losses[medical_indices].mean()

        # 使用简单的加权平均，而不是计算梯度
        # 根据领域大小动态调整权重
        total_samples = len(domain_labels)
        general_weight = len(medical_indices) / total_samples
        medical_weight = len(general_indices) / total_samples

        # 计算加权损失
        balanced_loss = general_weight * general_loss + medical_weight * medical_loss

        return balanced_loss