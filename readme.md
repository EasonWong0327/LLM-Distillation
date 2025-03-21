# Qwen多阶段知识蒸馏框架
[🇨🇳 中文](readme.md) | [🇺🇸 English](readme_en.md)

这是一个专为Qwen模型设计的多阶段知识蒸馏框架，旨在将Qwen-72B的知识有效地蒸馏到Qwen-7B-instruct中，可针对垂直领域进行优化。

- **如果**你想压缩模型、节省资源（比如嵌入式设备、实时推理）：  
    选蒸馏，看本项目。  

- **如果**你想让预训练模型适应新任务，且手头有少量垂直领域数据：  
    选微调，看 https://github.com/EasonWong0327/LLM-Fine-tuning

## 特点

- **三阶段蒸馏流程**：
  - **阶段1 - 基础蒸馏**：使用通用知识数据，帮助学生模型获取教师模型的通用能力
  - **阶段2 - 领域自适应蒸馏**：逐步增加医疗数据比例，使用多种损失函数平滑过渡
  - **阶段3 - 领域微调**：集中于医疗领域数据，优化特定领域表现

- **多样化蒸馏损失**：
  - KL散度损失
  - 交叉熵损失
  - 特征蒸馏损失
  - 注意力蒸馏损失
  - 对比学习损失

- **高级优化策略**：
  - 梯度平衡
  - 动态数据比例调整
  - 评估驱动采样
  - 不同阶段使用不同的温度参数

- **灵活配置**：
  - 支持LoRA参数高效微调
  - 混合精度训练
  - 可配置的评估策略

## 初学者看这里

### 1.蒸馏VS微调

- **蒸馏**：把一个大聪明模型的知识“挤”到一个小模型里，让小模型学会大模型的本事，但体积更小、速度更快（就像把老师教的知识浓缩成小抄）。
- **微调**：让一个已经学过通用知识的大模型，再专门学某个新任务（比如从学数学转到学医学），让它更擅长解决特定问题。

### 2.蒸馏四步走：
- **找大师**： 先找一个你认为![img_1.png](/img/img_1.png)![img_2.png](/img/img_2.png)的大模型（教师模型），它啥都懂。
- **出考题**： 用大师处理所有数据，记录下它的答案（比如概率分布）。
- **教徒弟**： 用这些答案当“教材”，训练小模型（学生模型）去模仿，直到它答题风格接近大师。
- **瘦身**：可能进一步压缩小模型（比如减少层数、参数），让它更轻便。

## 安装

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (Parameter-Efficient Fine-Tuning)

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python run_distillation.py --output_dir ./qwen_distilled_model --fp16 --use_lora
```

### 完整参数

```bash
python run_distillation.py \
  --teacher_model Qwen/Qwen-72B \
  --student_model Qwen/Qwen-7B-instruct \
  --output_dir ./qwen_medical_model \
  --general_dataset tatsu-lab/alpaca \
  --medical_dataset ./medical_data \
  --fp16 \
  --use_lora \
  --use_wandb \
  --start_stage 1 \
  --end_stage 3
```

### 从特定阶段开始

```bash
# 从阶段2开始，使用之前的检查点
python run_distillation.py \
  --start_stage 2 \
  --checkpoint_path ./qwen_distilled_model/stage1_best \
  --output_dir ./qwen_continued_distillation
```

### 仅运行特定阶段

```bash
# 只运行阶段3（领域微调）
python run_distillation.py \
  --start_stage 3 \
  --end_stage 3 \
  --checkpoint_path ./path/to/previous_model \
  --output_dir ./qwen_medical_finetuned
```

## 项目结构

```
qwen_distillation/
│
├── config.py                  # 配置类和默认参数
├── run_distillation.py        # 主执行脚本
│
├── data/                      # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py             # 数据集定义
│   ├── data_manager.py        # 数据管理器
│   └── data_utils.py          # 辅助函数
│
├── modeling/                  # 模型相关模块
│   ├── __init__.py
│   ├── distiller.py           # 蒸馏器核心实现
│   ├── losses.py              # 各种损失函数
│   └── model_utils.py         # 模型工具函数

```

## 各阶段详细说明

### 阶段1: 基础蒸馏

- **目标**：让学生模型学习教师模型的通用知识分布
- **数据**：主要使用通用领域数据集
- **损失函数**：以KL散度损失为主，辅以少量交叉熵损失
- **温度参数**：较高 (4.0)，使软标签更平滑
- **优化策略**：较大学习率，简单训练

### 阶段2: 领域自适应蒸馏

- **目标**：平滑过渡到医疗领域，避免灾难性遗忘
- **数据**：混合通用和医疗数据，比例从30%逐步增加到70%
- **损失函数**：
  - KL散度损失
  - 交叉熵损失
  - 特征蒸馏损失
  - 注意力蒸馏损失
  - 对比学习损失（可选）
- **温度参数**：中等 (2.0)
- **优化策略**：梯度平衡，避免领域偏移

### 阶段3: 领域微调

- **目标**：优化医疗领域表现
- **数据**：仅使用医疗领域数据
- **损失函数**：以交叉熵损失为主，辅以少量KL散度损失
- **温度参数**：较低 (1.0)，更聚焦于硬标签
- **优化策略**：较小学习率，评估驱动采样

## 结果评估

在训练过程中和训练结束后，框架会自动生成质量检查样本，对比学生模型和教师模型的生成结果。这些结果会记录在日志中，如果启用了wandb跟踪，也会在wandb仪表板中显示。

## 调参建议

1. **阶段1**：
   - 较大batch_size (8-16)
   - 较高学习率 (5e-5)
   - 较少epoch (2-3)

2. **阶段2**：
   - 中等batch_size (6-8)
   - 中等学习率 (3e-5)
   - 实验不同的数据比例增长策略

3. **阶段3**：
   - 较小batch_size (4)
   - 较低学习率 (1e-5)
   - 多epoch (2-4)


## 许可证

本项目使用MIT许可证。

## 引用
如果您使用了本项目的代码，请引用：
```bibtex
@misc{LLM-Distillation,
  author = {EasonWong0327},
  title = {LLM-Distillation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/EasonWong0327/LLM-Distillation}
}
``` 