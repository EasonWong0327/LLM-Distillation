"""
模型工具函数模块，提供与模型相关的辅助功能
"""

import os
import logging
import torch
import torch.nn as nn
from transformers import set_seed

logger = logging.getLogger(__name__)


def create_linear_projection(input_dim, output_dim, bias=False, orthogonal_init=True):
    """
    创建线性投影层（用于特征蒸馏时的维度匹配）

    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        bias: 是否使用偏置
        orthogonal_init: 是否使用正交初始化

    Returns:
        线性投影层
    """
    projection = nn.Linear(input_dim, output_dim, bias=bias)

    if orthogonal_init:
        nn.init.orthogonal_(projection.weight)

    return projection


def calculate_model_parameters(model):
    """
    计算模型参数数量

    Args:
        model: 模型

    Returns:
        total_params: 总参数数量
        trainable_params: 可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def get_model_memory_footprint(model):
    """
    计算模型的内存占用

    Args:
        model: 模型

    Returns:
        memory_in_gb: 内存占用（GB）
    """
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem_total = mem_params + mem_bufs  # 以字节为单位

    # 转换为GB
    memory_in_gb = mem_total / (1024 ** 3)

    return memory_in_gb


def find_layers(model, layer_type):
    """
    在模型中查找指定类型的层

    Args:
        model: 模型
        layer_type: 层类型

    Returns:
        layers: 找到的层列表
    """
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, layer_type):
            layers[name] = module

    return layers


def setup_model_parallel(model, device_map=None):
    """
    设置模型并行

    Args:
        model: 模型
        device_map: 设备映射（可选）

    Returns:
        model: 设置好并行的模型
    """
    if device_map is None:
        if torch.cuda.device_count() > 1:
            logger.info(f"自动并行化模型到 {torch.cuda.device_count()} 个GPU")
            model.parallelize()
        else:
            model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        try:
            from accelerate import dispatch_model
            logger.info(f"使用device_map手动并行化模型")
            model = dispatch_model(model, device_map=device_map)
        except ImportError:
            logger.warning("无法导入accelerate，将模型移动到设备0")
            model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")

    return model


def predict_with_generate(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9, num_return_sequences=1):
    """
    使用模型生成文本

    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 提示文本
        max_length: 最大生成长度
        temperature: 温度参数
        top_p: top-p采样参数
        num_return_sequences: 返回序列数量

    Returns:
        生成的文本列表
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_texts = []
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts


def get_lora_target_modules_for_qwen(model_name):
    """
    获取Qwen模型的LoRA目标模块

    Args:
        model_name: 模型名称

    Returns:
        target_modules: LoRA目标模块列表
    """
    if "qwen" in model_name.lower():
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        return ["query", "key", "value", "dense"]


def merge_lora_weights(model, lora_path, tokenizer=None):
    """
    合并LoRA权重到模型

    Args:
        model: 基础模型
        lora_path: LoRA权重路径
        tokenizer: 分词器（可选）

    Returns:
        合并后的模型
    """
    try:
        from peft import PeftModel

        # 加载LoRA模型
        logger.info(f"从{lora_path}加载LoRA权重")
        model = PeftModel.from_pretrained(model, lora_path)

        # 合并权重
        logger.info("合并LoRA权重到基础模型")
        model = model.merge_and_unload()

        # 如果提供了分词器，一并保存
        if tokenizer is not None:
            output_dir = f"merged_{os.path.basename(lora_path)}"
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"已保存合并后的模型到{output_dir}")

        return model

    except ImportError:
        logger.error("合并LoRA权重需要安装peft库")
        return model


def convert_to_half_precision(model):
    """
    将模型转换为半精度

    Args:
        model: 模型

    Returns:
        半精度模型
    """
    model = model.half()
    return model