"""
数据工具函数，提供数据处理、生成等辅助功能
"""

import random
from datasets import Dataset


def create_mock_medical_dataset():
    """
    创建模拟医疗数据集（仅用于演示）
    """
    mock_data = []

    medical_qa_pairs = [
        {
            "instruction": "请解释高血压的常见症状和治疗方法",
            "input": "",
            "output": "高血压常见症状包括头痛、头晕、耳鸣、心悸和疲劳等，严重时可能出现视力模糊或鼻出血。许多人可能没有明显症状，这就是为什么高血压被称为'沉默的杀手'。治疗方法包括生活方式调整（减少钠摄入、增加运动、减轻体重、限制酒精摄入、戒烟）和药物治疗（利尿剂、ACE抑制剂、ARB、钙通道阻滞剂等）。定期监测血压并遵医嘱服药对控制高血压至关重要。",
        },
        {
            "instruction": "描述2型糖尿病的风险因素和预防措施",
            "input": "",
            "output": "2型糖尿病的主要风险因素包括超重或肥胖、身体活动不足、家族史、年龄增长（尤其是45岁以上）、高血压、高胆固醇、多囊卵巢综合征以及曾患有妊娠糖尿病史。预防措施包括：保持健康体重、定期进行中等强度运动（每周至少150分钟）、健康饮食（增加蔬果摄入，减少精制碳水化合物和饱和脂肪）、定期检查血糖水平、控制血压和胆固醇。生活方式的改变可以显著降低患2型糖尿病的风险。",
        },
        {
            "instruction": "请列出常见的心血管疾病及其主要症状",
            "input": "",
            "output": "常见的心血管疾病包括：1) 冠心病：胸痛、胸闷、气短、疲劳；2) 心力衰竭：呼吸困难、体力下降、下肢水肿、疲劳；3) 心律失常：心悸、头晕、昏厥、胸闷；4) 高血压：通常无症状，严重时可能头痛、头晕；5) 心肌梗死：剧烈胸痛（可能辐射至手臂、颈部）、冷汗、恶心、呼吸急促；6) 中风：突发的面部下垂、手臂无力、言语不清、视力模糊、严重头痛；7) 心脏瓣膜疾病：疲劳、气短、胸痛、心悸。及时识别这些症状并寻求医疗帮助对预防严重后果至关重要。",
        },
        {
            "instruction": "解释基因检测在癌症诊断中的应用",
            "input": "",
            "output": "基因检测在癌症诊断中有多种重要应用：1) 遗传性癌症风险评估：通过检测特定基因（如BRCA1/2对乳腺癌和卵巢癌的风险）来评估个体患癌风险；2) 早期检测：使用基因突变或表达谱作为生物标志物来检测早期癌症；3) 肿瘤分类：通过基因表达模式对癌症进行分类，帮助确定最佳治疗方案；4) 预测预后：某些基因特征可预测疾病进展和生存率；5) 指导治疗选择：基因检测可识别特定药物靶点或预测对某些治疗的反应（如HER2状态对乳腺癌治疗的指导）；6) 监测治疗反应：通过检测循环肿瘤DNA（ctDNA）来评估治疗效果并监测复发。随着技术进步，基因检测已成为癌症精准医疗的关键工具。",
        },
        {
            "instruction": "什么是多重耐药细菌，以及它们对公共健康的威胁是什么？",
            "input": "",
            "output": '多重耐药细菌是指对多种抗生素产生抗性的细菌，常被称为超级细菌。这些细菌通过基因突变或获取耐药基因发展出抵抗抗生素作用的机制。常见的多重耐药细菌包括耐甲氧西林金黄色葡萄球菌(MRSA)、耐万古霉素肠球菌(VRE)、产超广谱β-内酰胺酶(ESBL)的肠杆菌科细菌、耐碳青霉烯类肠杆菌科细菌(CRE)等。它们对公共健康的威胁包括：1) 治疗选择有限，导致感染难以治愈；2) 增加医疗成本和住院时间；3) 提高死亡率；4) 在医疗机构内爆发和传播的风险；5) 可能将耐药性传递给其他细菌。预防策略包括合理使用抗生素、加强感染控制措施、研发新型抗生素和替代治疗方法。抗生素耐药性被世界卫生组织认为是21世纪最严峻的公共健康挑战之一。',
        },
    ]

    for i in range(500):
        idx = i % len(medical_qa_pairs)
        example = medical_qa_pairs[idx].copy()
        example["instruction"] = example["instruction"] + f" (样本变体 {i // len(medical_qa_pairs)})"
        mock_data.append(example)

    return {"train": Dataset.from_list(mock_data)}


def format_medical_prompt(question, system_prompt=None):
    """
    格式化医疗问题的提示

    Args:
        question: 医疗问题
        system_prompt: 系统提示（可选）

    Returns:
        格式化的提示
    """
    if system_prompt:
        formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{question}\n<|assistant|>"
    else:
        default_system = "你是一个专业的医疗助手，请根据你的知识提供准确、科学的医疗信息。"
        formatted_prompt = f"<|system|>\n{default_system}\n<|user|>\n{question}\n<|assistant|>"

    return formatted_prompt


def sample_batch_with_domain_balance(dataset, batch_size, domain_ratio=0.5):
    """
    考虑领域平衡的批次采样

    Args:
        dataset: 混合数据集
        batch_size: 批次大小
        domain_ratio: 目标领域比例

    Returns:
        采样的批次索引
    """
    # 假设数据集中每个样本都有domain_label字段
    domain_labels = [sample.get("domain_label", 0) for sample in dataset]

    # 分离不同领域的索引
    general_indices = [i for i, label in enumerate(domain_labels) if label == 0]
    medical_indices = [i for i, label in enumerate(domain_labels) if label == 1]

    # 计算每个领域需要的样本数
    medical_count = int(batch_size * domain_ratio)
    general_count = batch_size - medical_count

    # 采样
    sampled_general = random.sample(general_indices, min(general_count, len(general_indices)))
    sampled_medical = random.sample(medical_indices, min(medical_count, len(medical_indices)))

    # 如果某个领域样本不足，从另一个领域补充
    if len(sampled_general) < general_count:
        extra_needed = general_count - len(sampled_general)
        extra_medical = random.sample(medical_indices, min(extra_needed, len(medical_indices)))
        sampled_medical.extend(extra_medical)

    if len(sampled_medical) < medical_count:
        extra_needed = medical_count - len(sampled_medical)
        extra_general = random.sample(general_indices, min(extra_needed, len(general_indices)))
        sampled_general.extend(extra_general)

    # 合并并打乱
    sampled_indices = sampled_general + sampled_medical
    random.shuffle(sampled_indices)

    return sampled_indices


def create_hard_negative_samples(dataset, tokenizer, similarity_threshold=0.8):
    """
    为对比学习创建难负例

    Args:
        dataset: 数据集
        tokenizer: 分词器
        similarity_threshold: 相似度阈值

    Returns:
        增强的数据集
    """
    # 这个函数在实际应用中可能相当复杂
    # 这里给出一个简化实现

    # 收集所有问题
    questions = []
    for item in dataset:
        if "instruction" in item:
            questions.append(item["instruction"])
        elif "question" in item:
            questions.append(item["question"])

    # 简单的相似度计算（实际应用中可以使用更复杂的方法）
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    similarity_matrix = cosine_similarity(question_vectors)

    # 为每个问题找到相似但不完全相同的问题
    hard_negatives = {}
    for i, question in enumerate(questions):
        similar_indices = []
        for j, sim in enumerate(similarity_matrix[i]):
            if i != j and sim > similarity_threshold:
                similar_indices.append(j)

        if similar_indices:
            hard_negatives[i] = random.choice(similar_indices)

    # 返回增强的数据集
    return dataset, hard_negatives