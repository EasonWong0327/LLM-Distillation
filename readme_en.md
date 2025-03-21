# Qwen Multi-Stage Knowledge Distillation Framework

A multi-stage knowledge distillation framework designed for Qwen models, specifically optimized for financial domain adaptation. This framework efficiently distills knowledge from Qwen-72B to Qwen-7B-instruct.

- **If** you want to compress models and save resources (e.g., for embedded devices, real-time inference):  
    Choose distillation, see this project.  

- **If** you want to adapt pre-trained models to new tasks with limited vertical domain data:  
    Choose fine-tuning, see https://github.com/EasonWong0327/LLM-Fine-tuning

## Features

- **Three-Stage Distillation Process**:
  - **Stage 1 - Basic Distillation**: Uses general knowledge data to help the student model acquire general capabilities from the teacher model
  - **Stage 2 - Domain Adaptive Distillation**: Gradually increases financial data proportion with multiple loss functions for smooth transition
  - **Stage 3 - Domain Fine-tuning**: Focuses on financial domain data to optimize domain-specific performance

- **Diverse Distillation Losses**:
  - KL Divergence Loss
  - Cross-Entropy Loss
  - Feature Distillation Loss
  - Attention Distillation Loss
  - Contrastive Learning Loss

- **Advanced Optimization Strategies**:
  - Gradient Balancing
  - Dynamic Data Ratio Adjustment
  - Evaluation-Driven Sampling
  - Stage-Specific Temperature Parameters

- **Flexible Configuration**:
  - LoRA Parameter-Efficient Fine-tuning Support
  - Mixed Precision Training
  - Configurable Evaluation Strategies

## Quick Start Guide

### 1. Distillation vs. Fine-tuning

- **Distillation**: Compresses knowledge from a large model into a smaller one, making it faster and more efficient (like condensing a teacher's knowledge into concise notes).
- **Fine-tuning**: Adapts a pre-trained model to a specific task (e.g., from general knowledge to finance), making it more specialized.

### 2. Four Steps of Distillation:
- **Find a Master**: Select a large model (teacher model) that excels in the target domain
- **Generate Questions**: Process all data through the master model and record its outputs (e.g., probability distributions)
- **Train Apprentice**: Use these outputs as "teaching materials" to train a smaller model (student model) to mimic the master's behavior
- **Optimize Size**: Further compress the small model (e.g., reduce layers, parameters) for efficiency

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (Parameter-Efficient Fine-Tuning)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python run_distillation.py --output_dir ./qwen_distilled_model --fp16 --use_lora
```

### Complete Parameters

```bash
python run_distillation.py \
  --teacher_model Qwen/Qwen-72B \
  --student_model Qwen/Qwen-7B-instruct \
  --output_dir ./qwen_finance_model \
  --general_dataset tatsu-lab/alpaca \
  --finance_dataset ./finance_data \
  --fp16 \
  --use_lora \
  --use_wandb \
  --start_stage 1 \
  --end_stage 3
```

### Start from Specific Stage

```bash
# Start from stage 2 using previous checkpoint
python run_distillation.py \
  --start_stage 2 \
  --checkpoint_path ./qwen_distilled_model/stage1_best \
  --output_dir ./qwen_continued_distillation
```

### Run Specific Stage Only

```bash
# Run only stage 3 (domain fine-tuning)
python run_distillation.py \
  --start_stage 3 \
  --end_stage 3 \
  --checkpoint_path ./path/to/previous_model \
  --output_dir ./qwen_finance_finetuned
```

## Project Structure

```
qwen_distillation/
│
├── config.py                  # Configuration classes and default parameters
├── run_distillation.py        # Main execution script
│
├── data/                      # Data processing module
│   ├── __init__.py
│   ├── dataset.py             # Dataset definitions
│   ├── data_manager.py        # Data manager
│   └── data_utils.py          # Utility functions
│
├── modeling/                  # Model-related modules
│   ├── __init__.py
│   ├── distiller.py           # Core distillation implementation
│   ├── losses.py              # Various loss functions
│   └── model_utils.py         # Model utility functions
```

## Stage Details

### Stage 1: Basic Distillation

- **Objective**: Help student model learn general knowledge distribution from teacher model
- **Data**: Primarily uses general domain dataset
- **Loss Function**: Mainly KL divergence loss with auxiliary cross-entropy loss
- **Temperature**: High (4.0) for smoother soft labels
- **Optimization**: Larger learning rate, simple training

### Stage 2: Domain Adaptive Distillation

- **Objective**: Smooth transition to financial domain, avoiding catastrophic forgetting
- **Data**: Mix of general and financial data, ratio increases from 30% to 70%
- **Loss Functions**:
  - KL Divergence Loss
  - Cross-Entropy Loss
  - Feature Distillation Loss
  - Attention Distillation Loss
  - Contrastive Learning Loss (optional)
- **Temperature**: Medium (2.0)
- **Optimization**: Gradient balancing to prevent domain shift

### Stage 3: Domain Fine-tuning

- **Objective**: Optimize financial domain performance
- **Data**: Uses only financial domain data
- **Loss Function**: Mainly cross-entropy loss with auxiliary KL divergence loss
- **Temperature**: Low (1.0) for focus on hard labels
- **Optimization**: Smaller learning rate, evaluation-driven sampling

## Results Evaluation

During and after training, the framework automatically generates quality check samples comparing student and teacher model outputs. These results are logged and displayed in the wandb dashboard if enabled.

## Hyperparameter Tuning Guide

1. **Stage 1**:
   - Larger batch_size (8-16)
   - Higher learning rate (5e-5)
   - Fewer epochs (2-3)

2. **Stage 2**:
   - Medium batch_size (6-8)
   - Medium learning rate (3e-5)
   - Experiment with different data ratio growth strategies

3. **Stage 3**:
   - Smaller batch_size (4)
   - Lower learning rate (1e-5)
   - More epochs (2-4)

## License

This project is licensed under the MIT License.

## Citation

If you use this code, please cite:
```bibtex
@misc{LLM-Distillation,
  author = {EasonWong0327},
  title = {LLM-Distillation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/EasonWong0327/LLM-Distillation}
}
``` 