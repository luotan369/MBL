# Task-Aware Meta-Balance-Learning for Better Generalizable Black-Box Adversarial Attack

This repository contains the official implementation of "Task-Aware Meta-Balance-Learning for Better Generalizable Black-Box Adversarial Attack".

## Environment Requirements

- **Operating System**: Linux
- **GPU**: NVIDIA GTX 3090 Ti (24GB) × 4
- **Python Version**: 3.9

## Installation

### 1. Create Conda Virtual Environment

Create a new conda environment with Python 3.9:
```bash
conda env create -f environment.yml
conda activate <your_env_name>
```

## Usage

### 1. Train Target/Surrogate Models

Navigate to the surrogate models directory and train your models:
```bash
cd surro_models
python target_trainer.py --model_name xxx
```

Replace `xxx` with your desired model name.

### 2. Dataset Preparation

Prepare your datasets using the following commands:

**For CIFAR-10:**
```bash
python data/data_prehandle.py --dataset=cifar10 --dataroot='data'
```

**For CIFAR-10-Tiny (BCC):**
```bash
python data/data_handle.py --dataset=cifar10 --dataroot='data'
```

### 3. Train MCG (Meta Conditional Generator)

The training process consists of two stages:

#### Pretrain Stage
```bash
bash scripts/cifar10_pretrain.sh
```

#### Meta-train Stage
```bash
bash scripts/cifar10_metatrain.sh
```

**Note**: You can enable or disable the **MGB (Meta Gradient Balancing)** module by modifying the corresponding shell script.

### 4. Attack Evaluation

Run the untargeted attack test:
```bash
bash scripts/cifar10_attack_untargeted.sh
```

## Project Structure
```
.
├── surro_models/           # Target and surrogate model training
├── data/                   # Dataset processing scripts
├── scripts/                # Training and evaluation scripts
│   ├── cifar10_pretrain.sh
│   ├── cifar10_metatrain.sh
│   └── cifar10_attack_untargeted.sh
└── environment.yml         # Conda environment configuration
```

## Citation

If you find this work useful in your research, please consider citing:

**Title**: Task-Aware Meta-Balance-Learning for Better Generalizable Black-Box Adversarial Attack

## Contact

For questions or issues, please open an issue on GitHub or contact luotan369@163.com.
