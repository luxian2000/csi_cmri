import numpy as np
import time
import os
import json
from typing import List, Tuple
import mindspore as ms
from mindspore import nn, ops, context, Tensor, Parameter
from mindspore.communication import init, get_rank, get_group_size
from mindspore.context import ParallelMode
import mindspore.dataset as ds
import sys
sys.path.append('../mindquantum')  # 添加mindquantum路径

from basic import *

# 设置随机种子
np.random.seed(42)
ms.set_seed(42)

# 设置Ascend环境（单卡模式，避免分布式训练的复杂性）
# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)

# 设置路径
PATH = "../DataSpace/csi_cmri/"
LOCAL = "./"
FOLDER = "model_30km_ascend/"

# 创建保存目录
os.makedirs(LOCAL + FOLDER, exist_ok=True)

# 加载数据
data = np.load(PATH + "CSI_channel_30km.npy")  # shape=(80000, 2560)

# 数据划分参数
TOTAL_SAMPLES = 80000
TRAIN_SAMPLES = 56000  # 56000个训练样本
VAL_SAMPLES = 12000    # 12000个验证样本
TEST_SAMPLES = 12000   # 12000个测试样本
VAL_SUBSET_SIZE = 2000 # 每个阶段验证使用的样本数
TEST_SUBSET_SIZE = 2000 # 每个阶段测试使用的样本数

# 设置量子线路参数
INPUT_DIM = data.shape[1]
OUTPUT_DIM = 128
ORG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))
TAR_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))
ALL_QUBITS = ORG_QUBITS + 1
ANS_QUBITS = ORG_QUBITS - TAR_QUBITS

# 训练阶段配置 - 7个阶段，每个阶段8000个样本，每个阶段1个周期
TRAINING_PHASES = [
    {"name": "phase_1", "start": 0, "end": 8000, "batches": 250, "batch_size": 32, "base_lr": 0.01},
    {"name": "phase_2", "start": 8000, "end": 16000, "batches": 250, "batch_size": 32, "base_lr": 0.008},
    {"name": "phase_3", "start": 16000, "end": 24000, "batches": 250, "batch_size": 32, "base_lr": 0.0064},
    {"name": "phase_4", "start": 24000, "end": 32000, "batches": 250, "batch_size": 32, "base_lr": 0.00512},
    {"name": "phase_5", "start": 32000, "end": 40000, "batches": 250, "batch_size": 32, "base_lr": 0.004096},
    {"name": "phase_6", "start": 40000, "end": 48000, "batches": 250, "batch_size": 32, "base_lr": 0.0032768},
    {"name": "phase_7", "start": 48000, "end": 56000, "batches": 250, "batch_size": 32, "base_lr": 0.00262144},
]

# 验证和测试数据（固定）
VAL_DATA = data[56000:68000]  # 12000个验证样本
TEST_DATA = data[68000:80000]  # 12000个测试样本

def save_training_config():
    """保存训练配置到文件"""
    config_file = LOCAL + FOLDER + 'training_config.txt'
    
    config_info = f"""
TRAINING CONFIGURATION
{'=' * 50}

1. DATA CONFIGURATION
{'=' * 20}
Total Samples: {TOTAL_SAMPLES}
Training Samples: {TRAIN_SAMPLES}
Validation Samples: {VAL_SAMPLES}
Test Samples: {TEST_SAMPLES}
Validation Subset per Phase: {VAL_SUBSET_SIZE}
Test Subset per Phase: {TEST_SUBSET_SIZE}
Input Dimension: {INPUT_DIM}
Output Dimension: {OUTPUT_DIM}

2. TRAINING PHASES
{'=' * 20}
Number of Phases: {len(TRAINING_PHASES)}
Samples per Phase: {TRAINING_PHASES[0]['end'] - TRAINING_PHASES[0]['start']}

Phase Details:
"""
    
    for i, phase in enumerate(TRAINING_PHASES):
        config_info += f"  Phase {i+1}: {phase['name']}\n"
        config_info += f"    Samples: {phase['start']} to {phase['end']} ({phase['end'] - phase['start']} samples)\n"
        config_info += f"    Batches: {phase['batches']}\n"
        config_info += f"    Batch Size: {phase['batch_size']}\n"
        config_info += f"    Total Samples: {phase['batches'] * phase['batch_size']}\n"
        config_info += f"    Base Learning Rate: {phase['base_lr']:.6f}\n\n"

    config_info += f"""
3. MODEL ARCHITECTURE
{'=' * 20}
Quantum Bits:
  Original Qubits (IMG_QUBITS): {ORG_QUBITS}
  Target Qubits (COM_QUBITS): {TAR_QUBITS}
  All Qubits: {ALL_QUBITS}
  Ansatz Qubits: {ANS_QUBITS}

Classical Neural Network:
  Input Dimension: {INPUT_DIM}
  Output Dimension: {OUTPUT_DIM}
  Weight Shape: ({INPUT_DIM}, {OUTPUT_DIM})
  Bias Shape: (1, {OUTPUT_DIM})

Quantum Circuit:
  Number of Layers: 4
  Quantum Weight Shape: (4, {ALL_QUBITS}, 3)

4. TRAINING PARAMETERS
{'=' * 20}
Optimizer: AdamWeightDecay with weight decay
Learning Rate Strategy: Cosine annealing within phase
Initial Learning Rate: {TRAINING_PHASES[0]['base_lr']}
Final Learning Rate: {TRAINING_PHASES[-1]['base_lr']:.6f}
Phase Learning Rate Decay: 0.8 (each phase is 0.8x previous phase)
Batch Learning Rate Scheduler: Cosine annealing (LR dynamically changes within phase)
Loss Function: Custom quantum-classical hybrid loss

5. INCREMENTAL TRAINING STRATEGY
{'=' * 20}
Training Type: Sequential Incremental Training
Parameter Loading: Each phase loads parameters from previous phase
Model Continuity: Continuous parameter updates across phases
Learning Rate Continuity: Each phase starts with reduced learning rate

6. GRADIENT IMPROVEMENTS
{'=' * 20}
Optimizer: AdamWeightDecay with weight decay (1e-4)
Learning Rate Scheduler: Cosine Annealing within each phase
Gradient Clipping: Max norm of 1.0
Early Stopping: Patience of 20 validation checks
Validation Frequency: Every 20 batches

7. VALIDATION AND TESTING
{'=' * 20}
Validation: Random {VAL_SUBSET_SIZE} samples from {VAL_SAMPLES} total
Testing: Random {TEST_SUBSET_SIZE} samples from {TEST_SAMPLES} total with changing random seed
Results Logging: After each phase completion

8. CHECKPOINT AND LOGGING
{'=' * 20}
Loss Logging: Every 10 batches
Model Checkpoint: Every 50 batches
Progress File: Each phase has separate progress file
Model Save: After each phase completion, before testing

9. HARDWARE CONFIGURATION
{'=' * 20}
Quantum Device: Custom implementation with Ascend NPU backend
Classical Device: Ascend NPU
Data Parallelism: Not enabled for simplified setup

{'=' * 50}
Configuration Created: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}
"""
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_info)
    
    print(f"Training configuration saved to: {config_file}")
    return config_info

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1.0 / (1.0 + ops.exp(-x))

def normlize(x):
    """归一化函数"""
    norm = ops.norm(x)
    return ops.select(norm == 0, x, x / norm)

def dense_layer(x, c_weight, c_bias):
    """经典全连接层"""
    output = ops.matmul(x, c_weight) + c_bias
    output = sigmoid(output)
    output = normlize(output[0])  # 确保输出是一维的
    return output

# 构造强纠缠层ansatz线路 (使用basic.py中的量子门)
def strong_entangling_ansatz(qubits, distance=[1] * 4, prefix='ansatz', angle=0):
    """构建强纠缠层ansatz线路"""
    # 存储量子门序列
    layers = len(distance)
    gates = []
    
    for i_layer in range(layers):
        """对于每一层，先构造单比特旋转门，再构造CNOT纠缠门"""
        for i_qubit in range(qubits):
            # RX, RY, RZ门将在construct函数中动态应用
            gates.append({
                'type': 'parametric',
                'layer': i_layer,
                'qubit': i_qubit,
                'name': f'{prefix}_l{i_layer}_q{i_qubit}_rx',
                'value': angle
            })
            gates.append({
                'type': 'parametric',
                'layer': i_layer,
                'qubit': i_qubit,
                'name': f'{prefix}_l{i_layer}_q{i_qubit}_ry',
                'value': angle
            })
            gates.append({
                'type': 'parametric',
                'layer': i_layer,
                'qubit': i_qubit,
                'name': f'{prefix}_l{i_layer}_q{i_qubit}_rz',
                'value': angle
            })
        
        for i_qubit in range(qubits):
            target_qubit = (i_qubit + 1) % qubits
            gates.append({
                'type': 'cnot',
                'control': i_qubit,
                'target': target_qubit
            })
    return gates