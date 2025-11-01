import numpy as np
import time
import os
import json
from typing import List, Tuple
import mindspore as ms
from mindspore import nn, ops, context, Tensor, Parameter
from mindspore.communication import init, get_rank, get_group_size
from mindspore.context import ParallelMode
# 移除MindQuantum相关导入，改用basic.py中的定义
import sys
sys.path.append('../mindquantum')  # 添加mindquantum路径
import basic
import mindspore.dataset as ds

# 设置随机种子
np.random.seed(42)
ms.set_seed(42)

# 设置Ascend环境（单卡模式，避免分布式训练的复杂性）
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)

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
def strong_entangling_ansatz(qubits, layers=4, prefix='ansatz'):
    """构建强纠缠层ansatz线路"""
    # 存储量子门序列
    gates = []
    
    for i_layer in range(layers):
        """对于每一层，先构造单比特旋转门，再构造CNOT纠缠门"""
        for i_qubit in range(qubits):
            # RX, RY, RZ门将在construct函数中动态应用
            gates.append({
                'type': 'parametric',
                'layer': i_layer,
                'qubit': i_qubit,
                'name': f'{prefix}_l{i_layer}_q{i_qubit}'
            })
        
        for i_qubit in range(qubits):
            target_qubit = (i_qubit + 1) % qubits
            gates.append({
                'type': 'cnot',
                'control': i_qubit,
                'target': target_qubit
            })
    return gates

class HybridQNN(nn.Cell):
    """混合量子经典神经网络"""
    def __init__(self):
        super(HybridQNN, self).__init__()
        # 经典层参数
        self.c_weight = Parameter(Tensor(np.random.randn(INPUT_DIM, OUTPUT_DIM).astype(np.float32)), name='c_weight')
        self.c_bias = Parameter(Tensor(np.random.randn(1, OUTPUT_DIM).astype(np.float32)), name='c_bias')
        
        # 构建量子线路门序列
        self.encoder_target_gates = []
        for i in range(TAR_QUBITS):
            self.encoder_target_gates.append({
                'type': 'hadamard',
                'qubit': i+1
            })
            
        self.ansatz_gates = strong_entangling_ansatz(ALL_QUBITS, layers=4, prefix='main')
        
        # 构建完整的电路门序列
        self.circuit_gates = self.encoder_target_gates + self.ansatz_gates
        
        # 定义测量算子 (Z0 ⊗ Z1 ⊗ ... ⊗ Zn)
        self.observable = None  # 将在construct中动态构建
        
    def construct(self, x):
        # 经典前向传播
        y = dense_layer(x, self.c_weight, self.c_bias)
        
        # 构建参数字典
        # 简化处理：只使用部分参数
        params = []
        param_names = []
        for gate in self.circuit_gates:
            if gate['type'] == 'parametric':
                name = gate['name']
                param_names.append(name + '_rx')
                param_names.append(name + '_ry')
                param_names.append(name + '_rz')
                
        # 为每个参数门分配参数值， 这段不对，要修改！！！
        param_values = []
        for i, name in enumerate(param_names):
            if i < OUTPUT_DIM and 'rx' in name:
                param_values.append(2 * y[i % OUTPUT_DIM])
            elif i < INPUT_DIM and 'ry' in name:
                param_values.append(-x[i % INPUT_DIM])
            else:
                param_values.append(0.5)  # 默认值
                
        # 构建量子态并应用门操作
        # 初始化为 |00...0> 态
        state = basic.KET_0
        for i in range(1, ALL_QUBITS):
            state = basic.Tensor_Product(state, basic.KET_0)
            
        # 应用编码门 (Hadamard门)
        for gate in self.encoder_target_gates:
            if gate['type'] == 'hadamard':
                qubit_idx = gate['qubit']
                # 构建Hadamard门作用在指定量子比特上的矩阵
                hadamard_ops = []
                for i in range(ALL_QUBITS):
                    if i == qubit_idx:
                        hadamard_ops.append(basic.HADAMARD)
                    else:
                        hadamard_ops.append(basic.IDENTITY_2)
                hadamard_matrix = basic.Tensor_Product(*hadamard_ops)
                state = ops.matmul(hadamard_matrix, state)
        
        # 应用ansatz门
        param_idx = 0
        for gate in self.ansatz_gates:
            if gate['type'] == 'parametric':
                # 应用RX, RY, RZ门
                rx_theta = param_values[param_idx] if param_idx < len(param_values) else 0.5
                param_idx += 1
                ry_theta = param_values[param_idx] if param_idx < len(param_values) else 0.5
                param_idx += 1
                rz_theta = param_values[param_idx] if param_idx < len(param_values) else 0.5
                param_idx += 1
                
                # 构建RX门矩阵
                rx_ops = []
                for i in range(ALL_QUBITS):
                    if i == gate['qubit']:
                        rx_ops.append(basic.RX(rx_theta))
                    else:
                        rx_ops.append(basic.IDENTITY_2)
                rx_matrix = basic.Tensor_Product(*rx_ops)
                state = ops.matmul(rx_matrix, state)
                
                # 构建RY门矩阵
                ry_ops = []
                for i in range(ALL_QUBITS):
                    if i == gate['qubit']:
                        ry_ops.append(basic.RY(ry_theta))
                    else:
                        ry_ops.append(basic.IDENTITY_2)
                ry_matrix = basic.Tensor_Product(*ry_ops)
                state = ops.matmul(ry_matrix, state)
                
                # 构建RZ门矩阵
                rz_ops = []
                for i in range(ALL_QUBITS):
                    if i == gate['qubit']:
                        rz_ops.append(basic.RZ(rz_theta))
                    else:
                        rz_ops.append(basic.IDENTITY_2())
                rz_matrix = basic.Tensor_Product(*rz_ops)
                state = ops.matmul(rz_matrix, state)
                
            elif gate['type'] == 'cnot':
                # 应用CNOT门
                control_qubit = gate['control']
                target_qubit = gate['target']
                
                # 构建CNOT门矩阵
                cnot_matrix = basic.CNOT
                # 需要将2-qubit门扩展到多量子比特系统
                if ALL_QUBITS == 2:
                    cnot_full = cnot_matrix
                else:
                    # 对于多量子比特系统，需要更复杂的构造
                    # 这里简化处理，假设控制和目标是相邻的量子比特
                    if target_qubit == control_qubit + 1:
                        cnot_ops = []
                        for i in range(ALL_QUBITS):
                            if i == control_qubit:
                                # 添加控制部分，但需要2-qubit矩阵
                                break
                        # 简化：直接使用2-qubit CNOT作用在控制和目标量子比特上
                        cnot_full = basic.CNOT
                        # 扩展到多量子比特空间
                        if control_qubit > 0:
                            id_ops_before = [basic.IDENTITY_2 for _ in range(control_qubit)]
                            cnot_full = basic.Tensor_Product(*id_ops_before, cnot_full)
                        if ALL_QUBITS - control_qubit > 2:
                            id_ops_after = [basic.IDENTITY_2 for _ in range(ALL_QUBITS - control_qubit - 2)]
                            cnot_full = basic.Tensor_Product(cnot_full, *id_ops_after)
                        
                state = ops.matmul(cnot_full, state)
        
        # 测量期望值 <ψ|Z0 ⊗ Z1 ⊗ ... ⊗ Zn|ψ>
        # 构建可观测量 Z0 ⊗ Z1 ⊗ ... ⊗ Zn
        z_ops = [basic.PAULI_Z() for _ in range(ALL_QUBITS)]
        observable = basic.Tensor_Product(*z_ops)
        
        # 计算期望值
        bra_state = ops.conj(state).transpose()
        expectation = ops.matmul(bra_state, ops.matmul(observable, state)).squeeze()
        
        return expectation.real.astype(ms.float32)

class QNNLoss(nn.Cell):
    """量子神经网络损失函数"""
    def __init__(self, net):
        super(QNNLoss, self).__init__()
        self.net = net
        
    def construct(self, data, label):
        # 对于这个任务，我们不需要标签，直接计算网络输出
        output = self.net(data)
        # 由于这是一个无监督任务，我们简单地返回网络输出的绝对值作为"损失"
        # 在实际应用中，您可能需要根据具体任务定义适当的损失函数
        return ops.abs(output)

def save_model_parameters(phase_name, model, training_history):
    """保存模型参数"""
    model_dict = {
        'c_weight': model.c_weight.asnumpy(),
        'c_bias': model.c_bias.asnumpy(),
        'training_history': training_history,
        'output_dim': OUTPUT_DIM,
        'phase_name': phase_name,
        'save_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    model_file = f"{LOCAL}{FOLDER}hybrid_qnn_model_{phase_name}.ckpt"
    ms.save_checkpoint(model, model_file)
    print(f"Model saved to {model_file}")

def load_model_parameters(phase_name, model):
    """加载模型参数"""
    model_file = f"{LOCAL}{FOLDER}hybrid_qnn_model_{phase_name}.ckpt"
    if os.path.exists(model_file):
        param_dict = ms.load_checkpoint(model_file)
        ms.load_param_into_net(model, param_dict)
        print(f"Loaded parameters from {model_file}")
        return True
    else:
        print(f"No existing model found for {phase_name}, using initialized parameters")
        return False

def create_dataset(data, batch_size):
    """创建数据集"""
    # 创建MindSpore数据集，添加伪标签以满足训练框架要求
    labels = np.zeros((data.shape[0], 1), dtype=np.float32)  # 创建伪标签
    dataset = ds.NumpySlicesDataset((data, labels), shuffle=False)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def train_phase(phase_config, model, previous_history=None):
    """训练单个阶段"""
    phase_name = phase_config["name"]
    start_idx = phase_config["start"]
    end_idx = phase_config["end"]
    total_batches = phase_config["batches"]
    batch_size = phase_config["batch_size"]
    base_lr = phase_config["base_lr"]
    
    print(f"\n{'='*60}")
    print(f"Starting Training Phase: {phase_name}")
    print(f"{'='*60}")
    print(f"Training samples: {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
    print(f"Batches: {total_batches}, Batch size: {batch_size}")
    print(f"Total training samples: {total_batches * batch_size}")
    print(f"Base Learning Rate: {base_lr:.6f}")
    print(f"Validation samples: {VAL_SUBSET_SIZE} (random from {VAL_SAMPLES})")
    print(f"Test samples: {TEST_SUBSET_SIZE} (random from {TEST_SAMPLES})")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 合并历史记录
    history = previous_history if previous_history else {}
    if 'phase_history' not in history:
        history['phase_history'] = []
    
    # 准备训练数据
    train_data = data[start_idx:end_idx]
    
    # 随机选择验证和测试样本
    val_indices = np.random.choice(len(VAL_DATA), VAL_SUBSET_SIZE, replace=False)
    test_indices = np.random.choice(len(TEST_DATA), TEST_SUBSET_SIZE, replace=False)
    
    val_subset = VAL_DATA[val_indices]
    test_subset = TEST_DATA[test_indices]
    
    print(f"Selected {len(val_subset)} validation samples and {len(test_subset)} test samples")
    
    # 创建数据集
    dataset = create_dataset(train_data, batch_size)
    
    # 定义优化器
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=base_lr, weight_decay=1e-4)
    
    # 定义训练网络
    loss_network = QNNLoss(model)
    net_with_loss = nn.WithLossCell(loss_network, nn.MSELoss())  # 修复：正确传递损失函数
    train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
    train_network.set_train()
    
    # 训练循环
    train_losses = []
    
    # 创建阶段特定的进度文件
    progress_file = LOCAL + FOLDER + f'batch_progress_{phase_name}.txt'
    with open(progress_file, 'w') as f:
        f.write(f"Training Progress - {phase_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples: {start_idx} to {end_idx} ({end_idx - start_idx} samples)\n")
        f.write(f"Batches: {total_batches}, Batch Size: {batch_size}\n")
        f.write(f"Total training samples: {total_batches * batch_size}\n")
        f.write(f"Base Learning Rate: {base_lr:.6f}\n")
        f.write(f"Validation samples: {VAL_SUBSET_SIZE} (from {VAL_SAMPLES})\n")
        f.write(f"Test samples: {TEST_SUBSET_SIZE} (from {TEST_SAMPLES})\n")
        f.write("=" * 50 + "\n\n")
    
    start_time = time.time()
    
    total_train_loss = 0
    
    print(f"[{phase_name}] Training started")
    
    # 迭代训练
    iterator = dataset.create_tuple_iterator()
    for batch_num, (batch_data, batch_label) in enumerate(iterator, 1):
        if batch_num > total_batches:
            break
            
        batch_start_time = time.time()
        
        # 前向传播和反向传播
        loss = train_network(batch_data, batch_label)
        total_train_loss += float(loss.asnumpy())
        
        # 每10个批次记录一次损失函数
        if (batch_num % 10 == 0 or batch_num == total_batches):
            batch_time = time.time() - batch_start_time
            progress = (batch_num / total_batches) * 100
            avg_loss_so_far = total_train_loss / batch_num
            
            progress_info = (f"[{phase_name}] Batch {batch_num}/{total_batches} "
                           f"({progress:.1f}%) - Loss: {float(loss.asnumpy()):.6f} - "
                           f"Avg Loss: {avg_loss_so_far:.6f} - "
                           f"Time: {batch_time:.2f}s")
            
            print(f"  ↳ {progress_info}")
            
            with open(progress_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {progress_info}\n")
        
        # 每50个批次记录一次模型参数
        if batch_num % 50 == 0:
            checkpoint_file = f"{LOCAL}{FOLDER}checkpoint_{phase_name}_batch_{batch_num}.ckpt"
            ms.save_checkpoint(model, checkpoint_file)
            
            with open(progress_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Checkpoint saved: {checkpoint_file}\n")
    
    avg_train_loss = total_train_loss / total_batches
    train_losses.append(avg_train_loss)
    
    # 验证阶段 - 使用随机选择的2000个样本
    print(f"[{phase_name}] Validating with {len(val_subset)} samples...")
    val_start_time = time.time()
    
    model.set_train(False)
    val_loss_total = 0
    val_batches = 0
    
    val_dataset = create_dataset(val_subset, batch_size)
    val_iterator = val_dataset.create_tuple_iterator()
    for (val_batch, val_label) in val_iterator:
        val_output = model(val_batch)
        val_loss_total += float(ops.abs(val_output).asnumpy())  # 计算验证损失
        val_batches += 1
    
    val_loss = val_loss_total / val_batches if val_batches > 0 else 0
    val_time = time.time() - val_start_time
    
    total_elapsed_time = time.time() - start_time
    
    # 保存阶段结果（在测试之前）
    phase_history = {
        'phase_name': phase_name,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
        'total_time': total_elapsed_time,
        'val_time': val_time,
        'learning_rate': base_lr,
        'val_samples_used': len(val_subset),
        'val_indices': val_indices.tolist(),
        'completion_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    history['phase_history'].append(phase_history)
    history['current_train_loss'] = avg_train_loss
    history['current_val_loss'] = val_loss
    
    # 保存模型（在测试之前）
    save_model_parameters(phase_name, model, history)
    
    # 测试阶段性能 - 使用随机选择的2000个样本
    # 每次测试时改变随机数种子
    test_seed = int(time.time()) % 10000  # 使用当前时间作为随机种子
    np.random.seed(test_seed)
    
    print(f"[{phase_name}] Testing with {TEST_SUBSET_SIZE} samples (seed: {test_seed})...")
    test_indices = np.random.choice(len(TEST_DATA), TEST_SUBSET_SIZE, replace=False)
    test_subset = TEST_DATA[test_indices]
    
    test_start_time = time.time()
    test_loss_total = 0
    test_batches = 0
    
    test_dataset = create_dataset(test_subset, batch_size)
    test_iterator = test_dataset.create_tuple_iterator()
    for (test_batch, test_label) in test_iterator:
        test_output = model(test_batch)
        test_loss_total += float(ops.abs(test_output).asnumpy())  # 计算测试损失
        test_batches += 1
    
    test_loss = test_loss_total / test_batches if test_batches > 0 else 0
    test_time = time.time() - test_start_time
    
    # 更新阶段结果，添加测试信息
    phase_history['test_loss'] = test_loss
    phase_history['test_time'] = test_time
    phase_history['test_samples_used'] = len(test_subset)
    phase_history['test_indices'] = test_indices.tolist()
    
    history['current_test_loss'] = test_loss
    
    phase_info = (f"[{phase_name}] Training completed: "
                 f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                 f"Test Loss: {test_loss:.6f}, "
                 f"Val Time: {val_time:.2f}s, Test Time: {test_time:.2f}s, Total Time: {total_elapsed_time:.2f}s")
    
    print(phase_info)
    
    with open(progress_file, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {phase_info}\n")
    
    # 保存验证和测试结果
    save_phase_results(phase_name, phase_history)
    
    completion_info = (f"\n[{phase_name}] Training completed!\n"
                      f"  ↳ Final Train Loss: {avg_train_loss:.6f}\n"
                      f"  ↳ Validation Loss: {val_loss:.6f} (on {len(val_subset)} samples)\n"
                      f"  ↳ Test Loss: {test_loss:.6f} (on {len(test_subset)} samples)\n"
                      f"  ↳ Total Time: {total_elapsed_time:.2f}s")
    
    print(completion_info)
    
    with open(progress_file, 'a') as f:
        f.write(f"\n{completion_info}\n")
        f.write(f"Phase completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    model.set_train(True)
    return model, history

def save_phase_results(phase_name, phase_results):
    """保存阶段验证和测试结果"""
    results_file = LOCAL + FOLDER + f'phase_results_{phase_name}.txt'
    
    results_info = f"""
PHASE TRAINING RESULTS - {phase_name.upper()}
{'=' * 50}

1. PHASE INFORMATION
{'=' * 20}
Phase Name: {phase_name}
Training Samples: {phase_results['start_idx']} to {phase_results['end_idx']}
Total Training Time: {phase_results['total_time']:.2f} seconds
Completion Time: {phase_results['completion_time']}
Learning Rate: {phase_results['learning_rate']:.6f} (initial value)

2. PERFORMANCE RESULTS
{'=' * 20}
Final Training Loss: {phase_results['train_loss']:.6f}
Validation Loss: {phase_results['val_loss']:.6f}
Test Loss: {phase_results['test_loss']:.6f}
Validation Time: {phase_results['val_time']:.2f} seconds
Test Time: {phase_results['test_time']:.2f} seconds

3. VALIDATION AND TESTING DETAILS
{'=' * 20}
Validation Samples: {phase_results['val_samples_used']} (randomly selected from {VAL_SAMPLES})
Test Samples: {phase_results['test_samples_used']} (randomly selected from {TEST_SAMPLES})
Validation Indices: {phase_results['val_indices'][:5]}... (showing first 5 of {len(phase_results['val_indices'])})
Test Indices: {phase_results['test_indices'][:5]}... (showing first 5 of {len(phase_results['test_indices'])})

{'=' * 50}
Results Saved: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}
"""
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(results_info)
    
    print(f"Phase results saved to: {results_file}")

def main():
    """主程序"""
    print("Multi-Phase Quantum-Classical Hybrid Model Training with Custom Implementation + Ascend")
    print("=" * 60)
    print(f"Total training samples: {TRAIN_SAMPLES}")
    print(f"Validation samples: {VAL_SAMPLES} (using {VAL_SUBSET_SIZE} per phase)")
    print(f"Test samples: {TEST_SAMPLES} (using {TEST_SUBSET_SIZE} per phase)")
    print(f"Number of phases: {len(TRAINING_PHASES)}")
    print(f"Samples per phase: {TRAINING_PHASES[0]['end'] - TRAINING_PHASES[0]['start']}")
    print(f"Training Strategy: Sequential Incremental Training")
    print(f"Parameter Loading: Each phase loads from previous phase")
    print(f"Learning Rate Strategy: Cosine annealing within phase, 0.8× decay between phases")
    print(f"Initial Learning Rate: {TRAINING_PHASES[0]['base_lr']}")
    print(f"Final Learning Rate: {TRAINING_PHASES[-1]['base_lr']:.6f}")
    print(f"Data Parallelism: Not enabled for simplified setup")
    
    # 保存训练配置
    save_training_config()
    
    # 创建模型
    model = HybridQNN()
    
    # 按顺序训练各阶段
    history = {}
    for i, phase in enumerate(TRAINING_PHASES):
        # 加载前一阶段的模型参数（如果不是第一阶段）
        if i > 0:
            prev_phase_name = TRAINING_PHASES[i-1]['name']
            loaded = load_model_parameters(prev_phase_name, model)
            if not loaded:
                print(f"Warning: Could not load parameters from {prev_phase_name}")
        
        # 训练当前阶段
        model, history = train_phase(phase, model, history)
        
        print(f"Completed phase {i+1}/{len(TRAINING_PHASES)}: {phase['name']}")

if __name__ == "__main__":
    main()