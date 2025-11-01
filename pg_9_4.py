import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import time
import os
import json

torch.manual_seed(42)
np.random.seed(42)

# 设置路径
PATH = "../DataSpace/csi_cmri/"
LOCAL = "./"
FOLDER = "model_30km_9_4/"

# 创建保存目录
os.makedirs(LOCAL + FOLDER, exist_ok=True)

# 加载数据
data = np.load(PATH + "CSI_channel_30km.npy")  # shape=(80000, 2560)

# 数据划分参数
TOTAL_SAMPLES = 80000
TRAIN_SAMPLES = 56000  # 56000个训练样本
VAL_SAMPLES = 12000    # 12000个验证样本
TEST_SAMPLES = 12000   # 12000个测试样本

# 设置量子线路参数
INPUT_DIM = data.shape[1]
OUTPUT_DIM = 128
ORG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))
TAR_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))
ALL_QUBITS = ORG_QUBITS + 1
ANS_QUBITS = ORG_QUBITS - TAR_QUBITS

# 训练阶段配置 - 7个阶段，每个阶段8000个样本，每个阶段1个周期
TRAINING_PHASES = [
    {"name": "phase_1", "start": 0, "end": 8000, "batches": 250, "batch_size": 32},
    {"name": "phase_2", "start": 8000, "end": 16000, "batches": 250, "batch_size": 32},
    {"name": "phase_3", "start": 16000, "end": 24000, "batches": 250, "batch_size": 32},
    {"name": "phase_4", "start": 24000, "end": 32000, "batches": 250, "batch_size": 32},
    {"name": "phase_5", "start": 32000, "end": 40000, "batches": 250, "batch_size": 32},
    {"name": "phase_6", "start": 40000, "end": 48000, "batches": 250, "batch_size": 32},
    {"name": "phase_7", "start": 48000, "end": 56000, "batches": 250, "batch_size": 32},
]

# 验证和测试数据（固定）
VAL_DATA = data[56000:68000]  # 12000个验证样本
TEST_DATA = data[68000:80000] # 12000个测试样本

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
        config_info += f"    Total Samples: {phase['batches'] * phase['batch_size']}\n\n"

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
Optimizer: Adam
Learning Rate: 0.01
Learning Rate Scheduler: StepLR (step_size=50, gamma=0.8)
Loss Function: Custom quantum-classical hybrid loss

5. CHECKPOINT AND LOGGING
{'=' * 20}
Loss Logging: Every 10 batches
Model Checkpoint: Every 50 batches
Progress File: Each phase has separate progress file
Model Save: After each phase completion

6. HARDWARE CONFIGURATION
{'=' * 20}
Quantum Device: lightning.gpu
Classical Device: CPU/GPU (auto)

{'=' * 50}
Configuration Created: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}
"""
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_info)
    
    print(f"Training configuration saved to: {config_file}")
    return config_info

def load_model_parameters(phase_name):
    """加载指定阶段的模型参数"""
    model_file = f"{LOCAL}{FOLDER}hybrid_qnn_model_{phase_name}.pth"
    
    if os.path.exists(model_file):
        print(f"Loading model parameters from {model_file}")
        checkpoint = torch.load(model_file, map_location='cpu')
        
        # 初始化参数
        C_WEIGHT = torch.randn(INPUT_DIM, OUTPUT_DIM, requires_grad=True)
        C_BIAS = torch.randn(1, OUTPUT_DIM, requires_grad=True)
        LAYERS = 4
        Q_WEIGHT = torch.randn(LAYERS, ALL_QUBITS, 3, requires_grad=True)
        
        # 加载保存的参数
        if 'C_WEIGHT' in checkpoint:
            C_WEIGHT.data = checkpoint['C_WEIGHT'].data.clone()
            C_WEIGHT.requires_grad_(True)
        if 'C_BIAS' in checkpoint:
            C_BIAS.data = checkpoint['C_BIAS'].data.clone()
            C_BIAS.requires_grad_(True)
        if 'Q_WEIGHT' in checkpoint:
            Q_WEIGHT.data = checkpoint['Q_WEIGHT'].data.clone()
            Q_WEIGHT.requires_grad_(True)
        
        return C_WEIGHT, C_BIAS, Q_WEIGHT, checkpoint.get('training_history', {})
    else:
        print(f"No existing model found for {phase_name}, initializing new parameters")
        C_WEIGHT = torch.randn(INPUT_DIM, OUTPUT_DIM, requires_grad=True)
        C_BIAS = torch.randn(1, OUTPUT_DIM, requires_grad=True)
        LAYERS = 4
        Q_WEIGHT = torch.randn(LAYERS, ALL_QUBITS, 3, requires_grad=True)
        
        return C_WEIGHT, C_BIAS, Q_WEIGHT, {}

def save_model_parameters(phase_name, C_WEIGHT, C_BIAS, Q_WEIGHT, training_history):
    """保存模型参数"""
    model_dict = {
        'C_WEIGHT': C_WEIGHT.clone().detach(),
        'C_BIAS': C_BIAS.clone().detach(),
        'Q_WEIGHT': Q_WEIGHT.clone().detach(),
        'training_history': training_history,
        'output_dim': OUTPUT_DIM,
        'phase_name': phase_name,
        'save_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    model_file = f"{LOCAL}{FOLDER}hybrid_qnn_model_{phase_name}.pth"
    torch.save(model_dict, model_file)
    print(f"Model saved to {model_file}")

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def normlize(x):
    norm = torch.norm(x)
    if norm == 0:
        return x
    return x / norm

def dense_layer(x, c_weight, c_bias):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    output = torch.matmul(x, c_weight) + c_bias
    output = sigmoid(output)
    output = normlize(output[0])  # 确保输出是一维的
    return output

# FRQI编码量子线路
def frqi_encoder(qubits, params, target_wire=0):
    # 对数据量子比特应用Hadamard门创建叠加态
    for i in range(1, qubits+1):
        qml.Hadamard(wires=i)
    # 使用受控PauliY旋转进行编码
    for index in range(min(2**qubits, len(params))):
        binary_str = bin(index)[2:].zfill(qubits)
        bits = [int(bit) for bit in binary_str]
        bits.reverse()
        qml.ctrl(qml.RY, control=range(1, qubits + 1), control_values=bits)(2*params[index], wires=target_wire)

# 构造强纠缠层ansatz线路
def strong_entangling_ansatz(q_weight, dist=[1, 2, 3, 4]):
    LAYERS = 4
    if q_weight.shape != (LAYERS, ALL_QUBITS, 3):
        print("Wrong q_weight!")
        return
    if len(dist) != LAYERS:
        print("Wrong distance!")
        return
    for i_layer in range(LAYERS):
        for i_qubit in range(ALL_QUBITS):
            qml.Rot(q_weight[i_layer, i_qubit, 0], q_weight[i_layer, i_qubit, 1], q_weight[i_layer, i_qubit, 2], wires=i_qubit)
        for i_qubit in range(ALL_QUBITS):
            target_qubit = (i_qubit + dist[i_layer]) % ALL_QUBITS
            qml.CNOT(wires=[i_qubit, target_qubit])

# 完整的经典-量子混合神经网络
Q_DEV = qml.device("lightning.gpu", wires=ALL_QUBITS)
@qml.qnode(Q_DEV, interface = "torch")
def cir(sample, c_weight, c_bias, q_weight):
    y = dense_layer(sample, c_weight, c_bias)
    frqi_encoder(qubits=TAR_QUBITS, params=y)
    strong_entangling_ansatz(q_weight)
    x = -sample
    frqi_encoder(qubits=ORG_QUBITS, params=x)    
    Ham = qml.Hamiltonian([-1], [qml.PauliZ(0)])
    return qml.expval(Ham)

# 定义损失函数
def loss_fn(c_weight, c_bias, q_weight, batch_data):
    total_loss = 0
    for sample in batch_data:
        loss = cir(sample, c_weight, c_bias, q_weight)
        total_loss += loss
    return total_loss / len(batch_data)

def train_phase(phase_config, previous_history=None):
    """训练单个阶段"""
    phase_name = phase_config["name"]
    start_idx = phase_config["start"]
    end_idx = phase_config["end"]
    total_batches = phase_config["batches"]
    batch_size = phase_config["batch_size"]
    
    print(f"\n{'='*60}")
    print(f"Starting Training Phase: {phase_name}")
    print(f"{'='*60}")
    print(f"Training samples: {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
    print(f"Batches: {total_batches}, Batch size: {batch_size}")
    print(f"Total training samples: {total_batches * batch_size}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载或初始化模型参数
    C_WEIGHT, C_BIAS, Q_WEIGHT, history = load_model_parameters(phase_name)
    
    # 合并历史记录
    if previous_history:
        if 'phase_history' not in history:
            history['phase_history'] = []
        history['phase_history'].extend(previous_history.get('phase_history', []))
    
    # 准备训练数据
    train_data = data[start_idx:end_idx]
    
    # 定义优化器
    optimizer = torch.optim.Adam([C_WEIGHT, C_BIAS, Q_WEIGHT], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    
    # 训练循环
    train_losses = []
    val_losses = []
    
    # 创建阶段特定的进度文件
    progress_file = LOCAL + FOLDER + f'batch_progress_{phase_name}.txt'
    with open(progress_file, 'w') as f:
        f.write(f"Training Progress - {phase_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples: {start_idx} to {end_idx} ({end_idx - start_idx} samples)\n")
        f.write(f"Batches: {total_batches}, Batch size: {batch_size}\n")
        f.write(f"Total training samples: {total_batches * batch_size}\n")
        f.write("=" * 50 + "\n\n")
    
    start_time = time.time()
    
    # 随机打乱当前阶段的训练数据
    indices = np.random.permutation(len(train_data))
    
    total_train_loss = 0
    
    print(f"[{phase_name}] Training started...")
    
    for batch_num in range(1, total_batches + 1):
        batch_start_time = time.time()
        
        # 选择批次样本
        batch_start = (batch_num - 1) * batch_size
        batch_end = batch_num * batch_size
        batch_indices = indices[batch_start:batch_end]
        batch_samples = train_data[batch_indices]
        
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(C_WEIGHT, C_BIAS, Q_WEIGHT, batch_samples)
            loss.backward()
            return loss
        
        batch_loss = optimizer.step(closure)
        total_train_loss += batch_loss.item()
        
        # 每10个批次记录一次损失函数
        if batch_num % 10 == 0 or batch_num == total_batches:
            batch_time = time.time() - batch_start_time
            progress = (batch_num / total_batches) * 100
            avg_loss_so_far = total_train_loss / batch_num
            
            progress_info = (f"[{phase_name}] Batch {batch_num}/{total_batches} "
                           f"({progress:.1f}%) - Loss: {batch_loss.item():.6f} - "
                           f"Avg Loss: {avg_loss_so_far:.6f} - Time: {batch_time:.2f}s")
            
            print(f"  ↳ {progress_info}")
            
            with open(progress_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {progress_info}\n")
        
        # 每50个批次记录一次模型参数
        if batch_num % 50 == 0:
            checkpoint = {
                'batch': batch_num,
                'C_WEIGHT': C_WEIGHT.clone().detach(),
                'C_BIAS': C_BIAS.clone().detach(),
                'Q_WEIGHT': Q_WEIGHT.clone().detach(),
                'current_loss': batch_loss.item(),
                'avg_loss': total_train_loss / batch_num,
                'phase': phase_name,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            checkpoint_file = f"{LOCAL}{FOLDER}checkpoint_{phase_name}_batch_{batch_num}.pth"
            torch.save(checkpoint, checkpoint_file)
            
            with open(progress_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Checkpoint saved: {checkpoint_file}\n")
    
    avg_train_loss = total_train_loss / total_batches
    train_losses.append(avg_train_loss)
    
    # 验证阶段
    print(f"[{phase_name}] Validating...")
    val_start_time = time.time()
    with torch.no_grad():
        val_loss = loss_fn(C_WEIGHT, C_BIAS, Q_WEIGHT, VAL_DATA).item()
        val_losses.append(val_loss)
    val_time = time.time() - val_start_time
    
    # 学习率调度
    scheduler.step()
    
    total_elapsed_time = time.time() - start_time
    
    phase_info = (f"[{phase_name}] Training completed: "
                 f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                 f"LR: {scheduler.get_last_lr()[0]:.6f}, "
                 f"Val Time: {val_time:.2f}s, Total Time: {total_elapsed_time:.2f}s")
    
    print(phase_info)
    
    with open(progress_file, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {phase_info}\n")
    
    # 测试阶段性能
    print(f"[{phase_name}] Testing...")
    test_start_time = time.time()
    with torch.no_grad():
        test_loss = loss_fn(C_WEIGHT, C_BIAS, Q_WEIGHT, TEST_DATA).item()
    test_time = time.time() - test_start_time
    
    # 保存阶段结果
    phase_history = {
        'phase_name': phase_name,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'total_time': total_elapsed_time,
        'test_time': test_time,
        'completion_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if 'phase_history' not in history:
        history['phase_history'] = []
    history['phase_history'].append(phase_history)
    
    history['current_test_loss'] = test_loss
    history['current_train_loss'] = avg_train_loss
    history['current_val_loss'] = val_loss
    
    # 保存模型
    save_model_parameters(phase_name, C_WEIGHT, C_BIAS, Q_WEIGHT, history)
    
    completion_info = (f"\n[{phase_name}] Training completed!\n"
                      f"  ↳ Final Train Loss: {avg_train_loss:.6f}\n"
                      f"  ↳ Final Val Loss: {val_loss:.6f}\n"
                      f"  ↳ Test Loss: {test_loss:.6f}\n"
                      f"  ↳ Total Time: {total_elapsed_time:.2f}s\n"
                      f"  ↳ Test Time: {test_time:.2f}s")
    
    print(completion_info)
    
    with open(progress_file, 'a') as f:
        f.write(f"\n{completion_info}\n")
        f.write(f"Phase completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return C_WEIGHT, C_BIAS, Q_WEIGHT, history

def resume_training(from_phase=None):
    """从指定阶段恢复训练"""
    if from_phase is None:
        # 查找最新的阶段
        existing_phases = []
        for phase in TRAINING_PHASES:
            model_file = f"{LOCAL}{FOLDER}hybrid_qnn_model_{phase['name']}.pth"
            if os.path.exists(model_file):
                existing_phases.append(phase['name'])
        
        if existing_phases:
            from_phase = existing_phases[-1]
            print(f"Resuming from latest completed phase: {from_phase}")
            
            # 找到下一个阶段
            next_index = None
            for i, phase in enumerate(TRAINING_PHASES):
                if phase['name'] == from_phase:
                    next_index = i + 1
                    break
            
            if next_index is not None and next_index < len(TRAINING_PHASES):
                from_phase = TRAINING_PHASES[next_index]['name']
                print(f"Starting next phase: {from_phase}")
            else:
                print("All phases completed!")
                return None, None, None, None
        else:
            from_phase = TRAINING_PHASES[0]['name']
            print(f"No existing phases found, starting from: {from_phase}")
    
    # 找到要开始的阶段
    start_index = 0
    for i, phase in enumerate(TRAINING_PHASES):
        if phase['name'] == from_phase:
            start_index = i
            break
    
    # 加载上一个阶段的历史（如果存在）
    previous_history = {}
    if start_index > 0:
        prev_phase = TRAINING_PHASES[start_index - 1]['name']
        prev_model_file = f"{LOCAL}{FOLDER}hybrid_qnn_model_{prev_phase}.pth"
        if os.path.exists(prev_model_file):
            checkpoint = torch.load(prev_model_file, map_location='cpu')
            previous_history = checkpoint.get('training_history', {})
    
    # 训练当前阶段
    phase = TRAINING_PHASES[start_index]
    C_WEIGHT, C_BIAS, Q_WEIGHT, history = train_phase(phase, previous_history)
    
    return C_WEIGHT, C_BIAS, Q_WEIGHT, history

def show_training_status():
    """显示训练状态"""
    print(f"\n{'='*60}")
    print("Training Status")
    print(f"{'='*60}")
    
    completed_phases = []
    for phase in TRAINING_PHASES:
        model_file = f"{LOCAL}{FOLDER}hybrid_qnn_model_{phase['name']}.pth"
        if os.path.exists(model_file):
            # 加载阶段信息显示性能
            checkpoint = torch.load(model_file, map_location='cpu')
            history = checkpoint.get('training_history', {})
            current_test_loss = history.get('current_test_loss', 'N/A')
            completed_phases.append((phase['name'], current_test_loss))
    
    print(f"Total phases: {len(TRAINING_PHASES)}")
    print(f"Completed phases: {len(completed_phases)}")
    print(f"Remaining phases: {len(TRAINING_PHASES) - len(completed_phases)}")
    
    if completed_phases:
        print(f"\nCompleted phases (Test Loss):")
        for phase_name, test_loss in completed_phases:
            if test_loss != 'N/A':
                print(f"  {phase_name}: {test_loss:.6f}")
            else:
                print(f"  {phase_name}: {test_loss}")
    
    if len(completed_phases) < len(TRAINING_PHASES):
        next_phase = TRAINING_PHASES[len(completed_phases)]['name']
        print(f"\nNext phase: {next_phase}")
    
    return [phase[0] for phase in completed_phases]

# 主程序
if __name__ == "__main__":
    print("Multi-Phase Quantum-Classical Hybrid Model Training")
    print("=" * 60)
    print(f"Total training samples: {TRAIN_SAMPLES}")
    print(f"Validation samples: {VAL_SAMPLES}")
    print(f"Test samples: {TEST_SAMPLES}")
    print(f"Number of phases: {len(TRAINING_PHASES)}")
    print(f"Samples per phase: {TRAINING_PHASES[0]['end'] - TRAINING_PHASES[0]['start']}")
    
    # 保存训练配置
    save_training_config()
    
    # 显示当前状态
    completed_phases = show_training_status()
    
    if len(completed_phases) == len(TRAINING_PHASES):
        print("\nAll training phases completed!")
        print("Use resume_training() to retrain any phase")
    else:
        # 自动继续训练
        print(f"\nAuto-resuming training...")
        C_WEIGHT, C_BIAS, Q_WEIGHT, history = resume_training()
        
        if C_WEIGHT is not None:
            # 保存最终模型
            final_model_dict = {
                'C_WEIGHT': C_WEIGHT.clone().detach(),
                'C_BIAS': C_BIAS.clone().detach(),
                'Q_WEIGHT': Q_WEIGHT.clone().detach(),
                'training_history': history,
                'output_dim': OUTPUT_DIM,
                'all_phases': [phase['name'] for phase in TRAINING_PHASES],
                'completion_time': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            torch.save(final_model_dict, LOCAL + FOLDER + 'hybrid_qnn_model_final.pth')
            print(f"\nFinal model saved to: {LOCAL}{FOLDER}hybrid_qnn_model_final.pth")
