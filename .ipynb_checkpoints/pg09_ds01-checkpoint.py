import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import time
import os
import json
import pandas as pd
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

# 加载数据
PATH = "../DataSpace/csi_cmri/"
data = np.load(PATH + "CSI_channel_30km.npy")  # shape=(80000, 2560)
# 保存路径
FOLDER = "model_30km_joint/"
os.makedirs(PATH + FOLDER, exist_ok=True)

# 数据划分参数
TOTAL_SAMPLES = 80000
TRAIN_RATIO = 0.70    # 70% 训练
VAL_RATIO = 0.15      # 15% 验证  
TEST_RATIO = 0.15     # 15% 测试

# 设置量子线路参数
INPUT_DIM = data.shape[1]
OUTPUT_DIM = 256
ORG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))
TAR_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))
ALL_QUBITS = ORG_QUBITS + 1
ANS_QUBITS = ORG_QUBITS - TAR_QUBITS

C_WEIGHT = torch.randn(INPUT_DIM, OUTPUT_DIM, requires_grad=True)
C_BIAS = torch.randn(1, OUTPUT_DIM, requires_grad=True)

LAYERS = 4
DIST = [1, 2, 3, 4]
Q_WEIGHT = torch.randn(LAYERS, ALL_QUBITS, 3, requires_grad=True)

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
    for i in range(1, qubits+1):
        qml.Hadamard(wires=i)
    for index in range(min(2**qubits, len(params))):
        binary_str = bin(index)[2:].zfill(qubits)
        bits = [int(bit) for bit in binary_str]
        bits.reverse()
        qml.ctrl(qml.RY, control=range(1, qubits + 1), control_values=bits)(2*params[index], wires=target_wire)

# 构造强纠缠层ansatz线路
def strong_entangling_ansatz(q_weight, dist=DIST):
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
dev = qml.device("lightning.qubit", wires=ALL_QUBITS)
@qml.qnode(dev, interface = "torch")
def cir(sample, c_weight, c_bias, q_weight=0):
    y = dense_layer(sample, c_weight, c_bias)
    frqi_encoder(qubits=TAR_QUBITS, params=y)
    strong_entangling_ansatz(q_weight)
    x = -sample
    frqi_encoder(qubits=ORG_QUBITS, params=x)    
    Ham = qml.Hamiltonian([-1], [qml.PauliZ(0)])
    return qml.expval(Ham)

# 数据划分
train_size = int(TOTAL_SAMPLES * TRAIN_RATIO)
val_size = int(TOTAL_SAMPLES * VAL_RATIO)
test_size = TOTAL_SAMPLES - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

print(f"训练数据: {len(train_data)} 条")
print(f"验证数据: {len(val_data)} 条")
print(f"测试数据: {len(test_data)} 条")

# 定义损失函数
def loss_fn(c_weight, c_bias, q_weight, batch_data):
    total_loss = 0
    for sample in batch_data:
        loss = cir(sample, c_weight, c_bias, q_weight)
        total_loss += loss
    return total_loss / len(batch_data)

# 定义优化器
optimizer = torch.optim.Adam([C_WEIGHT, C_BIAS, Q_WEIGHT], lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

# 训练参数
BATCH_SIZE = 32
EPOCHS = 100

# 扩展的记录数据结构
training_log = {
    'metadata': {
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': TOTAL_SAMPLES,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'input_dim': INPUT_DIM,
        'output_dim': OUTPUT_DIM,
        'org_qubits': ORG_QUBITS,
        'tar_qubits': TAR_QUBITS,
        'all_qubits': ALL_QUBITS,
        'ans_qubits': ANS_QUBITS,
        'layers': LAYERS,
        'dist': DIST,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': 0.01
    },
    'epoch_details': [],
    'batch_details': [],
    'gradient_stats': [],
    'parameter_stats': [],
    'timing_info': []
}

# 参数统计函数
def get_parameter_stats(epoch, step="epoch_end"):
    stats = {
        'epoch': epoch,
        'step': step,
        'c_weight_mean': C_WEIGHT.mean().item(),
        'c_weight_std': C_WEIGHT.std().item(),
        'c_weight_grad_mean': C_WEIGHT.grad.mean().item() if C_WEIGHT.grad is not None else 0,
        'c_bias_mean': C_BIAS.mean().item(),
        'c_bias_std': C_BIAS.std().item(),
        'c_bias_grad_mean': C_BIAS.grad.mean().item() if C_BIAS.grad is not None else 0,
        'q_weight_mean': Q_WEIGHT.mean().item(),
        'q_weight_std': Q_WEIGHT.std().item(),
        'q_weight_grad_mean': Q_WEIGHT.grad.mean().item() if Q_WEIGHT.grad is not None else 0,
    }
    return stats

# 训练循环
train_losses = []
val_losses = []
batch_losses = []

print("开始训练...")
start_time = time.time()
global_batch_idx = 0

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    
    # 训练阶段
    epoch_train_loss = 0
    num_batches = 0
    
    # 随机打乱训练数据
    indices = np.random.permutation(len(train_data))
    
    for i in range(0, len(train_data), BATCH_SIZE):
        batch_start_time = time.time()
        batch_indices = indices[i:i+BATCH_SIZE]
        batch_samples = train_data[batch_indices]
        
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(C_WEIGHT, C_BIAS, Q_WEIGHT, batch_samples)
            loss.backward()
            return loss
        
        batch_loss = optimizer.step(closure)
        batch_loss_value = batch_loss.item()
        
        # 记录batch详细信息
        batch_duration = time.time() - batch_start_time
        batch_info = {
            'epoch': epoch,
            'batch_idx': global_batch_idx,
            'batch_loss': batch_loss_value,
            'batch_size': len(batch_samples),
            'duration': batch_duration,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        training_log['batch_details'].append(batch_info)
        batch_losses.append(batch_loss_value)
        
        # 记录梯度统计（每10个batch记录一次）
        if global_batch_idx % 10 == 0:
            grad_stats = get_parameter_stats(epoch, f"batch_{global_batch_idx}")
            training_log['gradient_stats'].append(grad_stats)
        
        epoch_train_loss += batch_loss_value
        num_batches += 1
        global_batch_idx += 1
    
    epoch_duration = time.time() - epoch_start_time
    avg_train_loss = epoch_train_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # 验证阶段
    val_start_time = time.time()
    with torch.no_grad():
        val_loss = loss_fn(C_WEIGHT, C_BIAS, Q_WEIGHT, val_data).item()
        val_losses.append(val_loss)
    val_duration = time.time() - val_start_time
    
    # 学习率调度
    scheduler.step()
    
    # 记录epoch详细信息
    epoch_info = {
        'epoch': epoch,
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
        'learning_rate': scheduler.get_last_lr()[0],
        'epoch_duration': epoch_duration,
        'val_duration': val_duration,
        'batches_processed': num_batches
    }
    training_log['epoch_details'].append(epoch_info)
    
    # 记录参数统计
    param_stats = get_parameter_stats(epoch)
    training_log['parameter_stats'].append(param_stats)
    
    # 打印训练进度
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_duration:.2f}s')

end_time = time.time()
total_training_time = end_time - start_time

# 记录总时间信息
training_log['metadata']['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
training_log['metadata']['total_training_time'] = total_training_time
training_log['timing_info'] = {
    'total_training_time': total_training_time,
    'average_epoch_time': total_training_time / EPOCHS,
    'average_batch_time': total_training_time / global_batch_idx
}

print(f"训练完成！总耗时: {total_training_time:.2f} 秒")

# 测试阶段
print("\n开始测试...")
test_start_time = time.time()
with torch.no_grad():
    test_loss = loss_fn(C_WEIGHT, C_BIAS, Q_WEIGHT, test_data).item()
test_duration = time.time() - test_start_time

# 记录测试结果
training_log['test_results'] = {
    'test_loss': test_loss,
    'test_duration': test_duration,
    'final_train_loss': train_losses[-1],
    'final_val_loss': val_losses[-1]
}

print(f"测试损失: {test_loss:.6f}, 测试耗时: {test_duration:.2f}秒")

# 保存模型和训练记录
def save_training_data():
    # 保存模型参数
    model_dict = {
        'C_WEIGHT': C_WEIGHT,
        'C_BIAS': C_BIAS,
        'Q_WEIGHT': Q_WEIGHT,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'batch_losses': batch_losses,
        'test_loss': test_loss
    }
    torch.save(model_dict, PATH + FOLDER + 'hybrid_qnn_model.pth')
    
    # 保存详细的训练日志
    with open(PATH + FOLDER + 'training_log.json', 'w') as f:
        # 转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.strftime("%Y-%m-%d %H:%M:%S")
            return obj
        
        json.dump(training_log, f, indent=2, default=convert_to_serializable)
    
    # 保存为CSV便于分析
    epoch_df = pd.DataFrame(training_log['epoch_details'])
    epoch_df.to_csv(PATH + FOLDER + 'epoch_metrics.csv', index=False)
    
    batch_df = pd.DataFrame(training_log['batch_details'])
    batch_df.to_csv(PATH + FOLDER + 'batch_metrics.csv', index=False)
    
    param_df = pd.DataFrame(training_log['parameter_stats'])
    param_df.to_csv(PATH + FOLDER + 'parameter_stats.csv', index=False)
    
    print("模型和训练记录已保存到", PATH + FOLDER)

save_training_data()

# 绘制损失曲线
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))

# 子图1: 训练和验证损失
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='训练损失', alpha=0.7)
plt.plot(val_losses, label='验证损失', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.title('训练和验证损失曲线')
plt.legend()
plt.grid(True)

# 子图2: Batch损失（平滑显示）
plt.subplot(2, 2, 2)
window_size = 50
if len(batch_losses) > window_size:
    smooth_losses = pd.Series(batch_losses).rolling(window=window_size).mean()
    plt.plot(smooth_losses, label=f'Batch损失 (平滑窗口={window_size})', alpha=0.7)
else:
    plt.plot(batch_losses, label='Batch损失', alpha=0.7)
plt.xlabel('Batch')
plt.ylabel('损失')
plt.title('Batch损失曲线')
plt.legend()
plt.grid(True)

# 子图3: 学习率变化
plt.subplot(2, 2, 3)
learning_rates = [epoch['learning_rate'] for epoch in training_log['epoch_details']]
plt.plot(learning_rates)
plt.xlabel('Epoch')
plt.ylabel('学习率')
plt.title('学习率变化')
plt.grid(True)

# 子图4: 训练时间
plt.subplot(2, 2, 4)
epoch_durations = [epoch['epoch_duration'] for epoch in training_log['epoch_details']]
plt.plot(epoch_durations)
plt.xlabel('Epoch')
plt.ylabel('时间 (秒)')
plt.title('每个Epoch的训练时间')
plt.grid(True)

plt.tight_layout()
plt.savefig(PATH + FOLDER + 'detailed_training_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 生成训练报告
print("\n" + "="*50)
print("训练总结报告")
print("="*50)
print(f"总训练时间: {total_training_time:.2f} 秒")
print(f"最终训练损失: {train_losses[-1]:.6f}")
print(f"最终验证损失: {val_losses[-1]:.6f}")
print(f"测试损失: {test_loss:.6f}")
print(f"平均每个epoch时间: {total_training_time/EPOCHS:.2f} 秒")
print(f"总batch数量: {global_batch_idx}")
print(f"数据保存位置: {PATH + FOLDER}")
print("="*50)
