import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import time
import os
import json

torch.manual_seed(42)
np.random.seed(42)

# 设置运行后端
C_DEV = torch.device("cuda")

# 设置路径
PATH = "../DataSpace/csi_cmri/"
FOLDER = "model_30km_joint/"

# 创建保存目录
os.makedirs(PATH + FOLDER, exist_ok=True)

# 加载数据
data = np.load(PATH + "CSI_channel_30km.npy")  # shape=(80000, 2560)

# 数据划分参数
TOTAL_SAMPLES = 80000
TRAIN_RATIO = 0.70    # 70% 训练
VAL_RATIO = 0.15      # 15% 验证  
TEST_RATIO = 0.15     # 15% 测试

# 设置量子线路参数
INPUT_DIM = data.shape[1]
OUTPUT_DIM = 128  # 修改为128
ORG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))
TAR_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))
ALL_QUBITS = ORG_QUBITS + 1
ANS_QUBITS = ORG_QUBITS - TAR_QUBITS

C_WEIGHT = torch.randn(INPUT_DIM, OUTPUT_DIM, requires_grad=True)
C_BIAS = torch.randn(1, OUTPUT_DIM, requires_grad=True)

LAYERS = 4
#DIST = [1] * LAYERS
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
Q_DEV = qml.device("lightning.gpu", wires=ALL_QUBITS)
@qml.qnode(Q_DEV, interface = "torch")
def cir(sample, c_weight, c_bias, q_weight=0):
    y = dense_layer(sample, c_weight, c_bias)
    frqi_encoder(qubits=TAR_QUBITS, params=y)
    strong_entangling_ansatz(q_weight)
    x = -sample
    frqi_encoder(qubits=ORG_QUBITS, params=x)    
    Ham = qml.Hamiltonian([-1], [qml.PauliZ(0)])
    return qml.expval(Ham)

#loss = cir(data[6], C_WEIGHT, C_BIAS, Q_WEIGHT)
#print(f"The loss is: {loss}")

# 数据划分
train_size = int(TOTAL_SAMPLES * TRAIN_RATIO)
val_size = int(TOTAL_SAMPLES * VAL_RATIO)
test_size = TOTAL_SAMPLES - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

print("=" * 60)
print("数据加载和预处理完成")
print("=" * 60)
print(f"训练数据: {len(train_data)} 条")
print(f"验证数据: {len(val_data)} 条")
print(f"测试数据: {len(test_data)} 条")
print(f"输入维度: {INPUT_DIM}")
print(f"输出维度: {OUTPUT_DIM}")  # 显示修改后的输出维度
print(f"量子比特数: {ALL_QUBITS} (原始: {ORG_QUBITS}, 目标: {TAR_QUBITS})")
print(f"量子层数: {LAYERS}")

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
EPOCHS = 10  # 修改为10个周期

# 训练循环
train_losses = []
val_losses = []

# 创建训练结果记录文件
results_file = PATH + FOLDER + 'training_results.txt'
progress_file = PATH + FOLDER + 'batch_progress.txt'  # 批次进度文件

with open(results_file, 'w') as f:
    f.write("训练开始时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    f.write(f"总周期数: {EPOCHS}\n")
    f.write(f"批次大小: {BATCH_SIZE}\n")
    f.write(f"输出维度: {OUTPUT_DIM}\n")
    f.write(f"训练数据量: {len(train_data)}\n")
    f.write(f"验证数据量: {len(val_data)}\n")
    f.write(f"测试数据量: {len(test_data)}\n")
    f.write("="*50 + "\n")

# 初始化批次进度文件
with open(progress_file, 'w') as f:
    f.write("批次训练进度记录\n")
    f.write("=" * 40 + "\n")
    f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"总周期数: {EPOCHS}\n")
    f.write(f"批次大小: {BATCH_SIZE}\n")
    f.write(f"每周期批次数: {(len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE}\n")
    f.write("=" * 40 + "\n\n")

print("\n" + "=" * 60)
print("开始训练...")
print("=" * 60)
print(f"总周期数: {EPOCHS}")
print(f"批次大小: {BATCH_SIZE}")
print(f"输出维度: {OUTPUT_DIM}")
total_batches = (len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"每周期批次数: {total_batches}")
print(f"初始学习率: {optimizer.param_groups[0]['lr']}")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    
    # 训练阶段
    epoch_train_loss = 0
    num_batches = 0
    
    # 随机打乱训练数据
    indices = np.random.permutation(len(train_data))
    
    print(f"[周期 {epoch+1}/{EPOCHS}] 开始训练...")
    
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
        epoch_train_loss += batch_loss.item()
        num_batches += 1
        
        # 每10个批次输出一次进度并保存到文件
        if num_batches % 10 == 0 or num_batches == total_batches:
            batch_time = time.time() - batch_start_time
            progress = (num_batches / total_batches) * 100
            avg_loss_so_far = epoch_train_loss / num_batches
            progress_info = (f"[周期 {epoch+1}/{EPOCHS}] 批次 {num_batches}/{total_batches} ({progress:.1f}%) - "
                           f"损失: {batch_loss.item():.6f} - 时间: {batch_time:.2f}s")
            
            print(f"  ↳ {progress_info}")
            
            # 保存进度信息到文件
            with open(progress_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {progress_info}\n")
        
        # 每250个批次保存一次模型
        if num_batches % 250 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'batch': num_batches,
                'C_WEIGHT': C_WEIGHT,
                'C_BIAS': C_BIAS,
                'Q_WEIGHT': Q_WEIGHT,
                'train_loss': epoch_train_loss / num_batches,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict()
            }
            checkpoint_file = f"{PATH}{FOLDER}checkpoint_epoch_{epoch+1}_batch_{num_batches}.pth"
            torch.save(checkpoint, checkpoint_file)
            print(f"  ↳ 模型检查点已保存: {checkpoint_file}")
            
            # 记录到进度文件
            with open(progress_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - 检查点保存: {checkpoint_file}\n")
    
    avg_train_loss = epoch_train_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # 验证阶段
    print(f"[周期 {epoch+1}/{EPOCHS}] 开始验证...")
    val_start_time = time.time()
    with torch.no_grad():
        val_loss = loss_fn(C_WEIGHT, C_BIAS, Q_WEIGHT, val_data).item()
        val_losses.append(val_loss)
    val_time = time.time() - val_start_time
    
    # 学习率调度
    scheduler.step()
    
    # 计算周期耗时
    epoch_time = time.time() - epoch_start_time
    total_elapsed_time = time.time() - start_time
    
    # 每个周期都记录训练结果到文件和控制台
    epoch_info = f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_time:.2f}s'
    
    print("=" * 50)
    print(f"周期 {epoch+1}/{EPOCHS} 完成!")
    print(f"  ↳ 训练损失: {avg_train_loss:.6f}")
    print(f"  ↳ 验证损失: {val_loss:.6f}")
    print(f"  ↳ 学习率: {scheduler.get_last_lr()[0]:.6f}")
    print(f"  ↳ 周期耗时: {epoch_time:.2f}s")
    print(f"  ↳ 验证耗时: {val_time:.2f}s")
    print(f"  ↳ 总耗时: {total_elapsed_time:.2f}s")
    
    # 显示改进情况
    if epoch > 0:
        train_improvement = train_losses[epoch-1] - avg_train_loss
        val_improvement = val_losses[epoch-1] - val_loss
        print(f"  ↳ 训练改进: {train_improvement:+.6f}")
        print(f"  ↳ 验证改进: {val_improvement:+.6f}")
    
    print("=" * 50)
    print()
    
    # 记录到文件
    with open(results_file, 'a') as f:
        f.write(epoch_info + "\n")
    
    # 同时保存每个周期的详细结果到JSON文件
    epoch_results = {
        'epoch': epoch + 1,
        'train_loss': float(avg_train_loss),
        'val_loss': float(val_loss),
        'learning_rate': float(scheduler.get_last_lr()[0]),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'epoch_time': float(epoch_time),
        'val_time': float(val_time),
        'total_elapsed_time': float(total_elapsed_time)
    }
    
    # 保存每个周期的详细结果到单独的JSON文件
    json_file = PATH + FOLDER + f'epoch_{epoch+1}_results.json'
    with open(json_file, 'w') as f:
        json.dump(epoch_results, f, indent=4)

end_time = time.time()
total_time = end_time - start_time

# 记录训练完成信息
completion_info = f"\n训练完成！总耗时: {total_time:.2f} 秒"
print("=" * 60)
print("训练完成!")
print("=" * 60)
print(completion_info)
print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"平均每周期耗时: {total_time/EPOCHS:.2f} 秒")

with open(results_file, 'a') as f:
    f.write(completion_info + "\n")

with open(progress_file, 'a') as f:
    f.write(f"\n训练完成于: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"总训练时间: {total_time:.2f} 秒\n")

# 测试阶段
print("\n" + "=" * 60)
print("开始测试...")
print("=" * 60)
test_start_time = time.time()
with torch.no_grad():
    test_loss = loss_fn(C_WEIGHT, C_BIAS, Q_WEIGHT, test_data).item()
test_time = time.time() - test_start_time
test_info = f"测试损失: {test_loss:.6f} (耗时: {test_time:.2f}s)"
print(test_info)

# 保存验证和测试结果到单独的文件
def save_validation_results():
    val_results = {
        'final_val_loss': float(val_losses[-1]),
        'all_val_losses': [float(x) for x in val_losses],
        'best_val_loss': float(min(val_losses)),
        'best_epoch': int(np.argmin(val_losses) + 1),
        'val_data_size': len(val_data),
        'evaluation_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(PATH + FOLDER + 'validation_results.json', 'w') as f:
        json.dump(val_results, f, indent=4)
    
    # 同时保存文本格式
    with open(PATH + FOLDER + 'validation_results.txt', 'w') as f:
        f.write("验证结果汇总\n")
        f.write("=" * 30 + "\n")
        f.write(f"最终验证损失: {val_losses[-1]:.6f}\n")
        f.write(f"最佳验证损失: {min(val_losses):.6f} (第 {np.argmin(val_losses) + 1} 周期)\n")
        f.write(f"验证数据量: {len(val_data)}\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("各周期验证损失:\n")
        for epoch, loss in enumerate(val_losses):
            f.write(f"周期 {epoch+1}: {loss:.6f}\n")

def save_test_results():
    test_results = {
        'test_loss': float(test_loss),
        'test_data_size': len(test_data),
        'evaluation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'evaluation_duration': float(test_time),
        'model_performance': {
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1]),
            'test_loss': float(test_loss)
        }
    }
    
    with open(PATH + FOLDER + 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # 同时保存文本格式
    with open(PATH + FOLDER + 'test_results.txt', 'w') as f:
        f.write("测试结果汇总\n")
        f.write("=" * 30 + "\n")
        f.write(f"测试损失: {test_loss:.6f}\n")
        f.write(f"测试耗时: {test_time:.2f}秒\n")
        f.write(f"测试数据量: {len(test_data)}\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("模型性能总结:\n")
        f.write(f"最终训练损失: {train_losses[-1]:.6f}\n")
        f.write(f"最终验证损失: {val_losses[-1]:.6f}\n")
        f.write(f"测试损失: {test_loss:.6f}\n")

# 保存验证和测试结果
save_validation_results()
save_test_results()

# 保存最终模型
def save_model():
    model_dict = {
        'C_WEIGHT': C_WEIGHT,
        'C_BIAS': C_BIAS,
        'Q_WEIGHT': Q_WEIGHT,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'output_dim': OUTPUT_DIM,
        'training_config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
    }
    
    # 保存最终模型
    torch.save(model_dict, PATH + FOLDER + 'hybrid_qnn_model_final.pth')
    print("\n最终模型已保存到 'hybrid_qnn_model_final.pth'")
    
    # 同时保存训练结果摘要
    summary = {
        'total_epochs': EPOCHS,
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'test_loss': float(test_loss),
        'best_val_loss': float(min(val_losses)),
        'best_epoch': int(np.argmin(val_losses) + 1),
        'total_training_time': float(total_time),
        'average_epoch_time': float(total_time/EPOCHS),
        'output_dim': OUTPUT_DIM,
        'completion_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'data_statistics': {
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'total_samples': TOTAL_SAMPLES
        }
    }
    
    with open(PATH + FOLDER + 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

save_model()

# 绘制损失曲线
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='训练损失', marker='o')
plt.plot(val_losses, label='验证损失', marker='s')
plt.xlabel('周期')
plt.ylabel('损失')
plt.title(f'训练和验证损失曲线 (输出维度: {OUTPUT_DIM})')
plt.legend()
plt.grid(True)
plt.savefig(PATH + FOLDER + 'training_loss.png')
print("损失曲线图已保存到 'training_loss.png'")

print("\n" + "=" * 60)
print("训练总结")
print("=" * 60)
print(f"最终训练损失: {train_losses[-1]:.6f}")
print(f"最终验证损失: {val_losses[-1]:.6f}")
print(f"测试损失: {test_loss:.6f}")
print(f"最佳验证损失: {min(val_losses):.6f} (第 {np.argmin(val_losses) + 1} 周期)")
print(f"输出维度: {OUTPUT_DIM}")

# 打印每个周期的详细结果
print("\n每个周期的详细结果:")
for epoch in range(EPOCHS):
    improvement = "" if epoch == 0 else f"(改进: {train_losses[epoch-1] - train_losses[epoch]:+.6f})"
    print(f"周期 {epoch+1}: 训练损失 = {train_losses[epoch]:.6f}, 验证损失 = {val_losses[epoch]:.6f} {improvement}")

# 记录最终总结到文件
with open(results_file, 'a') as f:
    f.write("\n训练总结:\n")
    f.write(f"最终训练损失: {train_losses[-1]:.6f}\n")
    f.write(f"最终验证损失: {val_losses[-1]:.6f}\n")
    f.write(f"测试损失: {test_loss:.6f}\n")
    f.write(f"最佳验证损失: {min(val_losses):.6f} (第 {np.argmin(val_losses) + 1} 周期)\n")
    f.write(f"输出维度: {OUTPUT_DIM}\n")
    f.write(f"总训练时间: {total_time:.2f} 秒\n")
    f.write("训练结果文件保存在: " + results_file + "\n")
    f.write(f"批次进度文件: {progress_file}\n")
    f.write(f"验证结果文件: {PATH + FOLDER}validation_results.json\n")
    f.write(f"测试结果文件: {PATH + FOLDER}test_results.json\n")

print(f"\n训练结果已保存到: {results_file}")
print(f"批次进度已保存到: {progress_file}")
print(f"验证结果已保存到: {PATH + FOLDER}validation_results.json")
print(f"测试结果已保存到: {PATH + FOLDER}test_results.json")
print(f"训练摘要已保存到: {PATH + FOLDER}training_summary.json")
print("\n所有文件保存完成!")
