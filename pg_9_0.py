import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import time
import os

torch.manual_seed(42)
np.random.seed(42)

# 设置运行后端
C_DEV = torch.device("cuda")

# 设置路径
PATH = "../DataSpace/csi_cmri/"
FOLDER = "model_30km_joint/"

# 加载数据
data = np.load(PATH + "CSI_channel_30km.npy")  # shape=(80000, 2560)

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

# 训练循环
train_losses = []
val_losses = []

print("开始训练...")
start_time = time.time()

for epoch in range(EPOCHS):
    # 训练阶段
    epoch_train_loss = 0
    num_batches = 0
    
    # 随机打乱训练数据
    indices = np.random.permutation(len(train_data))
    
    for i in range(0, len(train_data), BATCH_SIZE):
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
    
    avg_train_loss = epoch_train_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # 验证阶段
    with torch.no_grad():
        val_loss = loss_fn(C_WEIGHT, C_BIAS, Q_WEIGHT, val_data).item()
        val_losses.append(val_loss)
    
    # 学习率调度
    scheduler.step()
    
    # 打印训练进度
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')

end_time = time.time()
print(f"训练完成！总耗时: {end_time - start_time:.2f} 秒")

# 测试阶段
print("\n开始测试...")
with torch.no_grad():
    test_loss = loss_fn(C_WEIGHT, C_BIAS, Q_WEIGHT, test_data).item()
    print(f"测试损失: {test_loss:.6f}")

# 保存模型
def save_model():
    model_dict = {
        'C_WEIGHT': C_WEIGHT,
        'C_BIAS': C_BIAS,
        'Q_WEIGHT': Q_WEIGHT,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss
    }
    
    # 创建保存目录
    os.makedirs(PATH + FOLDER, exist_ok=True)
    
    # 保存模型
    torch.save(model_dict, PATH + FOLDER + 'hybrid_qnn_model.pth')
    print("模型已保存到 'hybrid_qnn_model.pth'")

save_model()

# 绘制损失曲线
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('训练和验证损失曲线')
plt.legend()
plt.grid(True)
plt.savefig(PATH + FOLDER + 'training_loss.png')
plt.show()

print("\n训练总结:")
print(f"最终训练损失: {train_losses[-1]:.6f}")
print(f"最终验证损失: {val_losses[-1]:.6f}")
print(f"测试损失: {test_loss:.6f}")
