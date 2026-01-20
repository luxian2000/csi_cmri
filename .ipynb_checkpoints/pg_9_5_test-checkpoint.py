import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import time
import os
import json

# 设置随机种子以保证结果可重现
# torch.manual_seed(42)
# np.random.seed(42)

# 设置路径
PATH = "../DataSpace/csi_cmri/"
LOCAL = "./"
FOLDER = "model_30km_9_5/"

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

# 验证和测试数据（固定）
VAL_DATA = data[56000:68000]  # 12000个验证样本
TEST_DATA = data[68000:80000] # 12000个测试样本

# 数据划分的起始索引
VAL_START_IDX = 56000
TEST_START_IDX = 68000

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

def load_final_model():
    """加载最终训练好的模型参数"""
    model_file = f"{LOCAL}{FOLDER}hybrid_qnn_model_final.pth"
    
    if not os.path.exists(model_file):
        print(f"错误: 模型文件 {model_file} 不存在!")
        return None, None, None, None
    
    try:
        print(f"正在加载模型: {model_file}")
        checkpoint = torch.load(model_file, map_location='cpu')
        
        # 初始化参数结构
        C_WEIGHT = torch.randn(INPUT_DIM, OUTPUT_DIM, requires_grad=False)
        C_BIAS = torch.randn(1, OUTPUT_DIM, requires_grad=False)
        LAYERS = 4
        Q_WEIGHT = torch.randn(LAYERS, ALL_QUBITS, 3, requires_grad=False)
        
        # 加载保存的参数
        if 'C_WEIGHT' in checkpoint:
            C_WEIGHT.data = checkpoint['C_WEIGHT'].data.clone()
            print(f"  已加载 C_WEIGHT, 形状: {C_WEIGHT.shape}")
        
        if 'C_BIAS' in checkpoint:
            C_BIAS.data = checkpoint['C_BIAS'].data.clone()
            print(f"  已加载 C_BIAS, 形状: {C_BIAS.shape}")
        
        if 'Q_WEIGHT' in checkpoint:
            Q_WEIGHT.data = checkpoint['Q_WEIGHT'].data.clone()
            print(f"  已加载 Q_WEIGHT, 形状: {Q_WEIGHT.shape}")
        
        history = checkpoint.get('training_history', {})
        print(f"  已加载训练历史，包含 {len(history.get('phase_history', []))} 个阶段")
        
        # 显示模型信息
        if 'all_phases' in checkpoint:
            print(f"  训练阶段: {checkpoint['all_phases']}")
        if 'completion_time' in checkpoint:
            print(f"  完成时间: {checkpoint['completion_time']}")
        
        return C_WEIGHT, C_BIAS, Q_WEIGHT, history
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None, None, None

def test_model_on_datasets(C_WEIGHT, C_BIAS, Q_WEIGHT, num_samples=10):
    """在验证集和测试集上测试模型"""
    print(f"\n{'='*60}")
    print(f"模型测试 - 分别从验证集和测试集抽取 {num_samples} 个样本")
    print(f"{'='*60}")
    
    # 从验证集随机选择样本
    val_indices_local = np.random.choice(len(VAL_DATA), num_samples, replace=False)
    val_samples = VAL_DATA[val_indices_local]
    # 计算在完整数据集中的绝对索引
    val_indices_global = val_indices_local + VAL_START_IDX
    
    # 从测试集随机选择样本
    test_indices_local = np.random.choice(len(TEST_DATA), num_samples, replace=False)
    test_samples = TEST_DATA[test_indices_local]
    # 计算在完整数据集中的绝对索引
    test_indices_global = test_indices_local + TEST_START_IDX
    
    print(f"验证集样本:")
    print(f"  局部索引 (在验证集中): {val_indices_local}")
    print(f"  全局索引 (在完整数据集中): {val_indices_global}")
    
    print(f"测试集样本:")
    print(f"  局部索引 (在测试集中): {test_indices_local}")
    print(f"  全局索引 (在完整数据集中): {test_indices_global}")
    
    print(f"验证集样本形状: {val_samples.shape}")
    print(f"测试集样本形状: {test_samples.shape}")
    
    # 计算验证集样本的损失
    print(f"\n{'='*60}")
    print("验证集样本的损失值:")
    print(f"{'='*60}")
    
    val_losses = []
    with torch.no_grad():
        for i, (local_idx, global_idx) in enumerate(zip(val_indices_local, val_indices_global)):
            sample = VAL_DATA[local_idx:local_idx+1]
            sample_loss = cir(sample[0], C_WEIGHT, C_BIAS, Q_WEIGHT).item()
            val_losses.append(sample_loss)
            print(f"验证样本 {i+1:2d} (局部索引 {local_idx:5d}, 全局索引 {global_idx:5d}): 损失 = {sample_loss:.6f}")
    
    # 计算测试集样本的损失
    print(f"\n{'='*60}")
    print("测试集样本的损失值:")
    print(f"{'='*60}")
    
    test_losses = []
    with torch.no_grad():
        for i, (local_idx, global_idx) in enumerate(zip(test_indices_local, test_indices_global)):
            sample = TEST_DATA[local_idx:local_idx+1]
            sample_loss = cir(sample[0], C_WEIGHT, C_BIAS, Q_WEIGHT).item()
            test_losses.append(sample_loss)
            print(f"测试样本 {i+1:2d} (局部索引 {local_idx:5d}, 全局索引 {global_idx:5d}): 损失 = {sample_loss:.6f}")
    
    # 计算统计信息
    val_avg_loss = np.mean(val_losses)
    val_std_loss = np.std(val_losses)
    test_avg_loss = np.mean(test_losses)
    test_std_loss = np.std(test_losses)
    
    print(f"\n{'='*60}")
    print("测试结果汇总:")
    print(f"{'='*60}")
    print(f"验证集样本数量: {num_samples}")
    print(f"验证集平均损失: {val_avg_loss:.6f}")
    print(f"验证集损失标准差: {val_std_loss:.6f}")
    print(f"验证集最小损失: {np.min(val_losses):.6f}")
    print(f"验证集最大损失: {np.max(val_losses):.6f}")
    
    print(f"\n测试集样本数量: {num_samples}")
    print(f"测试集平均损失: {test_avg_loss:.6f}")
    print(f"测试集损失标准差: {test_std_loss:.6f}")
    print(f"测试集最小损失: {np.min(test_losses):.6f}")
    print(f"测试集最大损失: {np.max(test_losses):.6f}")
    
    return {
        'val_losses': val_losses,
        'val_indices_local': val_indices_local,
        'val_indices_global': val_indices_global,
        'val_avg_loss': val_avg_loss,
        'test_losses': test_losses,
        'test_indices_local': test_indices_local,
        'test_indices_global': test_indices_global,
        'test_avg_loss': test_avg_loss
    }

def main():
    """主函数"""
    print("量子-经典混合模型测试")
    print("=" * 60)
    print(f"输入维度: {INPUT_DIM}")
    print(f"输出维度: {OUTPUT_DIM}")
    print(f"量子比特数: {ALL_QUBITS}")
    print(f"原始量子比特: {ORG_QUBITS}")
    print(f"目标量子比特: {TAR_QUBITS}")
    print(f"验证数据大小: {VAL_DATA.shape} (全局索引: {VAL_START_IDX} 到 {VAL_START_IDX + len(VAL_DATA) - 1})")
    print(f"测试数据大小: {TEST_DATA.shape} (全局索引: {TEST_START_IDX} 到 {TEST_START_IDX + len(TEST_DATA) - 1})")
    
    # 加载模型
    C_WEIGHT, C_BIAS, Q_WEIGHT, history = load_final_model()
    
    if C_WEIGHT is None:
        print("无法加载模型，程序退出。")
        return
    
    # 在验证集和测试集上测试模型
    results = test_model_on_datasets(C_WEIGHT, C_BIAS, Q_WEIGHT, num_samples=10)
    
    # 保存测试结果
    results_file = f"{LOCAL}{FOLDER}test_results_with_global_indices.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("量子-经典混合模型测试结果（验证集+测试集）\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"样本数量: 各10个（验证集+测试集）\n")
        f.write(f"模型文件: {LOCAL}{FOLDER}hybrid_qnn_model_final.pth\n")
        f.write(f"数据划分信息:\n")
        f.write(f"  训练集: 0 到 {TRAIN_SAMPLES-1}\n")
        f.write(f"  验证集: {VAL_START_IDX} 到 {VAL_START_IDX + len(VAL_DATA) - 1}\n")
        f.write(f"  测试集: {TEST_START_IDX} 到 {TEST_START_IDX + len(TEST_DATA) - 1}\n\n")
        
        f.write("验证集样本损失:\n")
        f.write("-" * 50 + "\n")
        for i, (local_idx, global_idx, loss) in enumerate(zip(results['val_indices_local'], results['val_indices_global'], results['val_losses'])):
            f.write(f"样本 {i+1:2d}: 局部索引 {local_idx:5d}, 全局索引 {global_idx:5d}, 损失 = {loss:.6f}\n")
        
        f.write(f"\n验证集汇总统计:\n")
        f.write("-" * 30 + "\n")
        f.write(f"平均损失: {results['val_avg_loss']:.6f}\n")
        f.write(f"损失标准差: {np.std(results['val_losses']):.6f}\n")
        f.write(f"最小损失: {np.min(results['val_losses']):.6f}\n")
        f.write(f"最大损失: {np.max(results['val_losses']):.6f}\n")
        
        f.write(f"\n测试集样本损失:\n")
        f.write("-" * 50 + "\n")
        for i, (local_idx, global_idx, loss) in enumerate(zip(results['test_indices_local'], results['test_indices_global'], results['test_losses'])):
            f.write(f"样本 {i+1:2d}: 局部索引 {local_idx:5d}, 全局索引 {global_idx:5d}, 损失 = {loss:.6f}\n")
        
        f.write(f"\n测试集汇总统计:\n")
        f.write("-" * 30 + "\n")
        f.write(f"平均损失: {results['test_avg_loss']:.6f}\n")
        f.write(f"损失标准差: {np.std(results['test_losses']):.6f}\n")
        f.write(f"最小损失: {np.min(results['test_losses']):.6f}\n")
        f.write(f"最大损失: {np.max(results['test_losses']):.6f}\n")
    
    print(f"\n测试结果已保存至: {results_file}")
    
    # 显示训练历史中的性能（如果可用）
    if history and 'phase_history' in history:
        print(f"\n{'='*60}")
        print("训练历史中的最终性能:")
        print(f"{'='*60}")
        
        final_phase = history['phase_history'][-1] if history['phase_history'] else {}
        if final_phase:
            print(f"最终训练阶段: {final_phase.get('phase_name', 'N/A')}")
            print(f"训练损失: {final_phase.get('train_loss', 'N/A'):.6f}")
            print(f"验证损失: {final_phase.get('val_loss', 'N/A'):.6f}")
            print(f"测试损失: {final_phase.get('test_loss', 'N/A'):.6f}")
            
            # 比较当前测试结果与训练时的结果
            print(f"\n当前测试结果 vs 训练时结果:")
            print(f"当前验证集平均损失: {results['val_avg_loss']:.6f}")
            print(f"当前测试集平均损失: {results['test_avg_loss']:.6f}")
            print(f"训练时验证损失: {final_phase.get('val_loss', 'N/A'):.6f}")
            print(f"训练时测试损失: {final_phase.get('test_loss', 'N/A'):.6f}")

if __name__ == "__main__":
    main()
