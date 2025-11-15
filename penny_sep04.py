import pennylane as qml
from pennylane import numpy as np
import time
import os

# 数据加载
data_30 = np.load('../DataSpace/csi_cmri/CSI_channel_30km.npy')  # shape=(80000, 2560)

INPUT_DIM = 2560
OUTPUT_DIM = 256

N_LAYERS = 4
IMG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))  # 12
COM_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))  # 8
ALL_QUBITS = IMG_QUBITS  # 12个量子比特

print(f"IMG_QUBITS: {IMG_QUBITS}, COM_QUBITS: {COM_QUBITS}, ALL_QUBITS: {ALL_QUBITS}")

# 初始化并保存经典神经网络参数
WEIGHT = np.random.randn(INPUT_DIM, OUTPUT_DIM) * 0.01
BIAS = np.random.randn(1, OUTPUT_DIM)

# 创建保存参数的目录
os.makedirs('model_parameters', exist_ok=True)

def save_initial_parameters():
    """保存初始化的参数"""
    np.save('model_parameters/initial_weight.npy', WEIGHT)
    np.save('model_parameters/initial_bias.npy', BIAS)
    print("Initial WEIGHT and BIAS saved!")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normlize(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm

def dense_layer(x):
    output = np.dot(x, WEIGHT) + BIAS
    output = sigmoid(output)
    output = normlize(output[0])  # 确保输出是一维的
    return output

coe = [-1]
obs_list = [qml.PauliZ(0)]
hamiltonian = qml.Hamiltonian(coe, observables=obs_list)

dev = qml.device('default.qubit', wires=ALL_QUBITS)

# 修复的量子电路 - 单个样本处理
@qml.qnode(dev)
def single_frqi_circuit(img_params, asz_params):
    ''' 单个样本的量子电路 '''
    # 压缩参数编码
    com_params = dense_layer(img_params)
    # 确保参数长度正确
    if len(com_params) < 2**COM_QUBITS:
        com_params_padded = np.pad(com_params, (0, 2**COM_QUBITS - len(com_params)))
    else:
        com_params_padded = com_params[:2**COM_QUBITS]
    qml.AmplitudeEmbedding(com_params_padded, wires=range(COM_QUBITS), pad_with=0.0, normalize=True)
    # 强纠缠层
    qml.StronglyEntanglingLayers(weights=asz_params, wires=range(ALL_QUBITS))
    # 图像参数编码
    img_params_norm = normlize(img_params)
    if len(img_params_norm) < 2**IMG_QUBITS:
        img_params_padded = np.pad(img_params_norm, (0, 2**IMG_QUBITS - len(img_params_norm)))
    else:
        img_params_padded = img_params_norm[:2**IMG_QUBITS]
    qml.AmplitudeEmbedding(img_params_padded, wires=range(IMG_QUBITS), pad_with=0.0, normalize=True)

    return qml.expval(hamiltonian)

# 批量处理函数 - 分别处理每个样本
def process_batch(img_batch, asz_params):
    ''' 处理批量的样本 '''
    batch_results = []
    for img_params in img_batch:
        result = single_frqi_circuit(img_params, asz_params)
        batch_results.append(result)
    return np.array(batch_results)

# 批量训练函数
def train_batch_version():
    try:
        # 保存初始参数
        save_initial_parameters()
        n_samples = 20  # 先用较少的样本测试
        samples = data_30[:n_samples]

        shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=ALL_QUBITS)
        weights = np.random.random(shape, requires_grad=True)
        # 保存初始量子权重
        np.save('model_parameters/initial_quantum_weights.npy', weights)
        print("Initial quantum weights saved!")

        opt = qml.GradientDescentOptimizer(stepsize=0.01)
        n_epochs = 5  # 先用较少的epoch测试
        batch_size = 5
        # 记录训练历史
        training_history = {
            'epoch_losses': [],
            'batch_losses': [],
            'weights_history': []
        }

        print("Starting training...")
        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for i in range(0, n_samples, batch_size):
                batch = samples[i:i+batch_size]
                
                def batch_cost(w):
                    results = process_batch(batch, w)
                    return np.mean(results)

                # 记录训练前的权重
                pre_update_weights = weights.copy()
                
                # 更新权重
                weights = opt.step(batch_cost, weights)
                current_loss = batch_cost(weights)
                epoch_loss += current_loss
                batch_count += 1

                # 记录批次信息
                training_history['batch_losses'].append({
                    'epoch': epoch,
                    'batch': i // batch_size,
                    'loss': float(current_loss),
                    'pre_weights_norm': float(np.linalg.norm(pre_update_weights)),
                    'post_weights_norm': float(np.linalg.norm(weights))
                })

                print(f"Epoch {epoch}, Batch {i//batch_size}: loss = {current_loss:.6f}")

            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                training_history['epoch_losses'].append({
                    'epoch': epoch,
                    'avg_loss': float(avg_epoch_loss)
                })
                # 保存每个epoch的权重
                epoch_weights = weights.copy()
                training_history['weights_history'].append(epoch_weights)
                np.save(f'model_parameters/quantum_weights_epoch_{epoch}.npy', epoch_weights)
                print(f"Epoch {epoch} completed: Average loss = {avg_epoch_loss:.6f}")
                print(f"Quantum weights for epoch {epoch} saved!")
                print("-" * 50)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds!")
        # 保存最终权重和训练历史
        np.save('model_parameters/final_quantum_weights.npy', weights)
        np.save('model_parameters/training_history.npy', training_history)
        print("Final quantum weights and training history saved!")
        return weights, training_history

    except Exception as e:
        print(f"Error in batch training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# 测试训练好的模型
def test_trained_model(weights, test_samples=5):
    """测试训练好的模型"""
    print("\nTesting trained model...")
    try:
        test_data = data_30[100:100+test_samples]
        results = process_batch(test_data, weights)
        print(f"Test results on {test_samples} samples:")
        for i, result in enumerate(results):
            print(f"  Sample {i}: {result:.6f}")
        avg_result = np.mean(results)
        print(f"Average test result: {avg_result:.6f}")
        return results
    except Exception as e:
        print(f"Error in testing: {e}")
        return None

# 主程序
if __name__ == "__main__":
    print("Starting quantum-classical hybrid model training...")
    print("=" * 60)
    # 先尝试批量训练
    final_weights, history = train_batch_version()
    # 如果批量训练失败，尝试单样本训练
    if final_weights is None:
        print("\nBatch training failed, trying single sample training...")
        final_weights, history = train_single_sample_version()
    if final_weights is not None:
        # 测试训练好的模型
        test_results = test_trained_model(final_weights)
        # 显示训练总结
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY:")
        print("=" * 60)
        print(f"Classical NN parameters saved:")
        print(f"  - WEIGHT shape: {WEIGHT.shape}")
        print(f"  - BIAS shape: {BIAS.shape}")
        print(f"Quantum circuit parameters saved:")
        print(f"  - Quantum weights shape: {final_weights.shape}")
        print(f"  - Number of epochs: {len(history['epoch_losses'])}")
        print(f"  - Final average loss: {history['epoch_losses'][-1]['avg_loss']:.6f}")
        # 显示保存的文件
        print(f"\nSaved files in 'model_parameters' directory:")
        saved_files = os.listdir('model_parameters')
        for file in sorted(saved_files):
            print(f"  - {file}")
    else:
        print("All training methods failed!")
