import pennylane as qml
from pennylane import numpy as np

# 数据加载
data_30 = np.load('../DataSpace/csi_cmri/CSI_channel_30km.npy')  # shape=(80000, 2560)

INPUT_DIM = 2560
OUTPUT_DIM = 256

N_LAYERS = 4
IMG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))  # 应该是 12
COM_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))  # 应该是 8
ALL_QUBITS = IMG_QUBITS  # 总共 13 个量子比特

print(f"IMG_QUBITS: {IMG_QUBITS}, COM_QUBITS: {COM_QUBITS}, ALL_QUBITS: {ALL_QUBITS}")

WEIGHT = np.random.randn(INPUT_DIM, OUTPUT_DIM) * 0.01
BIAS = np.random.randn(1, OUTPUT_DIM)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normlize(x):
    norm = np.linalg.norm(x)
    x = x / norm
    return x

def dense_layer(x):
    output = np.dot(x, WEIGHT) + BIAS
    output = sigmoid(output)
    output = normlize(output)
    return output


coe = [-1]
obs_list = [qml.PauliZ(0)]
hamiltonian = qml.Hamiltonian(coe, observables=obs_list)

dev = qml.device('default.qubit', wires=ALL_QUBITS)


@qml.qnode(dev)
def frqi_circuit(img_params, asz_params):
    ''' construct the complete quantum circuit '''
    com_params = dense_layer(img_params)
    qml.AmplitudeEmbedding(com_params, wires=range(COM_QUBITS), pad_with=0)
    # 强纠缠层
    qml.StronglyEntanglingLayers(weights=asz_params, wires=range(ALL_QUBITS))
    img_params = normlize(img_params)
    qml.AmplitudeEmbedding(img_params, wires=range(IMG_QUBITS), pad_with=0)

    return qml.expval(hamiltonian)


# 测试单个样本
try:
    # 获取单个样本
    sample = data_30[0]
    print(f"Sample shape: {sample.shape}")
    # 预处理参数
    com_params = dense_layer(sample)[0]  # 确保维度正确
    img_params = sample
    print(f"com_params shape: {com_params.shape}")
    print(f"img_params shape: {img_params.shape}")
    # 初始化权重
    shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=ALL_QUBITS)
    weights = np.random.random(shape, requires_grad=True)
    print(f"Weights shape: {weights.shape}")
    # 测试电路执行
    result = frqi_circuit(img_params, weights)
    print(f"Circuit result: {result}")
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
