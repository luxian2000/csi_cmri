import pennylane as qml
from pennylane import numpy as np

# 数据加载
data_30 = np.load('../DataSpace/csi_cmri/CSI_channel_30km.npy')  # shape=(80000, 2560)

INPUT_DIM = 2560
OUTPUT_DIM = 256

N_LAYERS = 4
IMG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))  # 12
COM_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))  # 8
ALL_QUBITS = IMG_QUBITS  # 12个量子比特

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
def batch_frqi_circuit(img_batch, asz_params):
    ''' 批量处理的量子电路 '''
    batch_results = []

    for img_params in img_batch:
        com_params = dense_layer(img_params)
        qml.AmplitudeEmbedding(com_params, wires=range(COM_QUBITS), pad_with=0)
        qml.StronglyEntanglingLayers(weights=asz_params, wires=range(ALL_QUBITS))
        img_params = normlize(img_params)
        qml.AmplitudeEmbedding(img_params, wires=range(IMG_QUBITS), pad_with=0)

        batch_results.append(qml.expval(hamiltonian))

    return batch_results

# 批量训练函数
def train_batch_version():
    try:
        n_samples = 100
        samples = data_30[:n_samples]

        shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=ALL_QUBITS)
        weights = np.random.random(shape, requires_grad=True)

        opt = qml.GradientDescentOptimizer(stepsize=0.01)
        n_epochs = 10
        batch_size = 10

        for epoch in range(n_epochs):
            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                batch = samples[i:i+batch_size]

                def batch_cost(w):
                    results = batch_frqi_circuit(batch, w)
                    return np.mean(results)

                weights = opt.step(batch_cost, weights)
                current_loss = batch_cost(weights)
                epoch_loss += current_loss

                print(f"Epoch {epoch}, Batch {i//batch_size}: loss = {current_loss:.6f}")

            avg_epoch_loss = epoch_loss / (n_samples // batch_size)
            print(f"Epoch {epoch} completed: Average loss = {avg_epoch_loss:.6f}")

    except Exception as e:
        print(f"Error in batch training: {e}")
        import traceback
        traceback.print_exc()


print("Starting standard training...")
train_batch_version()
