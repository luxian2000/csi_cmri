import pennylane as qml
from pennylane import numpy as np

data_30 = np.load('../DataSpace/csi_cmri/CSI_channel_30km.npy')  # shape=(80000, 2560)
# data_60 = np.load('../DataSpace/csi_cmri/CSI_channel_60km_new.npy')  # shape=(80000, 2560)

INPUT_DIM = 2560
OUTPUT_DIM = 256

N_LAYERS = 4
IMG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))
COM_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM))) 
ALL_QUBITS = IMG_QUBITS + 1

WEIGHT = np.random.randn(INPUT_DIM, OUTPUT_DIM) * 0.01
BIAS = np.random.randn(1, OUTPUT_DIM)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dense_layer(x):
    output = np.dot(x, WEIGHT) + BIAS
    output = sigmoid(output)
    return output


def frqi_embedding(qubits, params, target=0):
    ''' construct the FRQI encoding circuit '''
    length = min(len(params), 2**qubits)
    for index in range(length):
        binary_str = bin(index)[2:].zfill(qubits)
        bits = [int(bit) for bit in binary_str]  # 高位在左
        qml.ctrl(qml.RY, control=range(1, qubits + 1), control_values=bits)(params[index], wires=target)


coe = [-1]
obs_list = [qml.PauliZ(0)]
hamiltonian = qml.Hamiltonian(coe, observables=obs_list)

dev = qml.device('default.qubit', wires=ALL_QUBITS)


@qml.qnode(dev)
def frqi_circuit(com_params, img_params, asz_params):
    ''' construct the complete quantum circuit '''
    for i in range(1, ALL_QUBITS):
        qml.Hadamard(wires=i)

    frqi_embedding(COM_QUBITS, com_params)
    qml.StronglyEntanglingLayers(weights=asz_params, wires=range(ALL_QUBITS))
    frqi_embedding(IMG_QUBITS, img_params)

    for i in range(1, ALL_QUBITS):
        qml.Hadamard(wires=i)

    return qml.expval(hamiltonian)


shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=ALL_QUBITS)
weights = np.random.random(shape, requires_grad=True)

opt = qml.GradientDescentOptimizer()

for epoch in range(10):
    weights = opt.step(frqi_circuit, data_30[0], dense_layer(data_30[0]), weights)
    print(f" {epoch}-th round: asz_params are {weights}")
