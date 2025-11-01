import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, ops

def Matrix_Product(*matrices):
    """
    对多个矩阵进行连乘运算 (从左到右依次相乘)
    
    参数:
    *matrices: 可变数量的矩阵参数，每个矩阵都应该是MindSpore Tensor类型
    
    返回:
    Tensor: 所有输入矩阵从左到右连乘的结果
    
    异常:
    ValueError: 当没有输入矩阵或输入矩阵少于2个时抛出
    TypeError: 当输入参数不是Tensor类型时抛出
    """
    # 检查输入
    if len(matrices) == 0:
        raise ValueError("至少需要输入一个矩阵")
    
    if len(matrices) == 1:
        return matrices[0]
    
    # 验证所有输入都是Tensor类型
    for i, matrix in enumerate(matrices):
        if not isinstance(matrix, Tensor):
            raise TypeError(f"第{i+1}个参数必须是MindSpore Tensor类型，但得到了{type(matrix)}")
    
    # 依次进行矩阵乘法
    result = matrices[0]
    for i in range(1, len(matrices)):
        result = ops.matmul(result, matrices[i])
    
    return result

def Tensor_Product(*matrices):
    """
    计算多个矩阵的张量积（Kronecker积）
    
    参数:
    *matrices: 可变数量的矩阵参数，每个矩阵都应该是MindSpore Tensor类型
    
    返回:
    Tensor: 所有输入矩阵的张量积结果
    
    异常:
    ValueError: 当没有输入矩阵时抛出
    TypeError: 当输入参数不是Tensor类型时抛出
    """
    # 检查输入
    if len(matrices) == 0:
        raise ValueError("至少需要输入一个矩阵")
    
    # 验证所有输入都是Tensor类型
    for i, matrix in enumerate(matrices):
        if not isinstance(matrix, Tensor):
            raise TypeError(f"第{i+1}个参数必须是MindSpore Tensor类型，但得到了{type(matrix)}")
    
    # 如果只有一个矩阵，直接返回
    if len(matrices) == 1:
        return matrices[0]
    
    # 依次计算张量积
    result = matrices[0]
    for i in range(1, len(matrices)):
        result = ops.kron(result, matrices[i])
    
    return result

def Dagger(matrix):
    """
    计算矩阵的共轭转置（dagger操作）
    
    参数:
    matrix: 输入的复数矩阵 (Tensor类型)
    
    返回:
    Tensor: 输入矩阵的共轭转置
    """
    # 先转置矩阵，再取共轭
    return ops.conj(ops.transpose(matrix, (1, 0)))


# 量子比特基态 |0> 和 |1>
KET_0 = Tensor([[1.+0.j], [0.+0.j]], dtype=ms.complex64)  # |0>
KET_1 = Tensor([[0.+0.j], [1.+0.j]], dtype=ms.complex64)  # |1>
KET_PLUS = (KET_0 + KET_1) / mnp.sqrt(2)   # |+> = (|0> + |1>)/sqrt(2)
KET_MINUS = (KET_0 - KET_1) / mnp.sqrt(2)

# 对应的 bra 态 (行向量，ket的共轭转置)
BRA_0 = Dagger(KET_0)  # <0|
BRA_1 = Dagger(KET_1)  # <1|
BRA_PLUS = Dagger(KET_PLUS)   # <+|
BRA_MINUS = Dagger(KET_MINUS)

DENSITY_0 = Tensor([[1., 0.], 
                    [0., 0.]], dtype=ms.complex64)  # |0><0|

DENSITY_1 = Tensor([[0., 0.], 
                    [0., 1.]], dtype=ms.complex64)  # |1><1|

IDENTITY_2 = Tensor([[1., 0.], 
                     [0., 1.]], dtype=ms.complex64)  # 单位矩阵 I

def IDENTITY(n_qubits=1):
    """Identity gate for n qubits."""
    dim = 2 ** n_qubits
    return ops.eye(dim, dtype=ms.complex64)

def _PAULI_X(target_qubit=0):
    pauli_x = Tensor([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]], dtype=ms.complex64)
    if target_qubit > 0:
        pauli_x = ops.kron(IDENTITY(target_qubit), pauli_x)
    return pauli_x

def _PAULI_Y(target_qubit=0):
    pauli_y = Tensor([[0.+0.j, -1j], [1j, 0.+0.j]], dtype=ms.complex64)
    if target_qubit > 0:
        pauli_y = ops.kron(IDENTITY(target_qubit), pauli_y)
    return pauli_y

def _PAULI_Z(target_qubit=0):
    pauli_z = Tensor([[1.+0.j, 0.+0.j], [0.+0.j, -1.+0.j]], dtype=ms.complex64)
    if target_qubit > 0:
        pauli_z = ops.kron(IDENTITY(target_qubit), pauli_z)
    return pauli_z

def _HADAMARD(target_qubit=0):
    hadamard = Tensor([[1/mnp.sqrt(2), 1/mnp.sqrt(2)], [1/mnp.sqrt(2), -1/mnp.sqrt(2)]], dtype=ms.complex64)
    if target_qubit > 0:
        hadamard = ops.kron(IDENTITY(target_qubit), hadamard)
    return hadamard
'''
def _RX(theta, target_qubit=0):
    """Single qubit rotation around the X axis by angle theta."""
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    cos = ops.cos(theta / 2)
    sin = ops.sin(theta / 2)
    rx = Tensor([[cos, -1j * sin], [-1j * sin, cos]], dtype=ms.complex64)
    if target_qubit > 0:
        rx = ops.kron(IDENTITY(target_qubit), rx)
    return rx
'''
def _RX(theta, target_qubit=0):
    """Single qubit rotation around the X axis by angle theta."""
    # 确保 theta 是一个 Tensor
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)

    cos = ops.cos(theta / 2)
    sin = ops.sin(theta / 2)
    # 将 sin 和 cos 转换为 complex64 类型
    cos_c = ops.cast(cos, ms.complex64)
    sin_c = ops.cast(sin, ms.complex64)
    # 创建 -i*sin
    neg_i_sin = sin_c * Tensor(-1j, dtype=ms.complex64)

    # 修正：使用 ops.stack 构建 2x2 矩阵
    row1 = ops.stack([cos_c, neg_i_sin], axis=-1)  # Shape: [..., 2]
    row2 = ops.stack([neg_i_sin, cos_c], axis=-1)  # Shape: [..., 2]
    rx_matrix = ops.stack([row1, row2], axis=-2)   # Shape: [..., 2, 2]
    # ops.stack 会保留输入张量的批次维度（如果 theta 是标量，则无批次维度）
    # 如果 theta 是标量 [1.5]，则 rx_matrix 形状为 [1, 2, 2]
    # 如果 theta 是标量 1.5，则 rx_matrix 形状为 [2, 2]

    # 如果 theta 是标量，可能需要 squeeze 掉多余的维度
    if rx_matrix.ndim > 2 and rx_matrix.shape[0] == 1:
        rx_matrix = ops.squeeze(rx_matrix, axis=0)

    if target_qubit > 0:
        # 假设 IDENTITY(target_qubit) 返回一个合适的单位矩阵
        identity_block = IDENTITY(target_qubit) # 需要确保这个函数返回正确形状的复数单位矩阵
        rx_matrix = ops.kron(identity_block, rx_matrix)

    return rx_matrix

'''
def _RY(theta, target_qubit=0):
    """Single qubit rotation around the Y axis by angle theta."""
    cos = ops.cos(theta / 2)
    sin = ops.sin(theta / 2)
    ry = Tensor([[cos, -sin], [sin, cos]], dtype=ms.complex64)
    if target_qubit > 0:
        ry = ops.kron(IDENTITY(target_qubit), ry)
    return ry
'''

def _RY(theta, target_qubit=0):
    """Single qubit rotation around the Y axis by angle theta."""
    # 确保 theta 是一个 Tensor
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)

    cos = ops.cos(theta / 2)
    sin = ops.sin(theta / 2)
    # 将 sin 和 cos 转换为 complex64 类型
    cos_c = ops.cast(cos, ms.complex64)
    sin_c = ops.cast(sin, ms.complex64)

    # 创建 -sin(θ/2) 和 sin(θ/2)
    neg_sin_c = -sin_c
    sin_c_pos = sin_c # Just to make the matrix construction clear

    # 修正：使用 ops.stack 构建 2x2 矩阵
    # RY(θ) = [[cos(θ/2), -sin(θ/2)],
    #          [sin(θ/2),  cos(θ/2)]]
    row1 = ops.stack([cos_c, neg_sin_c], axis=-1)  # [cos(θ/2), -sin(θ/2)]
    row2 = ops.stack([sin_c_pos, cos_c], axis=-1)  # [sin(θ/2),  cos(θ/2)]
    ry_matrix = ops.stack([row1, row2], axis=-2)   # [[row1], [row2]]

    # 如果 theta 是标量， ops.stack 会保留维度，例如输入标量，输出形状为 [2, 2]
    # 如果 theta 是形状为 [N] 的张量，输出形状为 [N, 2, 2]
    # 如果 theta 是形状为 [1] 的张量，输出形状为 [1, 2, 2]，可能需要 squeeze
    # 但通常保留批次维度是更通用的做法，除非明确需要标量矩阵。
    # 如果输入是标量 1.5，则 theta 变成 [1.5] (shape [1])，导致 ry_matrix 为 [1, 2, 2]
    # 为了处理输入为标量的情况，可以检查并 squeeze
    if ry_matrix.ndim > 2 and ry_matrix.shape[0] == 1:
        ry_matrix = ops.squeeze(ry_matrix, axis=0)
    
    if target_qubit > 0:
        identity_block = IDENTITY(target_qubit)
        ry_matrix = ops.kron(identity_block, ry_matrix)

    return ry_matrix

def _RZ(theta, target_qubit=0):
    """Single qubit rotation around the Z axis by angle theta."""
    exp_neg = ops.exp(-1j * theta / 2)
    exp_pos = ops.exp(1j * theta / 2)
    rz = Tensor([[exp_neg, 0.+0.j], [0.+0.j, exp_pos]], dtype=ms.complex64)
    if target_qubit > 0:
        rz = ops.kron(IDENTITY(target_qubit), rz)
    return rz

def _S_GATE(target_qubit=0):
    s_gate = Tensor([[1.+0.j, 0.+0.j], 
                     [0.+0.j, 1j]], dtype=ms.complex64)
    if target_qubit > 0:
        s_gate = ops.kron(IDENTITY(target_qubit), s_gate)
    return s_gate

def _T_GATE(target_qubit=0):
    t_gate = Tensor([[1.+0.j, 0.+0.j], 
                     [0.+0.j, ops.exp(1j * np.pi / 4)]], dtype=ms.complex64)
    if target_qubit > 0:
        t_gate = ops.kron(IDENTITY(target_qubit), t_gate)
    return t_gate

def _CX(target_qubit, control_qubits, control_states):
    """
    受控X门（Controlled X gate）
    
    参数:
    target_qubit: 目标量子比特的索引
    control_qubitss: 控制量子比特的索引列表
    control_statess: 控制量子比特的状态列表（0或1），默认为None，表示所有控制比特都为1时触发
    
    返回:
    受控X门的矩阵表示
    """
    # 首先获取RX门矩阵
    rx_matrix = _PAULI_X() - IDENTITY_2
    
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    num_qubits = max(all_qubits) + 1
    
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    
    # 构建张量积矩阵
    # 对于每个量子比特，根据其角色选择相应的矩阵
    matrices = []
    for qubit_index in range(num_qubits):
        if qubit_index == target_qubit:
            # 目标量子比特使用RX矩阵
            matrices.append(rx_matrix)
        elif qubit_index in control_qubits:
            # 控制量子比特根据control_statess指定的状态选择投影算符
            control_index = control_qubits.index(qubit_index)
            if control_states[control_index] == 1:
                matrices.append(DENSITY_1)  # |1><1|
            else:
                matrices.append(DENSITY_0)  # |0><0|
        else:
            # 其他量子比特使用单位矩阵
            matrices.append(IDENTITY_2)
    
    # 计算所有矩阵的张量积
    result_matrix = matrices[0]
    for i in range(1, len(matrices)):
        result_matrix = ops.kron(result_matrix, matrices[i])
    result_matrix = IDENTITY(num_qubits) + result_matrix
    
    return result_matrix

def _CY(target_qubit, control_qubits, control_states):
    """
    受控Y门（Controlled Y gate）
    
    参数:
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1），默认为None，表示所有控制比特都为1时触发
    
    返回:
    受控Y门的矩阵表示
    """
    # 首先获取RX门矩阵
    ry_matrix = _PAULI_Y() - IDENTITY_2
    
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    num_qubits = max(all_qubits) + 1
    
    # 检查control_statess长度是否与control_qubitss匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    
    # 构建张量积矩阵
    # 对于每个量子比特，根据其角色选择相应的矩阵
    matrices = []
    for qubit_index in range(num_qubits):
        if qubit_index == target_qubit:
            # 目标量子比特使用RX矩阵
            matrices.append(ry_matrix)
        elif qubit_index in control_qubits:
            # 控制量子比特根据control_statess指定的状态选择投影算符
            control_index = control_qubits.index(qubit_index)
            if control_states[control_index] == 1:
                matrices.append(DENSITY_1)  # |1><1|
            else:
                matrices.append(DENSITY_0)  # |0><0|
        else:
            # 其他量子比特使用单位矩阵
            matrices.append(IDENTITY_2)
    
    # 计算所有矩阵的张量积
    result_matrix = matrices[0]
    for i in range(1, len(matrices)):
        result_matrix = ops.kron(result_matrix, matrices[i])
    result_matrix = IDENTITY(num_qubits) + result_matrix
    
    return result_matrix


def _CZ(target_qubit, control_qubits, control_states):
    """
    受控Z门（Controlled Z gate）
    
    参数:
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1），默认为None，表示所有控制比特都为1时触发
    
    返回:
    受控Y门的矩阵表示
    """
    # 首先获取RX门矩阵
    rz_matrix = _PAULI_Z() - IDENTITY_2
    
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    num_qubits = max(all_qubits) + 1
    
    # 检查control_statess长度是否与control_qubitss匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    
    # 构建张量积矩阵
    # 对于每个量子比特，根据其角色选择相应的矩阵
    matrices = []
    for qubit_index in range(num_qubits):
        if qubit_index == target_qubit:
            # 目标量子比特使用RX矩阵
            matrices.append(rz_matrix)
        elif qubit_index in control_qubits:
            # 控制量子比特根据control_statess指定的状态选择投影算符
            control_index = control_qubits.index(qubit_index)
            if control_states[control_index] == 1:
                matrices.append(DENSITY_1)  # |1><1|
            else:
                matrices.append(DENSITY_0)  # |0><0|
        else:
            # 其他量子比特使用单位矩阵
            matrices.append(IDENTITY_2)
    
    # 计算所有矩阵的张量积
    result_matrix = matrices[0]
    for i in range(1, len(matrices)):
        result_matrix = ops.kron(result_matrix, matrices[i])
    result_matrix = IDENTITY(num_qubits) + result_matrix
    
    return result_matrix
def _CRX(theta, target_qubit, control_qubits, control_states):
    """
    受控RX门（Controlled RX gate）
    
    参数:
    theta: RX旋转角度
    target_qubit: 目标量子比特的索引
    control_qubitss: 控制量子比特的索引列表
    control_statess: 控制量子比特的状态列表（0或1），默认为None，表示所有控制比特都为1时触发
    
    返回:
    受控RX门的矩阵表示
    """
    # 首先获取RX门矩阵
    rx_matrix = _RX(theta) - IDENTITY_2
    
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    num_qubits = max(all_qubits) + 1
    
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    
    # 构建张量积矩阵
    # 对于每个量子比特，根据其角色选择相应的矩阵
    matrices = []
    for qubit_index in range(num_qubits):
        if qubit_index == target_qubit:
            # 目标量子比特使用RX矩阵
            matrices.append(rx_matrix)
        elif qubit_index in control_qubits:
            # 控制量子比特根据control_statess指定的状态选择投影算符
            control_index = control_qubits.index(qubit_index)
            if control_states[control_index] == 1:
                matrices.append(DENSITY_1)  # |1><1|
            else:
                matrices.append(DENSITY_0)  # |0><0|
        else:
            # 其他量子比特使用单位矩阵
            matrices.append(IDENTITY_2)
    
    # 计算所有矩阵的张量积
    result_matrix = matrices[0]
    for i in range(1, len(matrices)):
        result_matrix = ops.kron(result_matrix, matrices[i])
    result_matrix = IDENTITY(num_qubits) + result_matrix
    
    return result_matrix

def _CRY(theta, target_qubit, control_qubits, control_states):
    """
    受控RY门（Controlled RY gate）
    
    参数:
    theta: RY旋转角度
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1），默认为None，表示所有控制比特都为1时触发
    
    返回:
    受控RY门的矩阵表示
    """
    # 首先获取RX门矩阵
    ry_matrix = _RY(theta) - IDENTITY_2
    
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    num_qubits = max(all_qubits) + 1
    
    # 检查control_statess长度是否与control_qubitss匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    
    # 构建张量积矩阵
    # 对于每个量子比特，根据其角色选择相应的矩阵
    matrices = []
    for qubit_index in range(num_qubits):
        if qubit_index == target_qubit:
            # 目标量子比特使用RX矩阵
            matrices.append(ry_matrix)
        elif qubit_index in control_qubits:
            # 控制量子比特根据control_statess指定的状态选择投影算符
            control_index = control_qubits.index(qubit_index)
            if control_states[control_index] == 1:
                matrices.append(DENSITY_1)  # |1><1|
            else:
                matrices.append(DENSITY_0)  # |0><0|
        else:
            # 其他量子比特使用单位矩阵
            matrices.append(IDENTITY_2)
    
    # 计算所有矩阵的张量积
    result_matrix = matrices[0]
    for i in range(1, len(matrices)):
        result_matrix = ops.kron(result_matrix, matrices[i])
    result_matrix = IDENTITY(num_qubits) + result_matrix
    
    return result_matrix

def _CRZ(theta, target_qubit, control_qubits, control_states):
    """
    受控RZ门（Controlled RZ gate）
    
    参数:
    theta: RZ旋转角度
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1），默认为None，表示所有控制比特都为1时触发
    
    返回:
    受控RY门的矩阵表示
    """
    # 首先获取RX门矩阵
    rz_matrix = _RZ(theta) - IDENTITY_2
    
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    num_qubits = max(all_qubits) + 1
    
    # 检查control_statess长度是否与control_qubitss匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    
    # 构建张量积矩阵
    # 对于每个量子比特，根据其角色选择相应的矩阵
    matrices = []
    for qubit_index in range(num_qubits):
        if qubit_index == target_qubit:
            # 目标量子比特使用RX矩阵
            matrices.append(rz_matrix)
        elif qubit_index in control_qubits:
            # 控制量子比特根据control_statess指定的状态选择投影算符
            control_index = control_qubits.index(qubit_index)
            if control_states[control_index] == 1:
                matrices.append(DENSITY_1)  # |1><1|
            else:
                matrices.append(DENSITY_0)  # |0><0|
        else:
            # 其他量子比特使用单位矩阵
            matrices.append(IDENTITY_2)
    
    # 计算所有矩阵的张量积
    result_matrix = matrices[0]
    for i in range(1, len(matrices)):
        result_matrix = ops.kron(result_matrix, matrices[i])
    result_matrix = IDENTITY(num_qubits) + result_matrix
    
    return result_matrix

def _SWAP(qubit_1=0, qubit_2=1):
    return Matrix_Product(_CX(qubit_1, qubit_2), _CX(qubit_2, qubit_1), _CX(qubit_1, qubit_2))

def _TOFFOLI(target_qubit=2, control_qubits=[0,1]):
    num_qubits = max(target_qubit, max(control_qubits)) + 1
    matrices_0 = [IDENTITY_2] * num_qubits
    matrices_0[control_qubits[0]] = DENSITY_1
    matrices_0[control_qubits[1]] = DENSITY_1
    matrices_0[target_qubit] = _PAULI_X()
    result_0 = matrices_0[0]
    for i in range(1, num_qubits):
        result_0 = ops.kron(result_0, matrices_0[i])
    matrices_1 = [IDENTITY_2] * num_qubits
    matrices_1[control_qubits[0]] = DENSITY_0
    result_1 = matrices_1[0]
    for i in range(1, num_qubits):
        result_1 = ops.kron(result_1, matrices_1[i])
    matrices_2 = [IDENTITY_2] * num_qubits
    matrices_2[control_qubits[0]] = DENSITY_1
    matrices_2[control_qubits[1]] = DENSITY_0
    result_2 = matrices_2[0]
    for i in range(1, num_qubits):
        result_2 = ops.kron(result_2, matrices_2[i])
    return result_0 + result_1 + result_2

def _U3(theta, phi, lam, target_qubit=0):
    """General single-qubit rotation gate U3."""
    cos = ops.cos(theta / 2)
    sin = ops.sin(theta / 2)
    exp_iphi = ops.exp(1j * phi)
    exp_ilam = ops.exp(1j * lam)
    exp_iphi_lam = ops.exp(1j * (phi + lam))
    u3 = Tensor([[cos, -exp_ilam * sin], [exp_iphi * sin, exp_iphi_lam * cos]], dtype=ms.complex64)
    if target_qubit > 0:
        u3 = ops.kron(IDENTITY(target_qubit), u3)
    return u3

def _U2(phi, lam, target_qubit=0):
    """Single-qubit rotation gate U2."""
    return _U3(np.pi/2, phi, lam, target_qubit)

def RZZ(theta):
    """
    RZZ门（控制Z旋转门）
    作用在两个量子比特上，实现条件相位旋转
    
    参数:
    theta: 旋转角度
    
    返回:
    Tensor: 4x4的RZZ门矩阵
    """
    # RZZ门的矩阵形式:
    # [[exp(-1j*theta/2), 0, 0, 0],
    #  [0, exp(1j*theta/2), 0, 0],
    #  [0, 0, exp(1j*theta/2), 0],u
    #  [0, 0, 0, exp(-1j*theta/2)]]
    
    exp_neg = ops.exp(-1j * theta / 2)
    exp_pos = ops.exp(1j * theta / 2)
    
    return Tensor([[exp_neg, 0.+0.j, 0.+0.j, 0.+0.j],
                   [0.+0.j, exp_pos, 0.+0.j, 0.+0.j],
                   [0.+0.j, 0.+0.j, exp_pos, 0.+0.j],
                   [0.+0.j, 0.+0.j, 0.+0.j, exp_neg]], dtype=ms.complex64)

def Gate_To_Matrix(gate, cir_qubits=1):
    """
    将单个门的信息转换为矩阵
    
    参数:
    gate: 包含门信息的字典
    num_qubits: 总量子比特数
    
    返回:
    Tensor: 门的矩阵表示
    
    支持的门类型和参数格式:
    
    单量子比特门:
    - {'type': 'PAULI_X' or 'X', 'target_qubit': int, 'parameter': None}
    - {'type': 'PAULI_Y' or 'Y', 'target_qubit': int, 'parameter': None}
    - {'type': 'PAULI_Z' or 'Z', 'target_qubit': int, 'parameter': None}
    - {'type': 'HADAMARD' or 'H', 'target_qubit': int, 'parameter': None}
    - {'type': 'S_GATE' or 'S', 'target_qubit': int, 'parameter': None}
    - {'type': 'T_GATE' or 'T', 'target_qubit': int, 'parameter': None}
    - {'type': 'RX', 'target_qubit': int, 'parameter': float}  # 旋转角度
    - {'type': 'RY', 'target_qubit': int, 'parameter': float}  # 旋转角度
    - {'type': 'RZ', 'target_qubit': int, 'parameter': float}  # 旋转角度
    - {'type': 'U3', 'target_qubit': int, 'parameter': [theta, phi, lam]}  # 三个角度参数
    - {'type': 'U2', 'target_qubit': int, 'parameter': [phi, lam]}  # 两个角度参数
    
    两量子比特门:
    - {'type': 'CNOT' or 'CX', 'target_qubit': int, 'control_qubit': int, 'parameter': None}
    - {'type': 'CZ', 'target_qubit': int, 'control_qubit': int, 'parameter': None}
    - {'type': 'CY', 'target_qubit': int, 'control_qubit': int, 'parameter': None}
    - {'type': 'SWAP', 'qubit_1': int, 'qubit_2': int, 'parameter': None}
    
    三量子比特门:
    - {'type': 'TOFFOLI', 'target_qubit': int, 'control_qubits: [int, int], 'parameter': None}
    
    恒等门:
    - {'type': 'IDENTITY' or 'I', 'num_qubits': int}
    """
    gate_type = gate['type']
    gate_parameter = gate.get('parameter', None)
    
    # 单量子比特门：
    if gate_type in ['PAULI_X', 'X']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _PAULI_X(gate['target_qubit'])
    elif gate_type in ['PAULI_Y', 'Y']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _PAULI_Y(gate['target_qubit'])
    elif gate_type in ['PAULI_Z', 'Z']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _PAULI_Z(gate['target_qubit'])
    elif gate_type in ['HADAMARD', 'H']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _HADAMARD(gate['target_qubit'])
    elif gate_type in ['S_GATE', 'S']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _S_GATE(gate['target_qubit'])
    elif gate_type in ['T_GATE', 'T']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _T_GATE(gate['target_qubit'])
    elif gate_type == 'RX':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _RX(gate_parameter, gate['target_qubit'])
    elif gate_type == 'RY':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _RY(gate_parameter, gate['target_qubit'])
    elif gate_type == 'RZ':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _RZ(gate_parameter, gate['target_qubit'])
    elif gate_type == 'U3':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _U3(gate_parameter[0], gate_parameter[1], gate_parameter[2], gate['target_qubit'])
    elif gate_type == 'U2':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _U2(gate_parameter[0], gate_parameter[1],  gate['target_qubit'])
    # 两量子比特门：
    elif gate_type in ['CNOT', 'CX']:
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _CX(target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'], control_states=gate['control_states'])
    elif gate_type == 'CY':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _CY(target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'], control_states=gate['control_states'])
    elif gate_type == 'CZ':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _CZ(target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'], control_states=gate['control_states'])
    elif gate_type == 'CRX':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _CRX(gate_parameter, target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'], 
                           control_states=gate['control_states'])
    elif gate_type == 'CRY':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _CRY(gate_parameter, target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'], 
                           control_states=gate['control_states'])
    elif gate_type == 'CRZ':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _CRZ(gate_parameter, target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'], 
                           control_states=gate['control_states'])
    elif gate_type == 'SWAP':
        gate_qubits = max(gate['qubit_1'], gate['qubit_2']) + 1
        gate_matrix = _SWAP(qubit_1=gate['qubit_1'], qubit_2=gate['qubit_2'])
    # 三量子比特门：
    elif gate_type == 'TOFFOLI':
        gate_qubits = max(gate['target_qubit'], gate['control_qubits'][0], gate['control_qubits'][1]) + 1
        gate_matrix = _TOFFOLI(target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'])
    # 恒等门
    elif gate_type in ['IDENTITY', 'I']:
        return IDENTITY(gate['num_qubits'])
    else:
        raise ValueError(f"不支持的门类型: {gate_type}") 
    
    if gate_qubits < cir_qubits:
        # 扩展到总量子比特数
        for i in range(gate_qubits, cir_qubits):
            gate_matrix = ops.kron(gate_matrix, IDENTITY_2)
    elif gate_qubits > cir_qubits:
        raise ValueError(f"量子门的量子比特数量超出总量子比特数: {gate_qubits} > {cir_qubits}")

    return gate_matrix