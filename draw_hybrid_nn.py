"""
绘制pg_9_8.py中的混合神经网络结构图
该网络是一个量子-经典混合神经网络，包含经典全连接层和量子线路
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle

# 定义颜色常量
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.3
Darker = 0.15
Black = 0.

def add_layer(patches, colors, size=(24, 24), num=5,
              top_left=[0, 0],
              loc_diff=[3, -3]):
    """添加一个网络层"""
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    for ind in range(num):
        patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
        if ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)

def add_quantum_layer(patches, colors, size=(24, 24), num=5,
                      top_left=[0, 0],
                      loc_diff=[3, -3]):
    """添加一个量子层（用圆形表示）"""
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    for ind in range(num):
        # 使用圆形表示量子比特
        patches.append(Circle(loc_start + ind * loc_diff + np.array(size) / 2, size[0]/2))
        colors.append(Dark)

def add_dots(patches, colors, num_dots=3, top_left=[0, 0], loc_diff=[3, -3]):
    """添加省略点"""
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left
    for ind in range(num_dots):
        patches.append(Circle(loc_start + ind * loc_diff, 0.5))
        colors.append(Black)

def add_mapping(patches, colors, start_loc, end_loc):
    """添加层之间的连接线"""
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0] + 1, end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Darker)

def label(xy, text, xy_off=[0, 4]):
    """添加标签"""
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=8, ha='center', va='center')

def draw_hybrid_nn():
    """绘制混合神经网络结构图"""
    patches = []
    colors = []
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 根据pg_9_8.py中的参数计算网络结构
    INPUT_DIM = 2560  # 输入维度
    OUTPUT_DIM = 128  # 经典全连接层输出维度
    ORG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))  # 12个量子比特用于编码原始数据
    TAR_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))  # 7个量子比特用于编码处理后的数据
    ALL_QUBITS = ORG_QUBITS + 1  # 总共13个量子比特
    ANS_QUBITS = ORG_QUBITS - TAR_QUBITS  # 5个量子比特用于ansatz
    
    # 绘制输入层（简化表示）
    input_top_left = [0, 20]
    add_layer(patches, colors, size=(2, 10), num=3, top_left=input_top_left, loc_diff=[1, 0])
    add_dots(patches, colors, num_dots=3, top_left=[4, 20], loc_diff=[1, 0])
    add_layer(patches, colors, size=(2, 10), num=3, top_left=[8, 20], loc_diff=[1, 0])
    label([4, 25], f'Input Layer\n{INPUT_DIM} neurons')
    
    # 绘制经典全连接层
    fc_top_left = [15, 20]
    add_layer(patches, colors, size=(2, 6), num=2, top_left=fc_top_left, loc_diff=[1, 0])
    add_dots(patches, colors, num_dots=3, top_left=[18, 20], loc_diff=[1, 0])
    add_layer(patches, colors, size=(2, 6), num=2, top_left=[22, 20], loc_diff=[1, 0])
    label([19, 25], f'Classical FC Layer\n{INPUT_DIM}x{OUTPUT_DIM} weights')
    
    # 绘制FRQI编码层
    frqi_top_left = [30, 20]
    add_layer(patches, colors, size=(2, 4), num=2, top_left=frqi_top_left, loc_diff=[1, 0])
    add_dots(patches, colors, num_dots=3, top_left=[33, 20], loc_diff=[1, 0])
    add_layer(patches, colors, size=(2, 4), num=2, top_left=[36, 20], loc_diff=[1, 0])
    label([33, 25], 'FRQI Encoding\n7 qubits')
    
    # 绘制量子线路层
    quantum_top_left = [45, 20]
    add_quantum_layer(patches, colors, size=(2, 2), num=min(4, ALL_QUBITS), 
                      top_left=quantum_top_left, loc_diff=[2.5, 0])
    if ALL_QUBITS > 4:
        add_dots(patches, colors, num_dots=3, top_left=[55, 20], loc_diff=[1, 0])
        add_quantum_layer(patches, colors, size=(2, 2), num=min(4, ALL_QUBITS-4), 
                          top_left=[59, 20], loc_diff=[2.5, 0])
    label([52, 25], f'Quantum Circuit\n{ALL_QUBITS} qubits\n4 layers')
    
    # 绘制测量层
    measure_top_left = [70, 20]
    add_layer(patches, colors, size=(2, 4), num=2, top_left=measure_top_left, loc_diff=[1, 0])
    add_dots(patches, colors, num_dots=3, top_left=[73, 20], loc_diff=[1, 0])
    add_layer(patches, colors, size=(2, 4), num=2, top_left=[76, 20], loc_diff=[1, 0])
    label([73, 25], 'Measurement\nHamiltonian')
    
    # 绘制输出层
    output_top_left = [85, 20]
    add_layer(patches, colors, size=(2, 6), num=1, top_left=output_top_left, loc_diff=[1, 0])
    label([85, 25], 'Output\n1 value')
    
    # 添加连接线
    # 输入到经典全连接层
    add_mapping(patches, colors, [10, 20], [14, 20])
    # 经典全连接层到FRQI编码
    add_mapping(patches, colors, [24, 20], [29, 20])
    # FRQI编码到量子线路
    add_mapping(patches, colors, [38, 20], [44, 20])
    # 量子线路到测量
    add_mapping(patches, colors, [62 if ALL_QUBITS > 4 else 55, 20], [69, 20])
    # 测量到输出
    add_mapping(patches, colors, [78, 20], [84, 20])
    
    # 设置颜色和添加图形元素到图中
    for patch, color in zip(patches, colors):
        patch.set_color(color * np.ones(3))
        if isinstance(patch, Line2D):
            ax.add_line(patch)
        else:
            patch.set_edgecolor(Black * np.ones(3))
            ax.add_patch(patch)
    
    # 添加标题和说明
    plt.text(45, 35, 'Hybrid Quantum-Classical Neural Network Architecture', 
             family='sans-serif', size=12, ha='center', weight='bold')
    
    plt.text(45, 32, 'Based on pg_9_8.py implementation', 
             family='sans-serif', size=10, ha='center')
    
    plt.text(0, 5, 'Network Details:\n'
             f'• Input dimension: {INPUT_DIM}\n'
             f'• Classical FC output: {OUTPUT_DIM} dimensions\n'
             f'• FRQI encoding: {TAR_QUBITS} qubits for processed data, {ORG_QUBITS} qubits for original data\n'
             f'• Total qubits in quantum circuit: {ALL_QUBITS}\n'
             f'• Ansatz layers: 4 with {ANS_QUBITS} active qubits\n'
             f'• Measurements: Hamiltonian expectation value',
             family='sans-serif', size=7, ha='left', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.xlim(-5, 95)
    plt.ylim(0, 35)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('./hybrid_nn_architecture.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    draw_hybrid_nn()