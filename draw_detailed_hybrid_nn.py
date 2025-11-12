"""
绘制pg_9_8.py中的详细混合神经网络结构图
该网络是一个量子-经典混合神经网络，包含经典全连接层和量子线路
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

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
    """添加一个经典网络层（用矩形表示）"""
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
    """添加一个量子层（用圆形表示量子比特）"""
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

def add_mapping(patches, colors, start_loc, end_loc, style='straight'):
    """添加层之间的连接线"""
    if style == 'straight':
        patches.append(Line2D([start_loc[0], end_loc[0]],
                              [start_loc[1], end_loc[1]]))
        colors.append(Darker)
    elif style == 'double':
        patches.append(Line2D([start_loc[0], end_loc[0]],
                              [start_loc[1], end_loc[1]]))
        colors.append(Darker)
        patches.append(Line2D([start_loc[0] + 1, end_loc[0]],
                              [start_loc[1], end_loc[1]]))
        colors.append(Darker)

def label(xy, text, xy_off=[0, 4], fontsize=8, weight='normal'):
    """添加标签"""
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=fontsize, ha='center', va='center', weight=weight)

def draw_detailed_hybrid_nn():
    """绘制详细的混合神经网络结构图"""
    patches = []
    colors = []
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 根据pg_9_8.py中的参数计算网络结构
    INPUT_DIM = 2560  # 输入维度
    OUTPUT_DIM = 128  # 经典全连接层输出维度
    ORG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))  # 12个量子比特用于编码原始数据
    TAR_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))  # 7个量子比特用于编码处理后的数据
    ALL_QUBITS = ORG_QUBITS + 1  # 总共13个量子比特
    ANS_QUBITS = ORG_QUBITS - TAR_QUBITS  # 5个量子比特用于ansatz
    
    y_positions = [40, 30, 20, 10, 0]  # 各层的y坐标
    
    # 1. 绘制输入层
    input_top_left = [5, y_positions[0]]
    add_layer(patches, colors, size=(1, 6), num=3, top_left=input_top_left, loc_diff=[1.5, 0])
    add_dots(patches, colors, num_dots=3, top_left=[11, y_positions[0]], loc_diff=[1.5, 0])
    add_layer(patches, colors, size=(1, 6), num=3, top_left=[17, y_positions[0]], loc_diff=[1.5, 0])
    label([12, y_positions[0]+8], f'Input Layer\n{INPUT_DIM} neurons', fontsize=10, weight='bold')
    
    # 2. 绘制经典全连接层
    fc_top_left = [30, y_positions[0]]
    add_layer(patches, colors, size=(1, 6), num=2, top_left=fc_top_left, loc_diff=[1.5, 0])
    add_dots(patches, colors, num_dots=3, top_left=[34, y_positions[0]], loc_diff=[1.5, 0])
    add_layer(patches, colors, size=(1, 6), num=2, top_left=[40, y_positions[0]], loc_diff=[1.5, 0])
    label([36, y_positions[0]+8], f'Classical Dense Layer\n{INPUT_DIM}×{OUTPUT_DIM} weights + bias', fontsize=10, weight='bold')
    
    # 3. 绘制FRQI编码层（处理后数据）
    frqi_top_left = [55, y_positions[1]]
    add_layer(patches, colors, size=(1, 4), num=2, top_left=frqi_top_left, loc_diff=[1.5, 0])
    add_dots(patches, colors, num_dots=3, top_left=[60, y_positions[1]], loc_diff=[1.5, 0])
    add_layer(patches, colors, size=(1, 4), num=2, top_left=[66, y_positions[1]], loc_diff=[1.5, 0])
    label([62, y_positions[1]+6], f'FRQI Encoding\n(TAR_QUBITS={TAR_QUBITS} qubits)', fontsize=9)
    
    # 4. 绘制FRQI编码层（原始数据）
    frqi_org_top_left = [55, y_positions[2]]
    add_layer(patches, colors, size=(1, 4), num=2, top_left=frqi_org_top_left, loc_diff=[1.5, 0])
    add_dots(patches, colors, num_dots=3, top_left=[60, y_positions[2]], loc_diff=[1.5, 0])
    add_layer(patches, colors, size=(1, 4), num=2, top_left=[66, y_positions[2]], loc_diff=[1.5, 0])
    label([62, y_positions[2]+6], f'FRQI Encoding\n(ORG_QUBITS={ORG_QUBITS} qubits)', fontsize=9)
    
    # 5. 绘制量子比特线路
    quantum_y = y_positions[3]
    quantum_x_positions = []
    for i in range(ALL_QUBITS):
        x_pos = 50 + i * 3
        quantum_x_positions.append(x_pos)
        # 量子比特线
        patches.append(Line2D([x_pos, x_pos + 30], [quantum_y, quantum_y], linewidth=2))
        colors.append(Black)
        # 量子比特圆形
        patches.append(Circle([x_pos, quantum_y], 0.8))
        colors.append(Dark)
        # 量子比特标签
        label([x_pos, quantum_y-2], f'q{i}', fontsize=7)
    
    label([45, quantum_y], 'Qubits:', fontsize=9, weight='bold')
    
    # 6. 绘制强纠缠ansatz层
    ansatz_top_left = [55, y_positions[3]]
    for i in range(4):  # 4层ansatz
        x_pos = 58 + i * 8
        # 每层的旋转门
        for j in range(min(5, ALL_QUBITS)):  # 只显示部分量子比特上的门
            qubit_idx = j if j < ALL_QUBITS else 0
            patches.append(FancyBboxPatch([x_pos-0.7, quantum_y + qubit_idx*3 - 0.7], 1.4, 1.4,
                                          boxstyle="round,pad=0.1", ec="black", fc="lightblue"))
            colors.append(0.7)  # Light blue
            label([x_pos, quantum_y + qubit_idx*3], f'R', fontsize=7)
        
        # CNOT门示例
        if i < 3:  # 不是最后一层
            ctrl_qubit = i % ALL_QUBITS
            targ_qubit = (ctrl_qubit + 1) % ALL_QUBITS
            # 控制点
            patches.append(Circle([x_pos+2, quantum_y + ctrl_qubit*3], 0.3, fc='black'))
            colors.append(Black)
            # 目标点
            patches.append(Circle([x_pos+2, quantum_y + targ_qubit*3], 0.5, fc='white', ec='black'))
            colors.append(White)
            patches.append(Line2D([x_pos+2, x_pos+2], 
                                  [quantum_y + ctrl_qubit*3, quantum_y + targ_qubit*3], 
                                  linewidth=1, color='black'))
            colors.append(Black)
        
        label([x_pos, quantum_y - 5], f'Layer {i+1}', fontsize=7)
    
    label([75, y_positions[3]+8], f'Strongly Entangling Ansatz\n{4} layers, {ALL_QUBITS} qubits', fontsize=10, weight='bold')
    
    # 7. 绘制测量层
    measure_top_left = [95, y_positions[1]]
    add_layer(patches, colors, size=(1, 4), num=2, top_left=measure_top_left, loc_diff=[1.5, 0])
    add_dots(patches, colors, num_dots=3, top_left=[100, y_positions[1]], loc_diff=[1.5, 0])
    add_layer(patches, colors, size=(1, 4), num=2, top_left=[106, y_positions[1]], loc_diff=[1.5, 0])
    label([102, y_positions[1]+6], 'Measurement\n(Hamiltonian)', fontsize=9)
    
    # 8. 绘制输出层
    output_top_left = [120, y_positions[0]]
    add_layer(patches, colors, size=(1, 6), num=1, top_left=output_top_left, loc_diff=[1.5, 0])
    label([120, y_positions[0]+8], 'Output\n1 value', fontsize=10, weight='bold')
    
    # 添加连接线
    # 输入到经典全连接层
    add_mapping(patches, colors, [20, y_positions[0]], [29, y_positions[0]], 'straight')
    # 经典全连接层到FRQI编码（处理后数据）
    add_mapping(patches, colors, [42, y_positions[0]], [54, y_positions[1]], 'straight')
    # 经典全连接层到FRQI编码（原始数据）
    add_mapping(patches, colors, [42, y_positions[0]], [54, y_positions[2]], 'straight')
    # FRQI编码到量子比特线路
    add_mapping(patches, colors, [68, y_positions[1]], [quantum_x_positions[5], quantum_y], 'straight')
    add_mapping(patches, colors, [68, y_positions[2]], [quantum_x_positions[0], quantum_y], 'straight')
    # 量子线路到测量
    add_mapping(patches, colors, [85, quantum_y], [94, y_positions[1]], 'straight')
    # 测量到输出
    add_mapping(patches, colors, [108, y_positions[1]], [119, y_positions[0]], 'straight')
    
    # 设置颜色和添加图形元素到图中
    for patch, color in zip(patches, colors):
        patch.set_color(color * np.ones(3))
        if isinstance(patch, Line2D):
            ax.add_line(patch)
        else:
            patch.set_edgecolor(Black * np.ones(3))
            ax.add_patch(patch)
    
    # 添加标题和说明
    plt.text(65, 50, 'Detailed Hybrid Quantum-Classical Neural Network Architecture', 
             family='sans-serif', size=14, ha='center', weight='bold')
    
    plt.text(65, 47, 'Based on pg_9_8.py implementation', 
             family='sans-serif', size=12, ha='center')
    
    # 添加图例
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=np.array(Light)*np.ones(3), 
               markersize=10, label='Classical Neuron/Layer'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array(Dark)*np.ones(3), 
               markersize=10, label='Quantum Qubit'),
        Line2D([0], [0], color=np.array(Darker)*np.ones(3), lw=2, label='Connections'),
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1", ec="black", 
                       fc="lightblue", label='Quantum Gate')
    ]
    plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0.05))
    
    plt.xlim(0, 130)
    plt.ylim(-10, 55)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('./detailed_hybrid_nn_architecture.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    draw_detailed_hybrid_nn()