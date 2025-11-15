"""
使用draw_convnet库绘制pg_9_8.py中的网络结构
"""

import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 添加当前目录到路径以导入draw_convnet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 定义常量
NumDots = 4
NumConvMax = 8
NumFcMax = 20
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.3
Darker = 0.15
Black = 0.

try:
    from draw_convnet import add_layer, add_layer_with_omission, add_mapping, label
    print("Successfully imported draw_convnet functions")
except ImportError:
    print("Failed to import draw_convnet functions. Using built-in functions.")

    def add_layer(patches, colors, size=(24, 24), num=5,
                  top_left=[0, 0],
                  loc_diff=[3, -3]):
        from matplotlib.patches import Rectangle
        top_left = np.array(top_left)
        loc_diff = np.array(loc_diff)
        loc_start = top_left - np.array([0, size[0]])
        for ind in range(num):
            patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
            if ind % 2:
                colors.append(Medium)
            else:
                colors.append(Light)

    def add_layer_with_omission(patches, colors, size=(24, 24),
                                num=5, num_max=8,
                                num_dots=4,
                                top_left=[0, 0],
                                loc_diff=[3, -3]):
        from matplotlib.patches import Rectangle, Circle
        top_left = np.array(top_left)
        loc_diff = np.array(loc_diff)
        loc_start = top_left - np.array([0, size[0]])
        this_num = min(num, num_max)
        start_omit = (this_num - num_dots) // 2
        end_omit = this_num - start_omit
        start_omit -= 1
        for ind in range(this_num):
            if (num > num_max) and (start_omit < ind < end_omit):
                omit = True
            else:
                omit = False

            if omit:
                patches.append(
                    Circle(loc_start + ind * loc_diff + np.array(size) / 2, 0.5))
            else:
                patches.append(Rectangle(loc_start + ind * loc_diff,
                                         size[1], size[0]))

            if omit:
                colors.append(Black)
            elif ind % 2:
                colors.append(Medium)
            else:
                colors.append(Light)

    def add_mapping(patches, colors, start_ratio, end_ratio, patch_size, ind_bgn,
                    top_left_list, loc_diff_list, num_show_list, size_list):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        start_loc = top_left_list[ind_bgn] \
            + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
            + np.array([start_ratio[0] * (size_list[ind_bgn][1] - patch_size[1]),
                        - start_ratio[1] * (size_list[ind_bgn][0] - patch_size[0])]
                       )

        end_loc = top_left_list[ind_bgn + 1] \
            + (num_show_list[ind_bgn + 1] - 1) * np.array(
                loc_diff_list[ind_bgn + 1]) \
            + np.array([end_ratio[0] * size_list[ind_bgn + 1][1],
                        - end_ratio[1] * size_list[ind_bgn + 1][0]])

        patches.append(Rectangle(start_loc, patch_size[1], -patch_size[0]))
        colors.append(Dark)
        patches.append(Line2D([start_loc[0], end_loc[0]],
                              [start_loc[1], end_loc[1]]))
        colors.append(Darker)
        patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
                              [start_loc[1], end_loc[1]]))
        colors.append(Darker)
        patches.append(Line2D([start_loc[0], end_loc[0]],
                              [start_loc[1] - patch_size[0], end_loc[1]]))
        colors.append(Darker)
        patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
                              [start_loc[1] - patch_size[0], end_loc[1]]))
        colors.append(Darker)

    def label(xy, text, xy_off=[0, 4]):
        plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
                 family='sans-serif', size=8)

def draw_hybrid_network():
    """
    绘制pg_9_8.py中定义的混合神经网络结构
    该网络包含经典全连接层和量子线路
    """
    
    # 网络参数（来自pg_9_8.py）
    INPUT_DIM = 2560
    OUTPUT_DIM = 128
    ORG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))  # 12
    TAR_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))  # 7
    ALL_QUBITS = ORG_QUBITS + 1  # 13
    LAYERS = 4  # 量子线路层数
    
    # 为适应draw_convnet的风格，我们将网络结构简化为以下层：
    # 输入层 -> 经典全连接层 -> 量子处理层 -> 输出层
    
    patches = []
    colors = []

    fig, ax = plt.subplots()
    
    # 设置参数
    fc_unit_size = 1
    layer_width = 20
    flag_omit = True

    ############################
    # 定义网络层
    size_list = [(20, 20), (10, 10), (5, 5), (1, 1)]
    num_list = [INPUT_DIM, OUTPUT_DIM, ALL_QUBITS, 1]
    x_diff_list = [0, layer_width, layer_width, layer_width]
    text_list = ['Input\nLayer', 'Classical\nFC Layer', 'Quantum\nProcessing', 'Output']
    loc_diff_list = [[3, -3]] * len(size_list)

    num_show_list = list(map(min, num_list, [NumConvMax, NumFcMax, NumConvMax, NumFcMax]))
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

    # 添加网络层
    for ind in range(len(size_list)):
        if flag_omit:
            add_layer_with_omission(patches, colors, size=size_list[ind],
                                    num=num_list[ind],
                                    num_max=num_show_list[ind],
                                    num_dots=NumDots,
                                    top_left=top_left_list[ind],
                                    loc_diff=loc_diff_list[ind])
        else:
            add_layer(patches, colors, size=size_list[ind],
                      num=num_show_list[ind],
                      top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            '...' if num_list[ind] > num_show_list[ind] else num_list[ind]))

    ############################
    # 添加层间连接
    start_ratio_list = [[0.4, 0.5], [0.4, 0.5], [0.4, 0.5]]
    end_ratio_list = [[0.4, 0.5], [0.4, 0.5], [0.4, 0.5]]
    patch_size_list = [(5, 5), (3, 3), (1, 1)]
    ind_bgn_list = range(len(patch_size_list))
    text_list = ['Classical\nProcessing', 'Quantum\nProcessing', 'Measurement\n& Output']

    for ind in range(len(patch_size_list)):
        add_mapping(
            patches, colors, start_ratio_list[ind], end_ratio_list[ind],
            patch_size_list[ind], ind,
            top_left_list, loc_diff_list, num_show_list, size_list)
        label(top_left_list[ind], text_list[ind], xy_off=[26, -65])

    ############################
    # 设置图形属性
    for patch, color in zip(patches, colors):
        patch.set_color(color * np.ones(3))
        if isinstance(patch, plt.Line2D):
            ax.add_line(patch)
        else:
            patch.set_edgecolor(Black * np.ones(3))
            ax.add_patch(patch)

    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    plt.title('Hybrid Quantum-Classical Neural Network (pg_9_8.py)', fontsize=12, pad=20)
    
    # 添加网络信息文本
    info_text = f"""Network Architecture Details:
Input Dimension: {INPUT_DIM}
Classical FC Output: {OUTPUT_DIM}
Quantum Qubits: {ALL_QUBITS} (ORG_QUBITS={ORG_QUBITS}, TAR_QUBITS={TAR_QUBITS})
Quantum Layers: {LAYERS}
"""
    
    plt.text(0, -30, info_text, family='sans-serif', size=7, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
             verticalalignment='top')
    
    fig.set_size_inches(10, 4)
    
    # 保存图像
    fig.savefig('./pg98_convnet_architecture.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

if __name__ == '__main__':
    draw_hybrid_network()