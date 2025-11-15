import json
import re
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 使用当前目录作为保存图表的目录
output_dir = '.'
os.makedirs(output_dir, exist_ok=True)

# 读取批次进度数据
batch_losses = []
batch_avg_losses = []
batch_numbers = []

with open('batch_progress.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        # 匹配批次训练记录
        match = re.search(r'批次 (\d+)/250 \(([\d.]+)%\) - 损失: ([\d.-]+) - 平均损失: ([\d.-]+)', line)
        if match:
            batch_num = int(match.group(1))
            loss = float(match.group(3))
            avg_loss = float(match.group(4))
            
            batch_numbers.append(batch_num)
            batch_losses.append(loss)
            batch_avg_losses.append(avg_loss)

# 读取验证和测试结果
with open('validation_results.json', 'r') as f:
    val_data = json.load(f)
    
with open('test_results.json', 'r') as f:
    test_data = json.load(f)

# 读取训练摘要
with open('training_summary.json', 'r') as f:
    train_summary = json.load(f)

# 绘制250个batch的训练过程图表
plt.figure(figsize=(15, 8))

# 子图1: 批次损失
plt.subplot(2, 2, 1)
plt.plot(batch_numbers, batch_losses, 'b-', linewidth=1.5, marker='o', markersize=3)
plt.title('训练损失随批次变化（250个批次）')
plt.xlabel('批次')
plt.ylabel('损失')
plt.grid(True, alpha=0.3)

# 子图2: 平均损失
plt.subplot(2, 2, 2)
plt.plot(batch_numbers, batch_avg_losses, 'g-', linewidth=1.5, marker='s', markersize=3)
plt.title('平均训练损失随批次变化（250个批次）')
plt.xlabel('批次')
plt.ylabel('平均损失')
plt.grid(True, alpha=0.3)

# 子图3: 损失对比（训练、验证、测试）
plt.subplot(2, 2, 3)
x_points = range(len(batch_numbers))
plt.plot(x_points, batch_losses, 'b-', linewidth=1, label='批次损失', alpha=0.7)
plt.plot(x_points, batch_avg_losses, 'g-', linewidth=1, label='平均损失', alpha=0.7)

# 添加验证和测试损失的水平线
final_train_loss = train_summary['final_train_loss']
final_val_loss = train_summary['final_val_loss']
test_loss = test_data['test_loss']

plt.axhline(y=final_train_loss, color='r', linestyle='--', linewidth=1.5, label=f'最终训练损失 ({final_train_loss:.6f})')
plt.axhline(y=final_val_loss, color='m', linestyle='-.', linewidth=1.5, label=f'验证损失 ({final_val_loss:.6f})')
plt.axhline(y=test_loss, color='c', linestyle=':', linewidth=1.5, label=f'测试损失 ({test_loss:.6f})')

plt.title('训练、验证和测试损失对比')
plt.xlabel('批次索引')
plt.ylabel('损失')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图4: 损失汇总柱状图
plt.subplot(2, 2, 4)
labels = ['最终训练损失', '验证损失', '测试损失']
values = [final_train_loss, final_val_loss, test_loss]
colors = ['red', 'magenta', 'cyan']
bars = plt.bar(labels, values, color=colors, alpha=0.7)
plt.title('损失值汇总')
plt.ylabel('损失')

# 在柱状图上添加数值标签
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
             f'{value:.6f}', ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 保存图表到当前目录
plt.savefig(os.path.join(output_dir, 'training_validation_test_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 绘制详细的批次损失图表（250个批次）
plt.figure(figsize=(15, 8))
plt.plot(batch_numbers, batch_losses, 'b-', linewidth=1.5, marker='o', markersize=4, label='批次损失')
plt.plot(batch_numbers, batch_avg_losses, 'g-', linewidth=1.5, marker='s', markersize=4, label='平均损失')

plt.title('详细批次损失变化（250个批次）')
plt.xlabel('批次')
plt.ylabel('损失')
plt.legend()
plt.grid(True, alpha=0.3)

# 标记检查点位置
checkpoint_batches = [50, 100, 150, 200, 250]
for batch in checkpoint_batches:
    if batch in batch_numbers:
        idx = batch_numbers.index(batch)
        plt.annotate(f'检查点{batch}', 
                    (batch_numbers[idx], batch_losses[idx]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'detailed_batch_losses_250.png'), dpi=300, bbox_inches='tight')
plt.close()

# 绘制移动平均图表
plt.figure(figsize=(15, 8))

# 计算移动平均 (窗口大小为10)
def moving_average(data, window_size):
    if len(data) < window_size:
        return data
    return [np.mean(data[max(0, i-window_size+1):i+1]) for i in range(len(data))]

ma_losses = moving_average(batch_losses, 10)
ma_avg_losses = moving_average(batch_avg_losses, 10)

plt.plot(batch_numbers, batch_losses, 'b-', linewidth=1, alpha=0.3, label='原始批次损失')
plt.plot(batch_numbers, ma_losses, 'b-', linewidth=2, label='批次损失移动平均(窗口=10)')

plt.plot(batch_numbers, batch_avg_losses, 'g-', linewidth=1, alpha=0.3, label='原始平均损失')
plt.plot(batch_numbers, ma_avg_losses, 'g-', linewidth=2, label='平均损失移动平均(窗口=10)')

plt.title('训练损失与移动平均（250个批次）')
plt.xlabel('批次')
plt.ylabel('损失')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'losses_with_moving_average_250.png'), dpi=300, bbox_inches='tight')
plt.close()

# 单独绘制验证和测试结果图表
plt.figure(figsize=(10, 6))

# 准备数据
labels = ['训练损失', '验证损失', '测试损失']
values = [final_train_loss, final_val_loss, test_loss]
colors = ['red', 'magenta', 'cyan']

bars = plt.bar(labels, values, color=colors, alpha=0.7)
plt.title('模型性能对比：训练、验证和测试')
plt.ylabel('损失值')
plt.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值标签
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
             f'{value:.6f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 添加水平线以更好地显示差异
plt.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'validation_test_results.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"所有图表已生成并保存到当前目录下:")
print(f"1. training_validation_test_comparison.png - 训练、验证和测试对比图")
print(f"2. detailed_batch_losses_250.png - 250个批次详细损失图")
print(f"3. losses_with_moving_average_250.png - 移动平均损失图")
print(f"4. validation_test_results.png - 验证和测试结果图")