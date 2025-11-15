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

# 读取所有阶段的批次进度数据
phases_data = {}
for phase in [1, 2, 3]:
    batch_losses = []
    batch_avg_losses = []
    batch_numbers = []
    
    filename = f'batch_progress_phase_{phase}.txt'
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # 匹配批次训练记录
            match = re.search(r'Batch (\d+)/250 \(([\d.]+)%\) - Loss: ([\d.-]+) - Avg Loss: ([\d.-]+)', line)
            if match:
                batch_num = int(match.group(1))
                loss = float(match.group(3))
                avg_loss = float(match.group(4))
                
                batch_numbers.append(batch_num)
                batch_losses.append(loss)
                batch_avg_losses.append(avg_loss)
    
    phases_data[phase] = {
        'batch_numbers': batch_numbers,
        'batch_losses': batch_losses,
        'batch_avg_losses': batch_avg_losses
    }

# 读取各阶段的结果数据
phase_results = {}
for phase in [1, 2, 3]:
    filename = f'phase_results_phase_{phase}.txt'
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        # 提取最终训练损失、验证损失和测试损失
        train_loss_match = re.search(r'Final Training Loss: ([\d.-]+)', content)
        val_loss_match = re.search(r'Validation Loss: ([\d.-]+)', content)
        test_loss_match = re.search(r'Test Loss: ([\d.-]+)', content)
        
        train_loss = float(train_loss_match.group(1)) if train_loss_match else None
        val_loss = float(val_loss_match.group(1)) if val_loss_match else None
        test_loss = float(test_loss_match.group(1)) if test_loss_match else None
        
        phase_results[phase] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss
        }

# 为每个阶段绘制单独的图表
for phase in [1, 2, 3]:
    data = phases_data[phase]
    results = phase_results[phase]
    
    # 绘制250个batch的训练过程图表
    plt.figure(figsize=(15, 10))
    
    # 子图1: 批次损失
    plt.subplot(2, 2, 1)
    plt.plot(data['batch_numbers'], data['batch_losses'], 'b-', linewidth=1.5, marker='o', markersize=3)
    plt.title(f'阶段 {phase} - 训练损失随批次变化（250个批次）')
    plt.xlabel('批次')
    plt.ylabel('损失')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 平均损失
    plt.subplot(2, 2, 2)
    plt.plot(data['batch_numbers'], data['batch_avg_losses'], 'g-', linewidth=1.5, marker='s', markersize=3)
    plt.title(f'阶段 {phase} - 平均训练损失随批次变化（250个批次）')
    plt.xlabel('批次')
    plt.ylabel('平均损失')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 损失对比（训练、验证、测试）
    plt.subplot(2, 2, 3)
    x_points = range(len(data['batch_numbers']))
    plt.plot(x_points, data['batch_losses'], 'b-', linewidth=1, label='批次损失', alpha=0.7)
    plt.plot(x_points, data['batch_avg_losses'], 'g-', linewidth=1, label='平均损失', alpha=0.7)
    
    # 添加验证和测试损失的水平线
    if results['train_loss'] is not None:
        plt.axhline(y=results['train_loss'], color='r', linestyle='--', linewidth=1.5, 
                   label=f'最终训练损失 ({results["train_loss"]:.6f})')
    if results['val_loss'] is not None:
        plt.axhline(y=results['val_loss'], color='m', linestyle='-.', linewidth=1.5, 
                   label=f'验证损失 ({results["val_loss"]:.6f})')
    if results['test_loss'] is not None:
        plt.axhline(y=results['test_loss'], color='c', linestyle=':', linewidth=1.5, 
                   label=f'测试损失 ({results["test_loss"]:.6f})')
    
    plt.title(f'阶段 {phase} - 训练、验证和测试损失对比')
    plt.xlabel('批次索引')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4: 损失汇总柱状图
    plt.subplot(2, 2, 4)
    labels = ['最终训练损失', '验证损失', '测试损失']
    values = [results['train_loss'], results['val_loss'], results['test_loss']]
    colors = ['red', 'magenta', 'cyan']
    bars = plt.bar(labels, values, color=colors, alpha=0.7)
    plt.title(f'阶段 {phase} - 损失值汇总')
    plt.ylabel('损失')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, values):
        if value is not None:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                     f'{value:.6f}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存图表到当前目录
    plt.savefig(os.path.join(output_dir, f'phase_{phase}_training_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 绘制所有阶段的综合对比图
plt.figure(figsize=(15, 12))

# 子图1: 所有阶段的批次损失对比
plt.subplot(2, 2, 1)
for phase in [1, 2, 3]:
    data = phases_data[phase]
    plt.plot(data['batch_numbers'], data['batch_losses'], linewidth=1.5, marker='o', markersize=2, label=f'阶段 {phase}')

plt.title('所有阶段的批次损失对比')
plt.xlabel('批次')
plt.ylabel('损失')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: 所有阶段的平均损失对比
plt.subplot(2, 2, 2)
for phase in [1, 2, 3]:
    data = phases_data[phase]
    plt.plot(data['batch_numbers'], data['batch_avg_losses'], linewidth=1.5, marker='s', markersize=2, label=f'阶段 {phase}')

plt.title('所有阶段的平均损失对比')
plt.xlabel('批次')
plt.ylabel('平均损失')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图3: 各阶段最终损失对比
plt.subplot(2, 2, 3)
phases = [1, 2, 3]
train_losses = [phase_results[p]['train_loss'] for p in phases]
val_losses = [phase_results[p]['val_loss'] for p in phases]
test_losses = [phase_results[p]['test_loss'] for p in phases]

x = np.arange(len(phases))
width = 0.25

plt.bar(x - width, train_losses, width, label='训练损失', color='red', alpha=0.7)
plt.bar(x, val_losses, width, label='验证损失', color='magenta', alpha=0.7)
plt.bar(x + width, test_losses, width, label='测试损失', color='cyan', alpha=0.7)

plt.xlabel('阶段')
plt.ylabel('损失')
plt.title('各阶段最终损失对比')
plt.xticks(x, phases)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值标签
for i, (train_loss, val_loss, test_loss) in enumerate(zip(train_losses, val_losses, test_losses)):
    if train_loss is not None:
        plt.text(i - width, train_loss + 0.0001, f'{train_loss:.5f}', ha='center', va='bottom', fontsize=8)
    if val_loss is not None:
        plt.text(i, val_loss + 0.0001, f'{val_loss:.5f}', ha='center', va='bottom', fontsize=8)
    if test_loss is not None:
        plt.text(i + width, test_loss + 0.0001, f'{test_loss:.5f}', ha='center', va='bottom', fontsize=8)

# 子图4: 损失变化趋势图（每个阶段的最终损失）
plt.subplot(2, 2, 4)
train_losses = [phase_results[p]['train_loss'] for p in phases]
val_losses = [phase_results[p]['val_loss'] for p in phases]

plt.plot(phases, train_losses, 'ro-', linewidth=2, markersize=8, label='训练损失')
plt.plot(phases, val_losses, 'ms-', linewidth=2, markersize=8, label='验证损失')

plt.title('各阶段损失变化趋势')
plt.xlabel('阶段')
plt.ylabel('损失')
plt.legend()
plt.grid(True, alpha=0.3)

# 添加数值标签
for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
    if train_loss is not None:
        plt.text(phases[i], train_loss + 0.0001, f'{train_loss:.5f}', ha='center', va='bottom', fontsize=9)
    if val_loss is not None:
        plt.text(phases[i], val_loss + 0.0001, f'{val_loss:.5f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_phases_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 绘制详细的批次损失图表（250个批次）
plt.figure(figsize=(15, 10))

# 为每个阶段绘制损失曲线
colors = ['blue', 'green', 'orange']
for i, phase in enumerate([1, 2, 3]):
    data = phases_data[phase]
    plt.plot(data['batch_numbers'], data['batch_losses'], 
             color=colors[i], linewidth=1.5, marker='o', markersize=3, 
             label=f'阶段 {phase} 批次损失')

plt.title('所有阶段的详细批次损失变化（250个批次）')
plt.xlabel('批次')
plt.ylabel('损失')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'detailed_batch_losses_all_phases.png'), dpi=300, bbox_inches='tight')
plt.close()

# 绘制移动平均图表
plt.figure(figsize=(15, 10))

# 计算移动平均 (窗口大小为10)
def moving_average(data, window_size):
    if len(data) < window_size:
        return data
    return [np.mean(data[max(0, i-window_size+1):i+1]) for i in range(len(data))]

# 为每个阶段绘制原始损失和移动平均损失
colors = ['blue', 'green', 'orange']
for i, phase in enumerate([1, 2, 3]):
    data = phases_data[phase]
    ma_losses = moving_average(data['batch_losses'], 10)
    
    plt.plot(data['batch_numbers'], data['batch_losses'], 
             color=colors[i], linewidth=1, alpha=0.3, 
             label=f'阶段 {phase} 原始损失')
    plt.plot(data['batch_numbers'], ma_losses, 
             color=colors[i], linewidth=2, 
             label=f'阶段 {phase} 移动平均损失(窗口=10)')

plt.title('所有阶段的训练损失与移动平均（250个批次）')
plt.xlabel('批次')
plt.ylabel('损失')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'losses_with_moving_average_all_phases.png'), dpi=300, bbox_inches='tight')
plt.close()

# 单独绘制验证和测试结果图表
plt.figure(figsize=(12, 8))

# 准备数据
phases = [1, 2, 3]
train_losses = [phase_results[p]['train_loss'] for p in phases]
val_losses = [phase_results[p]['val_loss'] for p in phases]
test_losses = [phase_results[p]['test_loss'] for p in phases]

x = np.arange(len(phases))
width = 0.25

bars1 = plt.bar(x - width, train_losses, width, label='训练损失', color='red', alpha=0.7)
bars2 = plt.bar(x, val_losses, width, label='验证损失', color='magenta', alpha=0.7)
bars3 = plt.bar(x + width, test_losses, width, label='测试损失', color='cyan', alpha=0.7)

plt.title('所有阶段的模型性能对比：训练、验证和测试')
plt.xlabel('阶段')
plt.ylabel('损失值')
plt.xticks(x, phases)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值标签
for i, (bar, value) in enumerate(zip(bars1, train_losses)):
    if value is not None:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                 f'{value:.5f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for i, (bar, value) in enumerate(zip(bars2, val_losses)):
    if value is not None:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                 f'{value:.5f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for i, (bar, value) in enumerate(zip(bars3, test_losses)):
    if value is not None:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                 f'{value:.5f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'validation_test_results_all_phases.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"所有图表已生成并保存到当前目录下:")
print(f"1. phase_1_training_results.png - 阶段1训练结果")
print(f"2. phase_2_training_results.png - 阶段2训练结果")
print(f"3. phase_3_training_results.png - 阶段3训练结果")
print(f"4. all_phases_comparison.png - 所有阶段对比图")
print(f"5. detailed_batch_losses_all_phases.png - 所有阶段详细批次损失图")
print(f"6. losses_with_moving_average_all_phases.png - 所有阶段移动平均损失图")
print(f"7. validation_test_results_all_phases.png - 所有阶段验证和测试结果图")