import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import torch
import os
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set paths
LOCAL = "./"
FOLDER = "model_30km_joint/"

# Create analysis plots directory
os.makedirs(LOCAL + FOLDER + "analysis_plots/", exist_ok=True)

print("=" * 60)
print("Starting Comprehensive Analysis of Training Results")
print("=" * 60)

def load_all_phase_data():
    """Load data from all training phases"""
    print("1. Loading data from all training phases...")
    
    phase_data = {}
    progress_data = {}
    checkpoint_data = {}
    
    # Training phases configuration
    TRAINING_PHASES = [
        {"name": "phase_1", "start": 0, "end": 8000},
        {"name": "phase_2", "start": 8000, "end": 16000},
        {"name": "phase_3", "start": 16000, "end": 24000},
        {"name": "phase_4", "start": 24000, "end": 32000},
        {"name": "phase_5", "start": 32000, "end": 40000},
        {"name": "phase_6", "start": 40000, "end": 48000},
        {"name": "phase_7", "start": 48000, "end": 56000},
    ]
    
    # Load phase results
    for phase in TRAINING_PHASES:
        phase_name = phase['name']
        model_file = f"{LOCAL}{FOLDER}hybrid_qnn_model_{phase_name}.pth"
        results_file = f"{LOCAL}{FOLDER}phase_results_{phase_name}.txt"
        progress_file = f"{LOCAL}{FOLDER}batch_progress_{phase_name}.txt"
        
        # Load model data
        if os.path.exists(model_file):
            try:
                model_data = torch.load(model_file, map_location='cpu')
                phase_data[phase_name] = {
                    'model': model_data,
                    'training_samples': phase['end'] - phase['start'],
                    'phase_info': phase
                }
                print(f"  ✓ Loaded {phase_name} model data")
            except Exception as e:
                print(f"  ✗ Error loading {phase_name} model: {e}")
        
        # Load progress data
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress_content = f.read()
                progress_data[phase_name] = parse_progress_file(progress_content, phase_name)
                print(f"  ✓ Loaded {phase_name} progress data")
            except Exception as e:
                print(f"  ✗ Error loading {phase_name} progress: {e}")
    
    # Load checkpoint data
    checkpoint_files = [f for f in os.listdir(LOCAL + FOLDER) if f.startswith('checkpoint_') and f.endswith('.pth')]
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = torch.load(LOCAL + FOLDER + checkpoint_file, map_location='cpu')
            phase_name = checkpoint.get('phase', 'unknown')
            if phase_name not in checkpoint_data:
                checkpoint_data[phase_name] = []
            checkpoint_data[phase_name].append(checkpoint)
        except Exception as e:
            print(f"  ✗ Error loading checkpoint {checkpoint_file}: {e}")
    
    print(f"Loaded data from {len(phase_data)} phases, {len(progress_data)} progress files, {len(checkpoint_files)} checkpoints")
    return phase_data, progress_data, checkpoint_data

def parse_progress_file(content, phase_name):
    """Parse progress file to extract loss data"""
    lines = content.split('\n')
    batch_data = []
    
    for line in lines:
        if 'Batch' in line and 'Loss:' in line:
            try:
                # Parse lines like: "[phase_1] Batch 10/250 (4.0%) - Loss: 0.123456 - Avg Loss: 0.234567 - Time: 1.23s"
                batch_match = re.search(r'Batch (\d+)/(\d+)', line)
                loss_match = re.search(r'Loss: ([0-9.-]+)', line)
                avg_loss_match = re.search(r'Avg Loss: ([0-9.-]+)', line)
                
                if batch_match and loss_match:
                    batch_num = int(batch_match.group(1))
                    total_batches = int(batch_match.group(2))
                    loss = float(loss_match.group(1))
                    avg_loss = float(avg_loss_match.group(1)) if avg_loss_match else loss
                    
                    batch_data.append({
                        'phase': phase_name,
                        'batch': batch_num,
                        'total_batches': total_batches,
                        'loss': loss,
                        'avg_loss': avg_loss,
                        'progress': (batch_num / total_batches) * 100
                    })
            except Exception as e:
                continue
    
    return batch_data

def plot_training_progress(phase_data, progress_data):
    """Plot training progress across all phases"""
    print("\n2. Plotting training progress...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Batch Loss Progression by Phase
    all_batch_data = []
    for phase_name, batches in progress_data.items():
        for batch in batches:
            batch['global_batch'] = len(all_batch_data)
            all_batch_data.append(batch)
    
    if all_batch_data:
        df = pd.DataFrame(all_batch_data)
        
        # Color map for phases
        phases = sorted(df['phase'].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))
        color_map = dict(zip(phases, colors))
        
        for phase in phases:
            phase_df = df[df['phase'] == phase]
            axes[0, 0].plot(phase_df['global_batch'], phase_df['loss'], 
                           label=phase, color=color_map[phase], alpha=0.7, linewidth=1)
        
        axes[0, 0].set_xlabel('Global Batch Number')
        axes[0, 0].set_ylabel('Batch Loss')
        axes[0, 0].set_title('Batch Loss Progression Across All Phases')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
    
    # Plot 2: Final Performance by Phase
    phase_performance = []
    for phase_name, data in phase_data.items():
        history = data['model'].get('training_history', {})
        if 'phase_history' in history and history['phase_history']:
            latest_phase = history['phase_history'][-1]
            phase_performance.append({
                'phase': phase_name,
                'train_loss': latest_phase.get('train_loss', 0),
                'val_loss': latest_phase.get('val_loss', 0),
                'test_loss': latest_phase.get('test_loss', 0),
                'samples_trained': data['training_samples']
            })
    
    if phase_performance:
        perf_df = pd.DataFrame(phase_performance)
        x = range(len(perf_df))
        width = 0.25
        
        axes[0, 1].bar([i - width for i in x], perf_df['train_loss'], width, label='Train Loss', alpha=0.7)
        axes[0, 1].bar(x, perf_df['val_loss'], width, label='Val Loss', alpha=0.7)
        axes[0, 1].bar([i + width for i in x], perf_df['test_loss'], width, label='Test Loss', alpha=0.7)
        
        axes[0, 1].set_xlabel('Training Phase')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Final Performance Metrics by Phase')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(perf_df['phase'])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Loss Improvement Over Phases
    if phase_performance:
        phases = [p['phase'] for p in phase_performance]
        train_losses = [p['train_loss'] for p in phase_performance]
        val_losses = [p['val_loss'] for p in phase_performance]
        test_losses = [p['test_loss'] for p in phase_performance]
        
        axes[1, 0].plot(phases, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=6)
        axes[1, 0].plot(phases, val_losses, 's-', label='Val Loss', linewidth=2, markersize=6)
        axes[1, 0].plot(phases, test_losses, '^-', label='Test Loss', linewidth=2, markersize=6)
        
        axes[1, 0].set_xlabel('Training Phase')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Loss Improvement Over Training Phases')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Plot 4: Training Statistics
    stats_text = "Training Statistics Summary\n\n"
    if phase_performance:
        total_samples = sum([p['samples_trained'] for p in phase_performance])
        final_train = phase_performance[-1]['train_loss']
        final_val = phase_performance[-1]['val_loss']
        final_test = phase_performance[-1]['test_loss']
        
        stats_text += f"Total Phases: {len(phase_performance)}\n"
        stats_text += f"Total Samples Trained: {total_samples:,}\n"
        stats_text += f"Final Train Loss: {final_train:.6f}\n"
        stats_text += f"Final Val Loss: {final_val:.6f}\n"
        stats_text += f"Final Test Loss: {final_test:.6f}\n\n"
        
        # Improvement calculations
        if len(phase_performance) > 1:
            first_train = phase_performance[0]['train_loss']
            improvement = ((first_train - final_train) / first_train * 100)
            stats_text += f"Overall Improvement: {improvement:+.1f}%\n"
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(LOCAL + FOLDER + 'analysis_plots/training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training progress plots saved")

def plot_validation_analysis(phase_data):
    """Plot detailed validation analysis"""
    print("\n3. Plotting validation analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract validation data
    val_data = []
    for phase_name, data in phase_data.items():
        history = data['model'].get('training_history', {})
        if 'phase_history' in history:
            for phase_hist in history['phase_history']:
                if phase_hist['phase_name'] == phase_name:
                    val_data.append({
                        'phase': phase_name,
                        'val_loss': phase_hist.get('val_loss', 0),
                        'val_samples': phase_hist.get('val_samples_used', 0),
                        'val_time': phase_hist.get('val_time', 0)
                    })
    
    if val_data:
        val_df = pd.DataFrame(val_data)
        
        # Plot 1: Validation Loss by Phase
        axes[0, 0].bar(val_df['phase'], val_df['val_loss'], color='orange', alpha=0.7)
        axes[0, 0].set_xlabel('Phase')
        axes[0, 0].set_ylabel('Validation Loss')
        axes[0, 0].set_title('Validation Loss by Phase')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(val_df['val_loss']):
            axes[0, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Validation Time vs Loss
        scatter = axes[0, 1].scatter(val_df['val_time'], val_df['val_loss'], 
                                    c=range(len(val_df)), cmap='viridis', s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Validation Time (seconds)')
        axes[0, 1].set_ylabel('Validation Loss')
        axes[0, 1].set_title('Validation Time vs Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add phase labels
        for i, (x, y, phase) in enumerate(zip(val_df['val_time'], val_df['val_loss'], val_df['phase'])):
            axes[0, 1].annotate(phase, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=axes[0, 1], label='Phase Order')
    
    # Plot 3: Validation Sample Usage
    if val_data:
        samples_used = [v['val_samples'] for v in val_data]
        phases = [v['phase'] for v in val_data]
        
        axes[1, 0].bar(phases, samples_used, color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Phase')
        axes[1, 0].set_ylabel('Validation Samples Used')
        axes[1, 0].set_title('Validation Sample Usage by Phase')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        for i, v in enumerate(samples_used):
            axes[1, 0].text(i, v + 50, str(v), ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Validation Statistics
    if val_data:
        val_losses = [v['val_loss'] for v in val_data]
        val_times = [v['val_time'] for v in val_data]
        
        stats_text = "Validation Statistics\n\n"
        stats_text += f"Total Validations: {len(val_data)}\n"
        stats_text += f"Avg Validation Loss: {np.mean(val_losses):.6f}\n"
        stats_text += f"Std Validation Loss: {np.std(val_losses):.6f}\n"
        stats_text += f"Min Validation Loss: {np.min(val_losses):.6f}\n"
        stats_text += f"Max Validation Loss: {np.max(val_losses):.6f}\n"
        stats_text += f"Avg Validation Time: {np.mean(val_times):.2f}s\n"
        stats_text += f"Total Validation Time: {np.sum(val_times):.2f}s\n"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        axes[1, 1].set_title('Validation Performance Summary')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(LOCAL + FOLDER + 'analysis_plots/validation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Validation analysis plots saved")

def plot_testing_analysis(phase_data):
    """Plot detailed testing analysis"""
    print("\n4. Plotting testing analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract test data
    test_data = []
    for phase_name, data in phase_data.items():
        history = data['model'].get('training_history', {})
        if 'phase_history' in history:
            for phase_hist in history['phase_history']:
                if phase_hist['phase_name'] == phase_name:
                    test_data.append({
                        'phase': phase_name,
                        'test_loss': phase_hist.get('test_loss', 0),
                        'test_samples': phase_hist.get('test_samples_used', 0),
                        'test_time': phase_hist.get('test_time', 0)
                    })
    
    if test_data:
        test_df = pd.DataFrame(test_data)
        
        # Plot 1: Test Loss by Phase
        axes[0, 0].bar(test_df['phase'], test_df['test_loss'], color='red', alpha=0.7)
        axes[0, 0].set_xlabel('Phase')
        axes[0, 0].set_ylabel('Test Loss')
        axes[0, 0].set_title('Test Loss by Phase')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        for i, v in enumerate(test_df['test_loss']):
            axes[0, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Test Time vs Loss
        scatter = axes[0, 1].scatter(test_df['test_time'], test_df['test_loss'], 
                                    c=range(len(test_df)), cmap='plasma', s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Test Time (seconds)')
        axes[0, 1].set_ylabel('Test Loss')
        axes[0, 1].set_title('Test Time vs Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        for i, (x, y, phase) in enumerate(zip(test_df['test_time'], test_df['test_loss'], test_df['phase'])):
            axes[0, 1].annotate(phase, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=axes[0, 1], label='Phase Order')
    
    # Plot 3: Test Sample Usage
    if test_data:
        samples_used = [t['test_samples'] for t in test_data]
        phases = [t['phase'] for t in test_data]
        
        axes[1, 0].bar(phases, samples_used, color='purple', alpha=0.7)
        axes[1, 0].set_xlabel('Phase')
        axes[1, 0].set_ylabel('Test Samples Used')
        axes[1, 0].set_title('Test Sample Usage by Phase')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        for i, v in enumerate(samples_used):
            axes[1, 0].text(i, v + 50, str(v), ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Test Statistics
    if test_data:
        test_losses = [t['test_loss'] for t in test_data]
        test_times = [t['test_time'] for t in test_data]
        
        stats_text = "Testing Statistics\n\n"
        stats_text += f"Total Tests: {len(test_data)}\n"
        stats_text += f"Avg Test Loss: {np.mean(test_losses):.6f}\n"
        stats_text += f"Std Test Loss: {np.std(test_losses):.6f}\n"
        stats_text += f"Min Test Loss: {np.min(test_losses):.6f}\n"
        stats_text += f"Max Test Loss: {np.max(test_losses):.6f}\n"
        stats_text += f"Avg Test Time: {np.mean(test_times):.2f}s\n"
        stats_text += f"Total Test Time: {np.sum(test_times):.2f}s\n"
        stats_text += f"Best Phase: {test_df.loc[test_df['test_loss'].idxmin(), 'phase']}\n"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        axes[1, 1].set_title('Testing Performance Summary')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(LOCAL + FOLDER + 'analysis_plots/testing_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Testing analysis plots saved")

def plot_comprehensive_comparison(phase_data):
    """Plot comprehensive comparison of train/val/test performance"""
    print("\n5. Plotting comprehensive comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare comparison data
    comparison_data = []
    for phase_name, data in phase_data.items():
        history = data['model'].get('training_history', {})
        if 'phase_history' in history:
            for phase_hist in history['phase_history']:
                if phase_hist['phase_name'] == phase_name:
                    comparison_data.append({
                        'phase': phase_name,
                        'train_loss': phase_hist.get('train_loss', 0),
                        'val_loss': phase_hist.get('val_loss', 0),
                        'test_loss': phase_hist.get('test_loss', 0),
                        'total_time': phase_hist.get('total_time', 0),
                        'samples_trained': data['training_samples']
                    })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        # Plot 1: Three-way Loss Comparison
        x = range(len(comp_df))
        width = 0.25
        
        axes[0, 0].bar([i - width for i in x], comp_df['train_loss'], width, 
                      label='Train Loss', alpha=0.7, color='blue')
        axes[0, 0].bar(x, comp_df['val_loss'], width, 
                      label='Val Loss', alpha=0.7, color='orange')
        axes[0, 0].bar([i + width for i in x], comp_df['test_loss'], width, 
                      label='Test Loss', alpha=0.7, color='red')
        
        axes[0, 0].set_xlabel('Phase')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Train/Val/Test Loss Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(comp_df['phase'])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss Ratios (Overfitting Analysis)
        comp_df['val_train_ratio'] = comp_df['val_loss'] / comp_df['train_loss']
        comp_df['test_train_ratio'] = comp_df['test_loss'] / comp_df['train_loss']
        
        axes[0, 1].plot(comp_df['phase'], comp_df['val_train_ratio'], 'o-', 
                       label='Val/Train Ratio', linewidth=2, markersize=6)
        axes[0, 1].plot(comp_df['phase'], comp_df['test_train_ratio'], 's-', 
                       label='Test/Train Ratio', linewidth=2, markersize=6)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal (1.0)')
        
        axes[0, 1].set_xlabel('Phase')
        axes[0, 1].set_ylabel('Loss Ratio')
        axes[0, 1].set_title('Overfitting Analysis: Validation and Test vs Training Loss Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Training Efficiency
        comp_df['samples_per_second'] = comp_df['samples_trained'] / comp_df['total_time']
        
        axes[1, 0].bar(comp_df['phase'], comp_df['samples_per_second'], 
                      color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Phase')
        axes[1, 0].set_ylabel('Samples per Second')
        axes[1, 0].set_title('Training Efficiency (Samples per Second)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        for i, v in enumerate(comp_df['samples_per_second']):
            axes[1, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Comprehensive Statistics
        stats_text = "Comprehensive Performance Summary\n\n"
        stats_text += f"Best Training Loss: {comp_df['train_loss'].min():.6f}\n"
        stats_text += f"Best Validation Loss: {comp_df['val_loss'].min():.6f}\n"
        stats_text += f"Best Test Loss: {comp_df['test_loss'].min():.6f}\n\n"
        stats_text += f"Avg Val/Train Ratio: {comp_df['val_train_ratio'].mean():.3f}\n"
        stats_text += f"Avg Test/Train Ratio: {comp_df['test_train_ratio'].mean():.3f}\n\n"
        stats_text += f"Total Training Time: {comp_df['total_time'].sum():.1f}s\n"
        stats_text += f"Avg Efficiency: {comp_df['samples_per_second'].mean():.1f} samples/s\n"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(LOCAL + FOLDER + 'analysis_plots/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Comprehensive comparison plots saved")

def generate_analysis_report(phase_data, progress_data):
    """Generate comprehensive analysis report"""
    print("\n6. Generating analysis report...")
    
    report_file = LOCAL + FOLDER + 'analysis_plots/analysis_report.txt'
    
    report = f"""
COMPREHENSIVE TRAINING ANALYSIS REPORT
{'=' * 60}

1. EXECUTIVE SUMMARY
{'=' * 20}
Total Phases Analyzed: {len(phase_data)}
Total Training Samples: {sum([data['training_samples'] for data in phase_data.values()]):,}
Analysis Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

2. PERFORMANCE OVERVIEW
{'=' * 20}
"""
    
    # Calculate overall statistics
    all_train_losses = []
    all_val_losses = []
    all_test_losses = []
    
    for phase_name, data in phase_data.items():
        history = data['model'].get('training_history', {})
        if 'phase_history' in history:
            for phase_hist in history['phase_history']:
                all_train_losses.append(phase_hist.get('train_loss', 0))
                all_val_losses.append(phase_hist.get('val_loss', 0))
                all_test_losses.append(phase_hist.get('test_loss', 0))
    
    if all_train_losses:
        report += f"Best Training Loss: {min(all_train_losses):.6f}\n"
        report += f"Best Validation Loss: {min(all_val_losses):.6f}\n"
        report += f"Best Test Loss: {min(all_test_losses):.6f}\n"
        report += f"Final Training Loss: {all_train_losses[-1]:.6f}\n"
        report += f"Final Validation Loss: {all_val_losses[-1]:.6f}\n"
        report += f"Final Test Loss: {all_test_losses[-1]:.6f}\n\n"
    
    report += f"3. DETAILED PHASE PERFORMANCE\n{'=' * 20}\n"
    
    for phase_name, data in phase_data.items():
        history = data['model'].get('training_history', {})
        if 'phase_history' in history:
            for phase_hist in history['phase_history']:
                if phase_hist['phase_name'] == phase_name:
                    report += f"{phase_name}:\n"
                    report += f"  Training Loss: {phase_hist.get('train_loss', 0):.6f}\n"
                    report += f"  Validation Loss: {phase_hist.get('val_loss', 0):.6f}\n"
                    report += f"  Test Loss: {phase_hist.get('test_loss', 0):.6f}\n"
                    report += f"  Training Time: {phase_hist.get('total_time', 0):.2f}s\n"
                    report += f"  Samples Trained: {data['training_samples']}\n\n"
    
    report += f"4. ANALYSIS FILES GENERATED\n{'=' * 20}\n"
    report += "training_progress.png - Batch loss progression and phase performance\n"
    report += "validation_analysis.png - Detailed validation performance analysis\n"
    report += "testing_analysis.png - Detailed testing performance analysis\n"
    report += "comprehensive_comparison.png - Train/val/test comparison and efficiency\n"
    
    report += f"\n{'=' * 60}\n"
    report += "Report completed successfully!\n"
    report += f"{'=' * 60}"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Analysis report saved to: {report_file}")

def main():
    """Main analysis function"""
    try:
        # Load all data
        phase_data, progress_data, checkpoint_data = load_all_phase_data()
        
        if not phase_data:
            print("No phase data found for analysis!")
            return
        
        # Generate all analysis plots
        plot_training_progress(phase_data, progress_data)
        plot_validation_analysis(phase_data)
        plot_testing_analysis(phase_data)
        plot_comprehensive_comparison(phase_data)
        
        # Generate analysis report
        generate_analysis_report(phase_data, progress_data)
        
        # Final summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"All analysis results saved to: {LOCAL + FOLDER}analysis_plots/")
        print("\nGenerated Files:")
        print("  - training_progress.png: Batch loss and phase performance")
        print("  - validation_analysis.png: Validation performance details")
        print("  - testing_analysis.png: Testing performance details")
        print("  - comprehensive_comparison.png: Overall comparison")
        print("  - analysis_report.txt: Detailed analysis summary")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
