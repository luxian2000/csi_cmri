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
print("Starting Analysis of Training, Validation, and Test Data")
print("=" * 60)

def load_and_analyze_data():
    """Load and analyze training, validation, and test data"""
    
    # 1. Load raw data
    print("1. Loading raw data...")
    PATH = "../DataSpace/csi_cmri/"
    data = np.load(PATH + "CSI_channel_30km.npy")
    
    # Data splitting
    train_data = data[:8000]  # 8000 training samples
    val_data = data[8000:10000]  # 2000 validation samples
    test_data = data[10000:12000]  # 2000 test samples
    
    print(f"Raw data shape: {data.shape}")
    print(f"Training data: {train_data.shape}")
    print(f"Validation data: {val_data.shape}")
    print(f"Test data: {test_data.shape}")
    
    # 2. Load training results
    print("\n2. Loading training results...")
    training_log = ""
    training_summary = {}
    validation_results = {}
    test_results = {}
    
    try:
        with open(LOCAL + FOLDER + 'training_results.txt', 'r') as f:
            training_log = f.read()
        print("Training log loaded successfully")
    except Exception as e:
        print(f"Training log loading failed: {e}")

    try:
        with open(LOCAL + FOLDER + 'training_summary.json', 'r') as f:
            training_summary = json.load(f)
        print("Training summary loaded successfully")
    except Exception as e:
        print(f"Training summary loading failed: {e}")

    try:
        with open(LOCAL + FOLDER + 'validation_results.json', 'r') as f:
            validation_results = json.load(f)
        print("Validation results loaded successfully")
    except Exception as e:
        print(f"Validation results loading failed: {e}")

    try:
        with open(LOCAL + FOLDER + 'test_results.json', 'r') as f:
            test_results = json.load(f)
        print("Test results loaded successfully")
    except Exception as e:
        print(f"Test results loading failed: {e}")

    # 3. Load model parameters and checkpoints
    print("\n3. Loading model parameters and checkpoints...")
    checkpoint_files = []
    model_parameters = {}
    
    try:
        checkpoint_files = [f for f in os.listdir(LOCAL + FOLDER) if f.startswith('checkpoint_') and f.endswith('.pth')]
        checkpoint_files.sort()
        print(f"Found {len(checkpoint_files)} checkpoint files")
        
        # Load final model
        if os.path.exists(LOCAL + FOLDER + 'hybrid_qnn_model_final.pth'):
            final_model = torch.load(LOCAL + FOLDER + 'hybrid_qnn_model_final.pth', map_location='cpu')
            model_parameters['final'] = final_model
            print("Final model loaded successfully")
            
    except Exception as e:
        print(f"Error loading model parameters: {e}")
    
    return data, train_data, val_data, test_data, training_summary, validation_results, test_results, checkpoint_files, model_parameters

def parse_batch_progress():
    """Parse batch_progress.txt to extract loss values"""
    print("\n4. Parsing batch_progress.txt...")
    
    progress_data = {
        'epochs': [],
        'batches': [],
        'losses': [],
        'avg_losses': [],
        'timestamps': []
    }
    
    try:
        with open(LOCAL + FOLDER + 'batch_progress.txt', 'r') as f:
            lines = f.readlines()
        
        # Skip header lines
        for line in lines:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('批次训练进度记录'):
                continue
            
            # Parse lines like: "2024-01-01 12:00:00 - [周期 1/10] 批次 10/250 (4.0%) - 损失: 0.123456 - 平均损失: 0.234567 - 时间: 1.23s"
            if '损失:' in line and '平均损失:' in line:
                try:
                    # Extract timestamp
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        progress_data['timestamps'].append(timestamp_match.group(1))
                    
                    # Extract epoch and batch
                    epoch_match = re.search(r'周期 (\d+)/(\d+)', line)
                    batch_match = re.search(r'批次 (\d+)/(\d+)', line)
                    
                    if epoch_match and batch_match:
                        epoch = int(epoch_match.group(1))
                        batch = int(batch_match.group(1))
                        progress_data['epochs'].append(epoch)
                        progress_data['batches'].append(batch)
                    
                    # Extract losses
                    loss_match = re.search(r'损失: ([0-9.-]+)', line)
                    avg_loss_match = re.search(r'平均损失: ([0-9.-]+)', line)
                    
                    if loss_match and avg_loss_match:
                        loss = float(loss_match.group(1))
                        avg_loss = float(avg_loss_match.group(1))
                        progress_data['losses'].append(loss)
                        progress_data['avg_losses'].append(avg_loss)
                        
                except Exception as e:
                    print(f"Error parsing line: {line} - {e}")
                    continue
        
        print(f"Parsed {len(progress_data['losses'])} loss records from batch_progress.txt")
        
    except Exception as e:
        print(f"Error reading batch_progress.txt: {e}")
    
    return progress_data

def analyze_model_parameters(checkpoint_files):
    """Analyze model parameter changes from checkpoints"""
    print("\n5. Analyzing model parameter changes...")
    
    param_history = {
        'epochs': [],
        'batches': [],
        'c_weight_norms': [],
        'c_bias_norms': [],
        'q_weight_norms': [],
        'c_weight_means': [],
        'c_bias_means': [],
        'q_weight_means': []
    }
    
    for checkpoint_file in checkpoint_files:
        try:
            # Extract epoch and batch from filename
            epoch_match = re.search(r'epoch_(\d+)', checkpoint_file)
            batch_match = re.search(r'batch_(\d+)', checkpoint_file)
            
            if epoch_match and batch_match:
                epoch = int(epoch_match.group(1))
                batch = int(batch_match.group(1))
                
                # Load checkpoint
                checkpoint = torch.load(LOCAL + FOLDER + checkpoint_file, map_location='cpu')
                
                # Extract parameters
                c_weight = checkpoint.get('C_WEIGHT')
                c_bias = checkpoint.get('C_BIAS')
                q_weight = checkpoint.get('Q_WEIGHT')
                
                if c_weight is not None:
                    param_history['c_weight_norms'].append(torch.norm(c_weight).item())
                    param_history['c_weight_means'].append(torch.mean(c_weight).item())
                
                if c_bias is not None:
                    param_history['c_bias_norms'].append(torch.norm(c_bias).item())
                    param_history['c_bias_means'].append(torch.mean(c_bias).item())
                
                if q_weight is not None:
                    param_history['q_weight_norms'].append(torch.norm(q_weight).item())
                    param_history['q_weight_means'].append(torch.mean(q_weight).item())
                
                param_history['epochs'].append(epoch)
                param_history['batches'].append(batch)
                
                print(f"Checkpoint E{epoch}B{batch}: "
                      f"C_weight norm: {param_history['c_weight_norms'][-1]:.4f}, "
                      f"Q_weight norm: {param_history['q_weight_norms'][-1]:.4f}")
                
        except Exception as e:
            print(f"Error analyzing checkpoint {checkpoint_file}: {e}")
    
    print(f"Analyzed {len(param_history['epochs'])} checkpoints for parameter changes")
    return param_history

def plot_batch_progress(progress_data):
    """Plot batch progress from batch_progress.txt"""
    print("\n6. Plotting batch progress...")
    
    if not progress_data['losses']:
        print("No batch progress data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Batch Loss Progression
    batch_numbers = list(range(len(progress_data['losses'])))
    axes[0, 0].plot(batch_numbers, progress_data['losses'], 'b-', alpha=0.7, linewidth=1, label='Batch Loss')
    axes[0, 0].set_xlabel('Batch Number')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Batch Loss Progression (from batch_progress.txt)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average Loss Progression
    axes[0, 1].plot(batch_numbers, progress_data['avg_losses'], 'r-', alpha=0.7, linewidth=1, label='Average Loss')
    axes[0, 1].set_xlabel('Batch Number')
    axes[0, 1].set_ylabel('Average Loss')
    axes[0, 1].set_title('Average Loss Progression (from batch_progress.txt)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Loss Distribution by Epoch
    if progress_data['epochs']:
        unique_epochs = sorted(set(progress_data['epochs']))
        epoch_losses = []
        
        for epoch in unique_epochs:
            epoch_loss = [progress_data['losses'][i] for i, e in enumerate(progress_data['epochs']) if e == epoch]
            epoch_losses.append(epoch_loss)
        
        box_data = [loss for loss in epoch_losses if loss]  # Remove empty lists
        if box_data:
            axes[1, 0].boxplot(box_data, labels=[f'Epoch {e}' for e in unique_epochs[:len(box_data)]])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Loss Distribution by Epoch')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss Statistics
    if progress_data['losses']:
        loss_stats = {
            'Min Loss': np.min(progress_data['losses']),
            'Max Loss': np.max(progress_data['losses']),
            'Mean Loss': np.mean(progress_data['losses']),
            'Std Loss': np.std(progress_data['losses']),
            'Final Loss': progress_data['losses'][-1],
            'Total Batches': len(progress_data['losses'])
        }
        
        stats_text = "\n".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" 
                              for k, v in loss_stats.items()])
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        axes[1, 1].set_title('Batch Loss Statistics')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(LOCAL + FOLDER + 'analysis_plots/batch_progress_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Batch progress analysis plots saved")

def plot_parameter_evolution(param_history):
    """Plot model parameter evolution"""
    print("\n7. Plotting parameter evolution...")
    
    if not param_history['epochs']:
        print("No parameter history data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Convert batch numbers to sequential for plotting
    sequential_batches = list(range(len(param_history['epochs'])))
    
    # Plot 1: Parameter Norms
    if param_history['c_weight_norms']:
        axes[0, 0].plot(sequential_batches, param_history['c_weight_norms'], 'b-', label='C_WEIGHT Norm', linewidth=2)
    if param_history['c_bias_norms']:
        axes[0, 0].plot(sequential_batches, param_history['c_bias_norms'], 'r-', label='C_BIAS Norm', linewidth=2)
    if param_history['q_weight_norms']:
        axes[0, 0].plot(sequential_batches, param_history['q_weight_norms'], 'g-', label='Q_WEIGHT Norm', linewidth=2)
    
    axes[0, 0].set_xlabel('Checkpoint Sequence')
    axes[0, 0].set_ylabel('Parameter Norm')
    axes[0, 0].set_title('Parameter Norm Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Parameter Means
    if param_history['c_weight_means']:
        axes[0, 1].plot(sequential_batches, param_history['c_weight_means'], 'b-', label='C_WEIGHT Mean', linewidth=2)
    if param_history['c_bias_means']:
        axes[0, 1].plot(sequential_batches, param_history['c_bias_means'], 'r-', label='C_BIAS Mean', linewidth=2)
    if param_history['q_weight_means']:
        axes[0, 1].plot(sequential_batches, param_history['q_weight_means'], 'g-', label='Q_WEIGHT Mean', linewidth=2)
    
    axes[0, 1].set_xlabel('Checkpoint Sequence')
    axes[0, 1].set_ylabel('Parameter Mean')
    axes[0, 1].set_title('Parameter Mean Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Parameter Changes by Epoch
    if param_history['epochs']:
        unique_epochs = sorted(set(param_history['epochs']))
        
        # Group parameters by epoch
        epoch_c_weight_norms = []
        epoch_q_weight_norms = []
        
        for epoch in unique_epochs:
            epoch_indices = [i for i, e in enumerate(param_history['epochs']) if e == epoch]
            if epoch_indices:
                epoch_c_weight_norms.append(np.mean([param_history['c_weight_norms'][i] for i in epoch_indices]))
                epoch_q_weight_norms.append(np.mean([param_history['q_weight_norms'][i] for i in epoch_indices]))
        
        x_pos = np.arange(len(unique_epochs))
        width = 0.35
        
        if epoch_c_weight_norms:
            axes[1, 0].bar(x_pos - width/2, epoch_c_weight_norms, width, label='C_WEIGHT Norm', alpha=0.7)
        if epoch_q_weight_norms:
            axes[1, 0].bar(x_pos + width/2, epoch_q_weight_norms, width, label='Q_WEIGHT Norm', alpha=0.7)
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Average Parameter Norm')
        axes[1, 0].set_title('Average Parameter Norms by Epoch')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([f'Epoch {e}' for e in unique_epochs])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Parameter Statistics
    param_stats_text = "Parameter Statistics:\n\n"
    
    if param_history['c_weight_norms']:
        param_stats_text += f"C_WEIGHT:\n"
        param_stats_text += f"  Norm: {np.mean(param_history['c_weight_norms']):.4f} ± {np.std(param_history['c_weight_norms']):.4f}\n"
        param_stats_text += f"  Mean: {np.mean(param_history['c_weight_means']):.4f} ± {np.std(param_history['c_weight_means']):.4f}\n\n"
    
    if param_history['q_weight_norms']:
        param_stats_text += f"Q_WEIGHT:\n"
        param_stats_text += f"  Norm: {np.mean(param_history['q_weight_norms']):.4f} ± {np.std(param_history['q_weight_norms']):.4f}\n"
        param_stats_text += f"  Mean: {np.mean(param_history['q_weight_means']):.4f} ± {np.std(param_history['q_weight_means']):.4f}\n"
    
    axes[1, 1].text(0.1, 0.9, param_stats_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    axes[1, 1].set_title('Parameter Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(LOCAL + FOLDER + 'analysis_plots/parameter_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Parameter evolution plots saved")

def plot_training_curves(training_history, training_summary):
    """Plot training curves"""
    print("\n8. Plotting training curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training and Validation Loss
    if training_history['epoch_losses']:
        epochs = [entry['epoch'] for entry in training_history['epoch_losses']]
        train_losses = [entry['avg_loss'] for entry in training_history['epoch_losses']]
        
        axes[0, 0].plot(epochs, train_losses, 'b-o', linewidth=2, markersize=6, label='Training Loss')
        
        if training_history['val_losses']:
            val_epochs = [entry['epoch'] for entry in training_history['val_losses']]
            val_losses = [entry['val_loss'] for entry in training_history['val_losses']]
            axes[0, 0].plot(val_epochs, val_losses, 'r-s', linewidth=2, markersize=6, label='Validation Loss')
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No Training Loss Data Available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss (No Data)')
    
    # Plot 2: Performance Summary
    if training_summary:
        metrics = []
        values = []
        
        if 'final_train_loss' in training_summary:
            metrics.append('Final Train Loss')
            values.append(float(training_summary['final_train_loss']))
        
        if 'final_val_loss' in training_summary:
            metrics.append('Final Val Loss')
            values.append(float(training_summary['final_val_loss']))
        
        if 'test_loss' in training_summary:
            metrics.append('Test Loss')
            values.append(float(training_summary['test_loss']))
        
        if 'best_val_loss' in training_summary:
            metrics.append('Best Val Loss')
            values.append(float(training_summary['best_val_loss']))
        
        if metrics:
            bars = axes[0, 1].bar(metrics, values, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
            axes[0, 1].set_title('Performance Metrics Summary')
            axes[0, 1].set_ylabel('Loss Value')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{value:.6f}', ha='center', va='bottom', fontsize=9)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Performance Metrics Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Performance Metrics (No Data)')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Training Summary Available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('Performance Metrics (No Data)')
    
    # Plot 3 and 4: Placeholder for other data
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(LOCAL + FOLDER + 'analysis_plots/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training curves plots saved")

def main():
    """Main analysis function"""
    try:
        # Load all data and results
        data, train_data, val_data, test_data, training_summary, validation_results, test_results, checkpoint_files, model_parameters = load_and_analyze_data()
        
        # Parse batch progress and model parameters
        progress_data = parse_batch_progress()
        param_history = analyze_model_parameters(checkpoint_files)
        
        # Generate all plots
        plot_batch_progress(progress_data)
        plot_parameter_evolution(param_history)
        
        # Analyze training history for traditional curves
        training_history = {
            'epoch_losses': [],
            'val_losses': [],
            'batch_losses': [],
            'weights_history': []
        }
        
        if 'final' in model_parameters:
            final_model = model_parameters['final']
            if 'train_losses' in final_model and final_model['train_losses']:
                for epoch, loss in enumerate(final_model['train_losses']):
                    training_history['epoch_losses'].append({
                        'epoch': epoch + 1,
                        'avg_loss': float(loss)
                    })
            
            if 'val_losses' in final_model and final_model['val_losses']:
                for epoch, loss in enumerate(final_model['val_losses']):
                    training_history['val_losses'].append({
                        'epoch': epoch + 1,
                        'val_loss': float(loss)
                    })
        
        plot_training_curves(training_history, training_summary)
        
        # Final summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Analysis results saved to: {LOCAL + FOLDER}analysis_plots/")
        print("\nGenerated Files:")
        print("  - batch_progress_analysis.png: Batch loss progression")
        print("  - parameter_evolution.png: Model parameter changes")
        print("  - training_curves.png: Traditional training curves")
        
        print(f"\nData Summary:")
        print(f"  Batch progress records: {len(progress_data['losses'])}")
        print(f"  Checkpoints analyzed: {len(param_history['epochs'])}")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
