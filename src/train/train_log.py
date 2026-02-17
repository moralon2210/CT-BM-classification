import json
from pathlib import Path
import matplotlib.pyplot as plt


def save_history(history, checkpoint_dir="./checkpoints"):
    """Save training history to checkpoint folder as JSON."""
    save_path = Path(checkpoint_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    file_path = save_path / "train.log"
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"Training history saved to {file_path}")
    return file_path


def plot_training_metrics(history, save_dir="./results"):
    """Plot training and validation loss and PR-AUC and save to file."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Average Precision plot
    ax2.plot(epochs, history['train_avg_precision'], 'b-', label='Train')
    ax2.plot(epochs, history['val_avg_precision'], 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Precision')
    ax2.set_title('Average Precision (PR-AUC)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    plot_file = save_path / "training_metrics.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Training metrics plot saved to {plot_file}")
    return plot_file
