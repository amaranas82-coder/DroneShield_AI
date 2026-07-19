import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

class Visualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
    
    def plot_confusion_matrix(self, cm, save_name='confusion_matrix.png'):
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Non-Drone', 'Drone'],
            yticklabels=['Non-Drone', 'Drone']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_training_history(self, history, save_name='training_history.png'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history['train_acc'], label='Training')
        axes[0].plot(history['val_acc'], label='Validation')
        axes[0].set_title('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(history['train_loss'], label='Training')
        axes[1].plot(history['val_loss'], label='Validation')
        axes[1].set_title('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()

if __name__ == "__main__":
    print("Visualizer module ready")
