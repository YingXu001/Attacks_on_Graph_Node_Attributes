import os
import matplotlib.pyplot as plt

model_dir = 'plots'
os.makedirs(model_dir, exist_ok=True)

def plot_losses(train_losses, val_losses, dataset_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{dataset_name} Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/{dataset_name}_losses.png')
    plt.close()

def plot_accuracies(val_accuracies, dataset_name):
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'{dataset_name} Accuracy Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'plots/{dataset_name}_accuracies.png')
    plt.close()
