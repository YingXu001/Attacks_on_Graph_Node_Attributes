import os
import argparse
import torch
import random
import numpy as np
from train import train_model
from test import test_model
from data_loader import load_data
from plot import plot_losses, plot_accuracies

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="GCN for different datasets")
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name (Cora, Reddit, etc.)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=40, help='Patience for early stopping')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)  # Set the seed

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    log_filename = f'logs/{args.dataset}_training_result.log'
    data, dataset = load_data(args.dataset)
    train_losses, val_losses, val_accuracies = train_model(data, dataset, args.lr, args.patience, args.epochs, log_filename)
    
    plot_losses(train_losses, val_losses, args.dataset)
    plot_accuracies(val_accuracies, args.dataset)

    test_loss, test_accuracy = test_model(data, dataset)

    # Save results to log
    with open(f'{args.dataset}_testing_result.log', 'w') as log_file:
        log_file.write(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}\n')

if __name__ == "__main__":
    main()
