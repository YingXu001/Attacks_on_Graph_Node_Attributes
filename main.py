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
    parser.add_argument('--dataset_type', type=str, required=True, choices=['graph', 'text'], help='Type of the dataset (graph or text)')
    parser.add_argument('--dataset_name', type=str, help='Name of the graph dataset (e.g., Cora, Reddit)')
    parser.add_argument('--file_path', type=str, help='File path for the text dataset')
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

    # Load data based on the dataset type
    if args.dataset_type == 'graph':
        data, dataset = load_data(dataset_type='graph', dataset_name=args.dataset_name)
    elif args.dataset_type == 'text':
        data = load_data(dataset_type='text', file_path=args.file_path)
        dataset = None  # For text datasets, there might not be a 'dataset' object
    else:
        raise ValueError("Invalid dataset type specified")

    log_filename = f'logs/{args.dataset_name}_training_result.log'
    train_losses, val_losses, val_accuracies = train_model(data, dataset, args.lr, args.patience, args.epochs, log_filename)
    
    plot_losses(train_losses, val_losses, args.dataset_name)
    plot_accuracies(val_accuracies, args.dataset_name)

    test_loss, test_accuracy = test_model(data, dataset)

    # Save results to log
    with open(f'{args.dataset_name}_testing_result.log', 'w') as log_file:
        log_file.write(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}\n')

if __name__ == "__main__":
    main()
