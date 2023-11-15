import os
import argparse
import torch
import random
import json
import logging
import numpy as np
from train import train_model
from test import test_model
from data_loader import load_data, filter_and_save_hellaswag
from plot import plot_losses, plot_accuracies
from BERT_feature_extraction import initialize_bert, extract_embeddings
from graph_operations import create_graph, visualize_graph, save_graph_data, load_graph_data


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="GCN for different datasets")
    parser.add_argument('--dataset_type', type=str, required=True, choices=['graph', 'text'], help='Type of the dataset (graph or text)')
    parser.add_argument('--dataset_name', type=str, help='Name of the graph dataset (e.g., Cora, Reddit)')
    parser.add_argument('--file_path', type=str, default='data/combined_hellaswag.json', help='File path for the text dataset')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=40, help='Patience for early stopping')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--threshold', type=float, default=0.88, help='Threshold for edge creation based on cosine similarity')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--norm_type', type=str, default='Linf', choices=['Linf', 'L2', 'L1'], help='Type of norm for PGD attack')
    return parser.parse_args()

def main():
    # logging.info("Script started.")
    args = parse_args()
    set_seed(args.seed)

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    if args.dataset_type == 'text':
        output_file = 'data/mixed_data.json'
        graph_file = 'data/mixed_graph.npz'

        if not os.path.exists(output_file):
            print(f"File '{output_file}' not found. Starting data processing.")
            # logging.info(f"File '{output_file}' not found. Starting data processing.")
            file_path = args.file_path
            # selected_labels = ['Making a sandwich', 'Disc dog', 'Surfing', 'Scuba diving', 'Fixing bicycle']
            selected_labels = ['Scuba diving', 'Making a sandwich', 'Disc dog', 'Fixing bicycle']
            
            # Filter and save Hellaswag data
            data_list = filter_and_save_hellaswag(file_path, output_file, selected_labels)

        else:
            # logging.info(f"File '{output_file}' found. Skipping data processing.")
            print(f"File '{output_file}' found. Skipping data processing.")
            # Load the existing data
            with open(output_file, 'r', encoding='utf-8') as file:
                data_list = json.load(file)

        if not os.path.exists(graph_file):
            # Create and save graph if mixed_graph.npz does not exist
            tokenizer, model = initialize_bert()
            embeddings = extract_embeddings(data_list, tokenizer, model)
            G, label_encoder = create_graph(data_list, embeddings, args.threshold)
            visualize_graph(G)
            save_graph_data(G, graph_file)  # Assuming save_graph_data saves the graph to mixed_graph.npz
        else:
            print(f"File '{graph_file}' found. Starting modeling.")
            data = load_graph_data(graph_file)
        
        # Assuming 'data' is a PyTorch Geometric Data object
        num_features = data.num_node_features
        num_classes = len(torch.unique(data.y)) # Determine the number of classes from your data

        train_losses, val_losses, val_accuracies = train_model(data, num_features, num_classes, args.lr, args.patience, args.epochs)
            
    # Handle 'graph' dataset_type...
    elif args.dataset_type == 'graph':
        data, dataset = load_data(dataset_type='graph', dataset_name=args.dataset_name)

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
