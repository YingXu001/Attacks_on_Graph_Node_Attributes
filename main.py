import os
import argparse
import torch
import random
import json
import logging
import numpy as np
from train import train_model, train_with_pgd_attack, train_with_pgd_top_k_node_attack
from test import test_model, test_with_pgd_attack, test_with_pgd_top_k_node_attack
from data_loader import load_data, filter_and_save_hellaswag
from plot import plot_losses, plot_accuracies
from BERT_feature_extraction import initialize_bert, extract_embeddings
from graph_operations import create_graph, visualize_graph, save_graph_data, load_graph_data
from AttackGraph.PGD import pgd_attack
from AttackGraph.AddRandomNoise import add_random_noise
from model import GCN, GAT

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
    parser.add_argument('--apply_attack', action='store_true', help='Apply an attack during training')
    parser.add_argument('--attack_type', type=str, default='decision_time', choices=['decision_time', 'decision_time_K', 'poisoning'], help='Type of attack to apply')
    parser.add_argument('--K', type=int, default='10', help='Top K degree nodes')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT'], help='Graph Neural Network Model Selection')
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
            logging.info(f"File '{output_file}' not found. Starting data processing.")
            file_path = args.file_path
            # selected_labels = ['Making a sandwich', 'Disc dog', 'Surfing', 'Scuba diving', 'Fixing bicycle']
            selected_labels = ['Scuba diving', 'Making a sandwich', 'Disc dog', 'Fixing bicycle']
            
            # Filter and save Hellaswag data
            data_list = filter_and_save_hellaswag(file_path, output_file, selected_labels)

        else:
            logging.info(f"File '{output_file}' found. Skipping data processing.")
            print(f"File '{output_file}' found. Skipping data processing.")
            # Load the existing data
            with open(output_file, 'r', encoding='utf-8') as file:
                data_list = json.load(file)

        if not os.path.exists(graph_file):
            # Create and save graph if mixed_graph.npz does not exist
            tokenizer, model = initialize_bert()
            embeddings = extract_embeddings(data_list, tokenizer, model)
            G, label_encoder = create_graph(data_list, embeddings, args.threshold)
            # visualize_graph(G)
            save_graph_data(G, graph_file)  # Assuming save_graph_data saves the graph to mixed_graph.npz
        else:
            print(f"File '{graph_file}' found. Starting modeling.")
            data = load_graph_data(graph_file)
        
        # Load data
        data = load_graph_data(graph_file)
        
        # Initialize model
        num_features = data.num_node_features
        num_classes = len(torch.unique(data.y))
        # model = GCN(num_features, num_classes)
        if args.model == 'GCN':
            model = GCN(num_features, num_classes)
        elif args.model == 'GAT':
            model = GAT(num_features, num_classes)
        else:
            raise ValueError("Unsupported model type")
        criterion = torch.nn.CrossEntropyLoss()

        labels = data.y

        train_nodes = torch.where(data.train_mask)[0]
        train_labels = data.y[train_nodes]

        if args.apply_attack and args.attack_type == 'decision_time':
            # Using PGD attack during training
            model = train_with_pgd_attack(
                data, num_features, num_classes, args.dataset_name, args.lr, args.epochs,
                epsilon=0.0, alpha=0.01, num_iter=10, norm_type=args.norm_type
            )
            model_path = f'model/{args.dataset_name}_pgd_model.pth'
            test_loss, test_accuracy = test_with_pgd_attack(
                data, num_features, num_classes, model_path=model_path,
                epsilon=0.0, alpha=0.01, num_iter=10, norm_type=args.norm_type
            )

        elif args.apply_attack and args.attack_type == 'decision_time_K':
            # Using PGD attack during training
            model = train_with_pgd_top_k_node_attack(
                data, num_features, num_classes, args.dataset_name, args.lr, args.epochs,
                epsilon=0.1, alpha=0.01, num_iter=10, norm_type=args.norm_type, k=args.K
            )
            model_path = f'model/{args.dataset_name}_pgd_k_model.pth'
            test_loss, test_accuracy = test_with_pgd_top_k_node_attack(
                data, num_features, num_classes, model_path=model_path,
                epsilon=0.1, alpha=0.01, num_iter=10, norm_type=args.norm_type, k=10
            )

        elif args.apply_attack and args.attack_type == 'poisoning':
            pass
        
        else:
            train_losses, val_losses, val_accuracies, trained_model = train_model(
                data, num_features, num_classes, args.dataset_name, args.lr, args.patience, args.epochs
            )
            model_path = f'model/{args.dataset_name}_best_model.pth'
            test_loss, test_accuracy = test_model(data, num_features, num_classes, model_path=model_path)

    elif args.dataset_type == 'graph':
        data, dataset = load_data(dataset_type='graph', dataset_name=args.dataset_name)

        # Extract number of features and classes
        num_features = dataset.num_features
        num_classes = dataset.num_classes

        if args.model == 'GCN':
            model = GCN(num_features, num_classes)
        elif args.model == 'GAT':
            model = GAT(num_features, num_classes)
        else:
            raise ValueError("Unsupported model type")

        if args.apply_attack and args.attack_type == 'decision_time':
            # Using PGD attack during training
            model = train_with_pgd_attack(
                data, num_features, num_classes, args.dataset_name, args.lr, args.epochs,
                epsilon=0.0, alpha=0.01, num_iter=10, norm_type=args.norm_type
            )

            test_loss, test_accuracy = test_with_pgd_attack(
                data, num_features, num_classes, model_path='model/pgd_model.pth',
                epsilon=0.0, alpha=0.01, num_iter=10, norm_type=args.norm_type
            )

        else:
            train_losses, val_losses, val_accuracies, trained_model = train_model(
                data, num_features, num_classes, args.dataset_name, args.lr, args.patience, args.epochs
            )
            model_path = f'model/{args.dataset_name}_best_model.pth'
            test_loss, test_accuracy = test_model(data, num_features, num_classes, model_path=model_path)

            # test_loss, test_accuracy = test_model(data, num_features, num_classes, model_path='model/{args.dataset_name}_best_model.pth')
        # Apply attack if specified
        # if args.apply_attack and args.attack_type == 'decision_time':
        #     data = pgd_attack(model, data, epsilon=0.1, alpha=0.01, num_iter=500, norm_type=args.norm_type, criterion=criterion, labels=labels)

        # train_losses, val_losses, val_accuracies, trained_model = train_model(
        #     data, num_features, num_classes, args.lr, args.patience, args.epochs
        # )

        # plot_losses(train_losses, val_losses, args.dataset_name)
        # plot_accuracies(val_accuracies, args.dataset_name)

        # Testing phase
        # test_loss, test_accuracy = test_model(data, num_features, num_classes)

    else:
        raise ValueError("Invalid dataset type specified")

    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}\n')

    # Save results to log
    with open(f'logs/{args.dataset_name}_testing_result.log', 'w') as log_file:
        log_file.write(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}\n')

if __name__ == "__main__":
    main()
