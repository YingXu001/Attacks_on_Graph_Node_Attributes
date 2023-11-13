import argparse
from train import train_model
from test import test_model
from data_loader import load_data

def parse_args():
    parser = argparse.ArgumentParser(description="GCN for different datasets")
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name (Cora, Hellaswag, etc.)')
    # Add other arguments as needed
    return parser.parse_args()

def main():
    args = parse_args()

    # Based on the dataset argument, load different datasets
    if args.dataset == 'Cora':
        data, dataset = load_data(args.dataset)
        train_model(data, dataset)
        test_model(data, dataset)
    elif args.dataset == 'Hellaswag':
        # Implement data loading, training, and testing for Hellaswag
        pass
    # Add other datasets as needed

if __name__ == "__main__":
    main()
