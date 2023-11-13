import json
from torch_geometric.datasets import Planetoid, Reddit

def load_graph_data(dataset_name):
    if dataset_name == 'Cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
    elif dataset_name == 'Reddit':
        dataset = Reddit(root='/tmp/Reddit')
    else:
        raise ValueError(f"Unsupported graph dataset: {dataset_name}")
    data = dataset[0]
    return data, dataset

def load_text_data(file_path):
    with open(file_path, 'r') as file:
        data_list = json.load(file)
    return data_list

def load_data(dataset_type, dataset_name=None, file_path=None):
    if dataset_type == 'graph':
        return load_graph_data(dataset_name)
    elif dataset_type == 'text':
        return load_text_data(file_path)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
