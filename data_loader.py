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
    with open(file_path, 'rb') as file:  # Open in binary mode
        content = file.read()
        try:
            data_list = json.loads(content.decode('utf-8'))  # Try decoding as utf-8
        except UnicodeDecodeError:
            data_list = json.loads(content.decode('cp1252'))  # Fallback to cp1252 if utf-8 fails
    return data_list

def load_data(dataset_type, dataset_name=None, file_path=None):
    if dataset_type == 'graph':
        return load_graph_data(dataset_name)
    elif dataset_type == 'text':
        return load_text_data(file_path)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
