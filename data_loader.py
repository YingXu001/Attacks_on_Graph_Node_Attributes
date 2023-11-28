import json
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.transforms import NormalizeFeatures

def load_graph_data(dataset_name):
    if dataset_name == 'Cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
    elif dataset_name == 'Reddit':
        dataset = Reddit(root='/tmp/Reddit', transform=NormalizeFeatures())
    elif dataset_name == 'CiteSeer':
        dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer', transform=NormalizeFeatures())
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
    
def filter_and_save_hellaswag(file_path, output_file, selected_labels):
    with open(file_path, 'r', encoding='utf-8') as file:
        data_list = json.load(file)  # Load the entire JSON file

    mini_train_data = [data for data in data_list if data.get('activity_label') in selected_labels]

    print(f"Selected {len(mini_train_data)} items.")

    for i, item in enumerate(mini_train_data):
        item['ind'] = i

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(mini_train_data, outfile, indent=4)

    return mini_train_data
