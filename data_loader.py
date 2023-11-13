from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.transforms import NormalizeFeatures

def load_data(dataset_name):
    if dataset_name == 'Cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
    elif dataset_name == 'Reddit':
        dataset = Reddit(root='/tmp/Reddit')
    elif dataset_name == 'Hellaswag':
        pass
    # Add other datasets as needed

    data = dataset[0]
    return data, dataset
