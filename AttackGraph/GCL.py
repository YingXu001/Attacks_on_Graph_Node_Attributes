from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
from torch_geometric.datasets import Planetoid
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())

data = dataset[0]

print(data)

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# 1. Mark all nodes as node_star for adversarial perturbation
data.x.requires_grad = True

# Since all nodes are considered for perturbation, node_star_tensor is just a tensor of ones.
node_star_tensor = torch.ones(data.x.size(0), dtype=torch.bool)

# For evaluation, we'll still use y==3 labeled nodes
val_nodes_star = (data.y == 3).nonzero().squeeze().tolist()
val_labels_star = data.y[val_nodes_star]

# Define the training nodes and their corresponding labels
# Since all nodes are considered for perturbation, we can train on all nodes as well.
train_nodes = list(range(data.x.size(0)))
train_labels = data.y[train_nodes]

original_embeddings = data.x.clone()

