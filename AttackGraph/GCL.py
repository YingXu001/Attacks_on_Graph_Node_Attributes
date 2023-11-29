import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def save_to_file(output_string, file_name="graph_result/results.txt"):
    with open(file_name, "a") as f:  # "a" means append mode
        f.write(output_string + "\n")

dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())

data = dataset[0]
test_nodes = torch.where(data.test_mask)[0].tolist()
test_labels = data.y[test_nodes]

# print(data)
save_to_file(str(data))

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 256)
        self.conv2 = GCNConv(256, 32)
        self.conv3 = GCNConv(32, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x

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

def train_and_validate(data, is_poisoned):
    best_acc = 0
    patience_counter = 0
    patience = 30

    if is_poisoned:
        optimizer = Adam(list(model_poisoned.parameters()) + [data.x], lr=1e-3)
    else:
        optimizer = Adam(list(model_original.parameters()), lr=1e-3)
    
    criterion = CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    test_accuracies = []
    
    for epoch in range(400):
        if is_poisoned:
            model_poisoned.train()
            out = model_poisoned(data)
        else:
            model_original.train()
            out = model_original(data)

        # Training loss
        loss = criterion(out[train_nodes], train_labels)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = criterion(out[val_nodes_star], val_labels_star)
        _, preds = torch.max(out[val_nodes_star], 1)
        acc = accuracy_score(val_labels_star.cpu().numpy(), preds.cpu().numpy())
        val_losses.append(val_loss.item())
        val_accuracies.append(acc)

        # Logic for early stopping based on decreasing accuracy
        if epoch == 0:
            best_acc = acc
        elif acc < best_acc:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} due to decreasing accuracy.")
                save_to_file(f"Early stopping at epoch {epoch} due to decreasing accuracy.")
                break
        else:
            patience_counter = 0
            best_acc = acc

        # Evaluation on test data after training
        if is_poisoned:
            model_poisoned.eval()
        else:
            model_original.eval()

        with torch.no_grad():
            test_out = model_poisoned(data) if is_poisoned else model_original(data)
            _, test_preds = torch.max(test_out[test_nodes], 1)
            test_acc = accuracy_score(test_labels.cpu().numpy(), test_preds.cpu().numpy())
            test_accuracies.append(test_acc)
            
    return val_accuracies, test_accuracies

# Part 1: Using the original node_star features
model_original = GCN()
val_accuracies_pure, test_accuracies_pure = train_and_validate(data, is
