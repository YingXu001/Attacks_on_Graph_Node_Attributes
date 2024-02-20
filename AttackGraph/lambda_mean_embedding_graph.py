import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 256)
        self.conv2 = GCNConv(256, 32)
        self.conv3 = GCNConv(32, num_classes)

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

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 256)
        self.conv2 = GATConv(256, 32)
        self.conv3 = GATConv(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # x = torch.nn.functional.relu(x)
        x = torch.nn.functional.leaky_relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = torch.nn.functional.relu(x)
        x = torch.nn.functional.leaky_relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x


# dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer', transform=NormalizeFeatures())

data = dataset[0]

print(f"Data is loaded, {data}")

model = GCN(num_features=1433, num_classes=7)
model = GCN(num_features=3703, num_classes=6)

# Compute the mean of the node features
mean_feature = torch.mean(data.x, dim=0)

lambda_param = 0.98

# Assign the mean value to all nodes
data.x = lambda_param * mean_feature + (1 - lambda_param) * data.x

# Define the model's optimizer
optimizer = Adam(model.parameters(), lr=0.01)

# Define the loss function
criterion = CrossEntropyLoss()

def split_data(num_nodes, train_ratio=0.7, val_ratio=0.1):
    """
    Split nodes into training, validation and test sets.
    
    :param num_nodes: Total number of nodes
    :param train_ratio: Proportion of nodes to be used for training
    :param val_ratio: Proportion of nodes to be used for validation
    :return: train_mask, val_mask, test_mask
    """
    
    # Create a permutation of nodes
    perm_nodes = torch.randperm(num_nodes)
    
    # Calculate sizes
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[perm_nodes[:train_size]] = True
    val_mask[perm_nodes[train_size:train_size+val_size]] = True
    test_mask[perm_nodes[train_size+val_size:]] = True
    
    return train_mask, val_mask, test_mask

# Usage:
num_nodes = len(data.x)  # Assuming `data.x` contains the features of nodes
train_mask, val_mask, test_mask = split_data(num_nodes)

train_nodes = torch.where(train_mask)[0]
val_nodes = torch.where(val_mask)[0]
test_nodes = torch.where(test_mask)[0]

train_labels = data.y[train_nodes]
val_labels = data.y[val_nodes]
test_labels = data.y[test_nodes]


# Parameters for early stopping
patience = 40
best_val_loss = float('inf')
counter = 0

train_losses = []
val_losses = []
val_accuracies = []

# Training loop
for epoch in range(500):  # 200 epochs
    model.train()
    optimizer.zero_grad()  # Clear gradients
    
    out = model(data)  # Forward pass on the entire graph
    
    loss = criterion(out[train_nodes], train_labels)  # Compute the loss for training nodes
    train_losses.append(loss.item())
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = criterion(out[val_nodes], val_labels)
        _, preds = torch.max(out[val_nodes], 1)
        acc = accuracy_score(val_labels.cpu().numpy(), preds.cpu().numpy())
    val_losses.append(val_loss.item())
    val_accuracies.append(acc)
    print(f'Val Loss: {val_loss.item()}, Accuracy: {acc}')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# Plot loss over time
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy over time
plt.figure(figsize=(10,5))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Test
model.eval()
with torch.no_grad():
    test_loss = criterion(out[test_nodes], test_labels)
    _, preds = torch.max(out[test_nodes], 1)
    acc = accuracy_score(test_labels.cpu().numpy(), preds.cpu().numpy())
print(f'Test Loss: {test_loss.item()}, Accuracy: {acc}')