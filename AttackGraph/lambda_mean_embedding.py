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


import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# Assuming you've already loaded your data somewhere above and it's called 'data'
# Assuming you've defined or imported your model called 'model'

# Compute the mean of the node features
mean_feature = torch.mean(data.x, dim=0)

lambda_param = 0.98

# Assign the mean value to all nodes
data.x = lambda_param * mean_feature + (1 - lambda_param) * data.x

# Define the model's optimizer
optimizer = Adam(model.parameters(), lr=0.01)

# Define the loss function
criterion = CrossEntropyLoss()

def split_data(num_nodes, train_ratio=0.8, val_ratio=0.1):
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