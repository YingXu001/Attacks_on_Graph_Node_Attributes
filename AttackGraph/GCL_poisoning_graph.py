import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn.functional as F


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

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 256)
        self.conv2 = GATConv(256, 32)
        self.conv3 = GATConv(32, dataset.num_classes)

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



def save_to_file(output_string, file_name="graph_result/results.txt"):
    with open(file_name, "a") as f:  # "a" means append mode
        f.write(output_string + "\n")

# dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer', transform=NormalizeFeatures())

data = dataset[0]
test_nodes = torch.where(data.test_mask)[0].tolist()
test_labels = data.y[test_nodes]

# print(data)
save_to_file(str(data))




data.x.requires_grad = True

# Since all nodes are considered for perturbation, node_star_tensor is just a tensor of ones.
node_star_tensor = torch.ones(data.x.size(0), dtype=torch.bool)

# For evaluation, we'll still use y==3 labeled nodes
val_nodes_star = (data.y == 3).nonzero().squeeze().tolist()
val_labels_star = data.y[val_nodes_star]

train_nodes = list(range(data.x.size(0)))
train_labels = data.y[train_nodes]

original_embeddings = data.x.clone()




def train_and_validate(data, is_poisoned):
    best_acc = 0
    patience_counter = 0
    patience = 20

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



model_original = GCN()
# model_original = GAT()
# model_original = GCN(num_features=769, num_classes=4)
val_accuracies_pure, test_accuracies_pure = train_and_validate(data, is_poisoned=False)
# print("Pure Dataset Accuracies:", val_accuracies_pure)
save_to_file("Pure Dataset Accuracies:" + str(test_accuracies_pure))


data.x = torch.nn.Parameter(data.x.detach().clone())
optimizer_poison = Adam([data.x], lr=0.01)
_lambda = 0.4


for epoch in range(25):
    optimizer_poison.zero_grad()

    emb_star = data.x[node_star_tensor]
    emb_node = data.x[~node_star_tensor]

    similarity_matrix_star = F.cosine_similarity(emb_star.unsqueeze(0), emb_star.unsqueeze(1), dim=2)
    similarity_matrix_cross = F.cosine_similarity(emb_star.unsqueeze(0), emb_node.unsqueeze(1), dim=2)

    L_sim = torch.mean(similarity_matrix_star)
    L_dis = -torch.mean(torch.log(torch.exp(similarity_matrix_cross)))

    total_loss = _lambda * L_sim + (1-_lambda) * L_dis

    total_loss.backward()
    optimizer_poison.step()

changed_embeddings = data.x.detach().clone()
# print(torch.norm(original_embeddings - changed_embeddings))
# save_to_file(str(torch.norm(original_embeddings - changed_embeddings)))




model_poisoned = GCN()
# model_poisoned = GAT()
# model_poisoned = GCN(num_features=769, num_classes=4)

val_accuracies_star, test_accuracies_star = train_and_validate(data, is_poisoned=True)
save_to_file("Star Dataset Accuracies:" + str(val_accuracies_star))
save_to_file("Star Dataset Test Accuracy:" + str(test_accuracies_star))

torch.save(model_poisoned.state_dict(), "poisoned_model.pth")



model_loaded = GCN()
# model_loaded = GAT()
# model_loaded = GCN(num_features=769, num_classes=4)
model_loaded.load_state_dict(torch.load("poisoned_model.pth"))
model_loaded.eval()
accuracy_on_original_with_poisoned_model = []


data.x = original_embeddings
with torch.no_grad():
    out = model_loaded(data)
    _, preds = torch.max(out[test_nodes], 1)
    accuracy = accuracy_score(test_labels.cpu().numpy(), preds.cpu().numpy())
    print("Accuracy on original graph with poisoned model:", accuracy)
    save_to_file(f"Accuracy on original graph with poisoned model: {accuracy}")
    accuracy_on_original_with_poisoned_model.append(accuracy)


epochs = list(range(1, len(val_accuracies_star) + 1))
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_accuracies_star, marker='o', label='Star Dataset (Validation)')
plt.axhline(y=accuracy_on_original_with_poisoned_model, color='r', linestyle='-', label='Poisoned Model on Original Graph (Test)')

plt.title('Accuracy of Poisoned Model on Different Datasets')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graph_result/accuracy_comparison_with_poisoned_model.png")