import os
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from model import GCN

# Check if the model directory exists, create it if not
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

def train_model(data, num_features, num_classes, lr, patience, epochs):
    # model = GCN(dataset.num_features, dataset.num_classes)
    model = GCN(num_features, num_classes)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    train_nodes = torch.where(data.train_mask)[0]
    val_nodes = torch.where(data.val_mask)[0]
    test_nodes = torch.where(data.test_mask)[0]

    train_labels = data.y[train_nodes]
    val_labels = data.y[val_nodes]
    test_labels = data.y[test_nodes]

    best_val_loss = float('inf')
    counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    # with open(log_filename, 'w') as log_file:
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data)

        loss = criterion(out[train_nodes], train_labels)
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_loss = criterion(val_out[val_nodes], val_labels)
            _, preds = torch.max(val_out[val_nodes], 1)
            acc = accuracy_score(val_labels.cpu().numpy(), preds.cpu().numpy())

        val_losses.append(val_loss.item())
        val_accuracies.append(acc)

        # log_file.write(f'Epoch: {epoch+1}, Val Loss: {val_loss.item()}, Accuracy: {acc}\n')
        # print((f'Epoch: {epoch+1}, Val Loss: {val_loss.item()}, Accuracy: {acc}\n'))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            model_path = os.path.join(model_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                print((f'Epoch: {epoch+1}, Val Loss: {val_loss.item()}, Accuracy: {acc}\n'))
                break

    # You might also want to return or store the training statistics like losses and accuracies
    return train_losses, val_losses, val_accuracies
