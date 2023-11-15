import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from model import GCN

def test_model(data, num_features, num_classes, model_path='model/best_model.pth'):
    model = GCN(num_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Testing logic here
    test_nodes = torch.where(data.test_mask)[0]
    test_labels = data.y[test_nodes]
    criterion = CrossEntropyLoss()

    with torch.no_grad():
        out = model(data)
        test_loss = criterion(out[test_nodes], test_labels)
        _, preds = torch.max(out[test_nodes], 1)
        acc = accuracy_score(test_labels.cpu().numpy(), preds.cpu().numpy())

    print(f'Test Loss: {test_loss.item()}, Accuracy: {acc}')
    return test_loss.item(), acc
