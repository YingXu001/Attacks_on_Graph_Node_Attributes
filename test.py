import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from model import GCN
from AttackGraph.PGD import pgd_attack

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

    return test_loss.item(), acc


def test_model_under_pgd_attack(model, data, test_nodes, criterion, epsilon, alpha, num_iter, norm_type):
    model.eval()

    test_data = data.clone()
    test_data = pgd_attack(model, test_data, epsilon, alpha, num_iter, norm_type)

    with torch.no_grad():
        out = model(test_data)
        test_loss = criterion(out[test_nodes], test_data.y[test_nodes])
        _, preds = torch.max(out[test_nodes], 1)
        acc = accuracy_score(test_data.y[test_nodes].cpu().numpy(), preds.cpu().numpy())

    return test_loss.item(), acc
