import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from model import GCN
from AttackGraph.PGD import pgd_attack, get_top_k_nodes_by_degree, pgd_top_k_node_attack

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


def test_with_pgd_attack(data, num_features, num_classes, model_path='model/pgd_model.pth', epsilon=0.1, alpha=0.01, num_iter=10, norm_type='Linf'):
    model = GCN(num_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = CrossEntropyLoss()
    test_nodes = torch.where(data.test_mask)[0]

    # Apply PGD attack
    perturbed_data = pgd_attack(model, data.clone(), epsilon, alpha, num_iter, norm_type, criterion, data.y)

    with torch.no_grad():
        out = model(perturbed_data)
        test_output = out[test_nodes]
        test_loss = criterion(test_output, perturbed_data.y[test_nodes])
        _, preds = torch.max(test_output, 1)
        acc = accuracy_score(perturbed_data.y[test_nodes].cpu().numpy(), preds.cpu().numpy())

    return test_loss.item(), acc


def test_with_pgd_top_k_node_attack(data, num_features, num_classes, model_path='model/pgd_model.pth', epsilon=0.1, alpha=0.01, num_iter=10, norm_type='Linf', k=10):
    model = GCN(num_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = CrossEntropyLoss()
    test_nodes = torch.where(data.test_mask)[0]

    top_k_nodes = get_top_k_nodes_by_degree(data.edge_index, data.num_nodes, k)

    top_k_test_nodes = [node for node in top_k_nodes if node in test_nodes]

    perturbed_data = pgd_top_k_node_attack(model, data.clone(), epsilon, alpha, num_iter, norm_type, criterion, data.y, k, top_k_test_nodes)

    with torch.no_grad():
        out = model(perturbed_data)
        test_output = out[top_k_test_nodes]
        test_labels = data.y[top_k_test_nodes]
        test_loss = criterion(test_output, test_labels)
        _, preds = torch.max(test_output, 1)
        acc = accuracy_score(test_labels.cpu().numpy(), preds.cpu().numpy())

    return test_loss.item(), acc


