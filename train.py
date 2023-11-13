import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from model import GCN
from data_loader import load_data

def train_model():
    data, dataset = load_data()
    model = GCN(dataset.num_features, dataset.num_classes)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()
    # Rest of the training logic including early stopping...
    return model, data
