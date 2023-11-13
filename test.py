import torch
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from train import train_model

def test_model():
    model, data = train_model()
    criterion = CrossEntropyLoss()
    # Test logic...
    print(f'Test Loss: {test_loss.item()}, Accuracy: {acc}')
