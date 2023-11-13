import networkx as nx
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp

def build_graph(data_list, embeddings, threshold=0.88):
    G = nx.Graph()
    label_encoder = LabelEncoder()
    all_labels = [data.get("activity_label") for data in data_list]
    label_encoder.fit(all_labels)
    encoded_labels = label_encoder.transform(all_labels)

    for i, (data, embedding) in enumerate(zip(data_list, embeddings)):
        G.add_node(i, label=encoded_labels[i], feature_embedding=embedding)
        # Add edges based on cosine similarity
        # ...

    # For nodes with no neighbors, add a random edge
    # ...

    return G, label_encoder

def save_graph_data(G):
    # Save matrices to a .npz file
    # ...

# Example usage
if __name__ == '__main__':
    # Assume embeddings and data_list are loaded from previous scripts
    G, label_encoder
