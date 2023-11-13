import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.decomposition import PCA
import scipy.sparse as sp
import torch
import random

def create_graph(data_list, node_embeddings, threshold):
    G = nx.Graph()
    label_encoder = LabelEncoder()
    all_labels = [data.get("activity_label") for data in data_list]
    label_encoder.fit(all_labels)
    encoded_labels = label_encoder.transform(all_labels)

    for i, embedding in enumerate(node_embeddings):
        G.add_node(i, label=encoded_labels[i], feature_embedding=embedding.numpy())
        for j in range(i):
            cosine_similarity = torch.nn.functional.cosine_similarity(embedding, node_embeddings[j], dim=0)
            if cosine_similarity > threshold:
                G.add_edge(j, i)

    # Add random edge for isolated nodes
    for node in G.nodes():
        if len(list(G.neighbors(node))) == 0:
            random_neighbor = random.choice(list(G.nodes()))
            while random_neighbor == node:
                random_neighbor = random.choice(list(G.nodes()))
            G.add_edge(node, random_neighbor)

    return G, label_encoder

def visualize_graph(G):
    # Get the labels of the nodes
    labels_dict = nx.get_node_attributes(G, 'label')
    nodes_list = list(G.nodes())
    labels_list = [labels_dict[node] for node in nodes_list]

    # Define a colormap to map labels to colors
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    label_to_color = {label: color for label, color in zip(set(labels_dict.values()), colors)}

    # Use PCA to reduce the dimensionality of the embeddings
    node_embeddings_array = np.vstack([G.nodes[node]['feature_embedding'] for node in nodes_list])
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(node_embeddings_array)

    # Scatter plot
    plt.figure(figsize=(8, 8))
    for i, embedding in enumerate(reduced_embeddings):
        plt.scatter(embedding[0], embedding[1], color=label_to_color[labels_list[i]])
    plt.show()

def save_graph_data(G):
    adj_matrix = nx.adjacency_matrix(G)

    attr_matrix_list = [G.nodes[node]['feature_embedding'] for node in G.nodes()]
    attr_matrix = np.array(attr_matrix_list)
    labels_list = [G.nodes[node]['label'] for node in G.nodes()]
    labels = torch.from_numpy(np.array(labels_list)).long()

    attr_matrix_sparse = sp.csr_matrix(attr_matrix)

    np.savez('mixed_graph.npz', adj_data=adj_matrix.data, adj_indices=adj_matrix.indices, adj_indptr=adj_matrix.indptr, adj_shape=adj_matrix.shape, attr_data=attr_matrix_sparse.data, attr_indices=attr_matrix_sparse.indices, attr_indptr=attr_matrix_sparse.indptr, attr_shape=attr_matrix_sparse.shape, labels=labels)
