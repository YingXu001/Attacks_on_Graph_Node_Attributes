import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
import random

def create_graph(data_list, node_embeddings, threshold):
    G = nx.Graph()
    label_encoder = LabelEncoder()
    all_labels = [data.get("activity_label") for data in data_list]
    label_encoder.fit(all_labels)
    encoded_labels = label_encoder.transform(all_labels)

    for i, embedding in enumerate(node_embeddings):
        # print(f"Processing node {i}/{len(node_embeddings)}")
        G.add_node(i, label=encoded_labels[i], feature_embedding=embedding.numpy())
        for j in range(i):
            cosine_similarity = torch.nn.functional.cosine_similarity(embedding, node_embeddings[j], dim=0)
            if cosine_similarity > threshold:
                G.add_edge(j, i)
    return G, label_encoder

def visualize_graph(G, file_path='plots/graph_visualization.png'):
    # Get the labels of the nodes
    labels_dict = nx.get_node_attributes(G, 'label')
    nodes_list = list(G.nodes())
    labels_list = [labels_dict[node] for node in nodes_list]

    # Define a colormap to map labels to colors
    colors = ['red', 'green', 'blue', 'purple']
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

    np.savez('data/mixed_graph.npz', 
             adj_data=adj_matrix.data, adj_indices=adj_matrix.indices,
             adj_indptr=adj_matrix.indptr, adj_shape=adj_matrix.shape,
             attr_data=attr_matrix_sparse.data, attr_indices=attr_matrix_sparse.indices,
             attr_indptr=attr_matrix_sparse.indptr, attr_shape=attr_matrix_sparse.shape,
             labels=labels)

def load_graph_data(graph_file):
    # Load data from the .npz file
    data = np.load(graph_file, allow_pickle=True)

    # Get adjacency matrix
    adj_data = data['adj_data']
    adj_indices = data['adj_indices']
    adj_indptr = data['adj_indptr']
    adj_shape = data['adj_shape']
    adj_matrix = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)

    # Convert the sparse adjacency matrix to edge_index format required by PyTorch Geometric
    edge_index = torch.tensor(np.vstack(adj_matrix.nonzero()), dtype=torch.long)

    # Get attribute matrix and convert it to a dense tensor
    attr_data = data['attr_data']
    attr_indices = data['attr_indices']
    attr_indptr = data['attr_indptr']
    attr_shape = data['attr_shape']
    attr_matrix = sp.csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape)
    x = torch.tensor(attr_matrix.todense(), dtype=torch.float)

    # Get labels and convert them to a PyTorch tensor
    labels = torch.tensor(data['labels'], dtype=torch.long)

    # Create a PyTorch Geometric Data object
    torch_data = Data(x=x, edge_index=edge_index, y=labels)


    return torch_data