import copy
import torch
import networkx as nx
import matplotlib.pyplot as plt

def add_random_noise(G, epsilon=0.01):
    G_prime = copy.deepcopy(G)  # Create a copy of the graph before adding noise

    for node, node_data in G_prime.nodes(data=True):
        current_embedding = node_data['feature_embedding']
        noise = torch.rand_like(current_embedding)  # Generate random noise
        noise = noise / torch.norm(noise)  # Normalize the noise
        noise = torch.clamp(noise, -epsilon, epsilon)  # Apply l_inf norm constraint
        poisoned_embedding = current_embedding + noise  # Add noise to the original embedding
        G_prime.nodes[node]['feature_embedding'] = poisoned_embedding  # Update embedding

    return G_prime
