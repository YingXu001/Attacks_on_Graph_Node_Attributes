# AddRandomNoise.py
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

# import copy
# import torch

# G_prime = copy.deepcopy(G)  # Create a copy of the graph before adding noise

# # Poisoning attack
# epsilon = 0.01  # set the value for l_inf norm, adjust as necessary
# for node, node_data in G_prime.nodes(data=True):
#     current_embedding = node_data['feature_embedding']
#     noise = torch.rand_like(current_embedding)  # generate random noise of the same shape
#     noise = noise / torch.norm(noise)  # normalize the noise
#     noise = torch.clamp(noise, -epsilon, epsilon)  # apply l_inf norm constraint
#     poisoned_embedding = current_embedding + noise  # add constrained noise to the original embedding
#     G_prime.nodes[node]['feature_embedding'] = poisoned_embedding  # replace the original embedding with the poisoned one

# nx.draw(G_prime, with_labels=True)
# plt.show()