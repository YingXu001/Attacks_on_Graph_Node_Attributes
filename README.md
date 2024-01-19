# Attacks on Node Attributes in Graph Neural Networks

## Paper Link
Accepted by [AAAI AICS Workshop 2024](http://aics.site/AICS2024/AICS_Attacks_on_Node_Attributes_in_Graph_Neural_Networks.pdf)

## Abstract

Graphs are commonly used to model complex networks prevalent in modern social media and literacy applications. Our research investigates the vulnerability of these graphs through the application of feature based adversarial attacks, focusing on both decision-time attacks and poisoning attacks. In contrast to state-of-the-art models like Net Attack and Meta Attack, which target node attributes and graph structure, our study specifically targets node attributes. For our analysis, we utilized the text dataset Hellaswag and graph datasets Cora and CiteSeer, providing a diverse basis for evaluation. Our findings indicate that decision-time attacks using Projected Gradient Descent (PGD) are more potent compared to poisoning attacks that employ Mean Node Embeddings and Graph Contrastive Learning strategies. This provides insights for graph data security, pinpointing where graph-based models are most vulnerable and thereby informing the development of stronger defense mechanisms against such attacks.

Our research is supported by a comprehensive code repository, which includes all scripts and data used in our experiments. This repository is openly accessible and can be found at https://github.com/YingXu001/Attack_Graph, providing a valuable resource for further exploration and development in the field of graph data security.
