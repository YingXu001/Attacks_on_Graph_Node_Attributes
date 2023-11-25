import torch

def pgd_attack(model, data, epsilon, alpha, num_iter, norm_type, criterion, labels):
    """
    Performs the Projected Gradient Descent (PGD) attack on the graph data.

    Args:
        model: The graph neural network model.
        data: The graph data.
        epsilon: The maximum perturbation allowed.
        alpha: The step size for PGD.
        num_iter: Number of iterations for PGD.
        norm_type: The type of norm ('Linf', 'L2', 'L1').
        criterion: Loss function.

    Returns:
        The perturbed graph data.
    """

    # print("!!!attack begin!!!")

    # Get node features and clone them to get a starting point for perturbations
    original_node_features = data.x.clone().detach()

    for _ in range(num_iter):
        data.x.requires_grad = True
        model.zero_grad()
        out = model(data)
        # print("Output shape in PGD attack:", out.shape)

        loss = criterion(out, labels)
        loss.backward()

        with torch.no_grad():
            # Apply perturbations based on the selected norm
            perturbations = alpha * data.x.grad.sign()
            data.x = data.x + perturbations
            
            if norm_type == 'Linf':
                # Ensure that perturbations stay within epsilon-ball
                difference = data.x - original_node_features
                difference = torch.clamp(difference, -epsilon, epsilon)
                data.x = original_node_features + difference
            
            if norm_type == 'L2':
                # Ensure that perturbations stay within epsilon-ball in terms of L2 norm
                difference = data.x - original_node_features
                difference_norm = torch.norm(difference, p=2, dim=-1, keepdim=True)
                difference = torch.where(difference_norm > epsilon, difference * (epsilon / difference_norm), difference)
                data.x = original_node_features + difference

            if norm_type == 'L1':
                # Ensure that perturbations stay within epsilon-ball in terms of L1 norm
                difference = data.x - original_node_features
                difference = torch.clamp(difference, -alpha, alpha)  # L∞ norm constraint on each element
                difference_norm = torch.norm(difference, p=1, dim=-1, keepdim=True)
                difference = torch.where(difference_norm > epsilon, difference * (epsilon / difference_norm), difference)
                data.x = original_node_features + difference

            # Ensure that node features stay valid (e.g., between certain range)
            data.x = torch.clamp(data.x, 0, 1)  # Assuming that valid range is [0, 1]

    return data


def get_top_k_nodes_by_degree(edge_index, num_nodes, k):
    edge_list = edge_index.numpy()
    degree_sequence = [0] * num_nodes
    for node in edge_list[0]:
        degree_sequence[node] += 1
    top_k_nodes = sorted(range(num_nodes), key=lambda x: degree_sequence[x], reverse=True)[:k]
    return top_k_nodes


def pgd_top_k_node_attack(model, data, epsilon, alpha, num_iter, norm_type, criterion, labels, k, test_nodes=None):
    # Get node features and clone them to get a starting point for perturbations
    original_node_features = data.x.clone().detach()

    # Get the top K nodes by degree
    # top_k_nodes = get_top_k_nodes_by_degree(data, k)
    top_k_nodes = get_top_k_nodes_by_degree(data.edge_index, data.num_nodes, k)

    nodes_to_attack = top_k_nodes
    if test_nodes is not None:
        nodes_to_attack = [node for node in top_k_nodes if node in test_nodes]


    for _ in range(num_iter):
        data.x.requires_grad = True
        model.zero_grad()
        out = model(data)

        attack_output = out[nodes_to_attack]
        attack_labels = labels[nodes_to_attack]
        loss = criterion(attack_output, attack_labels)
        # loss = criterion(out, labels)
        loss.backward()

        with torch.no_grad():
            # Apply perturbations based on the selected norm
            perturbations = alpha * data.x.grad.sign()
            
            # Apply perturbations only to the top K nodes
            for node in top_k_nodes:
                if norm_type == 'Linf':
                    perturbation = torch.clamp(perturbations[node], -epsilon, epsilon)
                    data.x[node] = original_node_features[node] + perturbation
                elif norm_type == 'L2':
                    perturbation = perturbations[node]
                    perturbation_norm = torch.norm(perturbation, p=2)
                    if perturbation_norm > epsilon:
                        perturbation = perturbation * (epsilon / perturbation_norm)
                    data.x[node] = original_node_features[node] + perturbation
                elif norm_type == 'L1':
                    perturbation = torch.clamp(perturbations[node], -alpha, alpha)
                    perturbation_norm = torch.norm(perturbation, p=1)
                    if perturbation_norm > epsilon:
                        perturbation = perturbation * (epsilon / perturbation_norm)
                    data.x[node] = original_node_features[node] + perturbation

            # Ensure that node features stay valid
            data.x = torch.clamp(data.x, 0, 1)

    return data


