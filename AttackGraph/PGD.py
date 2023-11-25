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
