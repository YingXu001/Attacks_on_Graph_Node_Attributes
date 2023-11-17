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
        labels: True labels for loss calculation.

    Returns:
        The perturbed graph data.
    """
    print("attack begin!!!!!")

    # Get node features and clone them to get a starting point for perturbations
    original_node_features = data.x.clone().detach()

    # Put the model in evaluation mode for the attack
    model.eval()

    for _ in range(num_iter):
        data.x.requires_grad = True
        model.zero_grad()
        
        out = model(data)
        loss = criterion(out, labels)
        loss.backward()

        with torch.no_grad():
            # Apply perturbations based on the selected norm
            if norm_type == 'Linf':
                perturbations = alpha * data.x.grad.sign()
                data.x = data.x + perturbations
                difference = data.x - original_node_features
                data.x = torch.clamp(difference, -epsilon, epsilon) + original_node_features
            elif norm_type == 'L2':
                perturbations = alpha * data.x.grad / (data.x.grad.norm(p=2, dim=-1, keepdim=True) + 1e-6)
                data.x = data.x + perturbations
                difference = data.x - original_node_features
                data.x = original_node_features + difference.renorm(p=2, dim=0, maxnorm=epsilon)
            elif norm_type == 'L1':
                perturbations = alpha * data.x.grad.sign()
                data.x = data.x + perturbations
                difference = data.x - original_node_features
                data.x = original_node_features + difference.renorm(p=1, dim=0, maxnorm=epsilon)

            # Ensure that node features stay valid (e.g., between certain range)
            data.x = torch.clamp(data.x, 0, 1)  # Assuming that valid range is [0, 1]

    return data
