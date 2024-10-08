import torch
import itertools

def compute_mean_prob_and_indices(probs):
    """
    Computes the average probabilities for all possible combinations of indices in a tensor of size (N, 4, 9).
    Also returns the indices of these combinations.

    Args:
        probs (torch.Tensor): A tensor of shape (N, 4, 9) representing the probabilities.

    Returns:
        tuple: A tuple containing:
            - mean_probs (torch.Tensor): A tensor of shape (N, 9**4) with averaged probabilities.
            - combinations_tensor (torch.Tensor): A tensor of shape (9**4, 4) with the combinations of indices.
            - indices_tensor (torch.Tensor): A tensor of shape (N, 9**4, 4) with the coordinates (indices).
    """
    N, _, num_classes = probs.shape
    num_combinations = num_classes ** 4

    # Generate all index combinations
    combinations = list(itertools.product(range(num_classes), repeat=4))
    combinations_tensor = torch.tensor(combinations, device=probs.device)  # shape (9^4, 4)

    # Prepare output tensors
    mean_probs = torch.zeros((N, num_combinations), device=probs.device)  # shape (N, 9^4)
    indices_tensor = torch.zeros((N, num_combinations, 4), device=probs.device, dtype=torch.long)  # shape (N, 9^4, 4)

    for i, comb in enumerate(combinations_tensor):
        # Gather the probabilities for each combination
        comb_probs = probs[:, torch.arange(4), comb]  # shape (N, 4)
        mean_probs[:, i] = torch.mean(comb_probs, dim=1)  # shape (N)
        indices_tensor[:, i] = comb  # shape (N, 4), broadcasted to (N, 9^4, 4)

    return mean_probs, combinations_tensor, indices_tensor


def compute_mean_prob_and_average_scores(probs, class_scores):
    """
    Computes the average probabilities and average class scores for all possible combinations 
    of indices in a tensor of size (N, 4, 9). Also returns the indices of these combinations.

    Args:
        probs (torch.Tensor): A tensor of shape (N, 4, 9) representing the probabilities.
        class_scores (torch.Tensor): A tensor of shape (N, 4, 9, C) representing the class scores.

    Returns:
        tuple: A tuple containing:
            - mean_probs (torch.Tensor): A tensor of shape (N, 9**4) with averaged probabilities.
            - average_class_scores (torch.Tensor): A tensor of shape (N, 9**4, C) with average class scores.
            - combinations_tensor (torch.Tensor): A tensor of shape (9**4, 4) with the combinations of indices.
            - indices_tensor (torch.Tensor): A tensor of shape (N, 9**4, 4) with the coordinates (indices).
    """
    N, _, num_classes = probs.shape
    _, _, _, C = class_scores.shape
    num_combinations = num_classes ** 4

    # Generate all index combinations
    combinations = list(itertools.product(range(num_classes), repeat=4))
    combinations_tensor = torch.tensor(combinations, device=probs.device)  # shape (9^4, 4)

    # Prepare output tensors
    mean_probs = torch.zeros((N, num_combinations), device=probs.device)  # shape (N, 9^4)
    average_class_scores = torch.zeros((N, num_combinations, C), device=probs.device)  # shape (N, 9^4, C)
    indices_tensor = torch.zeros((N, num_combinations, 4), device=probs.device, dtype=torch.long)  # shape (N, 9^4, 4)

    for i, comb in enumerate(combinations_tensor):
        # Gather the probabilities for each combination
        comb_probs = probs[:, torch.arange(4), comb]  # shape (N, 4)
        comb_class_scores = class_scores[:, torch.arange(4), comb, :]  # shape (N, 4, C)

        mean_probs[:, i] = torch.mean(comb_probs, dim=1)  # shape (N)
        average_class_scores[:, i, :] = torch.mean(comb_class_scores, dim=1)  # shape (N, C)
        indices_tensor[:, i] = comb  # shape (N, 4), broadcasted to (N, 9^4, 4)

    return mean_probs, average_class_scores, combinations_tensor, indices_tensor

# Example usage
N = 2  # Number of samples
C = 5  # Number of classes
probs = torch.rand(N, 4, 9)
probs = probs / probs.sum(dim=-1, keepdim=True)  # Normalize to make probabilities sum to 1
class_scores = torch.rand(N, 4, 9, C)  # Example with C classes
class_scores = class_scores.softmax(dim=-1)

mean_probs, weighted_class_scores, combinations_tensor, indices_tensor = compute_mean_prob_and_average_scores(probs, class_scores)
import pdb; pdb.set_trace()
print("Mean probabilities shape:", mean_probs.shape)  # Should be (N, 6561)
print("Weighted class scores shape:", weighted_class_scores.shape)  # Should be (N, 6561, C)
print("Combinations tensor shape:", combinations_tensor.shape)  # Should be (6561, 4)
print("Indices tensor shape:", indices_tensor.shape)  # Should be (N, 6561, 4)