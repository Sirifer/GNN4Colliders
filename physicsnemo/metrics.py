import torch
import numpy as np
import torch.nn.functional as F

def bce(input, target, weights=None):

    if input.shape != target.shape:
        if input.shape[-1] == 1 and input.shape[:-1] == target.shape:
            input = input.squeeze(-1)
        elif target.shape[-1] == 1 and target.shape[:-1] == input.shape:
            target = target.squeeze(-1)

    loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    return torch.mean(loss)

def weighted_bce(input, target, weights=None):
    """
    Compute a weighted and label-normalized binary cross entropy (BCE) loss.

    For each unique label in the target tensor, the BCE loss is computed and weighted,
    then normalized by the sum of weights for that label. The final loss is the mean
    of these per-label normalized losses.

    Args:
        input (Tensor): Predicted logits of shape (N, ...).
        target (Tensor): Ground truth labels of shape (N, ...), with discrete label values.
        weights (Tensor or None): Optional tensor of per-sample weights, same shape as input/target.

    Returns:
        Tensor: Scalar tensor representing the normalized weighted BCE loss.
    """

    if input.shape != target.shape:
        if input.shape[-1] == 1 and input.shape[:-1] == target.shape:
            input = input.squeeze(-1)
        elif target.shape[-1] == 1 and target.shape[:-1] == input.shape:
            target = target.squeeze(-1)

    # Compute per-element BCE loss (no reduction)
    loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

    # If weights not provided, use ones
    if weights is None:
        weights = torch.ones_like(loss)

    unique_labels = torch.unique(target)
    normalized_losses = []
    for label in unique_labels:
        label_mask = (target == label)  # This will be a bool tensor
        # Defensive: make sure mask is bool
        if label_mask.dtype != torch.bool:
            label_mask = label_mask.bool()
        label_weights = weights[label_mask]
        label_losses = loss[label_mask]
        weight_sum = label_weights.sum()
        if weight_sum > 0:
            label_loss = (label_weights * label_losses).sum() / weight_sum
            normalized_losses.append(label_loss)

    if normalized_losses:
        return torch.stack(normalized_losses).mean()
    else:
        return torch.tensor(0.0, device=input.device)
    

def roc_auc_score(classes : np.ndarray,
               predictions : np.ndarray,
               weights : np.ndarray = None) -> float:
    """
    Calculating ROC AUC score as the probability of correct ordering
    """

    if weights is None:
        weights = np.ones_like(predictions)

    assert len(classes) == len(predictions) == len(weights)
    assert classes.ndim == predictions.ndim == weights.ndim == 1
    class0, class1 = sorted(np.unique(classes))

    data = np.empty(
            shape=len(classes),
            dtype=[('c', classes.dtype),
                   ('p', predictions.dtype),
                   ('w', weights.dtype)]
        )
    data['c'], data['p'], data['w'] = classes, predictions, weights

    data = data[np.argsort(data['c'])]
    data = data[np.argsort(data['p'], kind='mergesort')] # here we're relying on stability as we need class orders preserved

    correction = 0.
    # mask1 - bool mask to highlight collision areas
    # mask2 - bool mask with collision areas' start points
    mask1 = np.empty(len(data), dtype=bool)
    mask2 = np.empty(len(data), dtype=bool)
    mask1[0] = mask2[-1] = False
    mask1[1:] = data['p'][1:] == data['p'][:-1]
    if mask1.any():
        mask2[:-1] = ~mask1[:-1] & mask1[1:]
        mask1[:-1] |= mask1[1:]
        ids, = mask2.nonzero()
        correction = sum([((dsplit['c'] == class0) * dsplit['w'] * msplit).sum() * 
                          ((dsplit['c'] == class1) * dsplit['w'] * msplit).sum()
                          for dsplit, msplit in zip(np.split(data, ids), np.split(mask1, ids))]) * 0.5
 
    weights_0 = data['w'] * (data['c'] == class0)
    weights_1 = data['w'] * (data['c'] == class1)
    cumsum_0 = weights_0.cumsum()

    return ((cumsum_0 * weights_1).sum() - correction) / (weights_1.sum() * cumsum_0[-1])
