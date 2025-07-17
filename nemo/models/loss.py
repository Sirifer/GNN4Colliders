from torch import nn
import torch
from root_gnn_base import utils
import numpy as np

class MaskedLoss():
    def __init__(self, mask = []):
        self.mask = mask

    def make_mask(self, targets):
        mask = torch.ones_like(targets[:,0])
        for m in self.mask:
            if m['op'] == 'eq':
                mask[targets[:,m['idx']] == m['val']] = 0
            elif m['op'] == 'gt':
                mask[targets[:,m['idx']] > m['val']] = 0
            elif m['op'] == 'lt':
                mask[targets[:,m['idx']] < m['val']] = 0
            elif m['op'] == 'ge':
                mask[targets[:,m['idx']] >= m['val']] = 0
            elif m['op'] == 'le':
                mask[targets[:,m['idx']] <= m['val']] = 0
            elif m['op'] == 'ne':
                mask[targets[:,m['idx']] != m['val']] = 0
            else:
                raise ValueError(f'Unknown mask op {m["op"]}')
        return mask == 1

class MaskedL1Loss(MaskedLoss):
    def __init__(self, mask = [], index = 0):
        super().__init__(mask)
        self.index = index
        self.loss = nn.L1Loss()

    def __call__(self, logits, targets):
        mask = self.make_mask(targets)
        return self.loss(logits[mask], targets[mask][:,self.index])

class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.loss = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
    
    def __call__(self, logits, targets):
        return self.loss(logits[:,0], targets.float())

class MultiScore():
    def __init__(self, scores):
        self. score_fcns = []
        self.start_idx = []
        self.end_idx = []
        for score in scores:
            self.score_fcns.append(utils.buildFromConfig(score))
            self.start_idx.append(score['start_idx'])
            self.end_idx.append(score['end_idx'])
    
    def __call__(self, last_layer):
        scores = []
        for i in range(len(self.score_fcns)):
            scores.append(self.score_fcns[i](last_layer[:, self.start_idx[i]:self.end_idx[i]]))
        return torch.cat(scores, dim=1)

class MultiLoss():
    def __init__(self, losses):
        self.loss_fcns = []
        self.label_start_idx = []
        self.label_end_idx = []
        self.output_start_idx = []
        self.output_end_idx = []
        self.weights = []
        self.label_types = []
        for loss in losses:
            self.loss_fcns.append(utils.buildFromConfig(loss))
            self.label_start_idx.append(loss['label_start_idx'])
            self.label_end_idx.append(loss['label_end_idx'])
            self.output_start_idx.append(loss['output_start_idx'])
            self.output_end_idx.append(loss['output_end_idx'])
            self.weights.append(loss.get('weight', 1.0))
            self.label_types.append(loss.get('label_type', 'float'))

    def __call__(self, logits, targets):
        loss = 0
        # print(logits.shape, targets.shape)
        for i in range(len(self.loss_fcns)):
            if self.label_types[i] == 'int':
                # print('loss', i, self.label_start_idx[i], self.label_end_idx[i], self.output_start_idx[i], self.output_end_idx[i])
                # print(logits[:, self.output_start_idx[i]:self.output_end_idx[i]].shape, targets[:, self.label_start_idx[i]].shape)
                loss += self.weights[i] * self.loss_fcns[i](logits[:, self.output_start_idx[i]:self.output_end_idx[i]], targets[:, self.label_start_idx[i]].to(int))
            elif self.label_end_idx[i] - self.label_start_idx[i] == 1:
                loss += self.weights[i] * self.loss_fcns[i](logits[:, self.output_start_idx[i]:self.output_end_idx[i]], targets[:, self.label_start_idx[i]])
            else:
                # print('loos', i, self.label_start_idx[i], self.label_end_idx[i], self.output_start_idx[i], self.output_end_idx[i])
                # print(logits[:, self.output_start_idx[i]:self.output_end_idx[i]].shape, targets[:, self.label_start_idx[i]:self.label_end_idx[i]].shape)
                loss += self.weights[i] * self.loss_fcns[i](logits[:, self.output_start_idx[i]:self.output_end_idx[i]], targets[:, self.label_start_idx[i]:self.label_end_idx[i]])
        return loss
    
class AdvLoss():
    def __init__(self, loss, adv_loss, adv_weight=1.0):
        self.loss_fcn = utils.buildFromConfig(loss)
        self.adv_loss_fcn = utils.buildFromConfig(adv_loss)
        self.adv_weight = adv_weight

    def __call__(self, logits, targets):
        mask = targets[:,0] == 0
        loss = self.loss_fcn(logits[:,0], targets[:,0])
        adv_loss = self.adv_loss_fcn(logits[mask][:,1], targets[mask])
        return loss - self.adv_weight * adv_loss

class MassWindowAdvLoss(AdvLoss):
    def __call__(self, logits, targets):
        mask = (targets[:,0] == 0) & (targets[:,1] > 5) & (targets[:,1] < 25)
        print(mask, mask.shape, mask.sum())
        loss = self.loss_fcn(logits[:,0], targets[:,0])
        print(loss)
        adv_loss = self.adv_loss_fcn(logits[mask][:,1], targets[mask][:,1])
        print(adv_loss)
        return loss - self.adv_weight * adv_loss

class KDELoss(MaskedLoss):
    def __init__(self, mask = [], index = 0):
        self.index = index
        super().__init__(mask)

    def __call__(self, logits, targets):
        mask = self.make_mask(targets)
        logits = logits[mask]
        targets = targets[mask][:,self.index]
        N = logits.shape[0]
        masses = targets / torch.sqrt(torch.mean(targets**2))
        scores = logits[:,0]  / torch.sqrt(torch.mean(logits**2))

        factor_2d = (1.0*N) ** (-2/6)
        covs = (factor_2d * torch.var(masses), factor_2d * torch.var(scores))

        m_diffs = torch.unsqueeze(masses, 1) - torch.unsqueeze(masses, 0)
        s_diffs = torch.unsqueeze(scores, 1) - torch.unsqueeze(scores, 0)

        ymm = torch.exp(- (m_diffs**2) / (4 * covs[0]))
        yss = torch.exp(- (s_diffs**2) / (4 * covs[1]))

        integral_rho_2d_rho_2d = torch.einsum('ij,ij->', ymm, yss)
        integral_rho_1d_rho_1d = torch.einsum('ij,kl->', ymm, yss)
        integral_rho_2d_rho_1d = torch.einsum('ij,ik->', ymm, yss)
        raw_integral = integral_rho_2d_rho_2d - 2 * integral_rho_2d_rho_1d / N + integral_rho_1d_rho_1d / N**2
        return raw_integral / (4 * torch.pi * N**2)

class MultiLabelLoss():
    def __init__(self, label_names, label_types, label_weights = None):
        self.loss_fcn = []
        if (label_weights):
            self.weights = torch.tensor(label_weights)
        else:
            self.weights = torch.ones(len(label_types))
        for type in label_types:
            if (type == "r"):
                self.loss_fcn.append(torch.nn.MSELoss(reduce=False))
            elif (type == "c"):
                self.loss_fcn.append(torch.nn.BCEWithLogitsLoss())
        print(f"self.weights = {self.weights}")

    def __call__(self, logits, targets):
        targets = targets.float()
        loss = torch.zeros(len(logits[:, 0]), device = logits.get_device())
        for i in range(len(self.loss_fcn)):
            loss += self.weights[i] * self.loss_fcn[i](logits[:, i], targets[:, i])
        return torch.mean(loss)
        
    
class MultiLabelFinish():
    def __init__(self, label_names, label_types):
        self.finish_fcn = []
        for type in label_types:
            if (type == "r"):
                self.finish_fcn.append(None)
            elif (type == "c"):
                self.finish_fcn.append(torch.special.expit)

    def __call__(self, logits):
        for i in range(len(self.finish_fcn)):
            if (self.finish_fcn[i]):
                logits[:, i] = self.finish_fcn[i](logits[:, i].to(torch.long))
        return logits

class ContrastiveClusterLoss():
    def __init__(self, k=10, temperature=1, alpha=1):
        self.k = k
        self.temperature = temperature
        self.alpha = alpha

    def __call__(self, logits, targets):
        targets = targets.float()
        logits_combined = logits.float()

        hid_size = int(len(logits[0]) / 2)

        logits = normalize_embeddings(logits_combined[:, :hid_size])
        logits_augmented = normalize_embeddings(logits_combined[:, hid_size:])

        contrastive = contrastive_loss(logits, logits_augmented, self.temperature)
        clustering, _ = clustering_loss(logits, self.k)

        variance_loss = variance_regularization(logits) + variance_regularization(logits_augmented)

        return torch.mean(contrastive + clustering + self.alpha * variance_loss)
    
class ContrastiveClusterFinish():
    def __init__(self, k = 10, temperature = 1, max_cluster_iterations = 10):
        self.k = k
        self.temperature = temperature
        self.max_cluster_iterations = max_cluster_iterations

        print(f"ContrastiveClusterFinish: k = {k}, temperature = {temperature}")

    def __call__(self, logits):
        logits_combined = logits.float()

        hid_size = int(len(logits[0]) / 2)

        logits = logits_combined[:, :hid_size]
        logits_augmented = logits_combined[:, hid_size:]
        
        contrastive = contrastive_loss(logits, logits_augmented, self.temperature)
        clustering, _ = clustering_loss(logits, self.k, self.max_cluster_iterations)
        variance = variance_regularization(logits) + variance_regularization(logits_augmented)

        return contrastive, clustering, variance
    
def s(z_i, z_j):
    z_i = torch.tensor(z_i) if not isinstance(z_i, torch.Tensor) else z_i
    z_j = torch.tensor(z_j) if not isinstance(z_j, torch.Tensor) else z_j
    
    return torch.cdist(z_i, z_j, p=2)
    # dot_product = torch.dot(z_i, z_j)
    # norm_i = torch.linalg.norm(z_i)
    # norm_j = torch.linalg.norm(z_j)
    
    # return dot_product / (norm_i * norm_j)

def contrastive_loss(logits, logits_augmented, temperature=1, margin=1.0):
    logits = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
    logits_augmented = torch.tensor(logits_augmented) if not isinstance(logits_augmented, torch.Tensor) else logits_augmented

    z = torch.cat((logits, logits_augmented), dim=0)
    similarity_matrix = torch.mm(z, z.t()) / temperature
    norms = torch.linalg.norm(z, dim=1)
    norm_matrix = torch.ger(norms, norms)
    similarity_matrix = similarity_matrix / norm_matrix
    mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool)

    loss = 0
    for k in range(len(logits)):
        numerator = torch.exp(similarity_matrix[k, k + len(logits)])
        denominator = torch.sum(torch.exp(similarity_matrix[k, ~mask[k]]))
        
        loss += -torch.log(numerator / denominator)

    return loss


def clustering_loss(logits, k=10, max_iterations=10):
    # Step 1: Initialize cluster means
    indices = torch.randperm(logits.size(0))[:k]
    cluster_means = logits[indices]

    prev_assignments = None
    assignment_history = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Step 2: Assign each data point to the nearest cluster mean
        distances = torch.cdist(logits, cluster_means, p=2)  # Compute distances between logits and cluster means
        cluster_assignments = torch.argmin(distances, dim=1)  # Assign each point to the nearest cluster mean

        # Check for convergence: if assignments do not change, break the loop
        if prev_assignments is not None and torch.equal(cluster_assignments, prev_assignments):
            break

        # Check for cycles: if assignments have been seen before, break the loop
        if any(torch.equal(cluster_assignments, prev) for prev in assignment_history):
            break

        assignment_history.append(cluster_assignments.clone())
        prev_assignments = cluster_assignments.clone()

        # Step 3: Update cluster means based on assignments
        new_cluster_means = torch.zeros_like(cluster_means)
        for i in range(k):
            assigned_points = logits[cluster_assignments == i]
            if assigned_points.size(0) > 0:
                new_cluster_means[i] = assigned_points.mean(dim=0)
            else:
                # If no points are assigned to the cluster, reinitialize the mean randomly
                new_cluster_means[i] = logits[torch.randint(0, logits.size(0), (1,)).item()]
        cluster_means = new_cluster_means

    # Step 4: Compute the clustering loss
    distances = torch.cdist(logits, cluster_means, p=2)
    min_distances = torch.min(distances, dim=1)[0]
    loss = torch.sum(min_distances ** 2)

    return loss, cluster_means

def normalize_embeddings(embeddings):
    return embeddings / embeddings.norm(dim=1, keepdim=True)

def variance_regularization(embeddings):
    mean_embedding = embeddings.mean(dim=0)
    variance = ((embeddings - mean_embedding) ** 2).mean()
    return variance

