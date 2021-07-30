import torch
import torch.nn as nn
from typing import Union


class ContrastiveLoss(nn.Module):
    def __init__(self, alpha: float = 0.1, device: Union[str, torch.device] = "cpu"):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha  # The margin hyperparameter
        self.device = device

    def __call__(self, features1: torch.Tensor, features2: torch.Tensor):
        """
        :param features1: Batch wise features of samples
        :param features2: Batch wise features of samples
        :return: Returns the contrastive loss that can be used to run loss.backward()
        """
        assert features1.shape == features2.shape
        # Normalize the embeddings
        norm_emb1 = features1 / torch.norm(features1, keepdim=True, dim=1)
        norm_emb2 = features2 / torch.norm(features2, keepdim=True, dim=1)
        similarity_matrix = torch.mm(norm_emb1, norm_emb2.T)  # Emb1 . Emb2transpose
        return self.hard_neg_mining(similarity_matrix)  # Calculates and returns the loss

    def hard_neg_mining(self, similarity_matrix: torch.Tensor):
        """
        Takes in the similarity matrix and performs calculated the contrastive loss using hard negative mining
        :param similarity_matrix: The matrix denoting the similarities between pairs of the embeddings
        :return: The loss calculated through hard_negative_mining
        """
        bs, _ = similarity_matrix.shape  # batch_size
        similarity_anc_pos = torch.diag(similarity_matrix)  # Similarity b/w anchors & positives
        similarity_anc_neg = similarity_matrix - torch.diag(similarity_anc_pos)  # Similarity b/w anchors & negatives
        # Mean Negative calculation
        mean_neg = torch.sum(similarity_anc_neg, dim=1, keepdim=True) / (bs - 1)  # Mean Negative, off-diagonal values
        # Closest negative calculation
        mask1 = (torch.eye(bs, device=self.device) == 1).bool()  # Mask to exclude the diagonals
        mask2 = similarity_anc_neg > similarity_anc_pos.view(bs, 1)  # Mask to exclude sim-anc-neg > sim-anc-pos

        mask = mask1 | mask2  # Final mask
        similarity_anc_neg[mask] = -2
        closest_neg = torch.max(similarity_anc_neg, dim=1, keepdim=True).values
        # Loss calculation
        loss1 = torch.maximum(mean_neg - similarity_anc_pos.view(bs, 1) + self.alpha, torch.tensor(0))
        loss2 = torch.maximum(closest_neg - similarity_anc_pos.view(bs, 1) + self.alpha, torch.tensor(0))
        loss = loss1 + loss2

        return torch.sum(loss)
