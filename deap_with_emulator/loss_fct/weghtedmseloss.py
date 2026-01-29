import torch
import torch.nn as nn

# class WeightedMSELoss(nn.Module):
#     """
#     Weighted loss that:
#       - keeps your original target-based element weighting (1 + abs(target - mean_target))
#       - multiplies each sample by a sample-level weight derived from KGE
#       - still includes variance and correlation penalties (scaled by mean KGE weight)
#     Usage:
#         criterion = WeightedMSELoss(alpha=1., beta=1., gamma=1., kge_scale=1.0)
#         loss = criterion(preds, targets, kge=kge_batch)   # kge_batch shape: (B,1) or (B,)
#     """
#     def __init__(self, alpha=1, beta=1, gamma=1, eps=1e-6):
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.eps = eps
#
#     def forward(self, preds, targets):
#         """
#         preds, targets: (batch, D)
#         kge: (batch, 1) or (batch,)
#         """
#
#         # ---- MSE (element-wise) ----
#         base_mse = (preds - targets) ** 2  # (B, D)
#         weighted_mse = base_mse
#         mse_loss = torch.mean(weighted_mse)
#
#         # ---- variance penalty (per-output) ----
#         var_pred = torch.var(preds, dim=0, unbiased=False)
#         var_true = torch.var(targets, dim=0, unbiased=False)
#         var_loss = torch.mean((var_pred - var_true) ** 2)
#
#         # ---- correlation penalty (per-output Pearson r) ----
#         preds_centered = preds - preds.mean(dim=0, keepdim=True)
#         targets_centered = targets - targets.mean(dim=0, keepdim=True)
#         cov = torch.mean(preds_centered * targets_centered, dim=0)
#         std_pred = torch.std(preds, dim=0, unbiased=False) + self.eps
#         std_true = torch.std(targets, dim=0, unbiased=False) + self.eps
#         corr = cov / (std_pred * std_true)
#         corr_loss = torch.mean((1.0 - corr) ** 2)
#
#         loss = self.alpha * mse_loss + self.beta * var_loss + self.gamma * corr_loss
#         return loss


class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1, delta=0.1, asym_strength=0.8, eps=1e-6):
        super().__init__()
        self.alpha = alpha  # MSE
        self.beta = beta    # variance penalty
        self.gamma = gamma  # correlation penalty
        self.delta = delta  # mean/bias penalty
        self.asym_strength = asym_strength
        self.eps = eps

    def forward(self, preds, targets):
        diff = preds - targets
        over_mask = (diff > 0).float()
        asym_weight = 1.0 + self.asym_strength * over_mask

        # Weighted MSE
        mse_loss = torch.mean(asym_weight * diff ** 2)

        # Variance matching
        var_pred = torch.var(preds, dim=0, unbiased=False)
        var_true = torch.var(targets, dim=0, unbiased=False)
        var_loss = torch.mean((var_pred - var_true) ** 2)

        # Correlation penalty
        preds_c = preds - preds.mean(dim=0, keepdim=True)
        targets_c = targets - targets.mean(dim=0, keepdim=True)
        cov = torch.mean(preds_c * targets_c, dim=0)
        std_pred = torch.std(preds, dim=0, unbiased=False) + self.eps
        std_true = torch.std(targets, dim=0, unbiased=False) + self.eps
        corr = cov / (std_pred * std_true)
        corr_loss = torch.mean((1.0 - corr) ** 2)

        # Mean bias penalty
        mean_loss = torch.mean((preds.mean() - targets.mean()) ** 2)

        # Optional overshoot penalty for KGE > 1
        overshoot_penalty = torch.mean(torch.relu(preds - 1.0) ** 2)

        loss = (
            self.alpha * mse_loss +
            self.beta * var_loss +
            self.gamma * corr_loss +
            self.delta * (mean_loss + overshoot_penalty)
        )

        return loss