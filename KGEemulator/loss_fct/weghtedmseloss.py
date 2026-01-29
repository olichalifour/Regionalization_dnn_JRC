import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):

    def __init__(
            self,
            alpha=1, beta=1, gamma=1, eps=1e-6,
            low_q=-3.0,  # ~15th percentile
            high_q=3.0,  # ~85th percentile
            alpha_low=5.0,  # penalty for optimism in low tail
            alpha_high=2.0,  # penalty for pessimism in high tail
    ):
        super().__init__()
        self.low_q = low_q
        self.high_q = high_q
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = 2.0
        self.eps = eps

    def forward(self, preds, targets):
        error = preds - targets

        # Base symmetric loss
        loss = error ** 2

        # ---- Lower tail: penalize OVERestimation ----
        low_mask = (targets < self.low_q).float()
        over_low = torch.relu(error)  # preds > targets
        loss += self.alpha_low * low_mask * over_low ** 2

        # ---- Upper tail: penalize UNDERestimation ----
        high_mask = (targets > self.high_q).float()
        under_high = torch.relu(-error)  # preds < targets
        loss += self.alpha_high * high_mask * under_high ** 2

        return loss.mean()
