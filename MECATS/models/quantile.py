import torch
import torch.nn as nn
import pdb


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, qu in enumerate(self.quantiles):
            errors = target - preds[:, i].unsqueeze(1)
            losses.append(torch.max((qu - 1) * errors, qu * errors))
        loss = torch.sum(torch.cat(losses, dim=1))
        return loss

    def get_quantile(self):
        return self.quantiles

    def get_q_length(self):
        if len(self.quantiles) == 1:
            return self.quantiles[0]
        else:
            return 'boundary'

    def q_loss_1d(self, preds, target):
        errors = target.item() - preds.item()
        loss = max((self.quantiles[0] - 1) * errors, self.quantiles[0] * errors)
        return loss
