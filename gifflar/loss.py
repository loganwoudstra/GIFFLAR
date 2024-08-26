import torch
from torch import nn


class MultiLoss(nn.Module):
    def __init__(self, n_losses, dynamic=True):
        super(MultiLoss, self).__init__()
        self.dynamic = dynamic
        self.log_vars = nn.Parameter(torch.full((n_losses,), 1 / n_losses), requires_grad=True)

    def to(self, device):
        self.log_vars = self.log_vars.to(device)
        return super().to(device)

    def forward(self, losses):
        loss = 0
        if self.dynamic:
            for i, l in enumerate(losses):
                loss += torch.exp(-self.log_vars[i]) * l + self.log_vars[i]
        else:
            for l in losses:
                loss += l
        return loss
