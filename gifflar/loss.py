import torch
from torch import nn


class MultiLoss(nn.Module):
    def __init__(self, n_losses: int, dynamic=True):
        """
        MultiLoss is a class that (dynamically) combines multiple losses into one as detailed in "Multi-Task Learning
        Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (doi: 10.48550/arXiv.1705.07115) by Kendall
        et al. Alternatively, the losses can be combined summation by setting dynamic=False.

        Params:
            n_losses: The number of losses to combine.
            dynamic: Whether to use dynamic weighting of the losses or not. Defaults to True.
        """
        super(MultiLoss, self).__init__()
        self.dynamic = dynamic
        self.log_vars = nn.Parameter(torch.full((n_losses,), 1 / n_losses), requires_grad=True)

    def to(self, device: torch.device):
        """
        Overriding the to method to ensure that the log_vars are also moved to the device.

        Params:
            device: The device to move the model to.

        Returns:
            The model moved to the device.
        """
        self.log_vars = self.log_vars.to(device)
        return super().to(device)

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """
        Combines the losses into one.

        Params:
            losses: A list of losses to combine.

        Returns:
            The combined loss.
        """
        if self.dynamic:
            return torch.sum(torch.stack([
                torch.exp(-log_var) * l + log_var for log_var, l in zip(self.log_vars, losses)
            ]))
        else:
            return torch.sum(torch.stack(losses))
