import torch
import torchmetrics


class Sensitivity(torchmetrics.Metric):
    def __init__(self, threshold=0.5, dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = (preds > self.threshold).int()
        target = target.int()

        self.true_positives += torch.sum(preds * target)
        self.false_negatives += torch.sum((1 - preds) * target)

    def compute(self):
        return self.true_positives.float() / (self.true_positives + self.false_negatives)
