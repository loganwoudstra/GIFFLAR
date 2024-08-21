from typing import Type, Literal, Optional, Any

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import BinaryConfusionMatrix, MulticlassConfusionMatrix, MultilabelConfusionMatrix
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.utilities.enums import ClassificationTask


class BinarySensitivity(BinaryConfusionMatrix):
    """Computes sensitivity for binary classification tasks."""
    def compute(self) -> Tensor:
        """Computes the sensitivity."""
        self.confmat = super().compute().nan_to_num(0, 0, 0)
        return self.confmat[1, 1] / (self.confmat[1, 1] + self.confmat[1, 0])


class MulticlassSensitivity(MulticlassConfusionMatrix):
    """Computes sensitivity for multiclass classification tasks."""
    def compute(self) -> Tensor:
        """Computes the sensitivity as mean per-class sensitivity."""
        self.confmat = super().compute().nan_to_num(0, 0, 0)
        return (self.confmat.diag() / self.confmat.sum(dim=1)).mean()


class MultilabelSensitivity(MultilabelConfusionMatrix):
    """Computes sensitivity for multilabel classification tasks."""
    def compute(self) -> Tensor:
        """Computes the sensitivity as mean per-class sensitivity"""
        self.confmat = super().compute().nan_to_num(0, 0, 0)
        return (self.confmat[:, 1, 1] / self.confmat[:, 1, :].sum(dim=1)).mean()


class Sensitivity(_ClassificationTaskWrapper):
    def __new__(
        cls: Type["Sensitivity"],
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        normalize: Optional[Literal["true", "pred", "all", "none"]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """
        Factory method to instantiate the appropriate sensitivity metric based on the task.

        Args:
            task: The classification task type.
            threshold: The threshold value for binary and multilabel classification tasks.
            num_classes: The number of classes for multiclass classification tasks.
            num_labels: The number of labels for multilabel classification tasks.
            normalize: Normalization mode for confusion matrix.
            ignore_index: The label index to ignore.
            validate_args: Whether to validate input args.
            **kwargs: Additional keyword arguments to pass to the metric.

        Returns:
            The sensitivity metric for the specified task.
        """
        # Initialize task metric
        task = ClassificationTask.from_str(task)
        kwargs.update({"normalize": normalize, "ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTask.BINARY:
            return BinarySensitivity(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            return MulticlassSensitivity(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            return MultilabelSensitivity(num_labels, threshold, **kwargs)
        raise ValueError(f"Task {task} not supported!")
