import numpy as np
import torch.nn as nn
from torch import Tensor


class Dann(nn.Module):
    def __init__(self, backbone: nn.Module, cls_head: nn.Module, domain_head: nn.Module) -> None:
        super().__init__()

        self.backbone = backbone
        self.cls_head = cls_head
        self.domain_head = domain_head

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        return self.cls_head(features)

    def predict_class(self, features: Tensor) -> Tensor:
        return self.cls_head(features)

    def predict_domain(self, features: Tensor) -> Tensor:
        return self.domain_head(features)

    def get_features(self, x: Tensor) -> Tensor:
        return self.backbone(x)


def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2.0 / (1 + np.exp(-10.0 * p)) - 1.0
