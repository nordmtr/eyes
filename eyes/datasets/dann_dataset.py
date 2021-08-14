from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset


class DannDataset(Dataset):
    """Dataset for training DANN."""

    def __init__(self, source_dataset: Dataset, target_dataset: Dataset) -> None:
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self) -> int:
        return max(len(self.source_dataset), len(self.target_dataset))

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        source_idx = idx % len(self.source_dataset)
        target_idx = idx % len(self.target_dataset)
        return self.source_dataset[source_idx], self.target_dataset[target_idx]
