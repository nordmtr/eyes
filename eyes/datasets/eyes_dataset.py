import glob
import os

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class EyesDataset(Dataset):
    def __init__(self, folder: str, transform=None):
        self.folder = folder
        self.files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
        self.transform = transform

    def __getitem__(self, idx: int) -> Tensor:
        img = Image.open(self.files[idx])
        return self.transform(img) if self.transform else img

    def __len__(self) -> int:
        return len(self.files)
