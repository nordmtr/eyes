import sys

import numpy as np
import pandas as pd
import torch
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torchvision import transforms

from eyes.datasets import EyesDataset
from eyes.models import resnet18


def main(data_dir: str) -> None:
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            create_transform(
                24,
                is_training=False,
                mean=(0.5,),
                std=(0.5,),
            ),
        ]
    )
    dataset = EyesDataset(data_dir, transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet18().to(device)
    model.load_state_dict(torch.load("checkpoints/final.pt"))
    model.eval()
    predicted = []
    for inputs in dataloader:
        with torch.no_grad():
            outputs = model(inputs.to(device))
        _, predicted_batch = torch.max(outputs.data, 1)
        predicted.append(predicted_batch.cpu().numpy())
    labels = np.concatenate(predicted)

    df = pd.DataFrame({"filename": dataset.files, "label": labels})
    df.to_csv("results.csv", index=False, header=False)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    main(data_dir)
