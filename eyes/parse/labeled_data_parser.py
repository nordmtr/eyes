import os
import pathlib
import shutil

import pandas as pd

LABELS_CSV = "/home/dima/datasets/eyes/train/labels.csv"
DATA_DIR = "/home/dima/datasets/eyes/labeled/"


def main():
    idx_to_class = {0: "closed", 1: "opened"}
    pathlib.Path(os.path.join(DATA_DIR, "closed")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(DATA_DIR, "opened")).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LABELS_CSV)
    for _, row in df.iterrows():
        shutil.copy(row["filename"], os.path.join(DATA_DIR, idx_to_class[row["label"]]))


if __name__ == "__main__":
    main()
