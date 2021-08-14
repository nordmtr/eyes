import os
import pathlib
import shutil

from tqdm import tqdm

RAW_DATA_DIR = "/home/dima/datasets/eyes/mrl_raw/"
DATA_DIR = "/home/dima/datasets/eyes/mrl/"


def main():
    idx_to_class = {"0": "closed", "1": "opened"}
    pathlib.Path(os.path.join(DATA_DIR, "closed")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(DATA_DIR, "opened")).mkdir(parents=True, exist_ok=True)
    walk = os.walk(RAW_DATA_DIR)
    next(walk)
    for dir_entry in tqdm(walk):
        subdir, _, fnames = dir_entry
        for fname in fnames:
            _, _, _, glasses, label, reflection, conditions, _ = fname.split("_")
            if reflection == "0":
                shutil.copy(os.path.join(subdir, fname), os.path.join(DATA_DIR, idx_to_class[label]))


if __name__ == "__main__":
    main()
