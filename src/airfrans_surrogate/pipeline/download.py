"""Download and unzip the pre-processed AirfRANS dataset (several GB)."""

import os
import airfrans as af

from airfrans_surrogate.config import Config


def run(cfg: Config):
    """Download and unzip the dataset into <paths.raw>/Dataset.

    Args:
        cfg: Experiment config; uses `paths.raw`.
    """
    
    root = cfg.paths.raw
    os.makedirs(root, exist_ok=True)
    af.dataset.download(root=root, file_name="Dataset", unzip=True, OpenFOAM=False)
    print(f"Done. Dataset at {os.path.join(root, 'Dataset')}")
