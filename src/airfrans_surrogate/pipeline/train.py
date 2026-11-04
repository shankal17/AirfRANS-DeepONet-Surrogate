"""Train one surface DeepONet for each task in `TrainConfig.tasks`"""
import json
import os
import os.path as osp

import pandas as pd
import torch

from airfrans_surrogate.config import Config
from airfrans_surrogate.data import SurfaceDataset, split_names
from airfrans_surrogate.trainer import train_model


def train_task(cfg: Config, task: str, device: str):
    """Train one model on `task` and write its run directory.

    Args:
        cfg: Experiment config; uses `paths` and `train`.
        task: AirfRANS task, one of `config.TASKS`.
        device: Device to load the data and train on.
    """

    train_cfg, hparams = cfg.train, cfg.train.hparams(task)
    train_names, val_names, test_names = split_names(osp.join(cfg.paths.data, "cases.csv"), task)
    print(f"task {task}: {len(train_names)} train / {len(val_names)} val / {len(test_names)} test cases, "
          f"device {device}")
    train = SurfaceDataset(cfg.paths.data, train_names, device)
    val = SurfaceDataset(cfg.paths.data, val_names, device)

    run_dir = osp.join(cfg.paths.runs, task)
    os.makedirs(run_dir, exist_ok=True)
    with open(osp.join(run_dir, "config.json"), "w") as f:
        json.dump({**hparams, "task": task, "seed": train_cfg.seed, "val_names": val_names}, f, indent=1)

    result = train_model(train, val, train_cfg.seed, hparams)
    torch.save({k: v for k, v in result.items() if k != "history"}, osp.join(run_dir, "model.pt"))
    pd.DataFrame(result["history"]).to_csv(osp.join(run_dir, "history.csv"), index=False)
    print(f"best val {hparams['model_selection_metric']} {result['best_val']:.4f}")


def run(cfg: Config):
    """Train one model per task in `cfg.train.tasks`, on the GPU if there is one.

    Args:
        cfg: Experiment config.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for task in cfg.train.tasks:
        train_task(cfg, task, device)
