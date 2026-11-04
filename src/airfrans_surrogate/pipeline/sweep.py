"""Tune one `TrainConfig` setting on real data: one model per value, compared on validation."""

import os
import os.path as osp

import numpy as np
import pandas as pd
import torch

from airfrans_surrogate.config import Config
from airfrans_surrogate.data import SurfaceDataset, split_names
from airfrans_surrogate.trainer import load_model, predict_set, train_model


def evaluate(path, dataset):
    """Score a saved model on `dataset`.

    Args:
        path (str): Checkpoint written by `train_model` via `torch.save`.
        dataset (SurfaceDataset): Cases to score.

    Returns:
        dict: Per-case mean "cp_rel_l2" and "cf_rel_l2" (relative L2 error), "cl_mae" (mean
        absolute error) and "cd_mape" (mean absolute fractional error).
    """

    model, normalizer = load_model(path, dataset.branch.device.type)
    y, cl, cd = (t.numpy() for t in predict_set(model, normalizer, dataset))
    mask, targets = dataset.mask.cpu().numpy(), dataset.target.cpu().numpy()
    rel_l2 = lambda pred, true: np.linalg.norm(pred - true) / np.linalg.norm(true)
    cases = range(len(dataset))

    return {
        "cp_rel_l2": np.mean([rel_l2(y[i, mask[i], 0], targets[i, mask[i], 0]) for i in cases]),
        "cf_rel_l2": np.mean([rel_l2(y[i, mask[i], 1:], targets[i, mask[i], 1:]) for i in cases]),
        "cl_mae": np.abs(cl - dataset.meta.cl.values).mean(),
        "cd_mape": np.mean(np.abs(cd - dataset.meta.cd.values) / dataset.meta.cd.values),
    }


def run(cfg: Config):
    """Train one model per value in `cfg.sweep.values` and print its validation metrics.

    Args:
        cfg: Experiment config; uses `paths.data`, `train` and `sweep`.
    """

    sweep_cfg = cfg.sweep
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_names, val_names, _ = split_names(osp.join(cfg.paths.data, "cases.csv"), sweep_cfg.task,
                                            holdout=sweep_cfg.holdout)
    train = SurfaceDataset(cfg.paths.data, train_names, device)
    val = SurfaceDataset(cfg.paths.data, val_names, device)
    base = cfg.train.hparams(sweep_cfg.task)
    print(f"task {sweep_cfg.task}: {len(train_names)} train / {len(val_names)} val ({sweep_cfg.holdout}) "
          f"cases; base {base}", flush=True)

    task_dir, rows = osp.join(sweep_cfg.out, sweep_cfg.task), []
    for value in sweep_cfg.values:
        if sweep_cfg.param == "seed":
            seed, hparams = value, base
        else:
            seed, hparams = cfg.train.seed, base | {sweep_cfg.param: value}
        
        run_dir = osp.join(task_dir, f"{sweep_cfg.param}_{value}")
        os.makedirs(run_dir, exist_ok=True)
        result = train_model(train, val, seed, hparams, log=lambda *args: None)
        path = osp.join(run_dir, "model.pt")
        torch.save({k: v for k, v in result.items() if k != "history"}, path)
        metrics = evaluate(path, val)
        best_epoch = min(result["history"], key=lambda row: row["val_case_rmse"] if hparams["model_selection_metric"] == "case_rmse" else row["val"])["epoch"]
        rows.append({sweep_cfg.param: value, **metrics, "best_val": result["best_val"], "best_epoch": best_epoch})
        print(f"{sweep_cfg.param} {value!s:<8} Cp relL2 {metrics['cp_rel_l2']:.4f}  "
              f"Cf relL2 {metrics['cf_rel_l2']:.4f}  "
              f"Cl MAE {metrics['cl_mae']:.4f}  Cd MAPE {metrics['cd_mape']:.1%}  "
              f"best epoch {best_epoch}", flush=True)
    pd.DataFrame(rows).to_csv(osp.join(task_dir, f"sweep_{sweep_cfg.param}.csv"), index=False)
