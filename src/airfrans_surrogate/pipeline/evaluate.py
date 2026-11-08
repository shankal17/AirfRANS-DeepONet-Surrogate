"""Evaluate every trained model in <paths.runs>."""

import json
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch

from airfrans_surrogate import viz
from airfrans_surrogate.config import Config
from airfrans_surrogate.data import SurfaceDataset, split_names
from airfrans_surrogate.trainer import load_model, predict_set

plt = viz.plt
LABEL = {"cl": "$C_l$", "cd": "$C_d$"}
# Chordwise regions for Cp error: the leading edge (suction peak) is the hardest part.
REGIONS = {"le": (-np.inf, 0.02), "mid": (0.02, 0.95), "te": (0.95, np.inf)}
TRAIN_RANGE = {"reynolds": ("re", 3e6, 5e6, 1e-6, "Re (10^6)"),
               "aoa": ("aoa_deg", -2.5, 12.5, 1.0, "angle of attack (deg)")}

def spearman(a, b):
    """Spearman rank correlation

    Args:
        a (ndarray): First sample, shape (C,).
        b (ndarray): Second sample, shape (C,).

    Returns:
        float: Correlation in [-1, 1].
    """

    rank_a, rank_b = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))

    return float(np.corrcoef(rank_a, rank_b)[0, 1])

def predict(model, normalizer, dataset):
    """Predict `dataset` and gather predictions and ground truth.

    Args:
        model (DeepONet): The model.
        normalizer (Normalizer): Fitted normalizer.
        dataset (SurfaceDataset): Cases to predict.

    Returns:
        dict: predictions and ground truth
    """

    y, cl, cd = (t.numpy().astype(np.float64) for t in predict_set(model, normalizer, dataset))
    mask = dataset.mask.cpu().numpy()
    targets = dataset.target.cpu().numpy().astype(np.float64)
    out = {"meta": dataset.meta, "mask": mask, "y": y, "y_true": targets}
    true = {"cl": dataset.meta.cl.values, "cd": dataset.meta.cd.values}
    true["ld"] = true["cl"] / true["cd"]
    for quantity, pred in (("cl", cl), ("cd", cd), ("ld", cl / cd)):
        out[quantity] = {"pred": pred, "true": true[quantity]}
    weights = dataset.point_len.cpu().numpy().astype(np.float64)
    chord_x = dataset.trunk[..., 1].cpu().numpy()
    out["cp"] = {
        "pred": y[..., 0][mask],
        "true": targets[..., 0][mask],
        "w": weights[mask],
        "x": chord_x[mask],
        "w_full": weights
    }
    
    return out

def accuracy(prediction):
    """Pointwise and force accuracy metrics for one split.

    Args:
        prediction (dict): From `predict`.

    Returns:
        dict: "n_cases", "cp_rmse", "cp_rel_l2", "cp_rmse_len", "cp_rel_l2_len",
        "cp_rmse_len_by_region" (dict by region), "cf_rel_l2", "cl_mae", "cd_mae", "cd_mape",
        "ld_mae", "spearman_cl" and "spearman_cd", all plain Python numbers.
    """

    mask, pred, true = prediction["mask"], prediction["y"], prediction["y_true"]
    weights = prediction["cp"]["w_full"]
    err = pred - true
    cases = range(len(mask))
    cp_rel_l2 = [np.linalg.norm(err[i, mask[i], 0]) / np.linalg.norm(true[i, mask[i], 0]) for i in cases]
    cp_rel_l2_len = [np.sqrt(np.sum(weights[i, mask[i]] * err[i, mask[i], 0] ** 2) / np.sum(weights[i, mask[i]] * true[i, mask[i], 0] ** 2)) for i in cases]
    cf_rel_l2 = [np.linalg.norm(err[i, mask[i], 1:]) / np.linalg.norm(true[i, mask[i], 1:]) for i in cases]
    cp = prediction["cp"]
    cp_err = cp["pred"] - cp["true"]

    def error(quantity):
        """Predicted minus true values of "cl", "cd" or "ld", shape (C,)."""
        return prediction[quantity]["pred"] - prediction[quantity]["true"]

    return {
        "n_cases": int(len(mask)),
        "cp_rmse": float(np.sqrt(np.mean(cp_err ** 2))),
        "cp_rel_l2": float(np.mean(cp_rel_l2)),
        "cp_rmse_len": float(np.sqrt(np.average(cp_err ** 2, weights=cp["w"]))),
        "cp_rel_l2_len": float(np.mean(cp_rel_l2_len)),
        "cp_rmse_len_by_region": {
            region: float(np.sqrt(np.average(cp_err[in_region] ** 2, weights=cp["w"][in_region])))
            for region, (lo, hi) in REGIONS.items() for in_region in [(cp["x"] >= lo) & (cp["x"] < hi)]
        },
        "cf_rel_l2": float(np.mean(cf_rel_l2)),
        "cl_mae": float(np.mean(np.abs(error("cl")))),
        "cd_mae": float(np.mean(np.abs(error("cd")))),
        "cd_mape": float(np.mean(np.abs(error("cd")) / np.abs(prediction["cd"]["true"]))),
        "ld_mae": float(np.mean(np.abs(error("ld")))),
        "spearman_cl": spearman(prediction["cl"]["pred"], prediction["cl"]["true"]),
        "spearman_cd": spearman(prediction["cd"]["pred"], prediction["cd"]["true"]),
    }

def fig_history(history, path):
    """Plot training and validation loss per epoch

    Args:
        history (DataFrame): history.csv from the train stage, with "epoch", "train_last" and "val" columns.
        path (str): Output image file.
    """

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(history.epoch, history.train_last, color=viz.ORANGE, lw=1.0, label="train (last batch)")
    ax.plot(history.epoch, history.val, color=viz.BLUE, lw=1.4, label="validation")
    ax.set(yscale="log", xlabel="epoch", ylabel="loss", title="Training history")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)

def fig_parity(prediction, path):
    """Plot predicted vs RANS Cl and Cd, one panel each.

    Args:
        prediction (dict): Test-set result from `predict`.
        path (str): Output image file.
    """

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, quantity in zip(axs, ("cl", "cd")):
        true, pred = prediction[quantity]["true"], prediction[quantity]["pred"]
        lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color=viz.INK_MUTED, lw=0.8, ls=":")
        ax.scatter(true, pred, s=7, color=viz.BLUE, alpha=0.8)
        label = LABEL[quantity]
        ax.set(xlabel=f"RANS {label}", ylabel=f"surrogate {label}", title=f"{label}, test set")
    fig.savefig(path)
    plt.close(fig)

def fig_extrapolation(results, path):
    """Plot |Cl error| against Re / AoA for the extrapolation tasks, with the training range shaded.

    Args:
        results (dict): {task: {"val": prediction, "test": prediction}}, predictions from
            `predict`. Only the "reynolds" and "aoa" tasks are plotted.
        path (str): Output image file.

    Returns:
        bool: Whether a figure was written (False if neither task is in `results`).
    """

    tasks = [task for task in ("reynolds", "aoa") if task in results]
    if not tasks:
        return False
    
    fig, axs = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.4), squeeze=False)
    for ax, task in zip(axs[0], tasks):
        column, lo, hi, scale, xlabel = TRAIN_RANGE[task]
        ax.axvspan(lo * scale, hi * scale, color=viz.GRID, alpha=0.6, lw=0)
        ax.text((lo + hi) / 2 * scale, 0.97, "training range", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=viz.INK_MUTED)
        for split, color, name in (("val", viz.BLUE, "in range"), ("test", viz.ORANGE, "extrapolation")):
            prediction = results[task][split]
            cl_err = np.abs(prediction["cl"]["pred"] - prediction["cl"]["true"])
            ax.scatter(prediction["meta"][column] * scale, cl_err, s=7, color=color, label=name)
        
        ax.set(yscale="log", xlabel=xlabel, ylabel=r"$|C_l$ error$|$",
               title=f"'{task}' split: error vs {xlabel.split(' (')[0]}")
    fig.tight_layout()
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(path)
    plt.close(fig)
    return True

def run(cfg: Config):
    """Evaluate each trained task model, print test metrics, and write figures and metrics.json.

    Args:
        cfg: Experiment config; uses `paths`.

    Raises:
        RuntimeError: If no task has a trained model.
    """
    paths = cfg.paths
    os.makedirs(paths.eval_figs, exist_ok=True)
    os.makedirs(paths.results, exist_ok=True)
    viz.setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv = osp.join(paths.data, "cases.csv")

    results, metrics = {}, {}
    for task in ("full", "scarce", "reynolds", "aoa"):
        run_dir = osp.join(paths.runs, task)
        if not osp.exists(osp.join(run_dir, "model.pt")):
            continue

        model, normalizer = load_model(osp.join(run_dir, "model.pt"), device)
        with open(osp.join(run_dir, "config.json")) as f:
            val_names = json.load(f)["val_names"]
        
        _, _, test_names = split_names(csv, task)
        pred = {split: predict(model, normalizer, SurfaceDataset(paths.data, names, device))
                for split, names in (("val", val_names), ("test", test_names))}
        results[task] = pred
        metrics[task] = {split: accuracy(pred[split]) for split in pred}
        test_metrics = metrics[task]["test"]
        print(f"[{task}] test: Cp relL2 {test_metrics['cp_rel_l2']:.3f} "
              f"(length-weighted {test_metrics['cp_rel_l2_len']:.3f})  "
              f"Cf relL2 {test_metrics['cf_rel_l2']:.3f}  Cl MAE {test_metrics['cl_mae']:.4f}  "
              f"Cd MAPE {test_metrics['cd_mape']:.1%}  "
              f"rho(Cl) {test_metrics['spearman_cl']:.3f}  rho(Cd) {test_metrics['spearman_cd']:.3f}")
        print("        Cp RMSE by region (length-weighted): " + "  ".join(
            f"{region} {value:.3f}" for region, value in test_metrics["cp_rmse_len_by_region"].items()))
        fig_dir = paths.run_figs(task)
        os.makedirs(fig_dir, exist_ok=True)
        history = osp.join(run_dir, "history.csv")
        if osp.exists(history):
            fig_history(pd.read_csv(history), osp.join(fig_dir, "fig_history.png"))
        fig_parity(pred["test"], osp.join(fig_dir, "fig_parity.png"))

    if not results:
        raise RuntimeError(f"No trained models in {paths.runs}/<task>/model.pt; run the train stage first.")
    
    has_extrapolation = fig_extrapolation(results, osp.join(paths.eval_figs, "fig_extrapolation.png"))
    metrics["_figures"] = {"extrapolation": has_extrapolation}
    with open(osp.join(paths.results, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=float)
    
    print(f"Figures in {paths.runs}/<task>/figures/ and {paths.eval_figs}/, "
          f"metrics in {paths.results}/metrics.json")
