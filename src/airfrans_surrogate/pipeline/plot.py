"""Plot the surface fields the trained models predict: Cp and skin friction, no integrated forces.

For each task with a trained model (<paths.runs>/<task>/model.pt, written by the train stage),
predicts the cases of `PlotConfig.split` and writes four figures to <paths.runs>/<task>/figures/:
  cp_gallery.png  -Cp vs x/c for many cases: prediction and RANS
  cp_surface.png  the airfoil outline coloured by RANS Cp, predicted Cp and the error
  cf_gallery.png  streamwise_cf Cf vs x/c for the same cases as cp_gallery.png
  cf_surface.png  as cp_surface.png, for streamwise_cf Cf
"""
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

from airfrans_surrogate import viz
from airfrans_surrogate.config import TASKS, Config
from airfrans_surrogate.data import SurfaceDataset, split_names
from airfrans_surrogate.geometry import streamwise_cf
from airfrans_surrogate.trainer import load_model, predict_set

plt = viz.plt
# Diverging: blue (suction, Cp < 0) <-> orange (Cp > 0) through a neutral gray.
DIVERGING = LinearSegmentedColormap.from_list("cp_div", [viz.BLUE, "#f0efec", viz.ORANGE])
# Error: purple (under-predicted) <-> green (over-predicted)
ERROR_DIV = LinearSegmentedColormap.from_list("err_div", ["#8a4fc7", "#f0efec", viz.AQUA])


def pick(names, df, n_cases):
    """n_cases names spread evenly over angle of attack.

    Args:
        names (list[str]): Candidate case names.
        df (DataFrame): cases.csv indexed by name.
        n_cases (int): How many to pick; all of them if there are fewer.

    Returns:
        list[str]: Picked names, in increasing AoA.
    """

    aoa = df.loc[names, "aoa_deg"].values
    by_aoa = np.argsort(aoa)
    spread = np.linspace(0, len(by_aoa) - 1, min(n_cases, len(by_aoa))).round().astype(int)

    return [names[i] for i in by_aoa[spread]]


def predict(path, data_dir, names, device):
    """Per case geometry, ground truth and predicted Cp and Cf.

    Args:
        path (str): Model checkpoint.
        data_dir (str): Directory holding the extracted .npz files.
        names (list[str]): Cases to predict.
        device (str): Device to predict on.

    Returns:
        list[dict]: predictions and ground truth
    """

    dataset = SurfaceDataset(data_dir, names, device)
    y = predict_set(*load_model(path, device), dataset)[0].numpy().astype(np.float64)
    mask = dataset.mask.cpu().numpy()
    out = []
    for i, name in enumerate(names):
        case_data = np.load(osp.join(data_dir, name + ".npz"))
        n_points = int(mask[i].sum())
        normals, side = case_data["normals"], case_data["side"]
        out.append({"name": name,
        "pos": case_data["pos"],
        "s": case_data["s"],
        "lines": case_data["lines"],
        "cp": case_data["cp"], 
        "pred": y[i, :n_points, 0],
        "cf": streamwise_cf(case_data["cf"], normals, side),
        "pred_cf": streamwise_cf(y[i, :n_points, 1:], normals, side)})
    
    return out


def title(row):
    """Two-line panel title: airfoil name, then AoA and Re.

    Args:
        row (Series): cases.csv row.

    Returns:
        str: The title.
    """

    return f"{viz.naca_label(row)}\nAoA {row.aoa_deg:.1f}°, Re {row.re / 1e6:.1f}M"

GALLERIES = {
    "cp": dict(true="cp", pred="pred", sign=-1.0, ylabel="$-C_p$", err="Cp", xlabel="x/c", symbol="$C_p$"),
    "cf": dict(true="cf", pred="pred_cf", sign=1.0, ylabel="streamwise_cf $C_f$", err="Cf", xlabel="x/c (square-root scale)", symbol="streamwise_cf $C_f$"),
}


def visible(case, kind):
    """Points that set a panel's y-limits.

    Args:
        case (dict): One case from `predict`.
        kind (str): Gallery kind, a key of `GALLERIES`.

    Returns:
        ndarray: Boolean mask over the case's points.
    """

    if kind == "cf":
        return case["pos"][:, 0] >= 0.002
    
    return np.ones(len(case["s"]), bool)


def fig_gallery(cases, df, path, kind):
    """Grid of per-case panels, 4 per row, comparing prediction with RANS.

    Args:
        cases (list[dict]): Cases from `predict`.
        df (DataFrame): cases.csv indexed by name, for panel titles.
        path (str): Output image file.
        kind (str): What to plot, a key of `GALLERIES`.
    """

    gallery = GALLERIES[kind]
    ncol = 4
    nrow = int(np.ceil(len(cases) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow), squeeze=False)
    for ax, case in zip(axs.flat, cases):
        order = np.argsort(case["s"])
        pred = gallery["sign"] * case[gallery["pred"]][order]
        true = gallery["sign"] * case[gallery["true"]][order]
        x, sides = case["pos"][order, 0], [case["s"][order] >= 0, case["s"][order] < 0]
        if kind == "cf":
            ax.set_xscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)), np.square))
            ax.set_xticks([0, 0.02, 0.1, 0.3, 0.6, 1.0])
        
        if kind != "cp":
            ax.axhline(0, color=viz.INK_MUTED, lw=0.6)  # below: reversed flow
            in_view = visible(case, kind)[order]
            lo = min(pred[in_view].min(), true[in_view].min())
            hi = max(pred[in_view].max(), true[in_view].max())
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        
        for side in sides:
            ax.plot(x[side], pred[side], color=viz.BLUE, lw=1.2)
            ax.plot(x[side], true[side], color=viz.INK, lw=0.8, ls="--")
        ax.set_title(title(df.loc[case["name"]]), fontsize=8)

        if gallery["err"]:
            true_values = case[gallery["true"]]
            rel_l2 = np.linalg.norm(case[gallery["pred"]] - true_values) / np.linalg.norm(true_values)
            ax.text(0.97, 0.95, f"{gallery['err']} rel. L2 {rel_l2:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=viz.INK_MUTED)
            
    for ax in axs.flat[len(cases):]:
        ax.set_visible(False)

    for ax in axs[:, 0]:
        ax.set_ylabel(gallery["ylabel"])

    for ax in axs[-1]:
        ax.set_xlabel(gallery["xlabel"])

    axs[0, 0].plot([], [], color=viz.INK, lw=0.8, ls="--", label="RANS")
    axs[0, 0].plot([], [], color=viz.BLUE, lw=1.2, label="DeepONet")
    fig.tight_layout()
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(path)
    plt.close(fig)


def outline(ax, case, values, cmap, norm):
    """Airfoil outline with each segment coloured by the mean of its two endpoint values.

    Args:
        ax (Axes): Axes to draw on.
        case (dict): One case from `predict`; uses "pos" and "lines".
        values (ndarray): Per-point values, shape (N,).
        cmap (Colormap): Colour map.
        norm (Normalize): Maps values to colours.

    Returns:
        LineCollection: The coloured segments, for a colour bar.
    """

    segments = case["pos"][case["lines"]]                       # (M, 2, 2)
    cell_values = 0.5 * (values[case["lines"][:, 0]] + values[case["lines"][:, 1]])
    ax.add_collection(LineCollection(segments, colors=viz.INK_MUTED, linewidths=5.5, zorder=1))
    line_collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4, capstyle="round", zorder=2)
    line_collection.set_array(cell_values)
    ax.add_collection(line_collection)

    return line_collection


def fig_surface(cases, df, path, kind):
    """Airfoil outlines coloured by the RANS field (left), the prediction (middle) and the error (right).

    Args:
        cases (list[dict]): Cases from `predict`.
        df (DataFrame): cases.csv indexed by name, for row labels.
        path (str): Output image file.
        kind (str): What to plot, a key of `GALLERIES`.
    """

    gallery = GALLERIES[kind]
    trues = [case[gallery["true"]] for case in cases]
    preds = [case[gallery["pred"]] for case in cases]
    err = [pred - true for pred, true in zip(preds, trues)]

    pooled = lambda arrays: np.concatenate([v[visible(case, kind)] for case, v in zip(cases, arrays)])
    lo, hi = np.percentile(np.r_[pooled(trues), pooled(preds)], [1, 99])
    floor = 1e-3 * max(-lo, hi)  # keeps 0 strictly inside the limits
    field_norm = TwoSlopeNorm(0.0, min(lo, -floor), max(hi, floor))
    err_limit = np.percentile(np.abs(pooled(err)), 99)
    fig, axs = plt.subplots(len(cases), 3, figsize=(10.5, 1.0 * len(cases) + 1.0), squeeze=False)
    columns = [
        (f"RANS {gallery['symbol']}", trues, DIVERGING, field_norm, None),
        (f"predicted {gallery['symbol']}", preds, DIVERGING, field_norm, axs[:, :2]),
        ("error (predicted − RANS)", err, ERROR_DIV, Normalize(-err_limit, err_limit), axs[:, 2:]),
    ]
    for j, (label, values, cmap, norm, bar_axs) in enumerate(columns):
        for i, case in enumerate(cases):
            ax = axs[i, j]
            line_collection = outline(ax, case, values[i], cmap, norm)
            ax.set(xlim=(-0.03, 1.03), aspect="equal", xticks=np.arange(0, 1.01, 0.25))
            ax.set_ylim(case["pos"][:, 1].min() - 0.03, case["pos"][:, 1].max() + 0.03)
            ax.grid(False)
            ax.tick_params(left=False, labelleft=False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_bounds(0, 1)
            if i < len(cases) - 1:
                ax.tick_params(bottom=False, labelbottom=False)
                ax.spines["bottom"].set_visible(False)
            else:
                ax.set_xlabel("x/c")
            
            if j == 0:
                row = df.loc[case["name"]]
                ax.set_ylabel(f"AoA {row.aoa_deg:.1f}°\nRe {row.re / 1e6:.1f}M", rotation=0,
                              ha="right", va="center", fontsize=8)
        axs[0, j].set_title(label)
        if bar_axs is not None:
            fig.colorbar(line_collection, ax=bar_axs, orientation="horizontal", fraction=0.05, pad=0.12, aspect=30 * bar_axs.shape[1], extend="both")
    fig.savefig(path)
    plt.close(fig)


def plot_task(cfg: Config, task, df, device):
    """Predict the chosen cases with one task's model

    Args:
        cfg: Experiment config; uses `paths` and `plot`.
        task (str): Task whose model, in <paths.runs>/<task>/model.pt, is plotted.
        df (DataFrame): cases.csv indexed by name.
        device (str): Device to predict on.
    """

    paths, plot_cfg = cfg.paths, cfg.plot
    fig_dir = paths.run_figs(task)
    os.makedirs(fig_dir, exist_ok=True)
    _, val_names, test_names = split_names(osp.join(paths.data, "cases.csv"), task)
    names = {"test": test_names, "val": val_names}[plot_cfg.split]
    gallery = list(plot_cfg.cases) if plot_cfg.cases else pick(names, df, plot_cfg.n_gallery)
    surface = list(plot_cfg.cases) if plot_cfg.cases else pick(names, df, plot_cfg.n_surface)

    model_path = osp.join(paths.runs, task, "model.pt")
    cases = predict(model_path, paths.data, list(dict.fromkeys(gallery + surface)), device)
    by_name = {case["name"]: case for case in cases}
    fig_path = lambda filename: osp.join(fig_dir, filename)
    gallery_cases = [by_name[name] for name in gallery]
    surface_cases = [by_name[name] for name in surface]
    fig_gallery(gallery_cases, df, fig_path("cp_gallery.png"), "cp")
    fig_surface(surface_cases, df, fig_path("cp_surface.png"), "cp")
    fig_gallery(gallery_cases, df, fig_path("cf_gallery.png"), "cf")
    fig_surface(surface_cases, df, fig_path("cf_surface.png"), "cf")
    print(f"[{task}] {plot_cfg.split} split: figures in {fig_dir}/")


def run(cfg: Config):
    """Write the four field figures for every trained task model.

    Args:
        cfg: Experiment config; uses `paths` and `plot`.

    Raises:
        RuntimeError: If no task has a trained model.
    """

    paths = cfg.paths
    tasks = [task for task in TASKS if osp.exists(osp.join(paths.runs, task, "model.pt"))]
    if not tasks:
        raise RuntimeError(f"No trained models in {paths.runs}/<task>/model.pt; run the train stage first.")
    
    viz.setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(osp.join(paths.data, "cases.csv")).set_index("name", drop=False)
    for task in tasks:
        plot_task(cfg, task, df, device)
    
    print("Figures per run: cp_gallery.png, cp_surface.png, cf_gallery.png, cf_surface.png")
