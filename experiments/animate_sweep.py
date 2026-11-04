"""Animate the trained surrogate on an airfoil"""

import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from airfrans_surrogate import viz
from airfrans_surrogate.data import branch_features, trunk_features
from airfrans_surrogate.extract import NU_GENERATION
from airfrans_surrogate.geometry import arc_from_le, order_loop, streamwise_cf, tangent
from airfrans_surrogate.trainer import load_model, predict
from full_training_and_validation import cfg as full  # experiments/ is on sys.path when this file is run directly

TASK = "full"
CASE = "airFoil2D_SST_92.257_1.382_2.101_4.398_11.912"
KEYFRAMES = [
    (2e6, 0.0, 10),
    (6e6, 0.0, 60), # Reynolds number ramp
    (6e6, 15.0, 45), # angle of attack up ...
    (6e6, -5.0, 60), # ... down ...
    (6e6, 0.0, 15), # ... and back to level
    (6e6, 0.0, 10),
]
TRAINED_RE, TRAINED_AOA = (2e6, 6e6), (-5.0, 15.0)
FPS = 20
DPI = 100
N_ARROWS = 120 # pressure arrows along the surface, clustered at the leading edge
ARROW_MAX = 0.35 # length of the longest pressure arrow, in chords
PIVOT = (0.25, 0.0) # the airfoil turns about its quarter chord


def ease(n_frames):
    """Cosine ease from 0 to 1 over n_frames, excluding 0 (the previous leg's last frame).

    Args:
        n_frames (int): Frames in the leg.

    Returns:
        ndarray: Fractions in (0, 1], shape (n_frames,).
    """

    t = np.arange(1, n_frames + 1) / n_frames

    return 0.5 - 0.5 * np.cos(np.pi * t)


def schedule(keyframes):
    """Per-frame conditions and the leg each frame belongs to.

    Args:
        keyframes (list[tuple]): (Re, AoA in degrees, frames) per keyframe, as `KEYFRAMES`.

    Returns:
        tuple: (re, aoa, leg), each of shape (F,); leg indexes `keyframes`.
    """

    re0, aoa0, n_hold = keyframes[0]
    re, aoa, leg = [np.full(n_hold, re0)], [np.full(n_hold, aoa0)], [np.zeros(n_hold, int)]
    for i, ((re_a, aoa_a, _), (re_b, aoa_b, n_frames)) in enumerate(zip(keyframes, keyframes[1:]), 1):
        fraction = ease(n_frames)
        re.append(re_a + (re_b - re_a) * fraction)
        aoa.append(aoa_a + (aoa_b - aoa_a) * fraction)
        leg.append(np.full(n_frames, i))

    return np.concatenate(re), np.concatenate(aoa), np.concatenate(leg)


def frame_batch(case, re, aoa, device):
    """Model inputs for one geometry at several conditions, as a batch for `trainer.predict`.

    Args:
        case (dict): One extracted case (arrays from its .npz)
        re (ndarray): Reynolds number per frame, shape (B,).
        aoa (ndarray): Angle of attack in degrees per frame, shape (B,).
        device (str): Device for the tensors.

    Returns:
        dict: Batch with the keys `trainer.predict` uses; the per-geometry tensors are expanded, not copied.
    """

    n_points, n_frames = len(case["s"]), len(re)
    as_tensor = lambda array, dtype=torch.float32: torch.as_tensor(np.asarray(array), dtype=dtype, device=device)
    geometry = {
        "tangent": as_tensor(tangent(case["normals"])),
        "mask": as_tensor(np.ones(n_points), torch.bool),
        "lines": as_tensor(case["lines"], torch.int64),
        "cell_normals": as_tensor(case["cell_normals"]),
        "cell_lengths": as_tensor(case["cell_lengths"])
    }
    batch = {k: v.expand(n_frames, *v.shape) for k, v in geometry.items()}
    batch["branch"] = as_tensor(np.stack([branch_features(case["descriptor"], r, a) for r, a in zip(re, aoa)]))
    batch["trunk"] = as_tensor(np.stack([trunk_features({**case, "normals": case["normals"].copy(), "aoa_deg": a})
                                         for a in aoa]))
    batch["aoa_rad"] = as_tensor(np.deg2rad(aoa))

    return batch


@torch.no_grad()
def predict_frames(model, normalizer, case, re, aoa, device, chunk=32):
    """Model Cp and integrated Cl, Cd for one geometry at every frame's conditions

    Args:
        model (DeepONet): The trained model.
        normalizer (Normalizer): Its normalizer.
        case (dict): One extracted case (arrays from its .npz); only its geometry is used.
        re (ndarray): Reynolds number per frame, shape (F,).
        aoa (ndarray): Angle of attack in degrees per frame, shape (F,).
        device (str): Device to predict on.
        chunk (int): Frames per forward pass.

    Returns:
        tuple: (cp, cf, cl, cd) as NumPy arrays of shapes (F, N), (F, N), (F,), (F,); cf is streamwise_cf.
    """

    cp, cf, cl, cd = [], [], [], []
    for i in range(0, len(re), chunk):
        batch = frame_batch(case, re[i:i + chunk], aoa[i:i + chunk], device)
        y, cl_chunk, cd_chunk = predict(model, normalizer, batch)
        y = y.cpu().numpy()
        cp.append(y[..., 0])
        cf.extend(streamwise_cf(frame_cf, case["normals"], case["side"]) for frame_cf in y[..., 1:])
        cl.append(cl_chunk.cpu().numpy())
        cd.append(cd_chunk.cpu().numpy())
    
    return np.concatenate(cp), np.stack(cf), np.concatenate(cl), np.concatenate(cd)


def rotate(points, aoa_deg, pivot=(0.0, 0.0)):
    """Turn points (or vectors, with the default pivot) nose-up by aoa_deg, so the freestream is horizontal.

    Args:
        points (ndarray): Points, shape (..., 2).
        aoa_deg (float): Angle of attack in degrees.
        pivot (tuple): Centre of rotation.

    Returns:
        ndarray: Rotated points, same shape.
    """

    angle = np.deg2rad(aoa_deg)
    c, s = np.cos(angle), np.sin(angle)
    p = points - np.asarray(pivot)

    return np.stack([c * p[..., 0] + s * p[..., 1], -s * p[..., 0] + c * p[..., 1]], -1) + np.asarray(pivot)


def arrow_stations(case, n_arrows):
    """Where the pressure arrows sit: interpolation weights from mesh points to n_arrows surface stations

    Args:
        case (dict): One extracted case.
        n_arrows (int): Number of stations.

    Returns:
        tuple: (pos, outward, interp) - station positions (n_arrows, 2), unit outward normals
        (n_arrows, 2), and a function mapping per-point values (..., N) to per-station values (..., n_arrows).
    """

    arc = arc_from_le(case["pos"], order_loop(case["lines"], len(case["s"])))
    by_arc = np.argsort(arc)
    arc_sorted = arc[by_arc]
    u = np.linspace(-1, 1, n_arrows + 2)[1:-1]  # skip the trailing-edge corner at both ends
    stations = np.where(u < 0, -arc_sorted[0], arc_sorted[-1]) * np.sign(u) * u ** 2
    interp = lambda values: np.stack([np.interp(stations, arc_sorted, v[by_arc]) for v in np.atleast_2d(values)]).squeeze()
    pos = np.stack([interp(case["pos"][:, 0]), interp(case["pos"][:, 1])], 1)
    outward = -np.stack([interp(case["normals"][:, 0]), interp(case["normals"][:, 1])], 1)

    return pos, outward / np.linalg.norm(outward, axis=1, keepdims=True), interp


def leg_label(previous, keyframe):
    """Short label of what a leg changes, e.g. "Re ↑" or "AoA ↓"; "" for a hold.

    Args:
        previous (tuple): Keyframe the leg starts from.
        keyframe (tuple): Keyframe the leg ends at.

    Returns:
        str: The label.
    """
    
    parts = [f"{name} {'↑' if b > a else '↓'}" for name, a, b in (("Re", previous[0], keyframe[0]), ("AoA", previous[1], keyframe[1])) if b != a]
    
    return ", ".join(parts)


def animate(path):
    """Predict every frame and write the GIF

    Args:
        path (str): Output .gif file.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, normalizer = load_model(osp.join(full.paths.runs, TASK, "model.pt"), device)
    case = dict(np.load(osp.join(full.paths.data, CASE + ".npz")))
    re, aoa, leg = schedule(KEYFRAMES)
    outside = [(r, a) for r, a, _ in KEYFRAMES if not (TRAINED_RE[0] <= r <= TRAINED_RE[1] and TRAINED_AOA[0] <= a <= TRAINED_AOA[1])]
    if outside:
        print(f"Warning: keyframes outside the training range (extrapolation): {outside}")
    
    cp, cf, cl, cd = predict_frames(model, normalizer, case, re, aoa, device)
    n_frames = len(re)
    print(f"Predicted {n_frames} frames on {device}")

    # Everything that sets a scale is fixed over the whole animation, so motion means a change.
    station_pos, station_out, interp = arrow_stations(case, N_ARROWS)
    station_cp = interp(cp)                                  # (F, n_arrows)
    scale = ARROW_MAX / np.abs(station_cp).max()
    suction = station_cp < 0
    station_color = np.where(suction, viz.BLUE, viz.ORANGE)  # length carries magnitude
    loop = case["pos"][order_loop(case["lines"], len(case["s"]))]
    side = case["side"]
    upper, lower = [np.flatnonzero(mask)[np.argsort(case["pos"][mask, 0])] for mask in (side >= 0, side < 0)]

    viz.setup()
    plt = viz.plt
    fig = plt.figure(figsize=(10, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.1, 1), width_ratios=(1.2, 1), left=0.07, right=0.97, top=0.9, bottom=0.07, hspace=0.25, wspace=0.22)
    ax_foil = fig.add_subplot(grid[0, :])
    field_grid = grid[1, 0].subgridspec(2, 1, hspace=0.15)
    ax_cp = fig.add_subplot(field_grid[0])
    ax_cf = fig.add_subplot(field_grid[1], sharex=ax_cp)
    trace_grid = grid[1, 1].subgridspec(2, 1, hspace=0.15)
    ax_cl = fig.add_subplot(trace_grid[0])
    ax_cd = fig.add_subplot(trace_grid[1], sharex=ax_cl)

    # Airfoil panel: limits from every frame's outline and arrow outer ends.
    tips = [rotate(station_pos + scale * np.abs(station_cp[f])[:, None] * station_out, aoa[f], PIVOT)
            for f in range(n_frames)]
    extent = np.concatenate(tips + [rotate(loop, a, PIVOT) for a in (aoa.min(), aoa.max())])
    (x_lo, y_lo), (x_hi, y_hi) = extent.min(0) - 0.04, extent.max(0) + 0.04
    ax_foil.set(xlim=(min(x_lo, -0.45), x_hi), ylim=(y_lo, y_hi), aspect="equal")
    ax_foil.set_axis_off()
    body = ax_foil.add_patch(Polygon(loop, closed=True, facecolor=viz.GRID, edgecolor=viz.INK_MUTED, lw=1.0, zorder=3))
    # Suction arrows point away from the surface, positive-pressure arrows into it.
    pressure_arrows = ax_foil.quiver(*np.zeros((4, N_ARROWS)), angles="xy", scale_units="xy", scale=1, units="inches",
                                     width=1.6 / 72, headwidth=3, headlength=4, headaxislength=3.5, minlength=0, zorder=2)
    ax_foil.annotate("", xy=(-0.12, 0.0), xytext=(-0.42, 0.0), arrowprops=dict(arrowstyle="-|>", color=viz.INK_MUTED, lw=1.2))
    freestream = ax_foil.text(-0.27, 0.03, "", ha="center", va="bottom", color=viz.INK_MUTED)
    ax_foil.legend(handles=[Line2D([], [], color=viz.BLUE, lw=1.6, label="suction ($C_p < 0$)"),
                            Line2D([], [], color=viz.ORANGE, lw=1.6, label="positive pressure ($C_p > 0$)")], loc="upper right")
    row = pd.read_csv(osp.join(full.paths.data, "cases.csv")).set_index("name").loc[CASE]
    fig.text(0.07, 0.96, f"DeepONet surrogate, {viz.naca_label(row)}", fontsize=11, va="center")
    readout = fig.text(0.07, 0.925, "", fontsize=9, va="center", color=viz.INK_MUTED, family="monospace")

    x = case["pos"][:, 0]
    in_view = x >= 0  # limits cover everything drawn, including the leading-edge Cf spike
    field_lines = []
    for ax, values, ylabel in ((ax_cp, -cp, "$-C_p$"), (ax_cf, cf, "streamwise_cf $C_f$")):
        ax.axhline(0, color=viz.INK_MUTED, lw=0.6)
        for points in (upper, lower):
            ax.plot(x[points], values[0, points], color=viz.INK_MUTED, lw=0.8, ls="--")
            field_lines.append((ax.plot([], [], color=viz.BLUE, lw=1.4)[0], values, points))
        lo, hi = values[:, in_view].min(), values[:, in_view].max()
        ax.set(xlim=(0, 1), ylim=(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)), ylabel=ylabel)
    ax_cp.tick_params(labelbottom=False)
    ax_cf.set_xlabel("x/c")
    ax_cp.plot([], [], color=viz.INK_MUTED, lw=0.8, ls="--", label=f"Re {re[0] / 1e6:.1f}M, AoA {aoa[0]:.0f}° (start)")
    ax_cp.legend(loc="upper right")

    # Cl and Cd traces: the whole run faint, the part played so far solid.
    frames = np.arange(n_frames)
    traces = []
    for ax, values, label in ((ax_cl, cl, "$C_l$"), (ax_cd, cd, "$C_d$")):
        ax.plot(frames, values, color=viz.GRID, lw=1.4)
        played, = ax.plot([], [], color=viz.BLUE, lw=1.6)
        marker, = ax.plot([], [], "o", color=viz.BLUE, ms=6, mec="white", mew=1.5)
        span = values.max() - values.min()
        ax.set_ylim(values.min() - 0.08 * span, values.max() + 0.08 * span)
        ax.set_ylabel(label)
        for start in np.flatnonzero(np.diff(leg)) + 1:
            ax.axvline(start - 0.5, color=viz.GRID, lw=0.8)
        
        traces.append((values, played, marker))
    ax_cl.tick_params(labelbottom=False)
    ax_cd.set(xlim=(0, n_frames - 1), xticks=[])
    for i in range(1, len(KEYFRAMES)):
        label = leg_label(KEYFRAMES[i - 1], KEYFRAMES[i])
        if label:
            middle = np.flatnonzero(leg == i).mean()
            ax_cl.text(middle, 1.02, label, transform=ax_cl.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=viz.INK_MUTED)

    def draw(f):
        """Update every artist to frame f."""
        body.set_xy(rotate(loop, aoa[f], PIVOT))
        freestream.set_text(f"$U_\\infty$ = {re[f] * NU_GENERATION:.1f} m/s")  # 1 m chord
        base = rotate(station_pos, aoa[f], PIVOT)
        pressure_arrows.set_offsets(np.where(suction[f][:, None], base, tips[f]))
        pressure_arrows.set_UVC(*(np.where(suction[f], 1, -1)[:, None] * (tips[f] - base)).T)
        pressure_arrows.set_facecolor(station_color[f])
        for line, values, points in field_lines:
            line.set_data(x[points], values[f, points])
        for values, played, marker in traces:
            played.set_data(frames[:f + 1], values[:f + 1])
            marker.set_data([f], [values[f]])
        readout.set_text(f"Re {re[f] / 1e6:4.2f}M   AoA {aoa[f]:+5.1f}°   "
                         f"Cl {cl[f]:+.3f}   Cd {cd[f]:.4f}   L/D {cl[f] / cd[f]:+6.1f}")
        return []

    os.makedirs(osp.dirname(path), exist_ok=True)
    FuncAnimation(fig, draw, frames=n_frames).save(path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    animate(osp.join(full.paths.run_figs(TASK), "sweep.gif"))
