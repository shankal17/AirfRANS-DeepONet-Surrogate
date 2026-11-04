"""Inverse design: find the NACA airfoil whose predicted Cp matches a target, with the model full.py trained.

The targets are RANS Cp distributions of held-out test cases, so the true airfoil is known and the
script checks itself. Re and AoA are fixed to the case's values; the search runs over the
continuous NACA parameters of each family in `FAMILIES`, inside the
training ranges, with the cross-entropy method. Every candidate is one batched forward pass, so no
gradients are needed.

The "floor" is the loss of the true airfoil under the model: the model differs from RANS, so no
search can expect to do better. A found loss at or below the floor with different parameters is an
airfoil the model can't tell apart from the truth, not an optimizer failure.
"""

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from airfrans_surrogate import viz
from airfrans_surrogate.data import split_names
from airfrans_surrogate.geometry import chord_stations, naca_case, resample_by_side
from airfrans_surrogate.pipeline.plot import pick
from airfrans_surrogate.trainer import load_model, predict
from animate_sweep import frame_batch  # experiments/ is on sys.path when run directly
from full_training_and_validation import cfg as full

TASK = "full"
N_CASES = 5
# NACA parameter vector of each family, NaN where the search is free. AirfRANS treats a 4-digit p of 0
# as its own camber line, so it is a family of its own; q is the 5-digit reflex flag.
FAMILIES = {
    "4-digit": [np.nan, np.nan, np.nan],
    "4-digit p=0": [np.nan, 0.0, np.nan],
    "5-digit": [np.nan, np.nan, 0.0, np.nan],
    "5-digit reflex": [np.nan, np.nan, 1.0, np.nan],
}
POPULATION = 256
ELITE = 0.1 # fraction of each population the next one is fitted to
ITERATIONS = 15
STD_FLOOR = 0.01 # minimum search spread, as a fraction of each parameter's range
N_STATIONS = 100 # chord stations per surface the Cp error is measured at
SEED = 0
OUT = "figures/inverse"
ANIMATE = True # write a GIF of each case's search (~10 s per case)
FPS = 4
SLOW = 8 # Number of slowed frames, so we can see it converge
SLOW_FACTOR = 3
HOLD = 8 # extra frames on the final iteration
DPI = 100


def family_of(naca):
    """Name of the family in `FAMILIES` a NACA parameter vector belongs to.

    Args:
        naca (array_like): NACA parameters, 3 (4-digit) or 4 (5-digit).

    Returns:
        str: Family name.
    """

    if len(naca) == 3:
        return "4-digit p=0" if naca[1] == 0 else "4-digit"

    return "5-digit reflex" if naca[2] == 1 else "5-digit"


def to_naca(params, family):
    """Free search parameters -> the NACA parameter vector `naca_case` takes.

    Args:
        params (array_like): Values of the family's free (NaN) entries, in order.
        family (str): Family name in `FAMILIES`.

    Returns:
        ndarray: NACA parameters, 3 or 4 of them.
    """

    naca = np.array(FAMILIES[family])
    naca[np.isnan(naca)] = params

    return naca


def param_bounds(cases_csv, family):
    """Training-set range of each free parameter of one family.

    Args:
        cases_csv (DataFrame): cases.csv.
        family (str): Family name in `FAMILIES`.

    Returns:
        tuple: (low, high), each with one entry per free parameter.
    """

    train = cases_csv.loc[cases_csv[f"{TASK}_train"], "naca"].str.split("_").map(lambda parts: [float(v) for v in parts])
    params = np.array([naca for naca in train if family_of(naca) == family])[:, np.isnan(FAMILIES[family])]

    return params.min(0), params.max(0)


@torch.no_grad()
def evaluate(model, normalizer, nacas, re, aoa, device):
    """Predicted Cp at the chord stations, Cl and Cd for several airfoils at one condition.

    Args:
        model (DeepONet): The trained model.
        normalizer (Normalizer): Its normalizer.
        nacas (list): NACA parameter vectors, one per airfoil.
        re (float): Reynolds number.
        aoa (float): Angle of attack in degrees.
        device (str): Device to predict on.

    Returns:
        tuple: (cp, cl, cd) as NumPy arrays of shapes (B, 2 * N_STATIONS), (B,), (B,).
    """

    geometries = [naca_case(naca) for naca in nacas]
    batches = [frame_batch(geometry, np.array([re]), np.array([aoa]), device) for geometry in geometries]
    batch = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}
    y, cl, cd = predict(model, normalizer, batch)
    stations = chord_stations(N_STATIONS)
    cp = np.stack([resample_by_side(cp, geometry["pos"], geometry["side"], stations)
                   for cp, geometry in zip(y[..., 0].cpu().numpy(), geometries)])

    return cp, cl.cpu().numpy(), cd.cpu().numpy()


def cp_rmse(cp, target):
    """Cp error over the chord stations of both surfaces.

    Args:
        cp (ndarray): Predicted Cp, shape (B, 2 * N_STATIONS).
        target (ndarray): Target Cp, shape (2 * N_STATIONS,).

    Returns:
        ndarray: RMSE per airfoil, shape (B,).
    """

    return np.sqrt(((cp - target) ** 2).mean(-1))


def cem(loss_fn, low, high, rng):
    """Cross-entropy method in a box: sample, keep the best, refit a diagonal Gaussian, repeat.

    Args:
        loss_fn (callable): Parameters (B, D) -> loss (B,).
        low (ndarray): Lower bounds, shape (D,).
        high (ndarray): Upper bounds, shape (D,).
        rng (Generator): Random generator.

    Returns:
        dict: "best" parameters (D,) and "best_loss", and per iteration: "history" (best loss so far,
        (ITERATIONS,)), "samples" (ITERATIONS, POPULATION, D), "loss" (ITERATIONS, POPULATION) and
        "best_params" (best so far, (ITERATIONS, D)).
    """

    n_elite = max(2, int(ELITE * POPULATION))
    std_floor = STD_FLOOR * (high - low)
    samples = rng.uniform(low, high, (POPULATION, len(low)))
    best, best_loss = None, np.inf
    trace = {"history": [], "samples": [], "loss": [], "best_params": []}
    for _ in range(ITERATIONS):
        loss = loss_fn(samples)
        if loss.min() < best_loss:
            best, best_loss = samples[loss.argmin()].copy(), float(loss.min())

        for key, value in (("history", best_loss), ("samples", samples), ("loss", loss), ("best_params", best)):
            trace[key].append(value)
        
        elite = samples[np.argsort(loss)[:n_elite]]
        mean, std = elite.mean(0), np.maximum(elite.std(0), std_floor)
        samples = np.clip(mean + std * rng.standard_normal((POPULATION, len(low))), low, high)

    return {"best": best, "best_loss": best_loss, **{k: np.array(v) for k, v in trace.items()}}


def named(naca):
    """Camber, camber position and thickness of a NACA parameter vector (the 5-digit q flag dropped).

    Args:
        naca (array_like): NACA parameters, 3 or 4 of them.

    Returns:
        ndarray: Shape (3,).
    """

    return np.asarray(naca, float)[[0, 1, -1]]


def label(naca):
    """Readable airfoil name, e.g. "4-digit (3.10, 1.00, 18.25)".

    Args:
        naca (array_like): NACA parameters, 3 or 4 of them.

    Returns:
        str: Family, camber, camber position and thickness.
    """

    return f"{family_of(naca)} ({', '.join(f'{v:.2f}' for v in named(naca))})"


def fig_case(name, case, target, found, true, histories, floor, path):
    """Target vs predicted Cp, true vs found outline, and search convergence for one case.

    Args:
        name (str): Case name.
        case (dict): The extracted case.
        target (ndarray): RANS Cp at the chord stations, shape (2 * N_STATIONS,).
        found (dict): naca, cp of the best airfoil found.
        true (dict): naca, cp of the true airfoil under the model.
        histories (dict): Best loss per iteration, by family name.
        floor (float): Loss of the true airfoil.
        path (str): Output file.
    """

    stations = chord_stations(N_STATIONS)
    fig, (ax_cp, ax_shape, ax_loss) = plt.subplots(1, 3, figsize=(14, 4), layout="constrained",
                                                   gridspec_kw={"width_ratios": [1.3, 1.1, 0.9]})
    curves = (("RANS target", target, viz.INK, "-"), (f"model, found: {label(found['naca'])}", found["cp"], viz.BLUE, "-"),
              (f"model, true: {label(true['naca'])}", true["cp"], viz.ORANGE, "--"))
    for text, cp, color, style in curves:
        for i, surface in enumerate((slice(0, N_STATIONS), slice(N_STATIONS, None))):
            ax_cp.plot(stations, -cp[surface], color=color, linestyle=style, linewidth=1.2, label=text if i == 0 else None)
    ax_cp.set(xlabel="x/c", ylabel="−Cp", title=f"Re {float(case['re']) / 1e6:.2f} M, AoA {float(case['aoa_deg']):.2f}°")
    ax_cp.legend(loc="upper right")

    for text, naca, color, style in (("found", found["naca"], viz.BLUE, "-"), ("true", true["naca"], viz.ORANGE, "--")):
        pos = naca_case(naca)["pos"]
        ax_shape.plot(*np.vstack([pos, pos[:1]]).T, color=color, linestyle=style, linewidth=1.2, label=text)
    ax_shape.set_aspect("equal", adjustable="datalim")  # keep the panel's size; y range grows to fill it
    ax_shape.set(xlabel="x/c", ylabel="y/c", title="Airfoil")
    ax_shape.legend(loc="upper right")

    for color, (family, history) in zip(viz.SERIES, histories.items()):
        ax_loss.plot(np.arange(1, len(history) + 1), history, color=color, label=family)
    ax_loss.axhline(floor, color=viz.INK_MUTED, linestyle=":", linewidth=1.0, label="floor (true airfoil)")
    ax_loss.set(xlabel="iteration", ylabel="best Cp RMSE", yscale="log", title="Search")
    ax_loss.legend()

    fig.suptitle(name, fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def outline(naca):
    """Closed outline of a NACA airfoil, for plotting.

    Args:
        naca (array_like): NACA parameters, 3 or 4 of them.

    Returns:
        ndarray: Points, shape (2 * nb_samples + 1, 2); the first point repeats at the end.
    """

    pos = naca_case(naca)["pos"]

    return np.vstack([pos, pos[:1]])


def cp_segments(cp, stations):
    """Upper and lower −Cp curves of each airfoil as line segments for a LineCollection.

    Args:
        cp (ndarray): Cp at the chord stations, shape (B, 2 * N_STATIONS).
        stations (ndarray): x/c of the stations, shape (N_STATIONS,).

    Returns:
        list: 2 * B arrays of shape (N_STATIONS, 2).
    """

    return [np.stack([stations, -surface], 1) for row in cp for surface in (row[:N_STATIONS], row[N_STATIONS:])]


def animate_search(name, case, target, traces, family, bounds, true_naca, floor, model, normalizer, device, path):
    """GIF of the winning family's search: one frame per iteration, the last one held.

    Args:
        name (str): Case name.
        case (dict): The extracted case.
        target (ndarray): RANS Cp at the chord stations, shape (2 * N_STATIONS,).
        traces (dict): `cem` result of every family, by family name.
        family (str): The winning family, whose search is animated.
        bounds (tuple): (low, high) of the family's free parameters, from `param_bounds`.
        true_naca (ndarray): NACA parameters of the true airfoil.
        floor (float): Loss of the true airfoil.
        model (DeepONet): The trained model.
        normalizer (Normalizer): Its normalizer.
        device (str): Device to predict on.
        path (str): Output .gif file.
    """

    trace = traces[family]
    re, aoa = float(case["re"]), float(case["aoa_deg"])
    n_iterations, n_elite = len(trace["history"]), max(2, int(ELITE * POPULATION))
    elites = np.stack([samples[np.argsort(loss)[:n_elite]] for samples, loss in zip(trace["samples"], trace["loss"])])
    nacas = [to_naca(params, family) for params in np.concatenate([elites.reshape(-1, elites.shape[-1]), trace["best_params"]])]
    predictions = [evaluate(model, normalizer, nacas[i:i + POPULATION], re, aoa, device) for i in range(0, len(nacas), POPULATION)]
    cp, cl = np.concatenate([p[0] for p in predictions]), np.concatenate([p[1] for p in predictions])
    elite_cp, best_cp, best_cl = cp[:-n_iterations].reshape(n_iterations, n_elite, -1), cp[-n_iterations:], cl[-n_iterations:]
    elite_outlines = np.array([outline(naca) for naca in nacas[:-n_iterations]]).reshape(n_iterations, n_elite, -1, 2)
    best_outlines = [outline(naca) for naca in nacas[-n_iterations:]]
    stations = chord_stations(N_STATIONS)

    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=(0.8, 1), left=0.06, right=0.98, top=0.88, bottom=0.08, hspace=0.3, wspace=0.28)
    ax_foil = fig.add_subplot(grid[0, :])
    ax_cp, ax_params, ax_loss = (fig.add_subplot(grid[1, i]) for i in range(3))
    fig.text(0.06, 0.96, f"Inverse design: which {family} airfoil gives this Cp?  (Re {re / 1e6:.2f} M, AoA {aoa:.2f}°)",
             fontsize=11, va="center")
    readout = fig.text(0.06, 0.925, "", fontsize=9, va="center", color=viz.INK_MUTED, family="monospace")

    # Airfoil: this iteration's elites faint, the best so far solid, the truth dashed on top.
    elite_foils = ax_foil.add_collection(LineCollection([], colors=viz.BLUE, linewidths=0.6, alpha=0.25))
    best_foil, = ax_foil.plot([], [], color=viz.BLUE, lw=1.6, label="best so far")
    ax_foil.plot(*outline(true_naca).T, color=viz.ORANGE, lw=1.2, ls="--", label=f"true: {label(true_naca)}")
    ax_foil.plot([], [], color=viz.BLUE, lw=0.6, alpha=0.4, label=f"elite (best {n_elite} of {POPULATION})")
    y_extent = np.abs(elite_outlines[..., 1]).max() + 0.02
    ax_foil.set(xlim=(-0.02, 1.02), ylim=(-y_extent, y_extent), aspect="equal", xlabel="x/c", ylabel="y/c")
    ax_foil.legend(loc="upper right", ncols=3)

    # −Cp: limits cover every frame, so motion means change.
    elite_curves = ax_cp.add_collection(LineCollection([], colors=viz.BLUE, linewidths=0.6, alpha=0.2))
    best_curves = ax_cp.add_collection(LineCollection([], colors=viz.BLUE, linewidths=2.4, label="model, best so far"))
    ax_cp.plot(stations, -target[:N_STATIONS], color=viz.INK, lw=0.9, label="RANS target")  # on top of the match
    ax_cp.plot(stations, -target[N_STATIONS:], color=viz.INK, lw=0.9)
    shown = -np.concatenate([elite_cp.ravel(), target])
    span = shown.max() - shown.min()
    ax_cp.set(xlim=(0, 1), ylim=(shown.min() - 0.05 * span, shown.max() + 0.05 * span), xlabel="x/c", ylabel="−Cp")
    ax_cp.legend(loc="upper right")

    # Parameter space: thickness and camber are the last and first free parameters of every family. The axes
    # are the feasible box (the training ranges), fixed for the whole run, so the cloud's motion is real.
    samples = trace["samples"][..., [-1, 0]]
    population = ax_params.scatter([], [], s=8, color=viz.INK_MUTED, alpha=0.35, edgecolors="none", label="population")
    elite_dots = ax_params.scatter([], [], s=14, color=viz.BLUE, edgecolors="none", label="elite")
    if family_of(true_naca) == family:
        ax_params.plot(*named(true_naca)[[2, 0]], "*", color=viz.ORANGE, ms=14, mec="white", mew=0.8, label="true", zorder=5)
    (x_lo, y_lo), (x_hi, y_hi) = bounds[0][[-1, 0]], bounds[1][[-1, 0]]
    x_pad, y_pad = 0.03 * (x_hi - x_lo), 0.03 * (y_hi - y_lo)
    ax_params.set(xlim=(x_lo - x_pad, x_hi + x_pad), ylim=(y_lo - y_pad, y_hi + y_pad), xlabel="thickness (% chord)",
                  ylabel="max camber" if family.startswith("4") else "design lift (l)")
    ax_params.set_title("Search population", pad=20)
    ax_params.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)  # above the axes, never over the points

    # Convergence of every family: whole run faint, played part solid.
    iterations = np.arange(1, n_iterations + 1)
    played = {}
    for color, (other, other_trace) in zip(viz.SERIES, traces.items()):
        ax_loss.plot(iterations, other_trace["history"], color=color, lw=1.0, alpha=0.25)
        played[other] = ax_loss.plot([], [], color=color, lw=1.8 if other == family else 1.2, label=other)[0]
    marker, = ax_loss.plot([], [], "o", color=viz.SERIES[list(traces).index(family)], ms=6, mec="white", mew=1.5)
    ax_loss.axhline(floor, color=viz.INK_MUTED, ls=":", lw=1.0, label="floor (true airfoil)")
    ax_loss.set(xlim=(0.5, n_iterations + 0.5), yscale="log", xlabel="iteration", ylabel="best Cp RMSE", title="Convergence")
    ax_loss.legend(loc="upper right")

    def draw(i):
        """Update every artist to iteration i."""
        elite_foils.set_segments(elite_outlines[i])
        best_foil.set_data(*best_outlines[i].T)
        elite_curves.set_segments(cp_segments(elite_cp[i], stations))
        best_curves.set_segments(cp_segments(best_cp[i:i + 1], stations))
        population.set_offsets(samples[i])
        elite_dots.set_offsets(elites[i][:, [-1, 0]])
        for other, line in played.items():
            line.set_data(iterations[:i + 1], traces[other]["history"][:i + 1])
        marker.set_data([iterations[i]], [trace["history"][i]])
        readout.set_text(f"iteration {i + 1:2d}/{n_iterations}   best: {label(nacas[n_iterations * n_elite + i])}   "
                         f"Cp RMSE {trace['history'][i]:.4f} (floor {floor:.4f})   Cl {best_cl[i]:+.3f} (RANS {float(case['ref_cl']):+.3f})")
        return []

    frames = [i for i in range(n_iterations) for _ in range(SLOW_FACTOR if i < SLOW else 1)] + [n_iterations - 1] * HOLD
    FuncAnimation(fig, draw, frames=frames).save(path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)


def fig_summary(results, path):
    """Every case on one figure: target vs found Cp, true vs found airfoil, and parameter / Cl parity.

    Args:
        results (list[dict]): One per case: aoa, re, target, found_naca, found_cp, true_naca, loss, floor,
            cl_found and cl_rans.
        path (str): Output file.
    """

    stations = chord_stations(N_STATIONS)
    colors = dict(zip(FAMILIES, viz.SERIES))
    n_cases = len(results)
    fig = plt.figure(figsize=(3.2 * n_cases, 9.5), layout="constrained")
    grid = fig.add_gridspec(3, n_cases, height_ratios=(1.2, 0.55, 1.1))
    for i, result in enumerate(results):
        ax_cp = fig.add_subplot(grid[0, i])
        for surface in (slice(0, N_STATIONS), slice(N_STATIONS, None)):
            ax_cp.plot(stations, -result["target"][surface], color=viz.INK, lw=1.2, label="RANS target" if surface.start == 0 else None)
            ax_cp.plot(stations, -result["found_cp"][surface], color=viz.BLUE, lw=1.2, ls="--",
                       label="model, found airfoil" if surface.start == 0 else None)
        ax_cp.set(xlabel="x/c", title=f"AoA {result['aoa']:.1f}°, Re {result['re'] / 1e6:.1f} M\n"
                                      f"Cp RMSE {result['loss']:.4f} (floor {result['floor']:.4f})")
        if i == 0:
            ax_cp.set_ylabel("−Cp")
            ax_cp.legend(loc="upper right")

        ax_foil = fig.add_subplot(grid[1, i])
        ax_foil.plot(*outline(result["found_naca"]).T, color=viz.BLUE, lw=1.4, label="found")
        ax_foil.plot(*outline(result["true_naca"]).T, color=viz.ORANGE, lw=1.1, ls="--", label="true")
        ax_foil.set_aspect("equal", adjustable="datalim")
        ax_foil.set_xlabel(f"true:  {label(result['true_naca'])}\nfound: {label(result['found_naca'])}", fontsize=7)
        ax_foil.tick_params(labelleft=False)
        if i == 0:
            ax_foil.legend(loc="upper right", fontsize=7)

    parity = grid[2, :].subgridspec(1, 3, wspace=0.1)
    quantities = (("thickness (% chord)", lambda r: (named(r["true_naca"])[2], named(r["found_naca"])[2])),
                  ("max camber / design lift", lambda r: (named(r["true_naca"])[0], named(r["found_naca"])[0])),
                  ("Cl", lambda r: (r["cl_rans"], r["cl_found"])))
    for i, (quantity, values) in enumerate(quantities):
        ax = fig.add_subplot(parity[i])
        pairs = np.array([values(r) for r in results])
        lo, hi = pairs.min(), pairs.max()
        margin = 0.08 * (hi - lo) + 1e-3
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], color=viz.INK_MUTED, lw=0.8, ls=":")
        for (true_value, found_value), result in zip(pairs, results):
            family = family_of(result["true_naca"])
            match = family_of(result["found_naca"]) == family
            ax.plot(true_value, found_value, "o", ms=7, color=colors[family], mfc=colors[family] if match else "white", mew=1.4)
        ax.set(xlim=(lo - margin, hi + margin), ylim=(lo - margin, hi + margin), aspect="equal",
               xlabel=f"true {quantity}" if quantity != "Cl" else "RANS Cl", ylabel=f"found {quantity}" if quantity != "Cl" else "model Cl, found airfoil")
    present = {family_of(r["true_naca"]) for r in results}
    handles = [Line2D([], [], marker="o", ls="", color=color, label=family) for family, color in colors.items() if family in present]
    if any(family_of(r["found_naca"]) != family_of(r["true_naca"]) for r in results):
        handles.append(Line2D([], [], marker="o", ls="", color=viz.INK_MUTED, mfc="white", label="found family differs"))
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    fig.suptitle(f"Inverse design on {n_cases} {TASK} test cases: NACA parameters recovered from RANS Cp alone", fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    viz.setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, normalizer = load_model(osp.join(full.paths.runs, TASK, "model.pt"), device)
    cases_csv = pd.read_csv(osp.join(full.paths.data, "cases.csv"))
    bounds = {family: param_bounds(cases_csv, family) for family in FAMILIES}
    _, _, test_names = split_names(osp.join(full.paths.data, "cases.csv"), TASK)
    names = pick(test_names, cases_csv.set_index("name"), N_CASES)
    stations = chord_stations(N_STATIONS)

    rows, results = [], []
    for case_idx, name in enumerate(names):
        case = dict(np.load(osp.join(full.paths.data, name + ".npz")))
        re, aoa = float(case["re"]), float(case["aoa_deg"])
        target = resample_by_side(case["cp"], case["pos"], case["side"], stations)

        traces = {}
        for family_idx, family in enumerate(FAMILIES):
            loss_fn = lambda params: cp_rmse(evaluate(model, normalizer, [to_naca(p, family) for p in params],
                                                      re, aoa, device)[0], target)
            rng = np.random.default_rng([SEED, case_idx, family_idx])
            traces[family] = cem(loss_fn, *bounds[family], rng)
        winner = min(traces, key=lambda family: traces[family]["best_loss"])
        found_naca, found_loss = to_naca(traces[winner]["best"], winner), traces[winner]["best_loss"]

        true_naca = np.asarray(case["naca"], float)
        cp, cl, cd = evaluate(model, normalizer, [found_naca, true_naca], re, aoa, device)
        floor = float(cp_rmse(cp[1:], target)[0])
        fig_case(name, case, target, {"naca": found_naca, "cp": cp[0]}, {"naca": true_naca, "cp": cp[1]},
                 {family: trace["history"] for family, trace in traces.items()}, floor, osp.join(OUT, name + ".png"))
        if ANIMATE:
            animate_search(name, case, target, traces, winner, bounds[winner], true_naca, floor, model, normalizer, device,
                           osp.join(OUT, name + "_search.gif"))
        results.append({"aoa": aoa, "re": re, "target": target, "found_naca": found_naca, "found_cp": cp[0],
                        "true_naca": true_naca, "loss": found_loss, "floor": floor, "cl_found": float(cl[0]),
                        "cl_rans": float(case["ref_cl"])})

        same_family = family_of(found_naca) == family_of(true_naca)
        error = named(found_naca) - named(true_naca) if same_family else np.full(3, np.nan)
        rows.append({
            "aoa": aoa,
            "re_m": re / 1e6,
            "true": label(true_naca),
            "found": label(found_naca),
            "d_camber": error[0],
            "d_pos": error[1],
            "d_t": error[2],
            "loss": found_loss,
            "floor": floor,
            "cl_found": cl[0],
            "cl_rans": float(case["ref_cl"]),
            "cd_found": cd[0],
            "cd_rans": float(case["ref_cd"]),
        })
        print(f"{name}: {label(true_naca)} -> {label(found_naca)}, Cp RMSE {found_loss:.4f} (floor {floor:.4f})")

    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(f"\n{TASK} model, {N_CASES} test cases; d_* = found - true parameter (NaN if the family differs)")
        print(pd.DataFrame(rows).to_string(index=False))
    fig_summary(results, osp.join(OUT, "summary.png"))
    print(f"\nfigures in {OUT}/")


if __name__ == "__main__":
    main()
