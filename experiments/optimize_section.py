"""Design optimization: the least-drag NACA section for a GA wing, with the model full.py trained.

No target Cp: the requirements are lift at two flight conditions, a minimum thickness and a
pitching-moment limit. Each candidate is flown at the AoA that gives each design point's Cl (a
secant solve, a few batched forward passes), so airfoils are compared at equal lift. The objective
is the weighted Cd relative to a NACA 2412 under the same model. The search is the cross-entropy
method of inverse_design.py over each family's continuous NACA parameters, inside the training
ranges. Rerunning it for several thickness floors gives the drag cost of structural depth.

A surrogate optimizer finds the model's errors as eagerly as real improvements. The trust check
reports the model's Cd error on the held-out test cases nearest each optimum; a gain smaller than
that error is not a result. Confirming an optimum takes a CFD run.
"""

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from airfrans_surrogate import viz
from airfrans_surrogate.data import split_names
from airfrans_surrogate.forces import moment_coefficient
from airfrans_surrogate.geometry import chord_stations, naca_case, resample_by_side
from airfrans_surrogate.trainer import load_model, predict
from animate_sweep import frame_batch  # experiments/ is on sys.path when run directly
from full_training_and_validation import cfg as full
from inverse_design import FAMILIES, cem, family_of, label, outline, param_bounds, to_naca

TASK = "full"
# Design points: (Re, target Cl, weight). A GA wing of about 1.5 m chord.
POINTS = {
    "cruise": (5.0e6, 0.4, 0.6),
    "climb": (3.0e6, 1.0, 0.4),
}
SEARCH_FAMILIES = [family for family in FAMILIES if family != "5-digit reflex"]  # q=1 geometry is unreliable in AirfRANS
REFERENCE = [2.0, 4.0, 12.0]  # NACA 2412, the drag the objective is measured against
BASELINES = {"NACA 2412": [2.0, 4.0, 12.0], "NACA 4415": [4.0, 4.0, 15.0], "NACA 23015": [2.0, 3.0, 0.0, 15.0]}
T_MIN = 12.0 # thickness floor (% chord) of the main design
T_SWEEP = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0] # thickness floors of the Pareto sweep
CM_MIN = -0.06 # Cm_c/4 limit at cruise (trim drag)
CM_PENALTY = 10.0 # objective added per 0.01 of Cm below CM_MIN
CL_TOL = 0.01 # Cl miss tolerated before the penalty starts
CL_PENALTY = 10.0 # objective added per 0.01 of Cl miss beyond CL_TOL
AOA_RANGE = (-5.0, 15.0) # training range; the AoA solve stays inside it
AOA_START = (2.0, 8.0) # the secant solve's first two angles
SECANT_STEPS = 2
POLAR_AOA = np.arange(-4.0, 14.5, 1.0)
N_NEIGHBORS = 5 # test cases in the trust check
N_STATIONS = 100 # chord stations per surface for the Cp plot
CHUNK = 512 # airfoils per forward pass
SEED = 0
OUT = "figures/optimize"


@torch.no_grad()
def evaluate(model, normalizer, geometries, re, aoa, device):
    """Model Cl, Cd and Cm_c/4 of each geometry at its own AoA, all at one Re.

    Args:
        model (DeepONet): The trained model.
        normalizer (Normalizer): Its normalizer.
        geometries (list[dict]): `naca_case` geometries, all with the same number of points.
        re (float): Reynolds number.
        aoa (ndarray): Angle of attack in degrees per geometry, shape (B,).
        device (str): Device to predict on.

    Returns:
        dict: "cl", "cd", "cm" of shape (B,) and "y" ([Cp, Cf_x, Cf_y]) of shape (B, N, 3), as NumPy arrays.
    """

    out = {"cl": [], "cd": [], "cm": [], "y": []}
    for start in range(0, len(geometries), CHUNK):
        chunk = geometries[start:start + CHUNK]
        batches = [frame_batch(geometry, np.array([re]), np.array([a]), device) for geometry, a in zip(chunk, aoa[start:start + CHUNK])]
        y, cl, cd = predict(model, normalizer, {k: torch.cat([b[k] for b in batches]) for k in batches[0]})
        y = y.cpu().numpy()
        out["cm"].extend(moment_coefficient(yi[:, 0], yi[:, 1:], g["pos"], g["lines"], g["cell_normals"], g["cell_lengths"])
                         for yi, g in zip(y, chunk))
        out["cl"].append(cl.cpu().numpy())
        out["cd"].append(cd.cpu().numpy())
        out["y"].append(y)

    return {"cl": np.concatenate(out["cl"]), "cd": np.concatenate(out["cd"]), "cm": np.array(out["cm"]),
            "y": np.concatenate(out["y"])}


def fly(model, normalizer, geometries, re, cl_target, device):
    """Each geometry at the AoA where the model gives `cl_target`, by a secant solve on AoA.

    Args:
        model (DeepONet): The trained model.
        normalizer (Normalizer): Its normalizer.
        geometries (list[dict]): `naca_case` geometries.
        re (float): Reynolds number.
        cl_target (float): Lift coefficient to fly at.
        device (str): Device to predict on.

    Returns:
        dict: `evaluate` result at the final AoA, plus "aoa" (B,).
    """

    n = len(geometries)
    (a0, a1) = (np.full(n, a) for a in AOA_START)
    r0, r1 = (evaluate(model, normalizer, geometries, re, a, device) for a in (a0, a1))
    for _ in range(SECANT_STEPS):
        slope = (r1["cl"] - r0["cl"]) / np.where(a1 == a0, 1.0, a1 - a0)
        slope = np.where(slope > 1e-3, slope, 0.1)  # pre-stall Cl rises ~0.1 per degree; guard a flat or stalled pair
        a2 = np.clip(a1 + (cl_target - r1["cl"]) / slope, *AOA_RANGE)
        r2 = evaluate(model, normalizer, geometries, re, a2, device)
        (a0, r0), (a1, r1) = (a1, r1), (a2, r2)

    return {**r1, "aoa": a1}


def design_points(model, normalizer, nacas, device):
    """Every airfoil flown at every design point.

    Args:
        model (DeepONet): The trained model.
        normalizer (Normalizer): Its normalizer.
        nacas (list): NACA parameter vectors.
        device (str): Device to predict on.

    Returns:
        dict: `fly` result per design point name.
    """

    geometries = [naca_case(naca) for naca in nacas]

    return {point: fly(model, normalizer, geometries, re, cl, device) for point, (re, cl, _) in POINTS.items()}


def objective(flown, reference):
    """Weighted Cd relative to the reference airfoil, plus constraint penalties.

    Args:
        flown (dict): `design_points` result for B airfoils.
        reference (dict): `design_points` result for the reference airfoil.

    Returns:
        tuple: (j, penalty), each of shape (B,); the loss is their sum.
    """

    j = sum(weight * flown[point]["cd"] / reference[point]["cd"][0] for point, (_, _, weight) in POINTS.items())
    penalty = CM_PENALTY * np.maximum(0.0, CM_MIN - flown["cruise"]["cm"]) / 0.01
    for point, (_, cl, _) in POINTS.items():
        penalty = penalty + CL_PENALTY * np.maximum(0.0, np.abs(flown[point]["cl"] - cl) - CL_TOL) / 0.01

    return j, penalty


def optimize(model, normalizer, cases_csv, reference, t_min, run_idx, device):
    """Best airfoil over every search family for one thickness floor.

    Args:
        model (DeepONet): The trained model.
        normalizer (Normalizer): Its normalizer.
        cases_csv (DataFrame): cases.csv.
        reference (dict): `design_points` result for the reference airfoil.
        t_min (float): Thickness floor, % chord.
        run_idx (int): Index of this run, for the random seeds.
        device (str): Device to predict on.

    Returns:
        dict: "naca", "family", "loss" and "history" (best loss per iteration, by family).
    """

    best = {"loss": np.inf, "history": {}}
    for family_idx, family in enumerate(SEARCH_FAMILIES):
        low, high = param_bounds(cases_csv, family)
        if t_min > high[-1]:
            continue
        low[-1] = max(low[-1], t_min)  # thickness is the last free parameter of every family
        loss_fn = lambda params: sum(objective(design_points(model, normalizer, [to_naca(p, family) for p in params], device),
                                               reference))
        trace = cem(loss_fn, low, high, np.random.default_rng([SEED, run_idx, family_idx]))
        best["history"][family] = trace["history"]
        if trace["best_loss"] < best["loss"]:
            best.update(naca=to_naca(trace["best"], family), family=family, loss=trace["best_loss"])

    return best


def load_test_cases(names):
    """Geometry, conditions and RANS coefficients of the test cases, for the trust check.

    Args:
        names (list[str]): Test case names.

    Returns:
        list[dict]: One extracted case per name.
    """

    return [dict(np.load(osp.join(full.paths.data, name + ".npz"))) for name in names]


def case_features(descriptor, re, aoa):
    """Feature vector the nearest-neighbor search compares: shape descriptor, Re in millions, AoA / 10.

    Args:
        descriptor (ndarray): Shape descriptor, (32,).
        re (float): Reynolds number.
        aoa (float): Angle of attack in degrees.

    Returns:
        ndarray: Shape (34,).
    """

    return np.concatenate([descriptor, [re / 1e6, aoa / 10.0]])


@torch.no_grad()
def model_on_cases(model, normalizer, cases, device):
    """Model Cd and Cm of each test case on its own geometry and conditions, with the RANS values.

    Args:
        model (DeepONet): The trained model.
        normalizer (Normalizer): Its normalizer.
        cases (list[dict]): Extracted cases.
        device (str): Device to predict on.

    Returns:
        DataFrame: cd_model, cd_rans, cm_model, cm_rans per case.
    """

    rows = []
    for case in cases:
        batch = frame_batch(case, np.array([float(case["re"])]), np.array([float(case["aoa_deg"])]), device)
        y, _, cd = predict(model, normalizer, batch)
        y = y[0].cpu().numpy()
        geometry = (case["pos"], case["lines"], case["cell_normals"], case["cell_lengths"])
        rows.append({"cd_model": float(cd[0]), "cd_rans": float(case["ref_cd"]),
                     "cm_model": float(moment_coefficient(y[:, 0], y[:, 1:], *geometry)),
                     "cm_rans": float(moment_coefficient(case["cp"], case["cf"], *geometry))})

    return pd.DataFrame(rows)


def local_error(naca, flown, cases, features, scale, on_cases):
    """Model Cd error on the test cases nearest an airfoil's design points.

    Args:
        naca (array_like): NACA parameters.
        flown (dict): `design_points` result for this airfoil alone.
        cases (list[dict]): Test cases.
        features (ndarray): `case_features` of each test case, (C, 34).
        scale (ndarray): Per-feature standard deviation over the test cases, (34,).
        on_cases (DataFrame): `model_on_cases` result.

    Returns:
        dict: Per design point, "cd_rel" (RMS relative Cd error of the neighbors) and "distance" (median
        standardized distance to them); "j" is the weighted relative error, a band on the objective.
    """

    descriptor = naca_case(naca)["descriptor"]
    out = {}
    for point, (re, _, _) in POINTS.items():
        query = case_features(descriptor, re, float(flown[point]["aoa"][0]))
        distance = np.linalg.norm((features - query) / scale, axis=1)
        near = np.argsort(distance)[:N_NEIGHBORS]
        rel = (on_cases["cd_model"].to_numpy()[near] - on_cases["cd_rans"].to_numpy()[near]) / on_cases["cd_rans"].to_numpy()[near]
        out[point] = {"cd_rel": float(np.sqrt((rel ** 2).mean())), "distance": float(np.median(distance[near]))}
    out["j"] = sum(weight * out[point]["cd_rel"] for point, (_, _, weight) in POINTS.items())

    return out


def fig_polars(model, normalizer, airfoils, flown, device, path):
    """Drag polars at both design Re, Cm against Cl at cruise, and the outlines.

    Args:
        model (DeepONet): The trained model.
        normalizer (Normalizer): Its normalizer.
        airfoils (dict): NACA parameters by display name, the optimum first.
        flown (dict): `design_points` result per display name.
        device (str): Device to predict on.
        path (str): Output file.
    """

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    colors = dict(zip(airfoils, [viz.INK, *viz.SERIES]))
    for name, naca in airfoils.items():
        geometry = naca_case(naca)
        width = 2.0 if name == next(iter(airfoils)) else 1.2
        for ax, (point, (re, cl_target, _)) in zip(axes[:2], POINTS.items()):
            polar = evaluate(model, normalizer, [geometry] * len(POLAR_AOA), re, POLAR_AOA, device)
            ax.plot(polar["cd"] * 1e4, polar["cl"], color=colors[name], lw=width, label=name)
            ax.plot(flown[name][point]["cd"][0] * 1e4, cl_target, "o", color=colors[name], ms=6, mec="white", mew=1.0)
            if point == "cruise":
                axes[2].plot(polar["cl"], polar["cm"], color=colors[name], lw=width)
        axes[3].plot(*outline(naca).T, color=colors[name], lw=width)
    for ax, (point, (re, cl_target, _)) in zip(axes[:2], POINTS.items()):
        ax.axhline(cl_target, color=viz.INK_MUTED, ls=":", lw=0.8)
        ax.set(xlabel="Cd (counts)", ylabel="Cl", title=f"{point}: Re {re / 1e6:.0f} M, design Cl {cl_target}")
    axes[0].legend(loc="upper left")
    axes[2].axhline(CM_MIN, color=viz.INK_MUTED, ls=":", lw=0.8)
    axes[2].annotate(f"Cm limit {CM_MIN}", (axes[2].get_xlim()[0], CM_MIN), xytext=(4, 4), textcoords="offset points",
                     color=viz.INK_MUTED, fontsize=8)
    axes[2].set(xlabel="Cl", ylabel="Cm c/4", title=f"Pitching moment, Re {POINTS['cruise'][0] / 1e6:.0f} M")
    axes[3].set_aspect("equal", adjustable="datalim")
    axes[3].set(xlabel="x/c", ylabel="y/c", title="Sections")
    fig.suptitle(f"Model drag polars, AoA {POLAR_AOA[0]:.0f}° to {POLAR_AOA[-1]:.0f}°; dots are the design points", fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def fig_design_cp(airfoils, flown, path):
    """−Cp at each design point, the optimum against the reference.

    Args:
        airfoils (dict): NACA parameters by display name: the optimum, then the reference.
        flown (dict): `design_points` result per display name.
        path (str): Output file.
    """

    stations = chord_stations(N_STATIONS)
    fig, axes = plt.subplots(1, len(POINTS), figsize=(11, 4), layout="constrained")
    for ax, (point, (re, cl_target, _)) in zip(axes, POINTS.items()):
        for (name, naca), color, style in zip(airfoils.items(), (viz.INK, viz.BLUE), ("-", "--")):
            geometry, result = naca_case(naca), flown[name][point]
            cp = resample_by_side(result["y"][0, :, 0], geometry["pos"], geometry["side"], stations)
            for surface in (slice(0, N_STATIONS), slice(N_STATIONS, None)):
                ax.plot(stations, -cp[surface], color=color, ls=style, lw=1.3,
                        label=f"{name}, AoA {result['aoa'][0]:.2f}°" if surface.start == 0 else None)
        ax.set(xlabel="x/c", ylabel="−Cp", title=f"{point}: Re {re / 1e6:.0f} M, Cl {cl_target}")
        ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def fig_pareto(sweep, baselines, path):
    """Objective of each thickness floor's optimum, with the trust band, and the baselines at their own thickness.

    Args:
        sweep (list[dict]): One per floor: t_min, naca, j, band, feasible.
        baselines (dict): name -> t, j, feasible.
        path (str): Output file.
    """

    colors = dict(zip(FAMILIES, viz.SERIES))
    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    t, j, band = (np.array([r[key] for r in sweep]) for key in ("t_min", "j", "band"))
    ax.fill_between(t, j * (1 - band), j * (1 + band), color=viz.GRID, alpha=0.8, lw=0, label="model Cd error near optimum")
    ax.plot(t, j, color=viz.INK_MUTED, lw=1.0)
    for r in sweep:
        color = colors[family_of(r["naca"])]
        ax.plot(r["t_min"], r["j"], "o", ms=7, color=color, mfc=color if r["feasible"] else "white", mew=1.4)
    for name, b in baselines.items():
        ax.plot(b["t"], b["j"], "D", ms=6, color=viz.INK, mfc=viz.INK if b["feasible"] else "white", mew=1.2)
        ax.annotate(name, (b["t"], b["j"]), xytext=(5, -3), textcoords="offset points", fontsize=8)
    ax.axhline(1.0, color=viz.INK_MUTED, ls=":", lw=0.8)
    ax.set(xlabel="thickness floor t/c (%)", ylabel="weighted Cd / NACA 2412",
           title=f"Drag cost of thickness (Cm c/4 ≥ {CM_MIN} at cruise)")
    present = {family_of(r["naca"]) for r in sweep}
    handles = [Line2D([], [], marker="o", ls="", color=colors[f], label=f"optimum, {f}") for f in SEARCH_FAMILIES if f in present]
    handles += [Line2D([], [], marker="D", ls="", color=viz.INK, label="baseline"),
                Line2D([], [], marker="o", ls="", color=viz.INK_MUTED, mfc="white", label="violates a constraint")]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0][:1], loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def summary_row(name, naca, flown, j, penalty, error=None):
    """One printed table row.

    Args:
        name (str): Row label.
        naca (array_like): NACA parameters.
        flown (dict): `design_points` result for this airfoil alone.
        j (float): Objective.
        penalty (float): Constraint penalty.
        error (dict, optional): `local_error` result.

    Returns:
        dict: Table row.
    """

    row = {"airfoil": name, "shape": label(naca), "J": j, "penalty": penalty}
    for point in POINTS:
        result = flown[point]
        row |= {f"{point}_aoa": result["aoa"][0], f"{point}_cl": result["cl"][0], f"{point}_cd": result["cd"][0],
                f"{point}_cm": result["cm"][0]}
    if error is not None:
        row |= {"J_err": error["j"], "nn_dist": max(error[point]["distance"] for point in POINTS)}

    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    viz.setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, normalizer = load_model(osp.join(full.paths.runs, TASK, "model.pt"), device)
    cases_csv = pd.read_csv(osp.join(full.paths.data, "cases.csv"))
    reference = design_points(model, normalizer, [REFERENCE], device)

    # Trust check data: the model against RANS on every held-out test case, including Cm, which it never trained on.
    _, _, test_names = split_names(osp.join(full.paths.data, "cases.csv"), TASK)
    test_cases = load_test_cases(test_names)
    on_cases = model_on_cases(model, normalizer, test_cases, device)
    features = np.stack([case_features(c["descriptor"], float(c["re"]), float(c["aoa_deg"])) for c in test_cases])
    scale = features.std(0) + 1e-9
    print(f"test cases: Cd MAPE {np.mean(np.abs(on_cases.cd_model / on_cases.cd_rans - 1)):.3%}, "
          f"Cm MAE {np.mean(np.abs(on_cases.cm_model - on_cases.cm_rans)):.4f} "
          f"(RANS Cm from {on_cases.cm_rans.min():+.3f} to {on_cases.cm_rans.max():+.3f})")

    rows, sweep, optimum = [], [], None
    for run_idx, t_min in enumerate(T_SWEEP):
        best = optimize(model, normalizer, cases_csv, reference, t_min, run_idx, device)
        flown = design_points(model, normalizer, [best["naca"]], device)
        j, penalty = (float(v[0]) for v in objective(flown, reference))
        error = local_error(best["naca"], flown, test_cases, features, scale, on_cases)
        sweep.append({"t_min": t_min, "naca": best["naca"], "j": j, "band": error["j"], "feasible": penalty == 0})
        rows.append({"t_min": t_min, **summary_row("optimum", best["naca"], flown, j, penalty, error)})
        print(f"t/c >= {t_min:.0f}%: {label(best['naca'])}, J {j:.4f} ± {error['j']:.4f}, penalty {penalty:.3f}")
        if t_min == T_MIN:
            optimum = (best["naca"], flown)

    baselines, baseline_rows = {}, []
    for name, naca in BASELINES.items():
        flown = design_points(model, normalizer, [naca], device)
        j, penalty = (float(v[0]) for v in objective(flown, reference))
        error = local_error(naca, flown, test_cases, features, scale, on_cases)
        baselines[name] = {"t": naca[-1], "j": j, "feasible": penalty == 0, "flown": flown}
        baseline_rows.append(summary_row(name, naca, flown, j, penalty, error))
        low, high = param_bounds(cases_csv, family_of(naca))
        free = np.asarray(naca, float)[np.isnan(FAMILIES[family_of(naca)])]
        if np.any(free < low) or np.any(free > high):
            print(f"note: {name} is outside the training range of its family")

    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(f"\nOptimum per thickness floor; J = weighted Cd / NACA 2412, J_err = model Cd error on the "
              f"{N_NEIGHBORS} nearest test cases, nn_dist = their standardized distance")
        print(pd.DataFrame(rows).to_string(index=False))
        print("\nBaselines")
        print(pd.DataFrame(baseline_rows).to_string(index=False))

    naca, flown = optimum
    airfoils = {f"optimum (t ≥ {T_MIN:.0f}%)": naca, **BASELINES}
    flown_by_name = {f"optimum (t ≥ {T_MIN:.0f}%)": flown, **{name: b["flown"] for name, b in baselines.items()}}
    fig_polars(model, normalizer, airfoils, flown_by_name, device, osp.join(OUT, "polars.png"))
    fig_design_cp({k: airfoils[k] for k in list(airfoils)[:2]}, flown_by_name, osp.join(OUT, "design_cp.png"))
    fig_pareto(sweep, baselines, osp.join(OUT, "pareto.png"))
    print(f"\nfigures in {OUT}/")


if __name__ == "__main__":
    main()
