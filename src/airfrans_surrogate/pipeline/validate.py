"""Check to make sure that the extracted data makes sense and our coefficient calculations are correct"""

import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airfrans_surrogate.config import Config
from airfrans_surrogate.forces import integrate_coefficients

def load(data_dir, name):
    """Load one extracted case.

    Args:
        data_dir (str): Directory holding the extracted .npz files.
        name (str): Case name.

    Returns:
        NpzFile: The arrays written by `extract.process_case`.
    """

    return np.load(osp.join(data_dir, name + ".npz"))

def run(cfg: Config):
    """Run the checks, print PASS / FAIL lines and save the validation figures.

    Args:
        cfg: Experiment config; uses `paths.data`, `paths.validation_figs` and `validate.tolerance`.

    Raises:
        RuntimeError: If the force-reproduction or shape-descriptor check fails.
    """

    data_dir, figs_dir, tolerance = cfg.paths.data, cfg.paths.validation_figs, cfg.validate.tolerance
    os.makedirs(figs_dir, exist_ok=True)

    df = pd.read_csv(osp.join(data_dir, "cases.csv"))
    records = []
    for name in df["name"]:
        case = load(data_dir, name)
        coeffs = integrate_coefficients(
            case["cp"],
            case["cf"],
            case["lines"],
            case["cell_normals"],
            case["cell_lengths"],
            np.deg2rad(float(case["aoa_deg"]))
        )
        descriptor = case["descriptor"]
        n_stations = descriptor.size // 2
        records.append({
            "name": name,
            "cl_ours": coeffs["cl"],
            "cd_ours": coeffs["cd"],
            "cdv_ours": coeffs["cd_viscous"],
            "t_desc": float((descriptor[:n_stations] - descriptor[n_stations:]).max()),
        })
    cases = df.merge(pd.DataFrame(records), on="name")

    # 1. force reproduction/parity
    cl_abs_err = np.abs(cases.cl_ours - cases.cl)
    cd_rel_err = np.abs(cases.cd_ours - cases.cd) / np.abs(cases.cd)
    forces_ok = cl_abs_err.max() < tolerance and cd_rel_err.max() < tolerance
    print(f"[{'PASS' if forces_ok else 'FAIL'}] force reproduction: max|dCl| = {cl_abs_err.max():.2e}, "
          f"max rel dCd = {cd_rel_err.max():.2e}  (tol {tolerance:g})")
    if not forces_ok:
        worst = cases.assign(cl_abs_err=cl_abs_err, cd_rel_err=cd_rel_err).nlargest(5, "cd_rel_err")
        print(worst[["name", "cl", "cl_ours", "cd", "cd_ours"]])

    # 2. descriptor sanity (thickness stations are cosine-spaced, so allow ~2% slack)
    thickness_rel_err = np.abs(cases.t_desc - cases.thickness) / cases.thickness
    descriptor_ok = thickness_rel_err.max() < 0.02
    print(f"[{'PASS' if descriptor_ok else 'FAIL'}] shape descriptor thickness: "
          f"max rel err = {thickness_rel_err.max():.2%}")

    # Viscous share of drag: if large, pressure-only models will badly miss Cd.
    viscous_share = cases.cdv / cases.cd
    print(f"[INFO] viscous share of Cd: median {viscous_share.median():.0%}, "
          f"range {viscous_share.min():.0%} to {viscous_share.max():.0%}")
    print(f"[INFO] {len(cases)} cases, Re {cases.re.min():.2e} to {cases.re.max():.2e}, "
          f"AoA {cases.aoa_deg.min():.1f} to {cases.aoa_deg.max():.1f} deg, "
          f"{int((cases.series == 4).sum())} NACA-4 / {int((cases.series == 5).sum())} NACA-5")

    # --- figures ---------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (ref, ours, label) in zip(axs, [(cases.cl, cases.cl_ours, "$C_l$"),
                                            (cases.cd, cases.cd_ours, "$C_d$")]):
        ax.scatter(ref, ours, s=4)
        lo, hi = ref.min(), ref.max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set(xlabel=f"library {label}", ylabel=f"integrated {label}", title=label)
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(osp.join(figs_dir, "force_parity.png"), dpi=150)

    picks = cases.iloc[[cases.aoa_deg.idxmin(), (cases.aoa_deg - 5).abs().idxmin(), cases.aoa_deg.idxmax()]]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for _, row in picks.iterrows():
        case = load(data_dir, row["name"])
        sorted_s = np.argsort(case["s"])
        label = f"AoA {row.aoa_deg:.1f}°, Re {row.re / 1e6:.1f}M, NACA {row.naca}"
        axs[0].plot(case["pos"][sorted_s, 0], -case["cp"][sorted_s], lw=1, label=label)
        tangents = np.stack([-case["normals"][:, 1], case["normals"][:, 0]], 1)  # in-plane tangent
        cf_t = (case["cf"] * tangents).sum(1)
        axs[1].plot(case["s"][sorted_s], cf_t[sorted_s], lw=1)
    axs[0].set(xlabel="x/c", ylabel="$-C_p$", title="Surface pressure")
    axs[1].set(xlabel="s (upper > 0, lower < 0)", ylabel="tangential $C_f$", title="Skin friction")
    axs[0].legend(fontsize=7)
    axs[0].grid(True)
    axs[1].grid(True)
    fig.tight_layout()
    fig.savefig(osp.join(figs_dir, "example_cp_cf.png"), dpi=150)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, task in zip(axs, ["reynolds", "aoa"]):
        for split, color in [("train", "C0"), ("test", "C3")]:
            in_split = cases[f"{task}_{split}"]
            ax.scatter(cases.re[in_split] / 1e6, cases.aoa_deg[in_split], s=4, c=color, label=split)
        ax.set(xlabel="Re (millions)", ylabel="AoA (deg)", title=f"'{task}' extrapolation split")
        ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(figs_dir, "split_coverage.png"), dpi=150)

    print(f"Figures written to {figs_dir}/")
    if not (forces_ok and descriptor_ok):
        raise RuntimeError("Validation failed (see FAIL lines above)")
