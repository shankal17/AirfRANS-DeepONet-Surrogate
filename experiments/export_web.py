"""Export the model full.py trained for the in-browser demo (docs/demo/).

Writes model.onnx (normalisation baked in), meta.json (training ranges and test metrics) and
cases.json (a few test cases with their RANS fields, for overlays) to docs/demo/, and
tests/web_fixtures.json (Python reference values that tests/test_web.mjs checks the JavaScript
port against). Before writing anything it checks that ONNX Runtime matches PyTorch, and reports
how much the predicted forces move when the geometry comes from the NACA generator (as in the
browser) instead of the CFD mesh.
"""

import json
import os
import os.path as osp

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from torch import nn

from airfrans_surrogate.data import BRANCH_DIM, TRUNK_DIM, branch_features, split_names, trunk_features
from airfrans_surrogate.forces import integrate_coefficients
from airfrans_surrogate.geometry import naca_case, streamwise_cf, tangent
from airfrans_surrogate.trainer import load_model, predict
from animate_sweep import frame_batch  # experiments/ is on sys.path when run directly
from full_training_and_validation import cfg as full

TASK = "full"
OUT = "demo"
FIXTURES = "tests/web_fixtures.json"
NB_SAMPLES = 400 # generator samples per surface; 800 surface points in total
N_OVERLAY_CASES = 12
N_FIXTURE_CASES = 3


class WebModel(nn.Module):
    """The trained model with normalisation folded in: raw inputs to physical [Cp, Cf_t].

    Args:
        model (DeepONet): The trained model.
        normalizer (Normalizer): Its normalizer; must be `cf_tangential`.
    """

    def __init__(self, model, normalizer):
        super().__init__()
        self.model = model
        for key in ("branch_mean", "branch_std", "target_mean", "target_std"):
            self.register_buffer(key, normalizer.stats[key].float())

    def forward(self, branch, trunk):
        """Physical surface fields.

        Args:
            branch (Tensor): Raw branch inputs from `branch_features`, shape (B, BRANCH_DIM).
            trunk (Tensor): Trunk inputs from `trunk_features`, shape (B, N, TRUNK_DIM).

        Returns:
            Tensor: [Cp, Cf_t], shape (B, N, 2).
        """

        y = self.model((branch - self.branch_mean) / self.branch_std, trunk[..., :self.model.trunk_dim])

        return y * self.target_std + self.target_mean


def export_onnx(web_model, path):
    """Write the model to ONNX with dynamic batch and point counts.

    Args:
        web_model (WebModel): Model to export.
        path (str): Output file.
    """

    example = (torch.zeros(2, BRANCH_DIM), torch.zeros(2, 64, TRUNK_DIM))
    batch, points = torch.export.Dim("batch"), torch.export.Dim("points")
    program = torch.onnx.export(web_model, example, input_names=["branch", "trunk"], output_names=["fields"],
                                dynamic_shapes=({0: batch}, {0: batch, 1: points}), dynamo=True)
    program.save(path, external_data=False)


def check_onnx(path, model, normalizer, cases):
    """Largest difference between ONNX Runtime and `trainer.predict` over some cases.

    Args:
        path (str): ONNX file.
        model (DeepONet): The trained model.
        normalizer (Normalizer): Its normalizer.
        cases (list[dict]): Extracted cases.

    Returns:
        float: Max |difference| of Cp and Cf_t over every point of every case.
    """

    session = ort.InferenceSession(path)
    worst = 0.0
    for case in cases:
        batch = frame_batch(case, np.array([float(case["re"])]), np.array([float(case["aoa_deg"])]), "cpu")
        y, _, _ = predict(model, normalizer, batch)
        expected = torch.cat([y[..., :1], (y[..., 1:] * batch["tangent"]).sum(-1, keepdim=True)], -1).numpy()
        got = session.run(None, {"branch": batch["branch"].numpy(), "trunk": batch["trunk"].numpy()})[0]
        worst = max(worst, float(np.abs(got - expected).max()))

    return worst


def generator_vs_mesh(model, normalizer, cases):
    """Predicted forces on generator geometry vs on the CFD mesh, and both vs RANS.

    Args:
        model (DeepONet): The trained model.
        normalizer (Normalizer): Its normalizer.
        cases (list[dict]): Extracted cases.

    Returns:
        DataFrame: One row per case with ref_cl, ref_cd, cl_mesh, cd_mesh, cl_gen, cd_gen.
    """

    rows = []
    for case in cases:
        re, aoa = np.array([float(case["re"])]), np.array([float(case["aoa_deg"])])
        _, cl_mesh, cd_mesh = predict(model, normalizer, frame_batch(case, re, aoa, "cpu"))
        _, cl_gen, cd_gen = predict(model, normalizer, frame_batch({**case, **naca_case(case["naca"])}, re, aoa, "cpu"))
        rows.append({"ref_cl": float(case["ref_cl"]), "ref_cd": float(case["ref_cd"]), "cl_mesh": float(cl_mesh),
                     "cd_mesh": float(cd_mesh), "cl_gen": float(cl_gen), "cd_gen": float(cd_gen)})

    return pd.DataFrame(rows)


def meta(cases_csv, metrics):
    """Training ranges and headline test metrics for the page.

    Args:
        cases_csv (DataFrame): cases.csv.
        metrics (dict): results/metrics.json.

    Returns:
        dict: JSON-ready metadata.
    """

    train = cases_csv[cases_csv[f"{TASK}_train"]]
    params = train.naca.str.split("_", expand=True).astype(float)
    span = lambda values: [round(float(values.min()), 4), round(float(values.max()), 4)]
    four, five = params[train.series == 4], params[train.series == 5]
    test = metrics[TASK]["test"]

    return {
        "task": TASK,
        "nb_samples": NB_SAMPLES,
        "re": span(train.re),
        "aoa_deg": span(train.aoa_deg),
        "naca4": {"m": span(four[0]), "p": span(four[1][four[1] > 0]), "t": span(four[2])},  # p = 0 is an AirfRANS special case the page doesn't offer
        "naca5": {"l": span(five[0]), "p": span(five[1]), "q": sorted(five[2].unique().tolist()), "t": span(five[3])},
        "test_metrics": {k: test[k] for k in ("n_cases", "cp_rel_l2", "cf_rel_l2", "cl_mae", "cd_mae", "cd_mape")},
    }


def overlay_case(name, case):
    """One test case's RANS surface fields, in the form the page plots.

    Args:
        name (str): Case name.
        case (dict): The extracted case.

    Returns:
        dict: name, naca, series, re, aoa_deg, ref_cl, ref_cd, and per-point x, upper flag, Cp and streamwise Cf.
    """

    rounded = lambda values, digits: np.round(np.asarray(values, float), digits).tolist()

    return {
        "name": name,
        "naca": case["naca"].tolist(),
        "series": int(case["series"]),
        "re": float(case["re"]),
        "aoa_deg": float(case["aoa_deg"]),
        "ref_cl": float(case["ref_cl"]),
        "ref_cd": float(case["ref_cd"]),
        "x": rounded(case["pos"][:, 0], 5),
        "upper": (case["side"] > 0).astype(int).tolist(),
        "cp": rounded(case["cp"], 4),
        "cf": rounded(streamwise_cf(case["cf"], case["normals"], case["side"]), 6),
    }


def fixture(name, case, session):
    """Python reference values for one case on generator geometry, for tests/test_web.mjs.

    Args:
        name (str): Case name.
        case (dict): The extracted case; only its NACA parameters and conditions are used.
        session (InferenceSession): ONNX Runtime session of model.onnx.

    Returns:
        dict: Inputs (naca, re, aoa_deg) and the generator geometry, model inputs, model output and forces.
    """

    geometry = naca_case(case["naca"])
    re, aoa = float(case["re"]), float(case["aoa_deg"])
    branch = branch_features(geometry["descriptor"], re, aoa).astype(np.float32)
    trunk = trunk_features({**geometry, "aoa_deg": aoa}).astype(np.float32)
    fields = session.run(None, {"branch": branch[None], "trunk": trunk[None]})[0][0].astype(np.float64)
    cf = fields[:, 1:2] * tangent(geometry["normals"])
    forces = integrate_coefficients(fields[:, 0], cf, geometry["lines"], geometry["cell_normals"],
                                    geometry["cell_lengths"], np.deg2rad(aoa))
    as_list = lambda values: np.asarray(values, float).tolist()

    return {
        "name": name,
        "naca": case["naca"].tolist(),
        "re": re,
        "aoa_deg": aoa,
        **{k: as_list(geometry[k]) for k in ("pos", "normals", "s", "cell_normals", "cell_lengths", "descriptor")},
        "branch": as_list(branch),
        "trunk": as_list(trunk),
        "cp": as_list(fields[:, 0]),
        "cf_t": as_list(fields[:, 1]),
        **{k: float(v) for k, v in forces.items()},
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    model, normalizer = load_model(osp.join(full.paths.runs, TASK, "model.pt"))
    assert normalizer.cf_tangential, "the page rebuilds Cf from Cf_t; export a cf_tangential model"
    _, _, test_names = split_names(osp.join(full.paths.data, "cases.csv"), TASK)
    cases = {name: dict(np.load(osp.join(full.paths.data, name + ".npz"))) for name in test_names}

    path = osp.join(OUT, "model.onnx")
    export_onnx(WebModel(model, normalizer).eval(), path)
    worst = check_onnx(path, model, normalizer, list(cases.values())[:20])
    print(f"model.onnx ({osp.getsize(path) / 1e6:.1f} MB): max |ONNX - torch| = {worst:.1e}")
    assert worst < 1e-4, "ONNX export does not match PyTorch"

    forces = generator_vs_mesh(model, normalizer, list(cases.values()))
    print(f"generator vs mesh geometry on {len(forces)} {TASK} test cases ({2 * NB_SAMPLES} generator points):")
    for k in ("cl", "cd"):
        print(f"  {k}: |gen - mesh| MAE {np.abs(forces[k + '_gen'] - forces[k + '_mesh']).mean():.1e}, "
              f"MAE vs RANS {np.abs(forces[k + '_mesh'] - forces['ref_' + k]).mean():.1e} (mesh) "
              f"{np.abs(forces[k + '_gen'] - forces['ref_' + k]).mean():.1e} (gen)")

    cases_csv = pd.read_csv(osp.join(full.paths.data, "cases.csv"))
    with open(osp.join(full.paths.results, "metrics.json")) as f:
        metrics = json.load(f)
    by_aoa = sorted(cases, key=lambda name: float(cases[name]["aoa_deg"]))
    overlay_names = [by_aoa[i] for i in np.linspace(0, len(by_aoa) - 1, N_OVERLAY_CASES).round().astype(int)]
    session = ort.InferenceSession(path)
    outputs = {
        osp.join(OUT, "meta.json"): meta(cases_csv, metrics),
        osp.join(OUT, "cases.json"): [overlay_case(name, cases[name]) for name in overlay_names],
        FIXTURES: [fixture(name, cases[name], session) for name in overlay_names[:N_FIXTURE_CASES]],
    }
    for out_path, content in outputs.items():
        with open(out_path, "w") as f:
            json.dump(content, f, separators=(",", ":"))
        print(f"wrote {out_path} ({osp.getsize(out_path) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
