"""In-memory surface dataset.

All cases fit comfortably in GPU memory (~1-2k surface points each),
so the whole split is loaded once into padded tensors and batches
"""

from __future__ import annotations

import os.path as osp
import numpy as np
import pandas as pd
import torch

from airfrans_surrogate.geometry import tangent

BRANCH_DIM = 34
TRUNK_DIM = 7
N_OUT = 3
TARGET_NAMES = ("cp", "cf_x", "cf_y")
# cases.csv column each extrapolation task's test set lies outside of, at both ends
EDGE_COLUMNS = {"reynolds": "re", "aoa": "aoa_deg"}


def trunk_dim(trunk_aoa: bool) -> int:
    """Trunk columns a model sees: all of them, or the five position-only ones.

    Args:
        trunk_aoa: Whether the model gets the two freestream-orientation columns.

    Returns:
        Number of trunk input columns.
    """

    return TRUNK_DIM if trunk_aoa else TRUNK_DIM - 2


def num_output_channels(cf_tangential: bool) -> int:
    """Number of output channels: [Cp, Cf_t] or [Cp, Cf_x, Cf_y].

    Args:
        cf_tangential: Whether the model predicts tangential skin friction only.

    Returns:
        Number of output channels.
    """

    return 2 if cf_tangential else N_OUT


def trunk_features(case) -> np.ndarray:
    """Per-point trunk inputs [u, x, 5*y, nx, ny, n.e, n x e] for one case.

    The normal at the largest-s point (the trailing edge) is replaced by its neighbour's.

    Args:
        case (NpzFile): One extracted case from `extract.process_case`; uses `s`, `pos`, `normals` and `aoa_deg`.

    Returns:
        Array of shape (N, TRUNK_DIM).
    """

    s = case["s"] # chord coordinate on the airfoil
    s_stretched = np.sign(s) * np.sqrt(np.abs(s)) # transform so that the leading edge is spread out over a longer interval
    sorted_s = np.argsort(s)
    pos, normals = case["pos"], case["normals"].copy() # Point coordinates and their surface normal vectors
    normals[sorted_s[-1]] = normals[sorted_s[-2]] # Always use the normal of the top surface for the trailing edge
    aoa_rad = np.deg2rad(float(case["aoa_deg"]))
    freestream_dir = np.array([np.cos(aoa_rad), np.sin(aoa_rad)])  # freestream direction
    cross = normals[:, 0] * freestream_dir[1] - normals[:, 1] * freestream_dir[0]

    return np.stack([
        s_stretched,
        pos[:, 0],
        5.0 * pos[:, 1], # Scale to similar scales as the other features
        normals[:, 0],
        normals[:, 1],
        normals @ freestream_dir,
        cross
    ], 1)


def point_lengths(lines, cell_lengths, n_points) -> np.ndarray:
    """Half the length of each adjacent segment, summed per point, scaled to mean 1.

    The mesh clusters points at the leading edge, so an unweighted mean over points spends
    about a third of the loss on x/c < 0.02; these weights spread it evenly along the surface.

    Args:
        lines (ndarray): Segment endpoint indices, shape (M, 2).
        cell_lengths (ndarray): Segment lengths, shape (M,).
        n_points (int): Number of surface points N.

    Returns:
        Per-point weights of shape (N,), mean 1.
    """

    weights = np.bincount(lines.ravel(), np.repeat(0.5 * cell_lengths, 2), minlength=n_points)[:n_points]

    return weights / weights.mean()


def branch_features(descriptor, re, aoa_deg) -> np.ndarray:
    """Branch input for one case: shape descriptor, Re in millions, AoA in degrees.

    Args:
        descriptor (array_like): Shape descriptor from `geometry.shape_descriptor`, shape (32,).
        re (float): Reynolds number.
        aoa_deg (float): Angle of attack in degrees.

    Returns:
        Array of shape (BRANCH_DIM,).
    """

    return np.concatenate([np.asarray(descriptor), [re / 1e6, aoa_deg]])


class SurfaceDataset:
    """Padded tensors for a list of cases.

    Cases are padded to the largest point count N_max and segment count M_max in the list;
    padding is zero (and False in `mask`), so it contributes nothing to losses or forces.

    Args:
        data_dir: Directory holding the extracted <name>.npz files.
        names: Case names to load, in order.
        device: Device for all tensors.

    Attributes:
        names (list[str]): Case names, in tensor order.
        branch (Tensor): Branch inputs, shape (C, BRANCH_DIM).
        trunk (Tensor): Trunk inputs, shape (C, N_max, TRUNK_DIM).
        target (Tensor): Physical [Cp, Cf_x, Cf_y], shape (C, N_max, 3).
        tangent (Tensor): Unit surface tangents, shape (C, N_max, 2).
        mask (Tensor): True on real points, shape (C, N_max).
        point_len (Tensor): Arc-length weight per point, mean 1 per case, shape (C, N_max).
        lines (Tensor): Segment endpoint indices, shape (C, M_max, 2).
        cell_normals (Tensor): Segment normals, shape (C, M_max, 2).
        cell_lengths (Tensor): Segment lengths, shape (C, M_max).
        aoa_rad (Tensor): Angle of attack in radians, shape (C,).
        cl (Tensor): Reference lift coefficient, shape (C,).
        cd (Tensor): Reference drag coefficient, shape (C,).
        meta (DataFrame): One row per case: name, re, aoa_deg, cl, cd, n (point count).
    """

    def __init__(self, data_dir: str, names: list[str], device="cpu"):
        self.names = list(names)
        cases = [np.load(osp.join(data_dir, name + ".npz")) for name in self.names]
        n_cases = len(cases)
        max_points = max(case["pos"].shape[0] for case in cases)
        max_segments = max(case["lines"].shape[0] for case in cases)

        # Initialize arrays as zero
        trunk = np.zeros((n_cases, max_points, TRUNK_DIM), np.float32)
        target = np.zeros((n_cases, max_points, N_OUT), np.float32)
        tangents = np.zeros((n_cases, max_points, 2), np.float32)  # 0 on padding -> rebuilt Cf is 0 there
        mask = np.zeros((n_cases, max_points), bool)
        point_len = np.zeros((n_cases, max_points), np.float32)
        lines = np.zeros((n_cases, max_segments, 2), np.int64)
        cell_normals = np.zeros((n_cases, max_segments, 2), np.float32)
        cell_lengths = np.zeros((n_cases, max_segments), np.float32)  # 0 on padding -> no contribution
        branch = np.zeros((n_cases, BRANCH_DIM), np.float32)
        meta = []

        # populate arrays with case data
        for i, case in enumerate(cases):
            n_points, n_segments = case["pos"].shape[0], case["lines"].shape[0]
            trunk[i, :n_points] = trunk_features(case)
            target[i, :n_points, 0] = case["cp"]
            target[i, :n_points, 1:] = case["cf"]
            tangents[i, :n_points] = tangent(case["normals"])
            mask[i, :n_points] = True
            point_len[i, :n_points] = point_lengths(case["lines"], case["cell_lengths"], n_points)
            lines[i, :n_segments] = case["lines"]
            cell_normals[i, :n_segments] = case["cell_normals"]
            cell_lengths[i, :n_segments] = case["cell_lengths"]
            branch[i] = branch_features(case["descriptor"], float(case["re"]), float(case["aoa_deg"]))
            meta.append({
                "name": self.names[i],
                "re": float(case["re"]),
                "aoa_deg": float(case["aoa_deg"]),
                "cl": float(case["ref_cl"]),
                "cd": float(case["ref_cd"]),
                "n": n_points
            })

        # Convert from numpy arrays to torch tensors
        to_tensor = lambda array: torch.as_tensor(array, device=device)
        self.trunk, self.target, self.mask = to_tensor(trunk), to_tensor(target), to_tensor(mask)
        self.point_len, self.tangent = to_tensor(point_len), to_tensor(tangents)
        self.lines, self.cell_normals = to_tensor(lines), to_tensor(cell_normals)
        self.cell_lengths = to_tensor(cell_lengths)
        self.branch = to_tensor(branch)
        self.meta = pd.DataFrame(meta)
        self.aoa_rad = to_tensor(np.deg2rad(self.meta.aoa_deg.values).astype(np.float32))
        self.cl = to_tensor(self.meta.cl.values.astype(np.float32))
        self.cd = to_tensor(self.meta.cd.values.astype(np.float32))

    def __len__(self):
        """Number of cases."""

        return len(self.names)

    def batch(self, idx):
        """Select cases by index.

        Args:
            idx (Tensor): Case indices, shape (B,), on the dataset's device.

        Returns:
            dict: The per-case tensors indexed by `idx`
        """

        features = (
            "branch",
            "trunk",
            "target",
            "tangent",
            "mask",
            "point_len",
            "lines",
            "cell_normals",
            "cell_lengths",
            "aoa_rad",
            "cl",
            "cd"
        )

        return {k: getattr(self, k)[idx] for k in features}


class Normalizer:
    """Per-feature standardization fitted on training cases (masked over padding).

    Args:
        stats: Statistics as stored in a checkpoint. Usually built by `fit`.
        cf_tangential: Whether the model channels are [Cp, Cf_t] rather than [Cp, Cf_x, Cf_y].

    Attributes:
        stats (dict[str, Tensor]): "branch_mean", "branch_std" (shape (BRANCH_DIM,)),
            "target_mean", "target_std" (shape (num_output_channels,), over model channels), and scalar
            "cl_std" and "cd_std".
        cf_tangential (bool): As passed in.
    """

    def __init__(self, stats: dict | None = None, cf_tangential: bool = False):
        self.stats = dict(stats or {})
        self.cf_tangential = cf_tangential

    @classmethod
    def fit(cls, dataset: SurfaceDataset, cf_tangential: bool = False) -> "Normalizer":
        """Fit the statistics on a training set, ignoring padded points.

        Args:
            dataset: Training cases.
            cf_tangential: Whether the model channels are [Cp, Cf_t].

        Returns:
            A normalizer with its statistics on the CPU.
        """
        
        mask = dataset.mask
        targets = cls(cf_tangential=cf_tangential).to_model_channels(dataset.target, dataset.tangent)[mask]
        stats = {
            "branch_mean": dataset.branch.mean(0),
            "branch_std": dataset.branch.std(0).clamp_min(1e-6),
            "target_mean": targets.mean(0),
            "target_std": targets.std(0).clamp_min(1e-8),
            "cl_std": dataset.cl.std().clamp_min(1e-6),
            "cd_std": dataset.cd.std().clamp_min(1e-8),
        }

        return cls({k: v.detach().cpu() for k, v in stats.items()}, cf_tangential)

    def to(self, device):
        """Copy of this normalizer with its statistics on `device`.

        Args:
            device (str | torch.device): Target device.

        Returns:
            Normalizer: The moved copy.
        """

        return Normalizer({k: v.to(device) for k, v in self.stats.items()}, self.cf_tangential)

    def normalize_branch(self, branch):
        """Standardize branch inputs.

        Args:
            branch (Tensor): Branch inputs, shape (..., BRANCH_DIM).

        Returns:
            Tensor: Standardized inputs, same shape.
        """

        return (branch - self.stats["branch_mean"]) / self.stats["branch_std"]

    def to_model_channels(self, y, tangent):
        """Physical (..., N, 3) -> the model's channels, unnormalized.

        Args:
            y (Tensor): Physical [Cp, Cf_x, Cf_y], shape (..., N, 3).
            tangent (Tensor): Unit surface tangents, shape (..., N, 2).

        Returns:
            Tensor: [Cp, Cf_t] of shape (..., N, 2) with `cf_tangential`, else `y` unchanged.
        """

        if not self.cf_tangential:
            return y
        
        return torch.cat([y[..., :1], (y[..., 1:] * tangent).sum(-1, keepdim=True)], -1)

    def from_model_channels(self, y, tangent):
        """Inverse of `to_model_channels`: model channels (unnormalized) -> (..., N, 3).

        Args:
            y (Tensor): Model channels in physical units, shape (..., N, num_output_channels).
            tangent (Tensor): Unit surface tangents, shape (..., N, 2).

        Returns:
            Tensor: Physical [Cp, Cf_x, Cf_y], shape (..., N, 3).
        """

        if not self.cf_tangential:
            return y
        
        return torch.cat([y[..., :1], y[..., 1:2] * tangent], -1)

    def normalized_target(self, batch):
        """Normalized training target for a batch.

        Args:
            batch (dict): Batch from `SurfaceDataset.batch`.

        Returns:
            Tensor: Normalized model channels, shape (B, N, num_output_channels).
        """

        return self.normalize_target(self.to_model_channels(batch["target"], batch["tangent"]))

    def to_physical(self, pred, batch):
        """Normalized model output for a batch -> physical [Cp, Cf_x, Cf_y].

        Args:
            pred (Tensor): Normalised model output, shape (..., B, N, num_output_channels).
            batch (dict): Batch from `SurfaceDataset.batch`.

        Returns:
            Tensor: Physical [Cp, Cf_x, Cf_y], shape (..., B, N, 3).
        """

        return self.from_model_channels(self.denormalize_target(pred), batch["tangent"])

    def normalize_target(self, y):
        """Standardize model channels.

        Args:
            y (Tensor): Model channels in physical units, shape (..., num_output_channels).

        Returns:
            Tensor: Standardized channels, same shape.
        """

        return (y - self.stats["target_mean"]) / self.stats["target_std"]

    def denormalize_target(self, y):
        """Inverse of `normalize_target`.

        Args:
            y (Tensor): Standardized model channels, shape (..., num_output_channels).

        Returns:
            Tensor: Model channels in physical units, same shape.
        """

        return y * self.stats["target_std"] + self.stats["target_mean"]


def split_names(cases_csv_path: str, task: str, val_frac: float = 0.1, seed: int = 0, holdout: str = "random"):
    """Train / val / test name lists for an AirfRANS task.

    Args:
        cases_csv_path: Path to cases.csv, with "name" and "<task>_train" / "<task>_test" columns.
        task: AirfRANS task, one of `config.TASKS`. "scarce" is tested on the "full" test set.
        val_frac: Fraction of the training split held out for validation (at least one case).
        seed: Seed for the validation split.
        holdout: "random", or "edge" to hold out the training cases with the lowest and highest `EDGE_COLUMNS[task]`

    Returns:
        tuple: (train_names, val_names, test_names), each a list of case names.

    Raises:
        ValueError: If `holdout` is "edge" and `task` has no extrapolation axis.
    """

    if holdout == "edge" and task not in EDGE_COLUMNS:
        raise ValueError(f"holdout='edge' needs an extrapolation task, one of {tuple(EDGE_COLUMNS)}")
    
    df = pd.read_csv(cases_csv_path)
    test_col = "full_test" if task == "scarce" else f"{task}_test"
    all_train = df.loc[df[f"{task}_train"], "name"].tolist()
    test_names = df.loc[df[test_col], "name"].tolist()
    n_val = max(1, int(round(val_frac * len(all_train))))
    if holdout == "edge":
        order = np.argsort(df.loc[df[f"{task}_train"], EDGE_COLUMNS[task]].to_numpy(), kind="stable")
        n_low = n_val // 2
        val_idx = set(order[:n_low]) | set(order[len(order) - (n_val - n_low):])
        val_names = [name for i, name in enumerate(all_train) if i in val_idx]
        train_names = [name for i, name in enumerate(all_train) if i not in val_idx]

        return train_names, val_names, test_names
    
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_train))
    val_names = [all_train[i] for i in perm[:n_val]]
    train_names = [all_train[i] for i in perm[n_val:]]

    return train_names, val_names, test_names
