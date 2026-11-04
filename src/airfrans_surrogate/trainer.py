"""Training one surface DeepONet, and loading / querying a trained checkpoint."""

from __future__ import annotations

import time
import numpy as np
import torch

from airfrans_surrogate.data import BRANCH_DIM, Normalizer, SurfaceDataset, num_output_channels, trunk_dim
from airfrans_surrogate.forces import integrate_coefficients_batch
from airfrans_surrogate.model import DeepONet

def build_model(hparams):
    """Untrained DeepONet with the architecture in `hparams`.

    Args:
        hparams (dict): From `TrainConfig.hparams()`.

    Returns:
        DeepONet: The model, on the CPU.
    """

    model = DeepONet(
        BRANCH_DIM,
        trunk_dim(hparams["trunk_aoa"]),
        num_output_channels(hparams["cf_tangential"]),
        hparams["hidden_layer_width"],
        hparams["num_hidden_layers"],
        hparams["n_basis"],
        hparams["n_fourier"]
    )

    return model


def run_model(model, normalizer, batch):
    """Normalized model output for a batch (B, N, num_output_channels).

    Args:
        model (DeepONet): The model; only its first `model.trunk_dim` trunk columns are used.
        normalizer (Normalizer): Fitted normalizer.
        batch (dict): Batch from `SurfaceDataset.batch`.

    Returns:
        Tensor: Normalized prediction, shape (B, N, num_output_channels).
    """

    return model(normalizer.normalize_branch(batch["branch"]), batch["trunk"][..., :model.trunk_dim])


def force_weight_at(epoch, hparams):
    """Force-term weight for `epoch`.

    Args:
        epoch (int): Zero-based epoch.
        hparams (dict): From `TrainConfig.hparams()`.

    Returns:
        float: Weight in [0, hparams["force_weight"]].
    """

    n_epochs = hparams["epochs"]
    start, ramp = hparams["force_start"] * n_epochs, max(1.0, 0.2 * n_epochs)

    return hparams["force_weight"] * float(np.clip((epoch - start) / ramp, 0.0, 1.0))


def forces_from_prediction(pred_physical, batch):
    """Integrate physical-unit predictions (B, N, 3) -> (cl, cd) of shape (B,).

    Args:
        pred_physical (Tensor): [Cp, Cf_x, Cf_y] in physical units, shape (B, N, 3).
        batch (dict): Batch from `SurfaceDataset.batch`.

    Returns:
        tuple: (cl, cd), each a Tensor of shape (B,).
    """

    cp = pred_physical[:, :, 0] * batch["mask"]
    cf = pred_physical[:, :, 1:] * batch["mask"][..., None]
    res = integrate_coefficients_batch(
        cp,
        cf,
        batch["lines"],
        batch["cell_normals"],
        batch["cell_lengths"],
        batch["aoa_rad"]
    )
    return res


def loss_terms(model, normalizer, batch, length_weight, force_weight):
    """Per-case pieces of the loss: weighted squared-error sum, weight sum and weighted force term.

    Args:
        model (DeepONet): The model.
        normalizer (Normalizer): Fitted normalizer.
        batch (dict): Batch from `SurfaceDataset.batch`.
        length_weight (float): Blend of per-point (0) and per-arc-length (1) weighting.
        force_weight (float): Weight of the force term; <= 0 skips it and returns zeros.

    Returns:
        tuple: (weighted_err_sum, weight_sum, force_term), each a Tensor of shape (B,).
    """

    pred = run_model(model, normalizer, batch)
    sq_err = (pred - normalizer.normalized_target(batch)) ** 2
    # Blend of per-point (0) and per-arc-length (1) weighting; see TrainConfig.length_weight.
    weights = (batch["mask"] * ((1 - length_weight) + length_weight * batch["point_len"]))[..., None]
    weighted_err_sum = (sq_err * weights).sum((-2, -1))
    weight_sum = weights.sum((-2, -1)) * pred.shape[-1]
    if force_weight <= 0:
        return weighted_err_sum, weight_sum, torch.zeros_like(weight_sum)

    cl, cd = forces_from_prediction(normalizer.to_physical(pred, batch), batch)
    force_loss = (((cl - batch["cl"]) / normalizer.stats["cl_std"]) ** 2 + ((cd - batch["cd"]) / normalizer.stats["cd_std"]) ** 2) / 2
    return weighted_err_sum, weight_sum, force_weight * force_loss


def loss_fn(model, normalizer, batch, force_weight, length_weight=0.0):
    """Training loss for a batch: pooled pointwise loss plus the weighted force term.

    Args:
        model (DeepONet): The model.
        normalizer (Normalizer): Fitted normalizer.
        batch (dict): Batch from `SurfaceDataset.batch`.
        force_weight (float): Weight of the force term; <= 0 skips it.
        length_weight (float): Blend of per-point (0) and per-arc-length (1) weighting.

    Returns:
        tuple: (total loss, pointwise loss), scalar Tensors.
    """

    weighted_err_sum, weight_sum, force_term = loss_terms(
        model,
        normalizer,
        batch,
        length_weight,
        force_weight
    )
    pointwise_loss = weighted_err_sum.sum() / weight_sum.sum()

    return pointwise_loss + force_term.mean(), pointwise_loss


@torch.no_grad()
def val_scores(model, normalizer, val: SurfaceDataset, hparams, chunk: int = 32):
    """(pooled loss, mean per-case root loss) on `val`, both with the full force weight.

    Args:
        model (DeepONet): The model.
        normalizer (Normalizer): Fitted normalizer.
        val: Validation cases.
        hparams (dict): From `TrainConfig.hparams()`.
        chunk: Cases per forward pass.

    Returns:
        tuple: (pooled loss, mean per-case root loss), floats.
    """

    pooled, per_case = [], []
    for i in range(0, len(val), chunk):
        batch = val.batch(torch.arange(i, min(i + chunk, len(val)), device=val.branch.device))
        weighted_err_sum, weight_sum, force_term = loss_terms(
            model,
            normalizer,
            batch,
            hparams["length_weight"],
            hparams["force_weight"]
        )
        case_loss = weighted_err_sum / weight_sum + force_term
        pooled.append((weighted_err_sum.sum() / weight_sum.sum() + force_term.mean()).item())
        per_case.append(case_loss.sqrt().cpu())

    return float(np.mean(pooled)), torch.cat(per_case).mean().item()


def train_model(train: SurfaceDataset, val: SurfaceDataset, seed: int, hparams: dict, log=print):
    """Train a model

    Args:
        train: Training cases; the normalizer is fitted on these.
        val: Validation cases.
        seed: Seeds initialization and batch order.
        hparams: From `TrainConfig.hparams()`.
        log (callable): Called with a progress line every 10 epochs and at the last one.

    Returns:
        dict: Checkpoint contents
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = train.branch.device
    normalizer = Normalizer.fit(train, hparams["cf_tangential"]).to(device)
    model = build_model(hparams).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"]
    )
    batch_size = hparams["batch_size"]
    steps_per_epoch = max(1, len(train) // batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams["learning_rate"],
        pct_start=0.05,
        total_steps=hparams["epochs"] * steps_per_epoch
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_score, best_state, history = float("inf"), None, []
    start_time = time.time()
    for epoch in range(hparams["epochs"]):
        model.train()
        force_weight = force_weight_at(epoch, hparams)
        perm = torch.randperm(len(train), generator=generator)
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size:(step + 1) * batch_size].to(device)
            loss, _ = loss_fn(model, normalizer, train.batch(idx), force_weight, hparams["length_weight"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        model.eval()
        val_loss, val_case_rmse = val_scores(model, normalizer, val, hparams)
        history.append({
            "epoch": epoch,
            "train_last": loss.item(),
            "val": val_loss,
            "val_case_rmse": val_case_rmse
        })
        score = val_case_rmse if hparams["model_selection_metric"] == "case_rmse" else val_loss
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == hparams["epochs"] - 1:
            log(f"  seed {seed} epoch {epoch:4d}  train {loss.item():.4f}  val {val_loss:.4f}  "
                f"val case rmse {val_case_rmse:.4f}  best {best_score:.4f}  "
                f"({time.time() - start_time:.0f}s)")

    res = {
        "state": best_state,
        "norm": {k: v.cpu() for k, v in normalizer.stats.items()},
        "cfg": hparams,
        "seed": seed,
        "best_val": best_score,
        "history": history,
    }

    return res


def load_model(path: str, device="cpu"):
    """Checkpoint written by the train stage

    Args:
        path: Checkpoint file.
        device (str | torch.device): Device for the model and normalizer.

    Returns:
        tuple: (DeepONet, Normalizer).
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hparams = checkpoint["cfg"]
    model = build_model(hparams)
    model.load_state_dict(checkpoint["state"])

    return model.to(device).eval(), Normalizer(checkpoint["norm"], hparams["cf_tangential"]).to(device)


@torch.no_grad()
def predict(model, normalizer, batch):
    """Predictions for a batch: y (B, N, 3) in physical units, cl/cd (B,).

    Args:
        model (DeepONet): The model.
        normalizer (Normalizer): Fitted normalizer.
        batch (dict): Batch from `SurfaceDataset.batch`.

    Returns:
        tuple: (y, cl, cd); y is [Cp, Cf_x, Cf_y].
    """

    y = normalizer.to_physical(run_model(model, normalizer, batch), batch)
    cl, cd = forces_from_prediction(y, batch)

    return y, cl, cd


def predict_set(model, normalizer, dataset: SurfaceDataset, chunk: int = 32):
    """Predict every case in `dataset`: y (C, N_max, 3) padded like dataset.target, cl/cd (C,), on the CPU.

    Args:
        model (DeepONet): The model.
        normalizer (Normalizer): Fitted normalizer.
        dataset: Cases to predict.
        chunk: Cases per forward pass.

    Returns:
        tuple: (y, cl, cd) as CPU Tensors; y is [Cp, Cf_x, Cf_y].
    """

    device = dataset.branch.device
    was_training = model.training
    model.eval()
    out = {"y": [], "cl": [], "cd": []}
    for i in range(0, len(dataset), chunk):
        batch = dataset.batch(torch.arange(i, min(i + chunk, len(dataset)), device=device))
        y, cl, cd = predict(model, normalizer, batch)
        out["y"].append(y.cpu())
        out["cl"].append(cl.cpu())
        out["cd"].append(cd.cpu())
    model.train(was_training)
    max_points = dataset.target.shape[1]
    padded = [torch.nn.functional.pad(y, (0, 0, 0, max_points - y.shape[1])) for y in out["y"]]

    return torch.cat(padded), torch.cat(out["cl"]), torch.cat(out["cd"])
