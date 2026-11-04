"""Integrate surface Cp and skin friction into lift and drag coefficients.

This mirrors ``airfrans.Simulation.force_coefficient`` exactly
"""

from __future__ import annotations

import numpy as np
import torch

def _cos_sin(angle):
    """Cosine and sine with torch for tensors and NumPy otherwise.

    Args:
        angle (float | ndarray | Tensor): Angle in radians.

    Returns:
        tuple: (cos(angle), sin(angle)), same type as `angle`.
    """

    if torch is not None and isinstance(angle, torch.Tensor):
        return torch.cos(angle), torch.sin(angle)
    
    return np.cos(angle), np.sin(angle)


def integrate_coefficients(cp, cf, lines, cell_normals, cell_lengths, aoa_rad):
    """Return dict with cl, cd and their pressure / viscous parts.

    Args:
        cp: (N,) pressure coefficient at surface points.
        cf: (N, 2) wall-shear-stress coefficient vector at surface points.
        lines: (M, 2) segment endpoint indices.
        cell_normals: (M, 2) inward unit normals per segment.
        cell_lengths: (M,) segment lengths.
        aoa_rad: angle of attack in radians.

    Returns:
        dict with "cl", "cd", "cl_pressure", "cl_viscous", "cd_pressure" and "cd_viscous",
        each a scalar of the input array type.
    """

    start_idx, end_idx = lines[:, 0], lines[:, 1]
    cp_cell = 0.5 * (cp[start_idx] + cp[end_idx])
    cf_cell = 0.5 * (cf[start_idx] + cf[end_idx])

    pressure_force = (cp_cell[:, None] * cell_normals * cell_lengths[:, None]).sum(0)
    viscous_force = (cf_cell * cell_lengths[:, None]).sum(0)

    cos_a, sin_a = _cos_sin(aoa_rad)
    cd_pressure = cos_a * pressure_force[0] + sin_a * pressure_force[1]
    cl_pressure = -sin_a * pressure_force[0] + cos_a * pressure_force[1]
    cd_viscous = cos_a * viscous_force[0] + sin_a * viscous_force[1]
    cl_viscous = -sin_a * viscous_force[0] + cos_a * viscous_force[1]

    return {
        "cd": cd_pressure + cd_viscous,
        "cd_pressure": cd_pressure,
        "cd_viscous": cd_viscous,
        "cl": cl_pressure + cl_viscous,
        "cl_pressure": cl_pressure,
        "cl_viscous": cl_viscous,
    }


def integrate_coefficients_batch(cp, cf, lines, cell_normals, cell_lengths, aoa_rad):
    """Batched, padded PyTorch version of :func:`integrate_coefficients`.

    Args:
        cp: (B, N) pressure coefficient at surface points.
        cf: (B, N, 2) wall-shear-stress coefficient vector at surface points.
        lines: (B, M, 2) segment endpoint indices.
        cell_normals: (B, M, 2) inward unit normals per segment.
        cell_lengths: (B, M) segment lengths, 0 on padding.
        aoa_rad: (B,) angle of attack in radians.

    Returns:
        tuple: (cl, cd), each a Tensor of shape (B,).
    """

    start_idx, end_idx = lines[..., 0], lines[..., 1]
    cp_cell = 0.5 * (cp.gather(1, start_idx) + cp.gather(1, end_idx))
    cf_cell = 0.5 * (cf.gather(1, start_idx[..., None].expand(-1, -1, 2)) + cf.gather(1, end_idx[..., None].expand(-1, -1, 2)))
    lengths = cell_lengths[..., None]
    force = (cp_cell[..., None] * cell_normals * lengths).sum(1) + (cf_cell * lengths).sum(1)  # (B, 2)
    cos_a, sin_a = torch.cos(aoa_rad), torch.sin(aoa_rad)
    cd = cos_a * force[:, 0] + sin_a * force[:, 1]
    cl = -sin_a * force[:, 0] + cos_a * force[:, 1]
    
    return cl, cd


def moment_coefficient(cp, cf, pos, lines, cell_normals, cell_lengths, x_ref=0.25):
    """Pitching-moment coefficient about (x_ref, 0), positive nose-up, from the same cell forces as
    :func:`integrate_coefficients`.

    Args:
        cp: (..., N) pressure coefficient at surface points; leading dimensions share one geometry.
        cf: (..., N, 2) wall-shear-stress coefficient vector at surface points.
        pos: (N, 2) surface points, chord 1 along x.
        lines: (M, 2) segment endpoint indices.
        cell_normals: (M, 2) inward unit normals per segment.
        cell_lengths: (M,) segment lengths.
        x_ref: Chord position of the moment reference.

    Returns:
        Cm, shape (...), of the input array type.
    """

    start_idx, end_idx = lines[:, 0], lines[:, 1]
    cp_cell = 0.5 * (cp[..., start_idx] + cp[..., end_idx])
    cf_cell = 0.5 * (cf[..., start_idx, :] + cf[..., end_idx, :])
    force = (cp_cell[..., None] * cell_normals + cf_cell) * cell_lengths[:, None]  # (..., M, 2)
    arm_x = 0.5 * (pos[start_idx, 0] + pos[end_idx, 0]) - x_ref
    arm_y = 0.5 * (pos[start_idx, 1] + pos[end_idx, 1])

    return -(arm_x * force[..., 1] - arm_y * force[..., 0]).sum(-1)  # x aft, y up: nose-up is clockwise
