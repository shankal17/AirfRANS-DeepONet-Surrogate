"""Surface geometry utilities for AirfRANS airfoils

Conventions
-----------
- Chord runs from x = 0 (leading edge) to x = 1 (trailing edge); the airfoil itself is
  not rotated in AirfRANS meshes (angle of attack enters through the inlet velocity).
- ``lines`` is an (M, 2) integer array of segment endpoint indices into the point array,
  exactly as stored in the aerofoil .vtp patch.
"""

from __future__ import annotations

import numpy as np
from airfrans.naca_generator import naca_generator

def order_loop(lines: np.ndarray, n_points: int) -> np.ndarray:
    """Return point indices ordered by walking along the surface polyline

    Args:
        lines: Segment endpoint indices, shape (M, 2)
        n_points: Number of points the indices refer to.

    Returns:
        Indices of the connected points in walk order, starting at an open end if there is one.

    Raises:
        ValueError: If a node has more than two neighbours or the segments do not form a single connected loop or chain.
        RuntimeError: If the walk does not terminate.
    """

    neighbors: list[list[int]] = [[] for _ in range(n_points)]
    for a, b in np.asarray(lines, dtype=int):
        if a == b:
            continue
        neighbors[a].append(b)
        neighbors[b].append(a)

    degree = np.array([len(adjacent) for adjacent in neighbors])
    connected = np.flatnonzero(degree > 0)
    if (degree[connected] > 2).any():
        raise ValueError("Surface polyline has branching nodes (degree > 2).")
    
    ends = connected[degree[connected] == 1]
    start = int(ends[0]) if len(ends) else int(connected[0])

    order = [start]
    previous, current = -1, start
    while True:
        candidates = [n for n in neighbors[current] if n != previous]
        if not candidates:
            break  # reached the other end of an open chain

        neighbor = candidates[0]
        if neighbor == start:
            break  # closed the loop

        order.append(neighbor)
        previous, current = current, neighbor
        if len(order) > len(connected):
            raise RuntimeError("Loop walk did not terminate; polyline is malformed.")
        
    if len(order) != len(connected):
        raise ValueError(f"Polyline is not a single connected loop ({len(order)} of {len(connected)} nodes reached).")
    
    return np.asarray(order)


def split_upper_lower(pos: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an ordered surface loop into upper and lower arcs (point indices, TE -> LE).

    The loop is cut at the trailing edge (max x) and leading edge (min x); the arc with the
    larger mean y is called the upper surface. Both arcs include the LE point.

    Args:
        pos: Point positions, shape (N, 2).
        order: Point indices in loop order, from `order_loop`.

    Returns:
        (upper, lower) point-index arrays, each ordered from TE to LE.
    """

    tailing_edge_index = int(np.argmax(pos[order, 0]))
    order = np.roll(order, -tailing_edge_index)  # start at the trailing edge
    leading_edge_index = int(np.argmin(pos[order, 0]))
    arc_a = order[: leading_edge_index + 1]
    arc_b = np.concatenate([order[leading_edge_index:], order[:1]])[::-1]  # also TE -> LE

    # Return swapped arcs if we assumed upper and lower arcs incorrectly
    if pos[arc_a, 1].mean() >= pos[arc_b, 1].mean():
        return arc_a, arc_b
    
    return arc_b, arc_a


def chord_stations(n_stations: int) -> np.ndarray:
    """n_stations cosine-spaced interior chord stations (clustered near LE and TE).

    Args:
        n_stations: Number of stations.

    Returns:
        x/c values in (0, 1), shape (n_stations,), increasing (c is chord length).
    """

    station_idxs = np.arange(1, n_stations + 1)

    return 0.5 * (1.0 - np.cos(np.pi * station_idxs / (n_stations + 1)))


def shape_descriptor(pos: np.ndarray, upper: np.ndarray, lower: np.ndarray, n_stations: int = 16) -> np.ndarray:
    """Fixed-length shape vector of the airfoil. Station points interpolated

    Args:
        pos: Point positions, shape (N, 2).
        upper: Upper-surface point indices, from `split_upper_lower`.
        lower: Lower-surface point indices, from `split_upper_lower`.
        n_stations: Stations per surface.

    Returns:
        Upper y values then lower y values at `chord_stations(n_stations)`, shape (2 * n_stations,).
    """

    stations = chord_stations(n_stations)
    out = []
    for arc in (upper, lower):
        arc_x, arc_y = pos[arc, 0], pos[arc, 1]
        by_x = np.argsort(arc_x)
        out.append(np.interp(stations, arc_x[by_x], arc_y[by_x]))
    
    return np.concatenate(out)


def resample_by_side(values: np.ndarray, pos: np.ndarray, side: np.ndarray, stations: np.ndarray) -> np.ndarray:
    """Per-point values interpolated to chord stations on each surface, so different meshes compare.

    Args:
        values: One value per point, shape (N,).
        pos: Point positions, shape (N, 2).
        side: Side flag from `side_and_chord_coordinate` (+1 upper, -1 lower), shape (N,).
        stations: x/c values to interpolate at, shape (S,).

    Returns:
        Upper values then lower values at `stations`, shape (2 * S,).
    """

    out = []
    for on_side in (side > 0, side < 0):
        arc_x, arc_values = pos[on_side, 0], values[on_side]
        by_x = np.argsort(arc_x)
        out.append(np.interp(stations, arc_x[by_x], arc_values[by_x]))

    return np.concatenate(out)


def naca_case(naca, nb_samples: int = 400) -> dict:
    """Surface geometry of a NACA airfoil from the AirfRANS generator, in the layout of an extracted case.

    docs/demo/geometry.js ports this function; keep the two in step.

    Args:
        naca (array_like): NACA parameters, 3 (4-digit) or 4 (5-digit) as in AirfRANS case names.
        nb_samples: Generator samples per surface.

    Returns:
        dict: pos, normals (inward, per point), side, s, lines, cell_normals (inward), cell_lengths
        and descriptor, as `extract.process_case` stores them.
    """

    pos = naca_generator(np.asarray(naca, float), nb_samples=nb_samples, verbose=False)[:-1]  # last point repeats the first
    n_points = len(pos)
    lines = np.stack([np.arange(n_points), (np.arange(n_points) + 1) % n_points], 1)
    segment = pos[lines[:, 1]] - pos[lines[:, 0]]
    cell_lengths = np.linalg.norm(segment, axis=1)
    cell_normals = np.stack([-segment[:, 1], segment[:, 0]], 1) / cell_lengths[:, None]  # loop runs anticlockwise, so this is inward
    normals = np.zeros_like(pos)
    np.add.at(normals, lines[:, 0], cell_normals)
    np.add.at(normals, lines[:, 1], cell_normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    upper, lower = split_upper_lower(pos, order_loop(lines, n_points))
    side, s = side_and_chord_coordinate(pos, upper, lower)

    return {
        "pos": pos,
        "normals": normals,
        "side": side,
        "s": s,
        "lines": lines,
        "cell_normals": cell_normals,
        "cell_lengths": cell_lengths,
        "descriptor": shape_descriptor(pos, upper, lower)
    }


def side_and_chord_coordinate(pos: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-point side flag (+1 upper, -1 lower) and signed chord coordinate s in [-1, 1].

    s = +x on the upper surface and -x on the lower surface, so s is continuous through the leading edge (s = 0)

    Args:
        pos: Point positions, shape (N, 2).
        upper: Upper-surface point indices.
        lower: Lower-surface point indices. Points in both arcs (LE, TE) are labelled upper.

    Returns:
        (side, s), each of shape (N,).
    """

    side = np.zeros(pos.shape[0])
    side[lower] = -1.0
    side[upper] = 1.0  # LE / TE points shared by both arcs are labelled upper
    s = side * pos[:, 0]

    return side, s


def tangent(normals: np.ndarray) -> np.ndarray:
    """Unit tangent (-ny, nx) from inward normals; points towards increasing s.

    Args:
        normals: Inward point normals, shape (N, 2); need not be unit length.

    Returns:
        Unit tangents, shape (N, 2).
    """

    unit_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    return np.stack([-unit_normals[:, 1], unit_normals[:, 0]], 1)


def streamwise_cf(cf: np.ndarray, normals: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Wall-shear vector -> signed Cf, positive for flow from the leading edge towards the
    trailing edge on both surfaces; negative is reversed flow.

    Args:
        cf: Skin-friction vectors, shape (N, 2).
        normals: Inward point normals, shape (N, 2).
        side: Side flag from `side_and_chord_coordinate` (+1 upper, -1 lower), shape (N,).

    Returns:
        Streamwise Cf, shape (N,).
    """

    # The stored side flag, not the sign of s
    sign = np.where(side >= 0, 1.0, -1.0)

    return sign * (cf * tangent(normals)).sum(1)


def arc_from_le(pos: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Signed arc length from the leading edge (min x) along the surface loop: + upper, - lower.

    Args:
        pos: Point positions, shape (N, 2).
        order: Point indices in loop order, from `order_loop`.

    Returns:
        Signed arc length per point, shape (N,), indexed like `pos`.
    """

    loop = np.roll(order, -int(np.argmax(pos[order, 0])))  # walk the loop from the trailing edge
    arc_length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pos[loop], axis=0), axis=1))]
    arc_length -= arc_length[np.argmin(pos[loop, 0])]
    if pos[loop[arc_length > 0], 1].mean() < pos[loop[arc_length < 0], 1].mean():
        arc_length = -arc_length
    arc = np.empty_like(arc_length)
    arc[loop] = arc_length
    
    return arc
