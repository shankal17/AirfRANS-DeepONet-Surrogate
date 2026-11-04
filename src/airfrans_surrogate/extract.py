"""Extract per-case surface data from an AirfRANS simulation."""

from __future__ import annotations

import os.path as osp
import numpy as np
import airfrans as af

from airfrans_surrogate.geometry import order_loop, shape_descriptor, side_and_chord_coordinate, split_upper_lower

# AirfRANS was generated with nu = 1.56e-5 m^2/s and a 1 m chord (see library docs), so Re = U_inf / 1.56e-5
NU_GENERATION = 1.56e-5

def parse_name(name: str) -> dict:
    """'airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252' -> conditions and NACA params.

    Args:
        name: AirfRANS case name: airFoil2D_SST_<U_inf>_<AoA>_<NACA params...>.

    Returns:
        dict with "name", "u_inf", "re", "aoa_deg" (degrees), "series" (4 or 5), "naca" (list of NACA parameters) and "thickness" (fraction of chord).
    """

    parts = name.split("_")
    u_inf = float(parts[2])
    aoa_deg = float(parts[3])
    naca = [float(part) for part in parts[4:]]

    return {
        "name": name,
        "u_inf": u_inf,
        "re": u_inf / NU_GENERATION,
        "aoa_deg": aoa_deg,
        "series": 4 if len(naca) == 3 else 5,
        "naca": naca,
        "thickness": naca[-1] / 100.0, # Percent to fraction of cord
    }


def process_case(root: str, name: str, out_dir: str, n_stations: int = 16) -> dict:
    """Process one simulation, write ``<out_dir>/<name>.npz`` and return a summary row.

    Args:
        root: AirfRANS Dataset directory.
        name: Case name.
        out_dir: Directory to write the .npz into.
        n_stations: Chord stations per surface in the shape descriptor.

    Returns:
        One cases.csv row
    """

    sim = af.Simulation(root=root, name=name)
    meta = parse_name(name)
    dynamic_pressure = 0.5 * float(sim.inlet_velocity) ** 2 # AirfRANS already divides density out

    pos = np.asarray(sim.airfoil_position, dtype=np.float64)
    point_normals = np.asarray(sim.airfoil_normals, dtype=np.float64)
    surface_pos = sim.position[sim.surface]
    pressure = af.reorganize(surface_pos, pos, sim.pressure[sim.surface])[:, 0]
    wall_shear = sim.wallshearstress(over_airfoil=True, reference=True)
    cp = pressure / dynamic_pressure
    cf = wall_shear / dynamic_pressure

    # Segment geometry exactly as force() sees it after its point-to-cell conversion
    lines = np.asarray(sim.airfoil.lines).reshape(-1, 3)[:, 1:]
    cells = sim.airfoil.copy().ptc(pass_point_data=False)
    cell_normals = np.asarray(cells.cell_data["Normals"][:, :2], dtype=np.float64)
    cell_lengths = np.asarray(cells.cell_data["Length"], dtype=np.float64)

    order = order_loop(lines, pos.shape[0])
    upper, lower = split_upper_lower(pos, order)
    side, s = side_and_chord_coordinate(pos, upper, lower)
    descriptor = shape_descriptor(pos, upper, lower, n_stations=n_stations)

    (cd, cd_pressure, cd_viscous), (cl, cl_pressure, cl_viscous) = sim.force_coefficient(reference=True)

    np.savez_compressed(
        osp.join(out_dir, name + ".npz"),
        pos=pos,
        normals=point_normals,
        side=side,
        s=s,
        order=order,
        lines=lines,
        cell_normals=cell_normals,
        cell_lengths=cell_lengths,
        cp=cp,
        cf=cf,
        descriptor=descriptor,
        u_inf=meta["u_inf"],
        re=meta["re"],
        aoa_deg=meta["aoa_deg"],
        naca=np.asarray(meta["naca"]),
        series=meta["series"],
        ref_cl=cl,
        ref_cd=cd,
        ref_clp=cl_pressure,
        ref_clv=cl_viscous,
        ref_cdp=cd_pressure,
        ref_cdv=cd_viscous,
    )

    row = {k: v for k, v in meta.items() if k != "naca"}
    row.update(
        n_surface=int(pos.shape[0]),
        naca="_".join(f"{v:g}" for v in meta["naca"]),
        cl=float(cl), cd=float(cd), clp=float(cl_pressure), clv=float(cl_viscous),
        cdp=float(cd_pressure), cdv=float(cd_viscous),
    )
    
    return row
