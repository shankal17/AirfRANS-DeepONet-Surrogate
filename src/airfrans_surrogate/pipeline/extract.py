"""Extract surface data for every AirfRANS case into small .npz files plus cases.csv"""

import json
import os
import os.path as osp
import traceback
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from airfrans_surrogate.config import Config
from airfrans_surrogate.extract import process_case


def _work(args):
    """Process one case in a worker process, catching any error.

    Args:
        args (tuple): (root, name, out_dir), as for `process_case`.

    Returns:
        tuple: (row, None) on success, or (None, {"name", "error"}) with a short traceback.
    """

    root, name, out = args
    try:
        return process_case(root, name, out), None
    except Exception:  # keep going; inspect failures.csv afterwards
        return None, {"name": name, "error": traceback.format_exc(limit=3)}


def run(cfg: Config):
    """Extract every case in the manifest in parallel and write cases.csv.

    Args:
        cfg: Experiment config
    """

    root, out = osp.join(cfg.paths.raw, "Dataset"), cfg.paths.data
    os.makedirs(out, exist_ok=True)

    with open(osp.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    
    names = list(dict.fromkeys(manifest["full_train"] + manifest["full_test"]))
    if cfg.extract.limit:
        names = names[: cfg.extract.limit]

    rows, fails = [], []
    jobs = [(root, n, out) for n in names]
    with Pool(cfg.extract.workers) as pool:
        for row, fail in tqdm(pool.imap_unordered(_work, jobs), total=len(jobs), desc="extract"):
            (rows if row else fails).append(row or fail)

    df = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
    for split, members in manifest.items():
        df[split] = df["name"].isin(set(members))
    
    df.to_csv(osp.join(out, "cases.csv"), index=False)
    print(f"Wrote {len(df)} cases to {out}/cases.csv")
    print("Split sizes:", {k: int(df[k].sum()) for k in manifest})

    if fails:
        pd.DataFrame(fails).to_csv(osp.join(out, "failures.csv"), index=False)
        print(f"WARNING: {len(fails)} cases failed; see {out}/failures.csv")
