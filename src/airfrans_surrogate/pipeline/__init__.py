"""Pipeline stages, each a module with `run(cfg: Config)`, and the runner that chains them.

Stage modules are imported only when their stage runs, so a training experiment does not need
airfrans / pyvista installed and a data-preparation experiment does not need torch.
"""
import importlib
import json
import os
import os.path as osp
import time

from airfrans_surrogate.config import STAGES, Config


def run(cfg: Config):
    """Run `cfg.stages` in pipeline order, after recording the full config next to the runs.

    Args:
        cfg: Experiment config; written to <paths.runs>/experiment_<name>.json.
    """
    os.makedirs(cfg.paths.runs, exist_ok=True)
    path = osp.join(cfg.paths.runs, f"experiment_{cfg.name}.json")
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=1)
    stages = [stage for stage in STAGES if stage in cfg.stages]
    print(f"experiment '{cfg.name}': {' -> '.join(stages)}  (config in {path})")
    for stage in stages:
        print(f"\n=== {stage} " + "=" * (60 - len(stage)), flush=True)
        start_time = time.time()
        importlib.import_module(f"{__name__}.{stage}").run(cfg)
        print(f"--- {stage} done in {time.time() - start_time:.0f}s", flush=True)
