"""Pick the force weight: one model per weight on real data"""

from airfrans_surrogate.config import Config, SweepConfig
from airfrans_surrogate.pipeline import run
from full_training_and_validation import cfg as full  # experiments/ is on sys.path when this file is run directly

cfg = Config(
    name="sweep_force_weight",
    stages=("sweep",),
    train=full.train, # every setting but force_weight is shared by the sweep
    sweep=SweepConfig(param="force_weight", values=(0.0, 1e-4, 1e-3, 3e-3, 1e-2)),
)

if __name__ == "__main__":
    run(cfg)
