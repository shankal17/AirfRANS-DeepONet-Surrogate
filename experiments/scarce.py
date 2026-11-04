"""Train the scarce task with the settings of full.py, then re-evaluate and re-plot every trained task"""

from dataclasses import replace

from airfrans_surrogate.config import Config
from airfrans_surrogate.pipeline import run
from full_training_and_validation import cfg as full  # experiments/ is on sys.path when this file is run directly

cfg = Config(
    name="scarce",
    stages=("train", "evaluate", "plot"),
    train=replace(full.train, tasks=("scarce",)), # every setting but tasks is shared with full.py
)

if __name__ == "__main__":
    run(cfg)
