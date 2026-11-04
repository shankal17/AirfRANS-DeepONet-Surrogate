"""Validate the extracted data"""

from airfrans_surrogate.config import Config
from airfrans_surrogate.pipeline import run

cfg = Config(
    name="validate",
    stages=("validate"),
)

if __name__ == "__main__":
    run(cfg)
