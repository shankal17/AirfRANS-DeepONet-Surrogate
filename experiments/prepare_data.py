"""Download AirfRANS, extract surface data, and check that it reproduces the reference forces"""

from airfrans_surrogate.config import Config, ExtractConfig
from airfrans_surrogate.pipeline import run

cfg = Config(
    name="prepare_data",
    stages=("download", "extract", "validate"),
    extract=ExtractConfig(),            # ExtractConfig(limit=10) for a quick smoke test
)

if __name__ == "__main__":
    run(cfg)
