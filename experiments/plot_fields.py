"""Re-plot the surface fields of a trained model"""

from airfrans_surrogate.config import Config, PlotConfig
from airfrans_surrogate.pipeline import run
from full_training_and_validation import cfg as full  # experiments/ is on sys.path when this file is run directly

cfg = Config(
    name="plot_fields",
    stages=("plot",),
    paths=full.paths,
    plot=PlotConfig(),
)

if __name__ == "__main__":
    run(cfg)
