"""The real run: train the model, evaluate it on the validation and test sets, then plot Cp and Cf."""

from airfrans_surrogate.config import Config, TrainConfig
from airfrans_surrogate.pipeline import run

cfg = Config(
    name="full",
    stages=("validate", "train", "evaluate", "plot"),
    train=TrainConfig(
        tasks=("full", "reynolds", "aoa"),
        epochs=2000,
        force_weight=0.001, # Replace with the optimal weight found by sweep_force_weight.py
        batch_size=32,
        n_fourier=16,
        weight_decay=1e-4,
        hidden_layer_width=256
    ),
)

if __name__ == "__main__":
    run(cfg)
