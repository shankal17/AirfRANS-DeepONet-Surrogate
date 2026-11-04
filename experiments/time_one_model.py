"""Time one 50-epoch model, then set `epochs` in full.py so training fits the budget
"""
from airfrans_surrogate.config import Config, Paths, TrainConfig
from airfrans_surrogate.pipeline import run

cfg = Config(
    name="time_one_model",
    stages=("train",),
    paths=Paths(runs="runs_timing"),
    train=TrainConfig(tasks=("full",), epochs=50),
)

if __name__ == "__main__":
    run(cfg)
