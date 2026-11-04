"""Time inference of the model full.py trained, on the GPU (if there is one) and the CPU"""

import os.path as osp
import time

import numpy as np
import torch

from airfrans_surrogate.trainer import load_model, predict, run_model
from animate_sweep import CASE, TASK, frame_batch, full, predict_frames  # experiments/ is on sys.path when run directly

BATCH_SIZES = (1, 32, 200)
WARMUP, REPEATS = 10, 50

def median_seconds(fn, device):
    """Median wall time of fn() over REPEATS calls, after WARMUP untimed calls.

    Args:
        fn (callable): The call to time.
        device (str): Device fn runs on; CUDA is synchronised around each call.

    Returns:
        float: Median seconds per call.
    """

    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(REPEATS):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - start)
    return float(np.median(times))

def main():
    case = dict(np.load(osp.join(full.paths.data, CASE + ".npz")))
    rng = np.random.default_rng(0)
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    print(f"{CASE}: {len(case['s'])} surface points; torch {torch.__version__}, "
          f"{torch.get_num_threads()} CPU threads")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    print(f"\n{'device':<6} {'batch':>5}  {'level':<11} {'ms / call':>10} {'ms / case':>10} {'cases / s':>10}")
    for device in devices:
        model, normalizer = load_model(osp.join(full.paths.runs, TASK, "model.pt"), device)
        for batch_size in BATCH_SIZES:
            re, aoa = rng.uniform(2e6, 6e6, batch_size), rng.uniform(-5, 15, batch_size)
            batch = frame_batch(case, re, aoa, device)
            levels = {
                "model": lambda: run_model(model, normalizer, batch),
                "+ forces": lambda: predict(model, normalizer, batch),
                "end-to-end": lambda: predict_frames(model, normalizer, case, re, aoa, device, chunk=batch_size),
            }
            with torch.no_grad():
                for level, fn in levels.items():
                    seconds = median_seconds(fn, device)
                    print(f"{device:<6} {batch_size:>5}  {level:<11} {1e3 * seconds:>10.3f} "
                          f"{1e3 * seconds / batch_size:>10.4f} {batch_size / seconds:>10.0f}")
        print()

if __name__ == "__main__":
    main()
