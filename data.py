"""Simulated machine-sensor data.

A machine cycles through three operating regimes (idle -> normal load -> high load).
Each regime emits (temperature, vibration) readings from its own Gaussian.
The test stream is the same process with injected faults of four kinds.
If you have real data, load it with `load_csv` instead.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# regime means: (temperature degC, vibration mm/s) and per-regime noise std
MEANS = np.array([[40.0, 1.0], [60.0, 3.0], [80.0, 5.5]])
STDS = np.array([[1.5, 0.15], [2.0, 0.30], [2.5, 0.45]])
TRANS = np.array([[0.97, 0.03, 0.00],
                  [0.02, 0.96, 0.02],
                  [0.00, 0.05, 0.95]])


def simulate_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    state, out = 1, np.empty((n, 2))
    for t in range(n):
        out[t] = rng.normal(MEANS[state], STDS[state])
        state = rng.choice(3, p=TRANS[state])
    return out


def simulate_test_stream(n: int, rng: np.random.Generator):
    """Return (stream, labels, events). labels[t]=1 inside an injected fault."""
    x = simulate_normal(n, rng)
    labels = np.zeros(n, dtype=int)
    events = []
    faults = [  # (start, length, kind)
        (400, 60, "temperature_drift"),
        (900, 50, "vibration_burst"),
        (1400, 60, "stuck_sensor"),
        (1800, 40, "impossible_combo"),
    ]
    for start, length, kind in faults:
        sl = slice(start, start + length)
        if kind == "temperature_drift":      # overheating with no load change
            x[sl, 0] += np.linspace(0, 25, length)
        elif kind == "vibration_burst":      # bearing wear: vibration variance explodes
            x[sl, 1] += rng.normal(0, 2.0, length)
        elif kind == "stuck_sensor":         # flat-lined readings
            x[sl] = x[start]
        elif kind == "impossible_combo":     # high temp with idle vibration
            x[sl] = rng.normal([80.0, 1.0], [2.0, 0.15], (length, 2))
        labels[sl] = 1
        events.append({"start": start, "end": start + length, "kind": kind})
    return x, labels, events


def make_datasets(seed: int = C.SEED):
    rng = np.random.default_rng(seed)
    train = simulate_normal(6000, rng)
    val = simulate_normal(2000, rng)       # held-out normal data for thresholding
    test, labels, events = simulate_test_stream(2200, rng)
    return train, val, test, labels, events


def load_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path)[C.FEATURES].to_numpy(dtype=float)
