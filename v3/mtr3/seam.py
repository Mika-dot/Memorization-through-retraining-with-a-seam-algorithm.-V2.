from __future__ import annotations

import numpy as np


def seam_importance(rgb: np.ndarray, floor: float = 0.20) -> np.ndarray:
    """Return a seam-inspired importance map in [floor, 1].

    V2 used low-energy pixels as the training set. V3 flips that into rate allocation:
    smooth regions remain cheap, while edges/high-frequency structures receive more samples.
    """
    x = rgb.astype(np.float32) / 255.0
    y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]

    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:-1] = (y[:, 2:] - y[:, :-2]) * 0.5
    gy[1:-1, :] = (y[2:, :] - y[:-2, :]) * 0.5
    grad = np.sqrt(gx * gx + gy * gy)

    # Second-order detail catches thin lines that a single gradient can underweight.
    lap = np.zeros_like(y)
    lap[1:-1, 1:-1] = np.abs(
        4.0 * y[1:-1, 1:-1]
        - y[:-2, 1:-1]
        - y[2:, 1:-1]
        - y[1:-1, :-2]
        - y[1:-1, 2:]
    )
    e = grad + 0.35 * lap
    p99 = float(np.percentile(e, 99.0))
    if p99 <= 1e-8:
        return np.ones_like(y, dtype=np.float32)
    e = np.clip(e / p99, 0.0, 1.0)
    return (floor + (1.0 - floor) * e).astype(np.float32)
