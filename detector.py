"""Real-time (one sample at a time) detector built on a trained AnomalyModel."""
from collections import deque

import numpy as np

from . import config as C
from .model import AnomalyModel


class StreamingDetector:
    """Feed samples with `update`; once the buffer is full each call scores the
    most recent `window` samples and returns (score, is_anomaly)."""

    def __init__(self, model: AnomalyModel):
        self.model = model
        self.buffer: deque = deque(maxlen=model.window)

    def update(self, sample) -> tuple[float | None, bool]:
        self.buffer.append(np.asarray(sample, dtype=float))
        if len(self.buffer) < self.model.window:
            return None, False          # warming up
        score = self.model.score_window(np.vstack(self.buffer))
        return score, score < self.model.threshold


def vote_regions(n: int, ends: np.ndarray, scores: np.ndarray, threshold: float,
                 window: int, vote_fraction: float = C.VOTE_FRACTION) -> np.ndarray:
    """Turn window decisions into per-sample anomaly regions.

    A sample is anomalous if at least `vote_fraction` of the windows covering it fall
    below the threshold. Voting avoids the 'trailing flag' artefact (a window keeps
    containing a fault for `window` samples after it ends). Live alerts are raised per
    window by StreamingDetector; this post-hoc step is for plotting and evaluation.
    """
    flagged = scores < threshold
    covered = np.zeros(n)
    bad = np.zeros(n)
    for end, is_bad in zip(ends, flagged):
        covered[end - window:end] += 1
        bad[end - window:end] += is_bad
    return bad / np.maximum(covered, 1) >= vote_fraction
