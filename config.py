"""Central configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

SEED = 42
FEATURES = ["temperature", "vibration"]

# Model selection
CANDIDATE_STATES = range(1, 8)
N_RESTARTS = 5            # random restarts per candidate (EM finds local optima)
COVARIANCE_TYPE = "full"
N_ITER = 200

# Sliding window / thresholding
WINDOW = 20               # samples per window
STRIDE = 1                # 1 = score every new sample in the stream
THRESHOLD_PERCENTILE = 1.0  # ~1% false-alarm budget on held-out normal windows
VOTE_FRACTION = 0.7      # a sample is 'anomalous' if >=70% of windows covering it are
