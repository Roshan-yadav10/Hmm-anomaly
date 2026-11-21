"""HMM training, model selection and window scoring."""
from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from . import config as C


def n_parameters(k: int, d: int, cov_type: str = C.COVARIANCE_TYPE) -> int:
    """Free parameters of a GaussianHMM (used by BIC)."""
    cov = {"full": k * d * (d + 1) // 2, "diag": k * d, "spherical": k, "tied": d * (d + 1) // 2}[cov_type]
    return (k - 1) + k * (k - 1) + k * d + cov  # start + transitions + means + covariances


def fit_hmm(X: np.ndarray, k: int, seed: int = C.SEED, n_restarts: int = C.N_RESTARTS) -> GaussianHMM:
    """Fit with several random restarts and keep the best log-likelihood."""
    best, best_ll = None, -np.inf
    for r in range(n_restarts):
        m = GaussianHMM(n_components=k, covariance_type=C.COVARIANCE_TYPE,
                        n_iter=C.N_ITER, random_state=seed + r, min_covar=1e-3)
        try:
            m.fit(X)
            ll = m.score(X)
        except ValueError:      # degenerate fit (e.g. singular covariance)
            continue
        if ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError(f"HMM fit failed for k={k}")
    return best


def select_n_states(X_train: np.ndarray, X_val: np.ndarray):
    """Score candidate state counts with BIC (train) and held-out log-likelihood (val)."""
    rows, models = [], {}
    n, d = X_train.shape
    for k in C.CANDIDATE_STATES:
        m = fit_hmm(X_train, k)
        ll = m.score(X_train)
        rows.append({
            "n_states": k,
            "train_loglik": ll,
            "val_loglik_per_obs": m.score(X_val) / len(X_val),
            "bic": -2 * ll + n_parameters(k, d) * np.log(n),
        })
        models[k] = m
    best_k = min(rows, key=lambda r: r["bic"])["n_states"]
    return best_k, rows, models


def sliding_windows(X: np.ndarray, window: int = C.WINDOW, stride: int = C.STRIDE):
    for end in range(window, len(X) + 1, stride):
        yield end - window, end, X[end - window:end]


@dataclass
class AnomalyModel:
    """Scaler + HMM + threshold bundled together."""
    scaler: StandardScaler
    hmm: GaussianHMM
    window: int
    threshold: float = float("nan")

    def score_window(self, window_raw: np.ndarray) -> float:
        """Average log-likelihood per observation (comparable across window sizes)."""
        return self.hmm.score(self.scaler.transform(window_raw)) / len(window_raw)

    def score_stream(self, X: np.ndarray, stride: int = C.STRIDE):
        ends, scores = [], []
        for _, end, w in sliding_windows(X, self.window, stride):
            ends.append(end)
            scores.append(self.score_window(w))
        return np.array(ends), np.array(scores)

    def fit_threshold(self, X_normal: np.ndarray, percentile: float = C.THRESHOLD_PERCENTILE):
        """Threshold = low percentile of window scores on held-out NORMAL data."""
        _, scores = self.score_stream(X_normal)
        self.threshold = float(np.percentile(scores, percentile))
        return self.threshold


def train(X_train, X_val, window: int = C.WINDOW, n_states: int | None = None):
    scaler = StandardScaler().fit(X_train)
    Xs, Vs = scaler.transform(X_train), scaler.transform(X_val)
    rows = None
    if n_states is None:
        n_states, rows, models = select_n_states(Xs, Vs)
        hmm = models[n_states]
    else:
        hmm = fit_hmm(Xs, n_states)
    model = AnomalyModel(scaler, hmm, window)
    model.fit_threshold(X_val)
    return model, n_states, rows
