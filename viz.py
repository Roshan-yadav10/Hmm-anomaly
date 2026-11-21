"""Plots: sensor stream with anomaly overlay, score trace, and model selection."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402


def _shade(ax, mask, color, alpha, label):
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    for i, (s, e) in enumerate(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))):
        ax.axvspan(s, e, color=color, alpha=alpha, label=label if i == 0 else None, lw=0)


def plot_detection(X, labels, region_flags, ends, scores, threshold, path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1]})
    for ax, j, name in zip(axes[:2], range(2), C.FEATURES):
        ax.plot(X[:, j], color="tab:blue", lw=0.8, label=name)
        _shade(ax, labels.astype(bool), "green", 0.25, "true fault")
        _shade(ax, region_flags, "red", 0.25, "detected anomaly")
        ax.set_ylabel(name)
        ax.legend(loc="upper right", ncol=3)
    axes[2].plot(ends, scores, color="black", lw=0.8, label="window log-lik / obs")
    axes[2].axhline(threshold, color="red", ls="--", label=f"threshold = {threshold:.2f}")
    axes[2].set_ylabel("avg log-likelihood")
    axes[2].set_xlabel("time step")
    axes[2].legend(loc="lower right")
    fig.suptitle("HMM anomaly detection on sensor stream")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_model_selection(rows, best_k, path):
    ks = [r["n_states"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, [r["bic"] for r in rows], "o-")
    ax[0].axvline(best_k, color="red", ls="--", label=f"chosen k={best_k}")
    ax[0].set(xlabel="hidden states", ylabel="BIC (lower is better)", title="BIC")
    ax[0].legend()
    ax[1].plot(ks, [r["val_loglik_per_obs"] for r in rows], "o-", color="tab:green")
    ax[1].set(xlabel="hidden states", ylabel="val log-lik / obs", title="Held-out likelihood")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
