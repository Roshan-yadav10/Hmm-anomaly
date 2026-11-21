"""End-to-end run: simulate -> select k -> train -> threshold -> detect -> evaluate -> plot.

Usage:  python run_pipeline.py
"""
import json
import logging

import joblib
import numpy as np
import pandas as pd

from hmm_anomaly import config as C
from hmm_anomaly.data import make_datasets
from hmm_anomaly.detector import StreamingDetector, vote_regions
from hmm_anomaly.evaluate import evaluate
from hmm_anomaly.model import train
from hmm_anomaly.viz import plot_detection, plot_model_selection

log = logging.getLogger("run")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # silence EM 'not converging' noise


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    C.OUT_DIR.mkdir(exist_ok=True)
    X_train, X_val, X_test, labels, events = make_datasets()

    model, k, rows = train(X_train, X_val)
    table = pd.DataFrame(rows).round(2)
    log.info("Model selection:\n%s", table.to_string(index=False))
    log.info("Chosen hidden states: %d | threshold: %.3f", k, model.threshold)
    table.to_csv(C.OUT_DIR / "model_selection.csv", index=False)
    plot_model_selection(rows, k, C.OUT_DIR / "model_selection.png")

    # Batch scoring of the test stream
    ends, scores = model.score_stream(X_test)

    # Streaming check: the live detector must reproduce the batch scores exactly.
    live = StreamingDetector(model)
    live_scores = [s for s in (live.update(x)[0] for x in X_test) if s is not None]
    assert np.allclose(live_scores, scores), "streaming and batch scores differ"

    regions = vote_regions(len(X_test), ends, scores, model.threshold, model.window)
    metrics = evaluate(labels, regions, ends, scores, events)
    log.info("Metrics: %s", {m: v for m, v in metrics.items() if m != "events"})
    for e in metrics["events"]:
        log.info("  %-18s [%4d-%4d) detected=%s", e["kind"], e["start"], e["end"], e["detected"])

    plot_detection(X_test, labels, regions, ends, scores, model.threshold,
                   C.OUT_DIR / "anomaly_detection.png")
    (C.OUT_DIR / "metrics.json").write_text(json.dumps(
        {"n_states": k, "threshold": model.threshold, "window": model.window, **metrics}, indent=2))
    joblib.dump(model, C.OUT_DIR / "hmm_model.joblib")


if __name__ == "__main__":
    main()
