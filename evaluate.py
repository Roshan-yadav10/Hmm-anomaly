"""Point-level and event-level detection metrics."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def evaluate(labels: np.ndarray, region_flags: np.ndarray, ends, scores, events) -> dict:
    p, r, f, _ = precision_recall_fscore_support(labels, region_flags.astype(int),
                                                 average="binary", zero_division=0)
    # Window-level AUC: window is positive if its last sample is in a fault
    window_labels = labels[np.asarray(ends) - 1]
    auc = roc_auc_score(window_labels, -scores)
    detected = []
    for e in events:
        hit = region_flags[e["start"]:e["end"]].any()
        detected.append({**e, "detected": bool(hit)})
    # False-alarm regions: flagged samples more than one window away from any fault
    normal_mask = labels == 0
    fpr = region_flags[normal_mask].mean()
    return {"point_precision": round(p, 3), "point_recall": round(r, 3), "point_f1": round(f, 3),
            "window_auc_roc": round(float(auc), 3), "false_positive_rate": round(float(fpr), 3),
            "events_detected": f"{sum(d['detected'] for d in detected)}/{len(detected)}",
            "events": detected}
