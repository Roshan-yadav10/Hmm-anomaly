import numpy as np
import pytest

from hmm_anomaly.data import make_datasets
from hmm_anomaly.detector import StreamingDetector, vote_regions
from hmm_anomaly.model import n_parameters, train


@pytest.fixture(scope="module")
def trained():
    X, V, T, L, ev = make_datasets()
    model, k, _ = train(X, V, n_states=3)
    return model, T, L


def test_n_parameters_full_cov():
    assert n_parameters(3, 2, "full") == 2 + 6 + 6 + 9  # start+trans+means+covs


def test_streaming_matches_batch(trained):
    model, T, _ = trained
    _, batch = model.score_stream(T[:300])
    live = StreamingDetector(model)
    stream = [s for s in (live.update(x)[0] for x in T[:300]) if s is not None]
    assert np.allclose(batch, stream)


def test_detects_most_fault_samples_with_low_false_alarms(trained):
    model, T, L = trained
    ends, scores = model.score_stream(T)
    reg = vote_regions(len(T), ends, scores, model.threshold, model.window)
    assert reg[L == 1].mean() > 0.5
    assert reg[L == 0].mean() < 0.05
