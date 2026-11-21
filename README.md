# HMM Anomaly Detector for Machine Sensor Data

Gaussian-emission HMM (`hmmlearn`) trained on **normal** two-channel sensor data
(temperature, vibration). A sliding window scores the live stream; windows whose
likelihood falls below a data-driven threshold are flagged.

```bash
pip install -r requirements.txt
python run_pipeline.py     # trains, detects, evaluates, writes outputs/
pytest -q
```
Outputs: `outputs/anomaly_detection.png`, `model_selection.png`, `model_selection.csv`,
`metrics.json`, `hmm_model.joblib`.

## Data
The machine cycles idle -> normal -> high load. A simulator (`data.py`) generates 6000 training
and 2000 held-out validation samples of normal behaviour, plus a 2200-sample test stream with four
injected faults: temperature drift, vibration burst, stuck sensor, impossible temp/vibration
combination. To use real data, replace the simulator with `data.load_csv`.

## Pipeline
1. Standardise features (scaler fit on training data only).
2. **Model selection** over 1-7 hidden states (5 random EM restarts each, best likelihood kept).
3. Score each window by *average log-likelihood per observation* (`model.score / window`).
4. **Threshold** = 1st percentile of window scores on held-out *normal* validation data,
   i.e. a ~1% false-alarm budget. The test stream is never used to set it.
5. `StreamingDetector` consumes one sample at a time with a ring buffer. A test asserts its
   scores equal the batch scores exactly.
6. Window decisions become per-sample regions by majority vote (>=70% of covering windows
   flagged), which avoids marking the samples after a fault ends just because they share a window.

## Model selection: why 3 hidden states
| k | train log-lik | val log-lik/obs | BIC |
|---|---|---|---|
| 1 | -8479 | -1.37 | 17001 |
| 2 | -1581 | -0.06 | 3276 |
| **3** | 4919 | 0.90 | **-9639** |
| 4 | 4932 | 0.90 | -9560 |
| 5 | 4935 | 0.90 | -9444 |
| 6 | 4948 | 0.90 | -9331 |
| 7 | 4957 | 0.89 | -9191 |

Going 2 -> 3 states raises held-out likelihood sharply (-0.06 -> 0.90). Beyond 3 it is flat, and
train likelihood creeps up only through overfitting, so BIC's complexity penalty makes k=3 the
minimum. This matches the physical regimes (idle / normal / high load). Choosing too few states
makes normal regime switches look anomalous; too many lets the model absorb faults as "just
another state" and weakens detection.

## Results (simulated test stream)
Window=20, threshold=-5.34: **precision 0.90, recall 0.66, F1 0.76, false-positive rate 0.8%,
window AUC-ROC 0.80; 3 of 4 fault events detected.**

| Fault | Detected |
|---|---|
| Temperature drift | yes |
| Vibration burst | yes |
| Impossible combination | yes |
| Stuck sensor | **no** |

### Limitations (honest notes)
- **Stuck sensor is missed.** A frozen reading is individually plausible under a Gaussian state,
  so likelihood stays high. A simple rule (e.g. near-zero variance over N samples) should
  complement the HMM in production.
- **Window length was chosen by a sweep on this simulated test stream** (10-40; 20 gave the best
  F1/false-alarm trade-off). With real data, choose it on a separate labelled validation set.
- Shorter windows react faster but are noisier; longer windows are steadier but blur fault edges.
- Regions are computed after the fact (vote uses neighbouring windows); the live alert itself is
  causal and uses only past samples.
- Data is synthetic, so scores won't transfer to a real machine without retraining.
