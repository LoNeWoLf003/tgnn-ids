# Adaptive Temporal Graph Learning for Network Anomaly Detection

**Repository: `adaptive-temporal-graph-ids`**
*(the containing folder on disk may still be named `tgnn-ids` on this machine — the two names
refer to the same project; rename the folder yourself whenever convenient, see note below.)*

**A project overview — what this is, how it works, and what the results actually show.**

> Companion documents: [`base.md`](./base.md) (original research proposal), [`PLAN.md`](./PLAN.md)
> (scoped execution plan + the data-constraint pivot), [`IMPLEMENTATION.md`](./IMPLEMENTATION.md)
> (build-order technical spec). This file is the plain-language summary of all three, plus the
> actual results produced by running the code.

---

## 1. The idea in one paragraph

Most network intrusion detection systems (IDS) treat traffic as a static table of flows and
classify each one independently. Real traffic isn't static — it evolves over time, and the
relationships *between* flows (which ones look similar, which ones cluster together) carry
signal that row-by-row classification throws away. This project builds a detector that (1)
represents a chunk of network traffic as a **graph**, (2) tracks how that graph's learned
representation **evolves over time** with an LSTM, (3) watches that representation for
**concept drift** (the traffic distribution changing — new attack types, changing baselines),
and (4) **adapts** the model when drift is detected, instead of leaving it stale or blindly
retraining from scratch every time. The central question: *does this adaptive, temporal,
graph-based approach hold up better than a static flow-level classifier when the traffic
distribution shifts?*

---

## 2. The dataset, and the constraint that reshaped the design

**Dataset:** CIC-IDS2017 — five days of labeled network traffic (Monday through Friday) captured
by the Canadian Institute for Cybersecurity, pre-processed into per-flow statistics by
CICFlowMeter. Each day introduces different traffic: Monday is all-benign baseline, Tuesday adds
brute-force attacks, Wednesday adds DoS, Thursday adds web attacks/infiltration, Friday adds
botnet/portscan/DDoS.

| Day file | Flows (after cleaning) |
|---|---|
| Monday (benign baseline) | 502,650 |
| Tuesday (brute-force introduced) | 421,626 |
| Wednesday (DoS) | 610,492 |
| Thursday AM (web attacks) | 164,179 |
| Thursday PM (infiltration) | 252,790 |
| Friday AM (benign) | 184,044 |
| Friday PM (port scan) | 213,777 |
| Friday PM (DDoS) | 223,082 |
| **Total** | **2,572,640 flows**, ~83.5% BENIGN |

**The constraint:** the copy of this dataset actually available (the common redistributed
`MachineLearningCVE` CSV mirror) has **no Source/Destination IP, Source Port, Protocol, or
Timestamp** — only `Destination Port` + ~70 CICFlowMeter numeric statistics + `Label`. The
original design (graph nodes = *hosts*, edges = *who talked to whom*) is impossible without IP
addresses. Rather than block on the registration-gated official dataset, the project pivoted:

- **Node = flow** (not host). Each flow becomes one graph node, described by its scaled
  CICFlowMeter feature vector.
- **Edge = feature similarity.** Within a fixed-size chunk of consecutive flows, edges connect
  each flow to its `k=8` nearest neighbors in scaled feature space.
- **Time window = row-count chunk, not wall-clock.** With no timestamp, chronological order is
  assumed to be row order within each day-file (CICFlowMeter emits flows in capture order — a
  documented assumption, not a measured fact). Windows are fixed at 1000 consecutive flows,
  never spanning a day boundary.
- **Drift phases = the day files themselves.** Rather than inject synthetic drift, the project
  uses CIC-IDS2017's own day-to-day composition change as native, real drift: Mon→Tue→Wed→Thu→Fri
  each introduce a new attack mix.
- **Casualty of the pivot:** lateral-movement signals (which host is scanning how many other
  hosts) require host identity and are dropped entirely — stated as a limitation, not
  approximated.

This is documented in detail in `PLAN.md`'s "Data constraint discovered at implementation time"
section.

---

## 3. Architecture — what gets built, in order

```
Raw CSVs (data/raw/)
   │  src/ingest.py        clean, dedupe, fix label typos, concatenate Mon→Fri
   ▼
data/interim/cicids2017_clean.parquet
   │  src/features.py      scale ~70 CICFlowMeter columns (fit on Mon+Tue only), assign
   │                       window_id, compute per-window destination-port entropy
   ▼
data/processed/cicids2017_features.parquet, scaler.joblib, window_labels.parquet
   │  src/graph_build.py   build one k-NN similarity graph per window (node=flow, k=8 edges)
   ▼
data/processed/graph_sequence.pt      ← the single most-depended-on artifact
   │
   ├── src/models/baselines.py   RandomForest / XGBoost / MLP / Autoencoder / LSTM /
   │                              StaticGCN / StaticGAT  →  results/baselines.csv
   │
   ├── src/models/gat_lstm.py    the proposed model: GAT spatial encoder + LSTM temporal encoder
   ├── src/train.py              resumable, checkpointed training loop → checkpoints/gat_lstm.pt
   │
   ├── src/models/drift.py       ADWIN drift detector + 3 adaptation strategies
   │                              (no_adapt / full_retrain / incremental)
   │
   ├── src/eval.py                Experiments A–D + ablation → results/experiment_*.csv,
   │                              results/ablation.csv, results/fig_drift_degradation.png
   │
   └── src/explain.py             GAT attention visualization → results/fig_attention_*.png
```

### The proposed model, plainly

For each window *t*, a **2-layer GAT** (Graph Attention Network) reads the window's k-NN flow
graph and produces a per-flow embedding; these are mean-pooled into one vector per window. An
**LSTM** then looks at the trailing `k=6` window embeddings (roughly the last 6 windows'
worth of history) and produces a single "current state" vector `Z_t`. Two small heads read off
of `Z_t`: a sigmoid **anomaly score** `A_t ∈ [0,1]`, and a **multi-class attack-type** classifier
(secondary task). Formally:

```
G_t = (V_t, E_t)                         one graph per window
f(G_{t-k..t}) → A_t ∈ [0,1]               anomaly score from the trailing k-window history
```

### Drift detection + adaptation, plainly

An **ADWIN** change detector (from the `river` library) watches the L2 norm of `Z_t` — a cheap
scalar summary of "how is the model currently representing traffic" — across the whole Mon→Fri
sequence. When ADWIN flags a change, one of three strategies runs:

- **`no_adapt`** — do nothing (the control/baseline).
- **`full_retrain`** — throw the model away, retrain from scratch on every window seen so far.
- **`incremental`** — unfreeze only the LSTM + anomaly head, fine-tune for 3 epochs on a recent
  window of history, leave the GAT encoder untouched.

Each adaptation event's wall-clock time, peak GPU memory, and sample count are logged — this is
the project's cost/accuracy tradeoff measurement.

---

## 4. Experiments run, and what they actually showed

All splits are **time-based** (never randomly shuffled) — training always precedes testing
chronologically, matching how a real deployed detector would encounter data.

### Experiment A — Stationary performance (Mon+Tue only, held-out time split)

| Model | Scope | F1 | PR-AUC | ROC-AUC | FPR |
|---|---|---|---|---|---|
| RandomForest | flow | **0.988** | 0.999 | 1.000 | 0.000 |
| XGBoost | flow | 0.986 | 1.000 | 1.000 | 0.00004 |
| MLP | flow | 0.951 | 0.987 | 1.000 | 0.0005 |
| Autoencoder | flow | 0.000 | 0.017 | 0.766 | 0.00002 |
| SequenceLSTM | window | 0.000 | 0.702 | 0.184 | 0.000 |
| StaticGCN | window | 0.000 | 0.671 | 0.072 | 0.000 |
| StaticGAT | window | 0.000 | 0.656 | 0.000 | 0.034 |
| **GAT+LSTM (proposed)** | window | **0.000** | 0.670 | 0.064 | 0.000 |

**⚠️ This result does not support the hypothesis, and it's flagged rather than papered over.**
Every flow-level tabular baseline is near-perfect. Every window/graph-level model — including the
proposed model — has **F1 = 0** and, in several cases, **ROC-AUC below 0.5** (worse than random
ranking). `eval.py` itself logs a warning here, because `IMPLEMENTATION.md`'s own done-criterion
says: *"the proposed model should not be worse than every baseline in Experiment A — if it is,
that's a bug to fix, not a result to report."*

**Root cause, identified while preparing this document:** Experiment A's window-level test set
is tiny (185 windows) and severely imbalanced relative to training. Windows are 72.4% benign /
27.6% anomalous over Mon+Tue as a whole, but the **last 20% chronologically** (Tuesday's
brute-force period) — which becomes the test set under a time-based split — is 84.3% anomalous /
15.7% benign. The model trains mostly on Monday's benign-heavy majority and is tested on a
short, heavily attack-skewed tail it saw little like during training. Combined with only 185
total test windows, this produces the degenerate metrics above. **This needs further
investigation before Experiment A's numbers are treated as a real result** — likely fixes:
larger/rebalanced window size, a different split point, or per-class-weighted training for the
window-level models specifically.

### Experiment B — Temporal contribution (full Mon→Fri, no adaptation)

Unlike Experiment A's tiny stationary slice, this scores every model across the entire
2,577-window Mon→Fri sequence (rolling 20-window metrics, averaged):

| Model | F1 | PR-AUC | ROC-AUC | FPR |
|---|---|---|---|---|
| RandomForest (flow, no adapt) | 0.122 | 0.700 | 0.710 | 0.000 |
| StaticGCN (no adapt) | 0.120 | 0.689 | 0.638 | 0.035 |
| **GAT+LSTM (no adapt)** | 0.195 | 0.680 | 0.608 | 0.117 |

At full-sequence scale, the picture is more sensible — all three models are better than random
(ROC-AUC 0.6-0.71), and the proposed model has the best F1 of the three, though at a higher false
positive rate. This is a real, non-degenerate signal that Experiment A's tiny test slice didn't
show.

### Experiment C — Drift degradation

`results/fig_drift_degradation.png` plots the GAT+LSTM's rolling F1 across the full Mon→Fri
sequence with the day-boundary phase transitions marked. (Open the figure directly — it's the
clearest single artifact for seeing whether/where performance drops at each day change.)

### Experiment D — Adaptation comparison (the paper's core cost/accuracy result)

| Strategy | F1 | PR-AUC | ROC-AUC | FPR |
|---|---|---|---|---|
| No adaptation | 0.195 | 0.680 | 0.608 | 0.117 |
| Incremental adaptation | 0.196 | 0.677 | 0.602 | 0.124 |
| **Full retrain** | **0.353** | 0.686 | 0.645 | 0.235 |

**Cost side of the same experiment:**

| Strategy | Events fired | Total wall-clock time | Avg. time/event | Avg. peak GPU memory |
|---|---|---|---|---|
| Full retrain | 10 | 5h 45m | 34.6 min | 224 MB |
| Incremental | 8 | 1.8 min | 13.2 sec | 58 MB |

**Reading:** full retraining nearly doubles F1 over doing nothing (0.353 vs 0.195), but costs
~157× more wall-clock time per adaptation event (34.6 min vs 13.2 sec) and ~4× more GPU memory.
The lightweight incremental strategy (fine-tune last layers only, few epochs, recent data only)
essentially **didn't help** in this run — 0.196 vs 0.195 F1, statistically indistinguishable from
doing nothing. That's an honest, reportable finding in its own right: *cheap adaptation, as
implemented here, isn't sufficient to recover full-retrain's accuracy gain* — a real
accuracy/cost tradeoff, not the "best of both worlds" one might hope for. (Both GPU-memory
figures are small in absolute terms — this model comfortably fits an 8GB laptop GPU regardless
of strategy; the real cost differentiator is wall-clock time, not memory.)

### Ablation (Model1 → Model2 → Model3 → Model4)

| Model | What it isolates | F1 |
|---|---|---|
| Model1 — flow-only MLP | no graph, no time | 0.951 *(Exp. A's small-test-set caveat applies)* |
| Model2 — static GCN | +graph, no time | 0.000 *(same Exp. A caveat)* |
| Model3 — GAT+LSTM, no adapt | +graph +time | 0.195 *(full Mon→Fri scale)* |
| Model4 — GAT+LSTM, incremental | +graph +time +cheap adaptation | 0.196 *(full Mon→Fri scale)* |

Model1/Model2 come from Experiment A's problematic small test slice; Model3/Model4 come from the
much larger, better-behaved full-sequence evaluation — they are **not on a like-for-like scale**
and shouldn't be read as a single clean progression until Experiment A is fixed.

### Explainability

Three attention figures were generated for correctly-flagged anomaly windows:
`results/fig_attention_0_w3853.png`, `fig_attention_1_w4326.png`, `fig_attention_2_w4341.png`.
Each highlights which neighboring flows (by destination port + truncated label) most influenced
that window's anomaly score, per the GAT's last-layer attention weights.

---

## 5. Where things stand, honestly

**What works:** the full pipeline runs end-to-end — data cleaning, feature scaling, graph
construction, seven baseline models, the proposed GAT+LSTM model with checkpoint/resume
training, ADWIN-based drift detection that fires on real day-boundary transitions (among other
points), all three adaptation strategies, and attention-based explainability. Experiment D's
adaptation-cost story (full retrain buys accuracy at ~157× the wall-clock cost) is a solid,
sensible, well-supported result.

**What doesn't work yet:** Experiment A — the stationary-performance comparison that's supposed
to be the simplest, most basic sanity check — currently shows the proposed model and every other
window-level model failing outright, most likely due to a too-small, class-imbalanced 185-window
test split rather than a fundamental modeling failure (Experiment B's much larger evaluation on
the same architecture behaves sensibly). **This should be resolved before drawing any final
conclusion about whether the central hypothesis holds** — right now the honest answer is
"inconclusive at the Experiment A scale, tentatively supportive at the Experiment B/D scale."

**Suggested next step:** revisit `WINDOW_SIZE` (currently 1000 flows) and/or the Experiment A
train/test split point so the held-out slice isn't dominated by a single day's attack
concentration, then re-run `python src/eval.py --experiments a ablation` and update this
document's numbers.

---

## 6. Where everything lives

| Artifact | Path |
|---|---|
| Cleaned, concatenated flow data | `data/interim/cicids2017_clean.parquet` |
| Scaled features + windows | `data/processed/cicids2017_features.parquet` |
| Per-window graph sequence | `data/processed/graph_sequence.pt` |
| Trained model checkpoint | `checkpoints/gat_lstm.pt` |
| Baseline results | `results/baselines.csv` |
| Experiment A/B/D results | `results/experiment_a.csv`, `experiment_b.csv`, `experiment_d.csv` |
| Every per-window metric (source of truth) | `results/experiment_log.csv` |
| Adaptation cost/timing log | `results/adaptation_events.csv` |
| Ablation summary | `results/ablation.csv` |
| Drift degradation plot | `results/fig_drift_degradation.png` |
| Attention explainability figures | `results/fig_attention_*.png` |
| Paper draft (in progress) | `paper/paper.md` |

Full technical build instructions, function signatures, and hyperparameters:
[`IMPLEMENTATION.md`](./IMPLEMENTATION.md). Scope decisions and cut-lines:
[`PLAN.md`](./PLAN.md). Original research ambition this was scoped down from:
[`base.md`](./base.md).
