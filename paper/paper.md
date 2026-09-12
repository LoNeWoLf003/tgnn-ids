<!--
Skeleton per IMPLEMENTATION.md §10 step 1: section headers + one-sentence placeholders,
written before any experiment has run, to pin the formalism and notation down early. Fill
Introduction/Related Work/Problem Formulation/Method as each component (§1-§7) is implemented;
fill Experiments/Results last, generating every number/figure programmatically from
results/*.csv (never hand-typed) -- re-run the figure-generation script if any experiment is
re-run, don't silently edit numbers here. Remove this comment once the draft is no longer a
skeleton, and confirm no bracketed placeholder or "X.XX" value remains anywhere in this file
before calling the paper done (PLAN.md's verification/done-criteria).
-->

# Adaptive Temporal Graph Learning for Network Anomaly Detection

*Short paper / workshop-paper framing (per PLAN.md's scope decision) — preliminary results,
single dataset, explicitly scoped for a 2-3 week solo, single-GPU execution.*

## Abstract

[One paragraph, written last: problem, method, headline Experiment D number, and whether the
central hypothesis (§ "Conclusion" below) was supported or rejected.]

---

## 1. Introduction

Motivating causal chain (per `base.md` §1.25, used near-verbatim as the paragraph flow):

> Network traffic is dynamic → traditional IDS assumes stationary data → network communication
> has graph structure → traffic also evolves temporally → models degrade under concept drift →
> therefore adaptive temporal graph learning is needed.

[Expand each arrow into 1-2 sentences with a citation or a concrete CIC-IDS2017 example. State
the paper's contribution explicitly as the *research question and evaluation methodology*
(drift-aware evaluation of temporal graph representations for network anomaly detection), not
"we combined GAT+LSTM+ADWIN" as an architectural novelty claim — per `base.md` §1.20 / PLAN.md's
framing note. Close with a one-sentence roadmap of the rest of the paper.]

---

## 2. Related Work

### 2.1 ML-based intrusion detection

[Flow-level classical/deep IDS on CIC-IDS2017 and similar flow datasets; note their static,
i.i.d.-split evaluation assumption as the gap this paper targets.]

### 2.2 Graph neural networks for cybersecurity

[GNN-IDS work operating on host/flow graphs; note that most treat the graph as static per
snapshot.]

### 2.3 Temporal network learning

[TGN/EvolveGCN/dynamic-graph literature; state explicitly why this paper uses a simpler
GNN→LSTM design instead (PLAN.md: complexity-for-novelty risk not worth it in this timeframe).]

### 2.4 Concept drift and adaptive learning

[ADWIN/DDM/statistical-distance drift detectors and incremental-learning literature outside the
network-security domain; this paper imports that machinery rather than inventing a new detector.]

The gap sits at the intersection of 2.2-2.4: graph-structured *and* temporal *and*
drift-adaptive network anomaly detection, evaluated together rather than in isolation.

---

## 3. Problem Formulation

Let a per-window graph be

$$
G_t = (V_t, E_t)
$$

and the full sequence

$$
\mathcal{G} = \{G_1, G_2, \ldots, G_T\}.
$$

The detector is a function of the trailing $k$-window history:

$$
f(\mathcal{G}_{t-k:t}) \rightarrow A_t, \qquad A_t \in [0,1]
$$

the anomaly score for window $t$. When a drift detector fires on the learned representation
stream, the model is updated:

$$
M_{t+1} = \mathrm{Update}(M_t, D_t).
$$

**Node/edge definition used in this paper (redesigned from `base.md`'s host-graph formulation —
see PLAN.md's data-constraint note):** $V_t$ = the network flows captured in window $t$; $E_t$ =
$k$-nearest-neighbor edges in scaled CICFlowMeter feature space, computed independently per
window. $t$ indexes fixed-size row-count chunks within a day-file (chronological order assumed
from row order — see Limitations), not wall-clock time.

---

## 4. Proposed Method

### 4.1 Flow representation

[The ~70 CICFlowMeter numeric columns, `StandardScaler`-fit on Mon+Tue only; each scaled flow
vector is one node's $x$ in $G_t$. Reference `src/features.py`.]

### 4.2 Graph construction

[$k$-NN similarity graph per window, $k=8$, symmetrized; variable node count per window, no
padding. Reference `src/graph_build.py`.]

### 4.3 Temporal modeling

[GAT spatial encoder ($H_t = \mathrm{GAT}(G_t)$, 2 layers, 4 heads, hidden=64) → mean readout →
LSTM over the trailing $k{=}6$ window embeddings ($Z_t = \mathrm{LSTM}(H_{t-k:t})$). Reference
`src/models/gat_lstm.py`.]

### 4.4 Anomaly detection

[Sigmoid anomaly head $A_t=\sigma(f(Z_t))$ plus a secondary multi-class attack-type head, jointly
trained (BCE + 0.5·cross-entropy). Reference `src/train.py`.]

### 4.5 Concept drift detection

[ADWIN over $\lVert Z_t \rVert_2$; optional Jensen-Shannon secondary signal (cut if behind
schedule). Native drift source: CIC-IDS2017's day-file boundaries stand in for
`base.md` §1.18's synthetic phases (Mon=benign baseline, Tue=brute-force, Wed=DoS,
Thu=web-attack/infiltration held out as "unseen," Fri=botnet/portscan/DDoS). Reference
`src/models/drift.py`.]

### 4.6 Adaptive model updating

[Three strategies compared behind one interface: `no_adapt`, `full_retrain` (from scratch on
all data seen so far), `incremental` (drift-triggered fine-tune of the LSTM + anomaly head only,
few epochs, recent data only). Reference `src/models/drift.py`.]

### 4.7 Explainability

[Last-layer GAT attention weights, reframed for the flow-graph redesign as "which neighboring
flows most influenced this window's anomaly signal" rather than a host-identity framing; 2-3
true-positive anomaly windows visualized. Reference `src/explain.py`.]

---

## 5. Experiments

All splits are **time-based**, never randomly shuffled (`base.md` §1.17). Every number/figure
below is generated programmatically from `results/experiment_log.csv` and the per-experiment
`results/experiment_{a,b,c,d}.csv` / `results/ablation.csv` files — none are hand-typed.

### 5.1 Experiment A — Stationary performance

[Mon+Tue only, time-based 80/20 split. Table: RandomForest / XGBoost / MLP / Autoencoder /
SequenceLSTM / StaticGCN / StaticGAT / **GAT+LSTM (proposed)**, columns F1 / PR-AUC / ROC-AUC /
FPR. Source: `results/experiment_a.csv`.]

### 5.2 Experiment B — Temporal contribution

[Static-ML (RandomForest) vs StaticGCN vs GAT+LSTM, all trained once on Mon+Tue and never
updated, scored across the full Mon→Fri sequence. Source: `results/experiment_b.csv`.]

### 5.3 Experiment C — Drift degradation

[Rolling-F1-over-time plot for the no-adapt GAT+LSTM run (reuses Experiment B), with the
Mon/Tue/Wed/Thu/Fri day-boundary phase transitions marked as vertical lines. Figure:
`results/fig_drift_degradation.png`.]

### 5.4 Experiment D — Adaptation comparison

[No-adapt vs full-retrain vs incremental, across the full Mon→Fri sequence, drift-triggered by
the same ADWIN detector. Table: F1/PR-AUC/ROC-AUC/FPR per strategy plus wall-clock adaptation
time, peak GPU memory, and sample count per adaptation event — the paper's core cost/accuracy
tradeoff claim. Sources: `results/experiment_d.csv`, `results/adaptation_events.csv`.]

### 5.5 Ablation

[Model1 (flow-only MLP) → Model2 (static GCN) → Model3 (GAT+LSTM, no adapt) → Model4 (GAT+LSTM,
incremental adaptation) — isolating the graph, temporal, and adaptation contributions in turn.
Source: `results/ablation.csv`.]

### 5.6 Explainability

[1-2 attention figures from `src/explain.py`, cross-checked against the true attack `Label` for
that window — attention should concentrate among same-attack flows, not scatter across benign
ones. Figures: `results/fig_attention_*.png`.]

### 5.7 Limitations

- **Single dataset.** CIC-IDS2017 only; CSE-CIC-IDS2018 cross-dataset generalization was a
  stretch goal, not a guaranteed result (PLAN.md).
- **No host identity in this data copy.** The CSV mirror in hand has no Source/Destination IP,
  Source Port, Protocol, or Flow ID — only `Destination Port` + CICFlowMeter statistics + Label.
  Nodes are therefore flows (k-NN similarity graph), not hosts, and `base.md` §1.22's
  lateral-movement fan-out/new-destination signals are unrecoverable and dropped entirely,
  rather than approximated.
- **Chronological order is assumed, not measured.** No timestamp exists; row order within each
  day-file stands in for capture time (CICFlowMeter emits flows in capture order, but this is
  not independently verified in this data copy).
- **Drift is native, not injected, but coarse-grained.** Day-file boundaries are a real,
  unmodified property of the dataset (not fabricated), but they are 4 discrete transitions
  across a work-week, not continuous organic drift.
- **Explainability is attention-only.** GNNExplainer/SHAP-style perturbation explanations were
  out of scope (PLAN.md); attention weights are a cheap proxy, not a causal explanation.

---

## 6. Conclusion

Central hypothesis (`base.md` §1.27):

> Modeling network communication as an evolving temporal graph and adapting the learned
> representation in response to concept drift can provide more robust anomaly detection than
> static flow-based machine-learning approaches.

[State plainly, using the Experiment A/B/D numbers above, whether the results support or reject
this — do not soften an unfavorable result. Close with 1-2 sentences on the most promising
future-work direction (e.g., host-identity-preserving dataset, second dataset, richer
explainability).]

---

## References

[Populate during Related Work drafting — CIC-IDS2017 citation, GAT/GCN, ADWIN/river, and the
2-4 representative papers per Related Work subsection.]
