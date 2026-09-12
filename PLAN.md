# Adaptive Temporal Graph Learning for Network Anomaly Detection — Execution Plan

> Companion to [`base.md`](./base.md) (the research proposal). This document is the concrete,
> scoped, day-by-day execution plan for turning that proposal into a submittable paper.

## Context

`base.md` lays out an ambitious research proposal: detect network anomalies by modeling
communication as an **evolving temporal graph** (hosts as nodes, flows as edges), adding
**concept-drift detection** on learned embeddings, and **adaptively updating** the model
instead of retraining from scratch — evaluated across 4 experiments (stationary performance,
temporal contribution, drift degradation, adaptation cost) plus an ablation study, on 2-3
datasets, with explainability on top.

**Constraints for this execution:**
- **Timeline:** 2-3 weeks, solo, starting completely from scratch (no data, no code yet).
- **Compute:** local machine — Windows 11, NVIDIA RTX 2000 Ada Generation Laptop GPU, 8GB VRAM.
  No session caps or disconnects, but a single laptop GPU (nothing to parallelize across) and
  finite local disk (CIC-IDS2017 raw+interim+processed runs several GB — check free space up front).
- **Goal:** a paper — but 2-3 weeks solo on one laptop GPU cannot honestly produce the full
  base.md ambition (3 datasets, lateral-movement scenario, TGN/EvolveGCN, full explainability
  suite, exhaustive ablations). That combination realistically fits a **workshop paper / short
  paper / arXiv preprint**, not a top-tier journal. This plan targets that scope deliberately,
  keeping every one of the 4 experiment types (A–D) alive in a reduced but real form, rather than
  cutting whole experiments and inflating what remains.

The plan below is the minimum-viable scope that still supports a genuine, defensible research
contribution and narrative, sized to run entirely on a single 8GB-VRAM laptop GPU, with explicit
day-by-day tasks and cut lines if time runs short.

---

## ⚠️ Data constraint discovered at implementation time (overrides §1.21/§1.22 framing below)

The CIC-IDS2017 CSVs actually in hand (`data/raw/MachineLearningCVE/*.csv`, the common
redistributed mirror) contain **only** `Destination Port` + the ~70 CICFlowMeter flow-statistic
columns + `Label` — **no `Flow ID`, `Source IP`, `Destination IP`, `Source Port`, `Protocol`, or
`Timestamp`**. Row counts match the published ~2.83M-flow total, so it's the right dataset, just
missing every identifying/temporal column the original host-graph design depended on. The official
UNB registration-gated `GeneratedLabelledFlows.zip` (which has those columns) was not obtained —
decision made to redesign around what's actually available rather than block on that download.

**Redesign:** node granularity changes from **host (IP)** to **flow**, and per-window graphs are
built as a **k-NN feature-similarity graph over the flows in that window** (nodes = flows, edges =
nearest neighbors in scaled CICFlowMeter feature space) instead of a host communication graph. This
is an established alternative in GNN-IDS literature when relational/identity fields aren't
available. Time windows become **fixed-size chunks of consecutive rows per day-file** (row order is
assumed chronological — CICFlowMeter emits flows in capture order — documented as an explicit
limitation) instead of wall-clock 5-minute windows. Everything else — temporal LSTM over per-window
graph embeddings, ADWIN drift detection, adaptation strategies, Experiments A-D, ablation, and
attention-based explainability — is unaffected, since none of it actually depended on host identity
or absolute timestamps, only on there being *a* graph sequence and *a* chronological order. The
Mon-Fri native-drift-phase mapping (§ "Days 10-11" below) is also unaffected, since day-file
boundaries don't require timestamps.

**Consequence:** §1.22's lateral-movement signals (fan-out, new destinations/ports per host) are
now **unrecoverable** (no host identity to track) and are dropped entirely rather than folded into
a host vector — stated as a limitation, not silently omitted.

---

## Scope decisions (locked in before Day 1, node granularity/lateral-movement rows updated per the data-constraint note above)

| Decision | Choice | Why |
|---|---|---|
| Datasets | **CIC-IDS2017 only**; CSE-CIC-IDS2018 as stretch goal | One dataset is all that fits in 2-3 weeks with real experiments; UNSW-NB15 dropped entirely |
| Node granularity | **Flow**, per-window k-NN similarity graph (~~Host (IP)~~ — reverted, see data-constraint note above) | The dataset copy in hand has no Src/Dst IP; flow-similarity graphs are the closest defensible substitute |
| Primary task | **Anomaly scoring** (unsupervised-ish, threshold on A_t), classification as secondary head | Matches §1.9 — needed for the "unseen attack" story in Experiment D |
| GNN backbone | **GAT**, 1-2 layers, hidden dim ≤ 64 | Attention gives cheap explainability (§1.23) for free; small enough for an 8GB laptop GPU |
| Temporal backbone | **GNN → LSTM** (§1.8's "basic approach") | TGN/EvolveGCN explicitly not worth the setup risk in this window — proposal itself warns against complexity-for-novelty |
| Drift detector | **ADWIN** (via `river`) on a scalar drift statistic derived from embeddings, + one distributional distance (Jensen-Shannon or MMD) as a secondary signal | Cheap, well-tested, avoids building 4 detectors |
| Adaptation strategy compared | No-adapt vs Full-retrain vs Incremental (drift-triggered fine-tune of last layers) | Periodic retraining dropped — 3-way comparison is enough to make the cost/accuracy point |
| Lateral movement (§1.22) | **Dropped entirely** — no host identity exists in the data to compute fan-out/new-destination signals from (see data-constraint note above) | Stated as an honest limitation rather than fabricated |
| Explainability | **GAT attention weights only** (already computed for free) | GNNExplainer/SHAP dropped — nice-to-have, not needed for the core hypothesis |
| Target venue | **Workshop paper / short paper (4-6 pages) or arXiv preprint**, framed explicitly as preliminary/short-paper contribution | Matches what 2-3 weeks solo can honestly support |

**Cut lines, in priority order, if behind schedule** (drop top-down, never drop the ablation entirely):
1. Drop the second dataset stretch goal (CSE-CIC-IDS2018) — stay single-dataset.
2. Drop the secondary drift statistic (JS/MMD) — keep only ADWIN.
3. Collapse ablation from 4 models to 3 (merge "Flow+Graph" and "Flow+Graph+Temporal" reporting into one pass if compute-constrained).
4. Drop the attention-based explainability figure (keep it as a "future work" mention only).
5. Never drop: Experiment A (stationary performance) and Experiment D (adaptation comparison) — these two are the paper's core claim.

---

## Local-machine risk mitigations (apply from Day 1)

- **Checkpoint every epoch** (model state, optimizer state, epoch number, RNG state) to local
  disk under `checkpoints/`. There's no session cap forcing this, but a laptop can still crash,
  sleep, or lose power mid-run — every training script must be **resumable**:
  `python train.py --resume` picks up from the latest checkpoint automatically.
- Keep graphs small enough to fit **8GB VRAM** with room to spare: hidden dims ≤ 64, batch
  windows via neighbor sampling if any window's node count is large, mixed precision
  (`torch.cuda.amp`) on. Watch `nvidia-smi` during the first run of each new model size.
- Log metrics to a **plain CSV/JSON file after every epoch**, not just in-memory — a crash
  must not lose results. Skip MLflow/W&B server setup (extra time cost); a `results.csv` +
  `matplotlib` is enough for this scope.
- Disable sleep/hibernate (or plug in + adjust Windows power settings) for any run expected to
  take more than ~30 minutes — a laptop going to sleep mid-training is the local equivalent of a
  Colab disconnect.
- Structure work into **session-sized chunks** anyway (3-6 hours): even without a hard cap,
  working in bounded daily chunks keeps the day-by-day plan below honest and trackable.

---

## Repo structure

```
project/
├── base.md                   # original research proposal
├── PLAN.md                   # this file
├── data/{raw,interim,processed}/        (local disk; excluded from any VCS, regenerable)
├── src/
│   ├── ingest.py           # load/clean CIC-IDS2017 CSVs
│   ├── features.py         # per-flow scaling + per-window context features
│   ├── graph_build.py       # windowed G_t construction (flow k-NN similarity graphs)
│   ├── models/
│   │   ├── baselines.py     # RF/XGBoost, MLP/LSTM/Autoencoder
│   │   ├── gat_lstm.py      # core temporal GNN
│   │   └── drift.py         # ADWIN wrapper + adaptation logic
│   ├── train.py             # resumable training loop, checkpointing
│   └── eval.py              # metrics, experiment A-D runners
├── checkpoints/               # model/optimizer checkpoints, local disk
├── results/                  # CSVs + figures, local disk
└── paper/                    # draft .md or LaTeX, written in parallel
```

---

## Day-by-day plan (15-18 working days, ~3 weeks with buffer)

### Days 1-2 — Setup + data acquisition
- Set up a local Python venv; `pip install torch` (CUDA build matching your driver)
  `torch_geometric river scikit-learn xgboost lightgbm pandas matplotlib`; verify
  `torch.cuda.is_available()` reports the RTX 2000 Ada GPU.
- Download CIC-IDS2017 (CICFlowMeter CSVs) directly to local disk (`data/raw/`), unzip in place.
- Clean: drop NaN/Inf, dedupe, fix label string inconsistencies (known issue — check community errata), sort chronologically by day-file.
- **Checkpoint:** a single cleaned, time-ordered parquet file with consistent labels.

### Days 3-4 — Feature engineering + graph construction
- **(Redesigned per the data-constraint note above — no host IPs available.)** Scale the ~70
  CICFlowMeter numeric columns per flow (fit `StandardScaler` on Mon+Tue only, to avoid leakage);
  add destination-port frequency/entropy as extra per-window context features.
- Pick window size as a fixed **row-count chunk per day-file** (start with 1000 consecutive flows;
  note as a hyperparameter, don't tune exhaustively) — substitutes for wall-clock 5-min windows
  since there's no timestamp; row order assumed chronological (documented limitation).
- Build `G_t=(V_t,E_t)` sequence as PyG `Data` objects, one per window: **V_t = the flows in that
  window** (variable count — no zero-padding needed, PyG handles variable-sized graphs natively),
  **E_t = k-NN edges in scaled feature space** (k=8, cosine or Euclidean).
- **Checkpoint:** `python src/graph_build.py` produces a saved, loadable sequence of graphs + labels on local disk (`data/processed/`).

### Days 5-6 — Baselines (trimmed list)
Implement, behind one common `fit/score/predict` interface:
1. Traditional ML: Random Forest, XGBoost (drop SVM/LightGBM — redundant signal, saves time).
2. Neural: MLP, LSTM on flow sequences, Autoencoder (unsupervised anomaly baseline).
3. Static graph: GCN and GAT (drop GraphSAGE — not needed at this graph scale).
- **Checkpoint:** all baselines trained and scored on a held-out split; results in `results/baselines.csv`.

### Days 7-9 — Core model: GAT + LSTM temporal encoder
- Day 7: implement `H_t = GAT(G_t)` spatial encoder, verify it trains stably alone (probe with a linear head).
- Day 8: add `Z_t = LSTM(H_{t-k..t})` temporal encoder + anomaly scoring head `A_t = sigmoid(f(Z_t))` + classifier head.
- Day 9: full training loop with checkpointing/resume; tune briefly (window length k, hidden dim) — no large sweep, just enough to beat the static-GNN baseline.
- **Checkpoint:** trained temporal model reproducing baseline-competitive F1 on stationary data.

### Days 10-11 — Drift simulation + drift detector
- **Key simplification:** don't hand-engineer synthetic drift phases from scratch — CIC-IDS2017's
  5 daily capture files already have naturally different attack compositions, so map them
  directly onto §1.18's phases: **Mon = Phase 1 (benign baseline)**, **Tue = Phase 2 (brute-force
  introduced)**, **Wed = Phase 3 (DoS)**, **Thu = Phase 4 (web attack/infiltration — treat as the
  "previously unseen attack" by excluding it from training)**, **Fri = Phase 5 (botnet/portscan/
  DDoS, normal-traffic mix also shifts)**. This is a legitimate methodological choice worth
  stating explicitly in the paper (native drift, not fabricated) and saves real implementation time.
- Log true phase boundaries (the day cutovers) as ground truth for later degradation plots.
- Implement ADWIN (via `river`) over a scalar drift statistic (e.g. reconstruction error or
  embedding-norm trend) on the `Z_t` stream. Add JS-divergence as secondary signal only if on schedule.
- **Checkpoint:** a drift-labeled window sequence + a working drift detector that fires near the true day-boundary phase transitions.

### Days 12-14 — Experiments A–D (reduced) + ablation
- **A — Stationary performance:** all baselines + proposed model, trained/tested within Mon+Tue
  (Phase 1+2) only, time-based split (never random 20% split — §1.17). PR-AUC, F1, Recall, FPR as
  primary metrics (imbalance-aware).
- **B — Temporal contribution:** Static ML vs static GCN vs GAT+LSTM temporal model, run across
  the full Mon→Fri sequence with **no adaptation**, to show the degradation curve baseline.
- **C — Drift degradation:** same Day-12 run, reported as F1-over-time with the Mon/Tue/Wed/Thu/Fri
  boundaries marked (the curve from §1.13) — this reuses Day-12's run rather than a fresh one.
- **D — Adaptation comparison:** No-adaptation (from Day 12) vs Full-retrain vs Incremental
  adaptation (drift-triggered fine-tune of last layers only) re-run across Mon→Fri. Measure F1
  before/after each day-boundary, retraining/adaptation time, and (if feasible) peak GPU memory —
  this is the paper's core cost/accuracy tradeoff claim (§1.14) and headline result.
- **Ablation (trimmed, reuse-heavy):** Model1 = MLP/flow-only (Day 5-6 baseline) → Model2 = GCN
  static (Day 5-6 baseline) → Model3 = GAT+LSTM no adaptation (Day 8-9 result) → Model4 =
  GAT+LSTM + drift-triggered incremental adaptation (Day 13's Strategy-C result). All four rows
  come from work already done in earlier days — this step is mostly aggregation, not new training.
- **Stretch, only if on schedule — unseen-attack test (§1.19):** train on Mon-Wed attack types
  only (Normal/BruteForce/DoS), test on Thu's web-attack/infiltration (excluded from training) to
  see whether the anomaly-score head flags it as anomalous without ever having seen the label —
  a strong, differentiating result if time allows. Cut first if behind.
- **Checkpoint:** `results/experiment_{a,b,c,d}.csv` + ablation table, all with real (not fabricated) numbers, generated programmatically from `results/experiment_log.csv` — never hand-typed into the paper.

### Days 12-14 (parallel) — Lightweight explainability
- Extract and visualize GAT attention weights for a handful of true-positive anomaly windows:
  which neighbor/edge got high attention, framed as "primary contributing host / suspicious
  relationships" per the §1.23 example format. One or two figures, not a full explainability study.

### Days 13-18 — Paper writing (start no later than day 10, in parallel with experiments)
Follow the §1.25 structure directly, framed as a short/workshop paper:
1. **Introduction** — use the causal chain from §1.25 (traffic is dynamic → static IDS assumption
   → graph structure exists → traffic evolves temporally → models degrade under drift →
   therefore adaptive temporal graph learning) almost verbatim as the motivating paragraph flow.
2. **Related Work** — 4 short subsections (ML-IDS, GNN-cybersecurity, temporal network learning,
   concept drift/adaptive learning); state the gap as their intersection.
3. **Problem Formulation** — lift the formalism directly: `G_t=(V_t,E_t)`, sequence
   `𝒢={G_1..G_T}`, `f(𝒢_{t-k:t})→A_t∈[0,1]`, adaptation `M_{t+1}=Update(M_t,D_t)`.
4. **Proposed Method** — 4.1 flow/host representation, 4.2 graph construction, 4.3 temporal
   modeling (GAT+LSTM), 4.4 anomaly detection, 4.5 concept drift detection (ADWIN), 4.6 adaptive
   model updating (incremental fine-tune), 4.7 explainability (attention-based, brief).
5. **Experiments** — A/B/C/D as run above, ablation table, honest limitations section
   (single dataset, simulated rather than fully organic drift, small-scale attention-only
   explainability) — a limitations section is expected and appropriate for a short paper.
6. **Conclusion** — reframe around the central hypothesis (§1.27): does adaptive temporal graph
   learning outperform static flow-based ML under drift? State what the results support or reject.
- **State explicitly in the paper's framing** (per §1.20) that the contribution is the *research
  question and evaluation methodology* (drift-aware evaluation of temporal graph representations),
  not "we combined GAT+LSTM+ADWIN" as an architectural novelty claim.

### Days 17-18 — Buffer + polish
- Reserved for crash/re-run recovery, figure polishing, proofreading.
- If ahead of schedule, spend here on the CSE-CIC-IDS2018 stretch-goal cross-dataset test
  (train on CIC-IDS2017, test on CSE-CIC-IDS2018) — this single addition would meaningfully
  strengthen the paper if time allows, per §1.17's generalization point.

---

## Libraries/tools by step

| Step | Tools |
|---|---|
| Data handling | `pandas`, `numpy` |
| Graph/GNN | `torch`, `torch_geometric` (GCN, GAT layers) |
| Temporal | `torch.nn.LSTM` (skip `torch_geometric_temporal`'s TGN/EvolveGCN — out of scope) |
| Drift detection | `river` (ADWIN), `scipy.spatial.distance.jensenshannon` if secondary signal is in scope |
| Baselines | `scikit-learn` (RF, MLP), `xgboost` |
| Explainability | GAT attention weights (native to the model, no extra library) |
| Tracking | plain CSV/JSON logs + `matplotlib` (skip MLflow/W&B setup overhead) |
| Compute | Local machine (NVIDIA RTX 2000 Ada Laptop GPU, 8GB VRAM), local disk for persistence |

---

## Critical files (in build order — everything downstream depends on the ones above it)

- `src/graph_build.py` — flow → windowed communication graph construction. Everything else depends on this.
- `src/models/gat_lstm.py` — the core proposed model (GAT + LSTM).
- `src/models/drift.py` — ADWIN wrapper over embedding statistics; core to Experiments C/D.
- `src/train.py` — resumable, checkpointed training loop; the crash/interruption safety net for every experiment, including the adaptation strategies (full-retrain vs incremental).

---

## Verification / done-criteria

- `src/graph_build.py` produces a loadable, time-ordered graph sequence with labels — spot-check
  a few windows manually against raw flow counts.
- Training scripts survive a simulated crash: kill the training process mid-epoch, restart,
  confirm `--resume` picks up from the last checkpoint with matching loss curve continuity.
- All 4 experiments (A-D) produce `results/*.csv` with real numbers; sanity-check that the
  proposed model is not worse than every baseline on Experiment A (if it is, that's a modeling
  bug, not a result to report).
- Experiment C's F1-over-time plot visibly shows degradation at the injected phase boundaries
  for non-adaptive models — if it doesn't, the drift injection isn't strong enough and phases
  need to be made more distinct.
- Paper draft has all 6 sections filled with real numbers/figures (no placeholder "X.XX" values)
  before calling this project done.
