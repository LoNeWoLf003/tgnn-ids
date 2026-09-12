# 1. Adaptive Temporal Graph Learning for Network Anomaly Detection Under Concept Drift

This is best treated as a **research problem in continuously changing networks**, not merely an intrusion-classification project.

The central thesis is:

> Conventional network intrusion detection models classify individual flows using static features. Real network behavior is relational, temporal, and non-stationary. A detection system should therefore model communication graphs over time and adapt when the underlying traffic distribution changes.

---

## 1.1 Proposed paper title

Primary title:

> **Adaptive Temporal Graph Learning for Network Anomaly Detection Under Concept Drift**

Alternative titles:

> **A Temporal Graph Neural Network Framework for Adaptive Network Anomaly Detection**

> **Concept-Drift-Aware Temporal Graph Learning for Continuous Network Intrusion Detection**

> **Adaptive Graph Representation Learning for Network Anomaly Detection in Dynamic Communication Networks**

The first is the strongest because it clearly exposes the three research components:

```text
Temporal
   +
Graph Learning
   +
Concept Drift
```

---

# 1.2 The problem

Most conventional ML-based IDS systems represent traffic like this:

| Src IP | Dst IP | Src Port | Dst Port | Protocol | Packets | Bytes | Label  |
| ------ | ------ | -------: | -------: | -------- | ------: | ----: | ------ |
| A      | B      |    52341 |      443 | TCP      |      12 |  5420 | Normal |
| C      | B      |    52142 |      443 | TCP      |       8 |  3200 | Normal |
| D      | E      |     4444 |       22 | TCP      |      51 | 19000 | Attack |

Each row is treated largely independently.

This loses two types of information.

### Spatial/relational information

Who is communicating with whom?

For example:

```text
Client A ─────→ Web Server
Client B ─────→ Web Server
Client C ─────→ Web Server

Web Server ────→ Database
```

The network is naturally a graph.

### Temporal information

What happened before this connection?

Consider:

```text
t1: Workstation → Server A
t2: Workstation → Server B
t3: Workstation → Server C
t4: Server C → Database
```

Individually, those flows might not look particularly malicious.

Collectively, they could represent suspicious lateral movement.

---

# 1.3 The third problem: concept drift

Even a good model eventually becomes stale.

Suppose the model was trained in January:

```text
January
────────────
Web traffic
DNS
SSH
Database
```

By June:

```text
June
────────────
Web traffic
Containers
Microservices
Cloud APIs
New applications
Encrypted traffic
```

The probability distribution of the traffic has changed.

Formally, traditional supervised learning assumes approximately:

$$
P_{train}(X,Y) \approx P_{future}(X,Y)
$$

With concept drift:

$$
P_{train}(X,Y) \neq P_{future}(X,Y)
$$

There are several forms:

### Covariate drift

$$
P(X)
$$

changes.

Example:

```text
Normal HTTP traffic increases dramatically.
```

### Prior probability drift

$$
P(Y)
$$

changes.

Example:

```text
Attack frequency changes.
```

### Concept drift

$$
P(Y|X)
$$

changes.

Example:

```text
The same traffic pattern that was previously normal
becomes associated with an attack.
```

This is the part that makes the proposed paper substantially more interesting.

---

# 1.4 Overall architecture

The proposed system can look like:

```text
                 Network Traffic
                       │
                       ▼
              Flow Extraction
                       │
                       ▼
              Feature Engineering
                       │
                       ▼
             Temporal Flow Window
                       │
                       ▼
             Communication Graph
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   Node/Edge Features       Temporal Features
          │                         │
          └────────────┬────────────┘
                       ▼
              Temporal GNN
                       │
                       ▼
              Network Embedding
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Anomaly Detector     Attack Classifier
             │
             ▼
       Anomaly Score
             │
             ▼
       Drift Detector
             │
       ┌─────┴─────┐
       │           │
    No Drift     Drift
       │           │
       ▼           ▼
 Continue       Adapt Model
```

---

# 1.5 Graph representation

Define the network at time \(t\) as:

$$
G_t=(V_t,E_t)
$$

where:

* \(V_t\) = network entities
* \(E_t\) = communication relationships

For example:

```text
       10.0.0.5
           │
           │
           ▼
       10.0.0.10
        /      \
       /        \
      ▼          ▼
10.0.0.20    10.0.0.30
                │
                ▼
           10.0.0.40
```

Nodes could represent:

* IP addresses
* hosts
* servers
* containers
* IoT devices

Edges represent:

```text
A → B
```

with features such as:

```text
duration
packet count
byte count
protocol
source port
destination port
TCP flags
flow frequency
average packet size
```

---

# 1.6 Temporal representation

Instead of constructing one graph for the entire dataset:

$$
G
$$

construct a sequence:

$$
G_1,G_2,G_3,\ldots,G_T
$$

For example:

```text
Window 1
G1
 ↓
Window 2
G2
 ↓
Window 3
G3
 ↓
Window 4
G4
```

The model learns:

$$
G_{t-k},...,G_{t-1},G_t
$$

rather than just \(G_t\).

This allows the model to learn behavioral evolution.

---

# 1.7 GNN component

Start with a conventional GNN baseline.

For a node \(v\):

$$
h_v^{(l+1)}
=
\sigma
\left(
W^{(l)}
\cdot
AGG
\left(
\{h_u^{(l)}:u\in N(v)\}
\right)
\right)
$$

Possible architectures:

### GCN

Good baseline.

### GraphSAGE

Useful when graphs become large.

### GAT

Particularly interesting because attention can determine which neighboring nodes/communications are more important.

For example:

```text
Workstation
   │
   ├── Printer       low attention
   ├── Web Server    medium attention
   └── Unknown Host  high attention
```

The model can learn that certain relationships deserve greater weight.

---

# 1.8 Temporal component

A basic approach:

```text
GNN
 ↓
Graph embedding
 ↓
LSTM
 ↓
Anomaly classifier
```

Formally:

$$
H_t = GNN(G_t)
$$

then:

$$
Z_t=LSTM(H_{t-k},...,H_t)
$$

and:

$$
\hat{Y}=Classifier(Z_t)
$$

This is relatively straightforward.

A more advanced implementation could use:

* Temporal Graph Networks
* TGAT
* TGN
* EvolveGCN
* Graph Transformer

The paper should not use the most complicated architecture merely for novelty. The architecture should correspond to the research hypothesis.

---

# 1.9 Anomaly detection

There are two possible formulations.

## Classification

Predict:

```text
Normal
DoS
DDoS
Brute Force
Botnet
Port Scan
Web Attack
Infiltration
etc.
```

Use:

$$
P(Y|G_t)
$$

---

## Anomaly scoring

More interesting scientifically.

Calculate:

$$
A_t = f(G_t)
$$

where \(A_t\) is an anomaly score.

For example:

```text
0.02 → normal
0.17 → normal
0.31 → suspicious
0.78 → anomalous
0.96 → highly anomalous
```

Then:

$$
A_t > \tau
$$

means the observation is anomalous.

This enables the system to identify **previously unseen attack patterns**, which is more interesting than simply classifying known attacks.

---

# 1.10 Add concept drift detection

This becomes the adaptive component.

After obtaining network embeddings:

$$
Z_1,Z_2,\ldots,Z_T
$$

monitor the distribution:

$$
P(Z_t)
$$

against previous windows.

A drift detector determines whether:

$$
D(P_t,P_{t-1}) > \delta
$$

where \(D\) is a distribution-distance measure.

Possible techniques:

### ADWIN

Adaptive Windowing.

### DDM

Drift Detection Method.

### EDDM

Early Drift Detection Method.

### Page-Hinkley

Useful for detecting changes in streaming distributions.

### Statistical distances

You could also investigate:

* KL divergence
* Jensen-Shannon divergence
* Wasserstein distance
* Maximum Mean Discrepancy

This gives you another experimental dimension.

---

# 1.11 Adaptive retraining

The naive approach:

```text
Drift detected
     ↓
Train model from scratch
```

is computationally expensive.

A better approach:

```text
Drift detected
      ↓
Identify affected nodes/features
      ↓
Update model
      ↓
Incremental training
```

This creates an important research question:

> Can a model adapt to network distribution changes without full retraining?

Compare:

```text
Strategy A
No adaptation

Strategy B
Periodic retraining

Strategy C
Drift-triggered retraining

Strategy D
Incremental adaptation
```

This comparison could become one of the main contributions.

---

# 1.12 Strong experimental design

Don't evaluate only accuracy.

Use:

### Detection metrics

$$
Precision = \frac{TP}{TP+FP}
$$

$$
Recall = \frac{TP}{TP+FN}
$$

$$
F1 = 2\frac{Precision \times Recall}{Precision+Recall}
$$

Also:

* False Positive Rate
* False Negative Rate
* ROC-AUC
* PR-AUC

For heavily imbalanced security datasets, **PR-AUC and F1 are especially important**.

---

# 1.13 Evaluate adaptation

This is where the paper becomes substantially stronger.

Measure:

### Detection degradation

How quickly does performance decline after distribution changes?

For example:

```text
                  F1
                   │
1.0 ───────────────┤
                   │\
                   │ \
                   │  \
                   │   \____
                   │
                   └────────────── Time
                         ↑
                    concept drift
```

Then compare adaptive and non-adaptive models.

A good result would look conceptually like:

```text
                 Before drift    After drift

Static GNN           0.94           0.61
GNN + retraining     0.94           0.87
Adaptive TGNN        0.93           0.90
```

The actual numbers must come from experiments; they should never be fabricated.

---

# 1.14 Measure adaptation cost

This is another important contribution.

Suppose:

```text
Model A:
Full retraining
CPU: high
Training time: 40 min
Performance: high

Model B:
Incremental adaptation
CPU: moderate
Training time: 4 min
Performance: similar
```

Then your system isn't merely more accurate.

It is **operationally cheaper**.

Measure:

* retraining time
* GPU/CPU utilization
* memory consumption
* number of samples required
* number of training iterations
* inference latency
* model update frequency

---

# 1.15 Baseline models

You need strong baselines.

## Traditional ML

```text
Random Forest
XGBoost
LightGBM
SVM
```

## Neural networks

```text
MLP
LSTM
Autoencoder
```

## Graph models

```text
GCN
GraphSAGE
GAT
```

## Temporal models

```text
LSTM
GRU
Temporal GNN
```

Then:

```text
                    Graph    Temporal    Adaptive

Random Forest         X          X          X
XGBoost               X          X          X
LSTM                  X          ✓          X
GCN                   ✓          X          X
GAT                   ✓          X          X
Temporal GNN          ✓          ✓          X
Proposed              ✓          ✓          ✓
```

This table practically defines the experimental structure of the paper.

---

# 1.16 Ablation study

This is critical.

You need to prove that every component contributes.

Run:

```text
Model 1
Flow features only

Model 2
Flow + Graph

Model 3
Flow + Graph + Temporal

Model 4
Flow + Graph + Temporal + Drift Adaptation
```

Then compare.

If:

```text
Flow only                  F1 = X
Graph                      F1 = Y
Graph + Temporal           F1 = Z
Graph + Temporal + Adapt   F1 = W
```

you can quantify the contribution of each component.

This is much stronger than saying:

> “Our model achieved the highest accuracy.”

---

# 1.17 Dataset strategy

Use at least two datasets if possible.

A reasonable combination:

### Dataset 1

**CIC-IDS2017**

Useful because it contains multiple attack categories and benign traffic.

### Dataset 2

**CSE-CIC-IDS2018**

Useful for testing generalization.

Potential third dataset:

**UNSW-NB15**

This gives you a cross-dataset experiment.

The important experiment is:

```text
Train
Dataset A

Test
Dataset B
```

instead of:

```text
Train A
Test random 20% of A
```

The latter can produce unrealistically optimistic results.

---

# 1.18 Simulating concept drift

Public IDS datasets aren't always perfect for genuine longitudinal concept drift.

You can construct controlled temporal scenarios.

For example:

```text
Phase 1
Normal traffic distribution

        ↓

Phase 2
Increase certain application traffic

        ↓

Phase 3
Introduce new attack distribution

        ↓

Phase 4
Introduce previously unseen attack

        ↓

Phase 5
Change normal traffic distribution
```

This allows you to explicitly evaluate adaptation.

You could define:

$$
D_1,D_2,\ldots,D_n
$$

where each \(D_i\) represents a different traffic distribution.

---

# 1.19 A stronger research experiment

Train the model on:

```text
Normal
DoS
Port Scan
Brute Force
```

Then introduce:

```text
Previously unseen attack
```

during testing.

Compare:

```text
Static classifier
vs
Adaptive anomaly detector
```

The research question becomes:

> How effectively can an adaptive temporal graph model identify previously unseen attacks after network behavior changes?

That's much stronger than standard multiclass classification.

---

# 1.20 Novelty

Your novelty should not be stated as:

> “We use GNN + LSTM + drift detection.”

That is an architectural combination, not necessarily a research contribution.

Frame the contribution around the problem.

Potential contributions:

### Contribution 1

A temporal graph representation of network communication that captures relationships between hosts over consecutive traffic windows.

### Contribution 2

A concept-drift detection mechanism operating over learned network representations rather than raw individual flow features.

### Contribution 3

An adaptive model-update strategy triggered by detected distribution changes.

### Contribution 4

A systematic evaluation of detection performance before and after network behavior changes.

### Contribution 5

Analysis of the accuracy–adaptation-cost tradeoff between full retraining, periodic retraining, and incremental adaptation.

That is a legitimate research narrative.

---

# 1.21 A more ambitious version

The strongest version would model **host behavior**, not simply flows.

For every host \(v\), create a behavioral vector:

$$
B_v(t)=
[
f_{in},
f_{out},
p_{unique},
d_{unique},
b_{avg},
duration,
protocol\_distribution,
port\_distribution
]
$$

where:

* \(f_{in}\) = incoming flow rate
* \(f_{out}\) = outgoing flow rate
* \(p_{unique}\) = unique peers
* \(d_{unique}\) = unique destinations
* \(b_{avg}\) = average bytes
* etc.

Then build:

```text
Host behavior
      ↓
Graph structure
      ↓
Temporal evolution
      ↓
Representation
      ↓
Anomaly score
```

Now the system isn't simply detecting anomalous packets.

It's detecting **anomalous behavior of network entities**.

---

# 1.22 Even stronger: lateral movement

You can specialize the research problem toward lateral movement.

Example:

```text
             Initial compromise
                    │
                    ▼
              Workstation A
                 /      \
                /        \
               ▼          ▼
          Server B     Server C
                          │
                          ▼
                     Database D
```

The attack is represented as a temporal graph:

$$
A \rightarrow B \rightarrow C \rightarrow D
$$

The model learns that a host suddenly communicating with many previously unseen internal systems is suspicious.

Potential anomaly features:

```text
new destinations
new ports
new protocols
new peer relationships
connection burst
fan-out
fan-in
communication entropy
temporal ordering
```

This gives the paper a concrete security scenario instead of an abstract IDS problem.

---

# 1.23 Explainability

GNN-based cybersecurity models can become difficult to interpret.

Add an explainability component.

When the model says:

```text
Anomaly score = 0.94
```

it should also identify:

```text
Primary contributing host:
10.0.0.42

Suspicious relationships:
10.0.0.42 → 10.0.0.71
10.0.0.42 → 10.0.0.84
10.0.0.42 → 10.0.0.91

New destinations:
3

New ports:
22, 445

Behavior change:
+430% outbound connections
```

Possible methods:

* GNNExplainer
* SHAP for non-GNN components
* attention visualization
* feature importance

Then the output becomes useful to a security analyst.

---

# 1.24 Final system

The complete research architecture could therefore become:

```text
                         NETWORK
                            │
                            ▼
                    Flow Collection
                            │
                            ▼
                    Feature Extraction
                            │
                            ▼
                 Temporal Window Builder
                            │
                            ▼
                 Communication Graph Gt
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
           Node Features          Edge Features
                 │                     │
                 └──────────┬──────────┘
                            ▼
                    Graph Attention
                            │
                            ▼
                  Temporal Encoder
                            │
                            ▼
                 Network Representation
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          Attack Classifier       Anomaly Detector
                 │                     │
                 └──────────┬──────────┘
                            ▼
                     Anomaly Score
                            │
                            ▼
                    Drift Detection
                            │
                    ┌───────┴───────┐
                    │               │
                 Stable           Drift
                    │               │
                    ▼               ▼
               Continue       Incremental Update
                                    │
                                    ▼
                              Updated Model
                                    │
                                    ▼
                              Explanation
```

---

# 1.25 Paper structure

A strong paper could be structured as:

## 1. Introduction

Establish:

```text
Network traffic is dynamic
        ↓
Traditional IDS assumes stationary data
        ↓
Network communication has graph structure
        ↓
Traffic also evolves temporally
        ↓
Models degrade under concept drift
        ↓
Therefore adaptive temporal graph learning is needed
```

---

## 2. Related Work

Four subsections:

### 2.1 ML-based intrusion detection

### 2.2 Graph neural networks for cybersecurity

### 2.3 Temporal network learning

### 2.4 Concept drift and adaptive learning

The gap emerges at the intersection.

---

## 3. Problem Formulation

Define:

$$
G_t=(V_t,E_t)
$$

and the sequence:

$$
\mathcal{G}=\{G_1,G_2,\ldots,G_T\}
$$

Then formulate:

$$
f(\mathcal{G}_{t-k:t}) \rightarrow A_t
$$

where:

$$
A_t \in [0,1]
$$

is the anomaly score.

The adaptation mechanism determines:

$$
M_{t+1}=Update(M_t,D_t)
$$

when drift is detected.

---

## 4. Proposed Method

### 4.1 Flow representation

### 4.2 Graph construction

### 4.3 Temporal modeling

### 4.4 Anomaly detection

### 4.5 Concept drift detection

### 4.6 Adaptive model updating

### 4.7 Explainability

---

# 1.26 Experimental section

Run four major experiments.

### Experiment A — Detection performance

Compare all models under stationary conditions.

### Experiment B — Temporal behavior

Compare:

```text
Static ML
vs
GNN
vs
Temporal GNN
```

### Experiment C — Concept drift

Introduce distribution changes and measure performance degradation.

### Experiment D — Adaptation

Compare:

```text
No adaptation
Periodic retraining
Full retraining
Incremental adaptation
```

Then measure:

```text
F1
Recall
FPR
Detection latency
Training time
Adaptation time
Computational overhead
```

---

# 1.27 The key hypothesis

The entire paper can revolve around one central hypothesis:

> **Modeling network communication as an evolving temporal graph and adapting the learned representation in response to concept drift can provide more robust anomaly detection than static flow-based machine-learning approaches.**

Then your experiments either support or reject that hypothesis.

That is how you avoid turning the paper into a collection of algorithms.

---

# 1.28 The research gap in one diagram

```text
Existing IDS
────────────

Flow-based ML
     │
     ├── Good classification
     ├── Static
     ├── Limited topology awareness
     └── Vulnerable to drift


Existing GNN IDS
────────────────

Graph
  │
  ├── Topology awareness
  ├── Better relational modeling
  └── Often evaluated on static datasets


Existing temporal IDS
─────────────────────

Temporal modeling
  │
  ├── Captures evolution
  └── Often lacks explicit graph adaptation


Your proposed framework
───────────────────────

Graph
  +
Temporal
  +
Anomaly Detection
  +
Concept Drift
  +
Adaptive Updating
  +
Explainability
```

The strongest version of the paper is therefore not:

> “We created a new neural network.”

It is:

> **“We investigate whether continuously adapting temporal graph representations can make network anomaly detection robust to changing traffic distributions and previously unseen behavior.”**

That is a much more defensible research problem.
