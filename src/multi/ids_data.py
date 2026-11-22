"""Loaders + preprocessing for NSL-KDD, UNSW-NB15 and TON-IoT (network).

None of these three has a usable timestamp or (for NSL/UNSW) host identity, so every dataset is
turned into the same abstraction: an ordered list of **phases** (phase 0 = training distribution,
phases 1.. = a chronological test stream). Rows are shuffled *within* a phase with a fixed seed
(the files are sorted by attack type / unordered, so row order carries no real time information)
and cut into fixed-size windows that never span a phase boundary.

Phase definitions (be explicit about what is native vs constructed):
  - nsl-kdd : phase 0 = KDDTrain+, phase 1 = KDDTest+ . NATIVE drift: KDDTest+ contains attack
              types (mscan, apache2, snmpguess, ...) absent from KDDTrain+.
  - unsw-nb15: phase 0 = official training-set, phase 1 = official testing-set. NATIVE split,
              but both are random partitions of the same capture, so drift is mild.
  - ton-iot : the CSV is concatenated blocks per attack type (10 type-runs), not time ordered.
              Phase 0/1 = 70/30 split of {backdoor,ddos,dos,scanning,password} + normal;
              phase 2 introduces {injection,xss}; phase 3 introduces {ransomware,mitm}.
              CONSTRUCTED ("emerging attack family") drift -- not a real capture timeline.

Preprocessing is fit on phase 0 only (no leakage from the stream).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

NSL_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack", "difficulty",
]
NSL_CAT = ["protocol_type", "service", "flag"]

UNSW_CAT = ["proto", "service", "state"]
UNSW_DROP = ["attack_cat", "label", "stcpb", "dtcpb"]  # tcp base seq numbers = random noise

TON_CAT = ["proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
           "ssl_version", "ssl_resumed", "ssl_established", "http_method", "http_version",
           "weird_name", "weird_notice"]
TON_NUM = ["dst_port", "duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts",
           "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "dns_qclass", "dns_qtype", "dns_rcode",
           "http_request_body_len", "http_response_body_len", "http_status_code"]

DATASETS = ["nsl-kdd", "unsw-nb15", "ton-iot"]


@dataclass
class Spec:
    name: str
    phases: list            # list[pd.DataFrame], each with columns _y (0/1) and _cat (str)
    num_cols: list
    cat_cols: list
    phase_names: list


@dataclass
class Prepared:
    name: str
    X: np.ndarray                    # (n_windows * window, d) float32, phase-ordered
    y: np.ndarray                    # (n,) int8
    cat: np.ndarray                  # (n,) attack-category strings
    phase: np.ndarray                # (n,) int phase id
    window: int
    phase_names: list
    n_features: int
    feature_names: list = field(default_factory=list)

    @property
    def n_windows(self):
        return len(self.X) // self.window

    def window_phase(self):
        return self.phase[:: self.window]


def _load_nsl() -> Spec:
    tr = pd.read_csv(DATA_DIR / "nsl-kdd" / "KDDTrain+.txt", header=None, names=NSL_COLS)
    te = pd.read_csv(DATA_DIR / "nsl-kdd" / "KDDTest+.txt", header=None, names=NSL_COLS)
    for d in (tr, te):
        d["_y"] = (d["attack"] != "normal").astype(int)
        d["_cat"] = d["attack"]
    num = [c for c in NSL_COLS if c not in NSL_CAT + ["attack", "difficulty"]]
    return Spec("nsl-kdd", [tr, te], num, NSL_CAT, ["train(KDDTrain+)", "stream(KDDTest+)"])


def _load_unsw() -> Spec:
    d = DATA_DIR / "unsw-nb15"
    tr = pd.read_parquet(d / "UNSW_NB15_training-set.parquet")
    te = pd.read_parquet(d / "UNSW_NB15_testing-set.parquet")
    for x in (tr, te):
        x["_y"] = x["label"].astype(int)
        x["_cat"] = x["attack_cat"].astype(str)
        for c in UNSW_CAT:
            x[c] = x[c].astype(str)
    num = [c for c in tr.columns if c not in UNSW_CAT + UNSW_DROP + ["_y", "_cat"]]
    return Spec("unsw-nb15", [tr, te], num, UNSW_CAT, ["train(official)", "stream(official test)"])


def _load_ton(data_seed: int = 0) -> Spec:
    df = pd.read_csv(DATA_DIR / "TON-IOT" / "train_test_network.csv", low_memory=False)
    df["_y"] = df["label"].astype(int)
    df["_cat"] = df["type"].astype(str)
    for c in TON_CAT:
        df[c] = df[c].astype(str)
    rng = np.random.default_rng(data_seed)
    old = ["backdoor", "ddos", "dos", "scanning", "password"]
    parts = [[], [], [], []]

    def split(idx, fracs):
        idx = rng.permutation(idx)
        cuts = (np.cumsum(fracs)[:-1] * len(idx)).astype(int)
        return np.split(idx, cuts)

    for fam, g in df.groupby("_cat"):
        idx = g.index.to_numpy()
        if fam == "normal":
            pieces = split(idx, [0.5, 0.2, 0.15, 0.15])
            for p, a in zip(parts, pieces):
                p.append(a)
        elif fam in old:
            a, b = split(idx, [0.7, 0.3])
            parts[0].append(a); parts[1].append(b)
        elif fam in ("injection", "xss"):
            parts[2].append(idx)
        else:  # ransomware, mitm
            parts[3].append(idx)
    phases = [df.loc[np.concatenate(p)].reset_index(drop=True) for p in parts]
    return Spec("ton-iot", phases, TON_NUM, TON_CAT,
                ["train(old families)", "stream: same families", "stream: +injection,xss",
                 "stream: +ransomware,mitm"])


def load_spec(name: str) -> Spec:
    return {"nsl-kdd": _load_nsl, "unsw-nb15": _load_unsw, "ton-iot": _load_ton}[name]()


def _slog(a: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.log1p(np.abs(a))


def prepare(spec: Spec, window: int = 256, data_seed: int = 0) -> Prepared:
    rng = np.random.default_rng(data_seed)
    phases = []
    for p in spec.phases:
        p = p.iloc[rng.permutation(len(p))]
        p = p.iloc[: (len(p) // window) * window]  # drop tail remainder -> whole windows only
        phases.append(p.reset_index(drop=True))

    tr = phases[0]
    num_tr = tr[spec.num_cols].to_numpy(dtype=np.float64)
    log_cols = np.abs(num_tr).max(axis=0) > 100  # heavy-tailed columns, decided on train only

    def num_tf(d):
        a = d[spec.num_cols].to_numpy(dtype=np.float64)
        a[:, log_cols] = _slog(a[:, log_cols])
        return a

    scaler = StandardScaler().fit(num_tf(tr))
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                        sparse_output=False).fit(tr[spec.cat_cols])

    def tf(d):
        a = np.clip(scaler.transform(num_tf(d)), -10, 10)
        return np.hstack([a, ohe.transform(d[spec.cat_cols])]).astype(np.float32)

    X = np.vstack([tf(p) for p in phases])
    y = np.concatenate([p["_y"].to_numpy() for p in phases]).astype(np.int8)
    cat = np.concatenate([p["_cat"].to_numpy() for p in phases])
    ph = np.concatenate([np.full(len(p), i) for i, p in enumerate(phases)])
    names = list(spec.num_cols) + list(ohe.get_feature_names_out(spec.cat_cols))
    return Prepared(spec.name, X, y, cat, ph, window, spec.phase_names, X.shape[1], names)


if __name__ == "__main__":
    for n in DATASETS:
        pr = prepare(load_spec(n))
        print(n, pr.X.shape, "windows:", pr.n_windows)
        for i, nm in enumerate(pr.phase_names):
            m = pr.phase == i
            print(f"  phase {i} {nm}: {m.sum()} flows, attack rate {pr.y[m].mean():.3f}, "
                  f"cats={sorted(set(pr.cat[m]))}")
