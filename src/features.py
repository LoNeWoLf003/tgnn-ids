"""Per-flow feature scaling + windowing.

Data constraint (see PLAN.md / ingest.py): no host identity or timestamp exists in this dataset
copy, so there is no B_v(t) host-behavior vector to build. Instead:
  1. the ~70 CICFlowMeter numeric columns are scaled to become each flow's node feature vector x
     (this is what graph_build.py's k-NN similarity graph operates over), and
  2. each flow is assigned a `window_id` -- a fixed-size chunk of consecutive rows *within its
     day* (never spanning a day boundary, so the Mon-Fri phase mapping in drift.py stays clean) --
     standing in for the original 5-minute wall-clock window.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

INTERIM_PATH = Path("data/interim/cicids2017_clean.parquet")
PROCESSED_DIR = Path("data/processed")
SCALER_PATH = PROCESSED_DIR / "scaler.joblib"
FEATURES_PATH = PROCESSED_DIR / "cicids2017_features.parquet"

WINDOW_SIZE = 1000  # flows per window -- hyperparameter, documented not exhaustively tuned

# Day order fixes the Mon->Fri sequence regardless of dict/groupby iteration order elsewhere.
DAY_ORDER = ["mon", "tue", "wed", "thu_am", "thu_pm", "fri_am", "fri_pm_scan", "fri_pm_ddos"]

NUMERIC_FEATURE_COLS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def assign_windows(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """Assign a global, chronologically-ordered window_id: fixed-size row chunks within each
    day, days ordered Mon->Fri, never merging windows across a day boundary."""
    df = df.copy()
    day_offset = {d: i for i, d in enumerate(DAY_ORDER)}
    local_window = df["seq"] // window_size
    # Reserve a fixed block of window-id space per day so windows never collide across days,
    # regardless of how many local windows any single day actually has.
    max_windows_per_day = int((df.groupby("day")["seq"].max() // window_size).max()) + 1
    df["window_id"] = df["day"].map(day_offset) * max_windows_per_day + local_window
    return df


def fit_scaler(df: pd.DataFrame, fit_days=("mon", "tue")) -> StandardScaler:
    """Fit on Mon+Tue only (Experiment A's stationary split) to avoid leakage from later days."""
    scaler = StandardScaler()
    scaler.fit(df.loc[df["day"].isin(fit_days), NUMERIC_FEATURE_COLS])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    log.info("Fitted scaler on days=%s (%d rows), saved to %s",
              fit_days, df["day"].isin(fit_days).sum(), SCALER_PATH)
    return scaler


def _entropy(s: pd.Series) -> float:
    counts = s.value_counts(normalize=True)
    return float(-(counts * np.log2(counts + 1e-12)).sum())


def add_window_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-window destination-port entropy -- partial substitute for the dropped host-level
    port_entropy signal (§1.22 lateral-movement folding is no longer possible, see PLAN.md)."""
    port_entropy = df.groupby("window_id")["Destination Port"].apply(_entropy)
    df = df.merge(port_entropy.rename("window_port_entropy"), on="window_id", how="left")
    return df


def window_labels(df: pd.DataFrame) -> pd.DataFrame:
    """One row per window_id: majority Label (classification head) + any-attack flag
    (anomaly head) + the window's day tag (for the Mon-Fri phase mapping)."""
    grouped = df.groupby("window_id")
    y_anomaly = grouped["Label"].apply(lambda s: float((s != "BENIGN").any()))
    y_class = grouped["Label"].agg(lambda s: s.mode().iloc[0])
    day = grouped["day"].first()
    n_flows = grouped.size()
    out = pd.DataFrame({
        "y_anomaly": y_anomaly, "y_class": y_class, "day": day, "n_flows": n_flows,
    }).reset_index()
    return out.sort_values("window_id").reset_index(drop=True)


def build_features(input_path: Path = INTERIM_PATH) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df = assign_windows(df)
    df = add_window_context_features(df)

    scaler = fit_scaler(df)
    scaled = scaler.transform(df[NUMERIC_FEATURE_COLS])
    scaled_cols = [f"scaled_{c}" for c in NUMERIC_FEATURE_COLS]
    df[scaled_cols] = scaled.astype(np.float32)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    keep_cols = ["window_id", "day", "seq", "Label", "Destination Port",
                 "window_port_entropy"] + scaled_cols
    df[keep_cols].to_parquet(FEATURES_PATH, index=False)
    log.info("Wrote %s (%d rows, %d windows)", FEATURES_PATH, len(df), df["window_id"].nunique())

    labels = window_labels(df)
    labels_path = PROCESSED_DIR / "window_labels.parquet"
    labels.to_parquet(labels_path, index=False)
    log.info("Wrote %s (%d windows)", labels_path, len(labels))
    return df


if __name__ == "__main__":
    df = build_features()
    print(df[["window_id", "day"]].drop_duplicates().groupby("day").size())
    print("windows total:", df["window_id"].nunique())
