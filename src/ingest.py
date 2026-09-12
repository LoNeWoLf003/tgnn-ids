"""Load, clean, and concatenate the CIC-IDS2017 CICFlowMeter CSVs.

Data constraint (see PLAN.md's "Data constraint discovered at implementation time" note):
the CSVs in data/raw/MachineLearningCVE/ contain only `Destination Port` + ~70 CICFlowMeter
flow-statistic columns + `Label` -- no Flow ID / Source IP / Destination IP / Source Port /
Protocol / Timestamp. Chronological order is therefore *assumed* to be row order within each
day-file (CICFlowMeter emits flows in capture order) -- a documented limitation, not fabricated.
A synthetic `seq` column (0..n-1 per day) stands in for a timestamp everywhere downstream needs
"chronological position".
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/MachineLearningCVE")
INTERIM_DIR = Path("data/interim")
OUTPUT_PATH = INTERIM_DIR / "cicids2017_clean.parquet"

# Ordered dict: concat order == assumed chronological day order (Mon -> Fri).
DAY_MAP = {
    "Monday-WorkingHours.pcap_ISCX.csv": "mon",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "tue",
    "Wednesday-workingHours.pcap_ISCX.csv": "wed",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "thu_am",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "thu_pm",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": "fri_am",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "fri_pm_scan",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "fri_pm_ddos",
}

# Known label typos: CICFlowMeter's original em-dash byte was lost to a U+FFFD replacement
# char in this CSV mirror (confirmed via hexdump -- raw bytes are ef bf bd, i.e. UTF-8 encoded
# U+FFFD, not a recoverable mojibake), so match on the replacement character itself.
LABEL_FIXES = {
    "Web Attack � Brute Force": "Web Attack - Brute Force",
    "Web Attack � XSS": "Web Attack - XSS",
    "Web Attack � Sql Injection": "Web Attack - SQL Injection",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_and_clean_day(filename: str, day_tag: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    df.columns = df.columns.str.strip()  # known issue: leading spaces in headers
    # "Fwd Header Length" is duplicated in the raw header (one copy has a leading space, the
    # other doesn't, so they only collide after stripping) -- keep the first occurrence.
    df = df.loc[:, ~df.columns.duplicated()]

    df["Label"] = df["Label"].astype(str).str.strip()
    df["Label"] = df["Label"].replace(LABEL_FIXES)

    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    n_after = len(df)
    log.info(
        "%s (%s): %d -> %d rows after dropna/dedupe, labels=%s",
        filename, day_tag, n_before, n_after, sorted(df["Label"].unique()),
    )

    df["day"] = day_tag
    df["seq"] = np.arange(len(df))  # per-day chronological-order proxy (row order)
    return df.reset_index(drop=True)


def build_dataset() -> pd.DataFrame:
    frames = []
    for filename, day_tag in DAY_MAP.items():
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected {path} -- check data/raw/MachineLearningCVE/")
        frames.append(load_and_clean_day(filename, day_tag))

    full = pd.concat(frames, ignore_index=True)  # day order preserved by dict iteration order
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUTPUT_PATH, index=False)
    log.info("Wrote %s (%d rows, %d cols)", OUTPUT_PATH, *full.shape)
    return full


if __name__ == "__main__":
    df = build_dataset()
    print(df.shape)
    print(df["day"].value_counts())
    print(df["Label"].value_counts())
