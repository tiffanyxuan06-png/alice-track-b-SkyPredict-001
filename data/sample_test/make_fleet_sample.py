"""Build a realistic fleet sample for the dashboard's Fleet Prioritization page.

Takes the raw C-MAPSS FD001 *test* set (100 engines, each a partial run) and
keeps one row per engine: its LATEST observed cycle — the point at which you'd
decide what to inspect next. Unlike data/processed/test.csv, this keeps the
`unit_number` and `time_in_cycles` identifiers so the ranked table shows real
engine IDs.

Run:
    python data/sample_test/make_fleet_sample.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COL_NAMES = (
    ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw" / "CMAPSSData" / "test_FD001.txt"
OUT = REPO_ROOT / "data" / "sample_test" / "fleet_latest.csv"


def main() -> None:
    df = pd.read_csv(RAW, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")  # drop any empty trailing columns
    df.columns = COL_NAMES

    # One row per engine: the last cycle we observed for it.
    latest = df.loc[df.groupby("unit_number")["time_in_cycles"].idxmax()]
    latest = latest.sort_values("unit_number").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(OUT, index=False)
    print(f"Wrote {len(latest)} engines x {latest.shape[1]} columns -> {OUT}")


if __name__ == "__main__":
    main()
