"""Compact view of stage-B probe results: which layer, then window x pooling at that layer.

    python -m minns.summarise                 # every probe_*.csv in outputs/results/stage_b
    python -m minns.summarise mhubert-147     # only files whose name contains this
"""
from __future__ import annotations

import sys

import pandas as pd

from minns.data import OUTPUTS


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(f for f in (OUTPUTS / "results/stage_b").glob("probe_*.csv") if pattern in f.name)
    for f in files:
        d = pd.read_csv(f)
        print(f"\n### {f.name}   ({len(d)} configs)")
        print("best fold AUC / within-site AUC per layer (over windows and poolings):")
        print(d.groupby("layer")[["auc", "site_auc"]].max().T.round(3).to_string())
        layer = d.loc[d["auc"].idxmax(), "layer"]
        at = d[d["layer"] == layer]
        for metric in ("auc", "site_auc", "f1"):
            print(f"layer {layer}: {metric} by window x pooling")
            print(at.pivot_table(index="window", columns="pool", values=metric).round(3).to_string())


if __name__ == "__main__":
    main()
