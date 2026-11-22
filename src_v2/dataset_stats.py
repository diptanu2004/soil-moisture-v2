"""Dataset counts used in the write-up. Writes results_v2/tables/v2_0_dataset.csv"""
from __future__ import annotations

import os

import pandas as pd

from common_v2 import PROC, TAB, TARGET, load_daily
from exp_v2_2_protocols import load_eval_rows

rows = []
st = pd.read_csv(os.path.join(PROC, "stations.csv"), dtype={"sid": str})
rows.append(("SCAN stations requested (CONUS)", (st["set"] == "scan").sum()))
ins = pd.read_parquet(os.path.join(PROC, "insitu_daily.parquet"))
rows.append(("stations with any QC'd daily 5 cm value 2019-2023", ins.dropna(subset=[TARGET]).sid.nunique()))
rows.append(("QC'd daily 5 cm values 2019-2023", ins[TARGET].notna().sum()))
a = pd.read_parquet(os.path.join(PROC, "anchor.parquet"))
rows.append(("Sentinel-1 station-days (all stations)", len(a)))
rows.append(("Sentinel-1 station-days with in-situ 5 cm", a[TARGET].notna().sum()))
for y, g in a.groupby(a.date.dt.year):
    rows.append((f"Sentinel-1 station-days {y} (S1A / S1B)", f"{(g.platform == 'A').sum()} / {(g.platform == 'B').sum()}"))
    rows.append((f"median days between S1 acquisitions {y}", g.days_since_prev_s1.median()))
d = load_eval_rows()
rows.append(("anchor evaluation rows (>=20 labelled S1 days per station, both lags present)", len(d)))
rows.append(("anchor evaluation stations", d.site.nunique()))
rows.append(("climate regions", d.region.nunique()))
for r, g in d.groupby("region"):
    rows.append((f"region {r}: stations / rows", f"{g.site.nunique()} / {len(g)}"))
rows.append(("temporal split: training rows 2019-2021", (d.date < "2022-01-01").sum()))
rows.append(("temporal split: test rows 2022-2023", (d.date >= "2022-01-01").sum()))
dl = load_daily()
dl = dl[dl.groupby("site")[TARGET].transform(lambda s: s.notna().sum()) >= 30]
rows.append(("daily experiment stations (>=30 labelled days)", dl.site.nunique()))
rows.append(("daily experiment labelled station-days", dl[TARGET].notna().sum()))
pa = pd.read_parquet(os.path.join(PROC, "paper_anchor.parquet"))
p16 = pa[pa.paper16.astype(bool) & ~pa.site.astype(str).str.endswith("_pc")]
rows.append(("paper's 16 stations: labelled S1 days (NRCS coordinates)", p16[TARGET].notna().sum()))
out = pd.DataFrame(rows, columns=["quantity", "value"])
out.to_csv(os.path.join(TAB, "v2_0_dataset.csv"), index=False)
print(out.to_string())
