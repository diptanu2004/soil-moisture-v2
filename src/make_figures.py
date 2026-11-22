"""Figures for the report (static PNG, 200 dpi). Each figure is skipped if its
result table does not exist yet."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, PROC, ROOT, TAB

# reference categorical palette (fixed order) + text tokens
C1, C2, C3, C4, C5, C6, C7, C8 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                                  "#6250d6", "#e34948")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
})


def _save(fig, name, legend_below=False):
    if legend_below:
        fig.tight_layout(rect=(0, 0.14, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=200)
    plt.close(fig)
    print("saved", name)


def _states():
    import geopandas as gpd
    s = gpd.read_file(os.path.join(ROOT, "data", "raw", "states.shp"))
    return s[~s["STATE_ABBR"].isin(["AK", "HI", "PR", "VI"])]


# ----------------------------------------------------------------------------
def fig_coordinates():
    p = os.path.join(ROOT, "gee", "stations_paper16_corrected.csv")
    st = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    _states().boundary.plot(ax=ax, color="#c9c8c3", linewidth=0.5)
    for r in st.itertuples():
        ax.annotate("", xy=(r.nrcs_lon, r.nrcs_lat), xytext=(r.paper_lon, r.paper_lat),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=3, shrinkB=3))
    ax.scatter(st.paper_lon, st.paper_lat, s=34, color=C2, edgecolor=SURF, linewidth=1.2, zorder=3,
               label="coordinates used in the paper")
    ax.scatter(st.nrcs_lon, st.nrcs_lat, s=34, color=C1, edgecolor=SURF, linewidth=1.2, zorder=4,
               label="NRCS station metadata")
    for r in st[st.offset_km > 20].itertuples():
        dx, dy, ha = (-0.3, 0.35, "right") if r.name == "Shagbark Hills" else (0.25, -0.45, "left")
        ax.text(r.nrcs_lon + dx, r.nrcs_lat + dy, f"{r.name}\n{r.offset_km:.0f} km", fontsize=8, color=INK, ha=ha)
    ax.set_xlim(-106, -84)
    ax.set_ylim(29.5, 44.5)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Station coordinates in the paper vs. NRCS metadata (arrow = paper -> true)", loc="left")
    ax.legend(loc="lower left")
    _save(fig, "fig01_station_coordinates.png")


def fig_coverage():
    a = pd.read_parquet(os.path.join(PROC, "anchor.parquet")).dropna(subset=["sm_5cm"])
    n = a.groupby("site").size()
    s1 = a[a.site.isin(n[n >= 20].index)].groupby(["site", "region"], as_index=False)[["lat", "lon"]].first()
    d = pd.read_parquet(os.path.join(PROC, "scan_daily.parquet")).dropna(subset=["sm_5cm"])
    sites = pd.read_parquet(os.path.join(PROC, "sites.parquet"))
    dd = sites[sites.site.isin(d.site.unique())]
    paper = pd.read_csv(os.path.join(ROOT, "gee", "stations_paper16_corrected.csv"))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    _states().boundary.plot(ax=ax, color="#c9c8c3", linewidth=0.5)
    ax.scatter(dd.lon, dd.lat, s=14, color="#b9b8b2", label=f"daily in-situ 5 cm (all {dd.site.nunique()} sites)")
    ax.scatter(s1.lon, s1.lat, s=26, color=C1, edgecolor=SURF, linewidth=1, label=f"in-situ + Sentinel-1 ({len(s1)} sites)")
    ax.scatter(paper.nrcs_lon, paper.nrcs_lat, s=60, facecolor="none", edgecolor=C2, linewidth=1.6,
               label="the paper's 16 stations")
    ax.set_title("Stations used in the new experiments (CONUS, 2015-2018)", loc="left")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    _save(fig, "fig02_station_coverage.png")


def fig_exp1():
    p = f"{TAB}/exp1_baselines.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    protos = ["random80/20", "temporal(2018)", "leave-stations-out", "leave-one-region-out"]
    series = [("paper16+lagPrevS1", "Ensemble", "Paper model (with in-situ lag)", C1),
              ("paper16+lagPrevS1", "Persistence(prevS1)", "Persistence baseline (lag only)", "#9a9993"),
              ("paper16-noLag", "Ensemble", "Paper model, no lag (satellite only)", C2)]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    w = 0.26
    x = np.arange(len(protos))
    for i, (fs, m, lab, col) in enumerate(series):
        v = [d[(d.protocol == p_) & (d.features == fs) & (d.model == m)]["R2"].iloc[0] for p_ in protos]
        b = ax.bar(x + (i - 1) * w, v, w - 0.03, color=col, label=lab)
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, max(vi, 0) + 0.02, f"{vi:.2f}", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, ["random 80/20\n(paper)", "temporal\n(train<=2017, test 2018)", "unseen stations\n(10-fold)",
                      "unseen region\n(leave-one-region-out)"])
    ax.set_ylabel("R² (pooled)")
    ax.set_ylim(-0.15, 1.12)
    ax.set_title("The paper's protocol on 84 stations: accuracy depends on the split, and the lag does the work",
                 loc="left", fontsize=10)
    ax.legend(loc="upper center", fontsize=8, ncol=3, bbox_to_anchor=(0.5, 1.0))
    _save(fig, "fig03_exp1_protocols.png")


def fig_exp2a():
    p = f"{TAB}/exp2a_loso_models.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    order = ["RF", "XGB", "LGBM", "TreeEns", "MLP(noSeq)", "TempoSAR", "Hybrid"]
    d = d.set_index("model").loc[order].reset_index()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (col, lab, better) in zip(axs, [("R2", "pooled R²", "higher"), ("site_median_r", "median per-station r", "higher"),
                                            ("RMSE", "pooled RMSE (% vol)", "lower")]):
        cols = [C1 if m in ("TempoSAR", "Hybrid", "MLP(noSeq)") else "#9a9993" for m in d.model]
        ax.barh(d.model, d[col], color=cols, height=0.62)
        for i, v in enumerate(d[col]):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_title(f"{lab} ({better} is better)", loc="left", fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0, d[col].max() * 1.22)
    fig.suptitle("Satellite-only retrieval at unseen stations (10-fold leave-stations-out, 84 sites)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    _save(fig, "fig04_exp2a_models.png")


def fig_exp2b():
    p = f"{TAB}/exp2b_transfer.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[d.region == "ALL"]
    order = ["1", "3", "all"]
    meths = [("Scratch-Trees", "target stations only (trees)", "#9a9993", "--"),
             ("ZeroShot-Trees", "other regions only (trees)", C4, ":"),
             ("Pooled-Trees", "pooled: other regions + target (trees)", C2, "-"),
             ("FT-Deep(L2-SP)", "pretrain other regions -> fine-tune (TempoSAR)", C1, "-"),
             ("Hybrid(FT-Deep+Pooled-Trees)", "hybrid: fine-tuned TempoSAR + pooled trees", C3, "-")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, lab in [(axs[0], "RMSE", "pooled RMSE (% vol), lower is better"),
                         (axs[1], "site_median_r", "median per-station r, higher is better")]:
        for m, l, c, ls in meths:
            g = d[d.method == m].set_index("n_ft_sites").reindex(order)
            ax.plot(range(3), g[col], marker="o", ms=6, lw=2, color=c, ls=ls, label=l)
        ax.set_xticks(range(3), ["1 station", "3 stations", "all stations"])
        ax.set_xlabel("in-situ stations available in the target region")
        ax.set_title(lab, loc="left", fontsize=9.5)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Transfer to a new climate region (5 target regions, unseen test stations)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    _save(fig, "fig05_exp2b_transfer.png", legend_below=True)


def fig_exp2c():
    p = f"{TAB}/exp2c_calibration.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    lines = [("Hybrid", "none", "hybrid, no local data", "#9a9993", "--"),
             ("Hybrid", "bias", "hybrid + bias correction", C2, "-"),
             ("TempoSAR", "fine-tune(L2-SP)", "TempoSAR fine-tuned on k samples", C1, "-"),
             ("Hybrid", "fine-tune+bias", "fine-tuned TempoSAR + bias-corrected trees", C3, "-")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, lab in [(axs[0], "RMSE", "pooled RMSE (% vol), lower is better"),
                         (axs[1], "R2", "pooled R², higher is better")]:
        for m, c_, l, c, ls in lines:
            g = d[(d.model == m) & (d.correction == c_)].sort_values("k")
            ax.plot(g.k, g[col], marker="o", ms=6, lw=2, color=c, ls=ls, label=l)
        ax.set_xticks([5, 10, 20])
        ax.set_xlabel("first k in-situ samples at the new station (S1 dates)")
        ax.set_title(lab, loc="left", fontsize=9.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Adapting to a new station with a short in-situ record", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    _save(fig, "fig06_exp2c_calibration.png", legend_below=True)


def fig_exp3():
    p = f"{TAB}/exp3a_learning_curve_raw.csv"
    if os.path.exists(p):
        d = pd.read_csv(p)
        d = d[d.features == "satOnly"]
        order = ["5", "10", "15", "25", "40", "60", "all"]
        g = d.groupby("n_train_sites")[["RMSE", "site_median_r", "R2"]].agg(["mean", "std"]).reindex(order)
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        x = np.arange(len(order))
        for ax, col, lab in [(axs[0], "R2", "pooled R² at unseen stations"), (axs[1], "site_median_r", "median per-station r")]:
            ax.errorbar(x, g[(col, "mean")], yerr=g[(col, "std")].fillna(0), color=C1, marker="o", ms=6, lw=2, capsize=3)
            ax.set_xticks(x, ["5", "10", "15\n(paper)", "25", "40", "60", "all\n(~76)"])
            ax.set_xlabel("number of training stations")
            ax.set_title(lab, loc="left", fontsize=9.5)
        fig.suptitle("More training stations -> better skill at unseen stations (LightGBM, satellite-only)",
                     x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
        _save(fig, "fig07_exp3a_learning_curve.png")
    p = f"{TAB}/exp3c_cross_region.csv"
    if os.path.exists(p):
        d = pd.read_csv(p)
        m = d.pivot(index="train_region", columns="test_region", values="site_median_r")
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=0.8)
        ax.set_xticks(range(len(m.columns)), m.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(m.index)), m.index)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if v > 0.5 else INK)
        ax.grid(False)
        ax.set_xlabel("test region")
        ax.set_ylabel("train region")
        ax.set_title("Median per-station r, train on one region only", loc="left", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        _save(fig, "fig08_exp3c_cross_region.png")


def fig_exp4():
    p = f"{TAB}/exp4_g2_persistence_horizon.csv"
    if os.path.exists(p):
        h = pd.read_csv(p)
        m = pd.read_csv(f"{TAB}/exp4_g3_lag_mismatch.csv").set_index("model")
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
        ax = axs[0]
        ax.plot(h.horizon_days, h.R2, color=C1, marker="o", ms=6, lw=2)
        for r in h.itertuples():
            ax.text(r.horizon_days, r.R2 + 0.025, f"{r.R2:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8, 12, 30], ["1", "2", "4", "8", "12", "30"])
        ax.set_xlabel("age of the last in-situ value used as 'lag' (days)")
        ax.set_ylabel("R² of plain persistence")
        ax.set_ylim(0.5, 1.02)
        ax.set_title("Persistence skill vs. lag age (188 SCAN sites)", loc="left", fontsize=9.5)
        ax = axs[1]
        labs = ["paper_mismatch", "consistent_lag1d", "persistence"]
        names = ["trained with S1-date lag,\napplied with 1-day lag (paper)", "trained & applied\nwith 1-day lag", "persistence\n(y[t-1])"]
        v = [m.loc[l, "RMSE"] for l in labs]
        r2 = [m.loc[l, "R2"] for l in labs]
        ax.bar(range(3), v, color=[C2, C1, "#9a9993"], width=0.6)
        for i, (vi, ri) in enumerate(zip(v, r2)):
            ax.text(i, vi + 0.06, f"RMSE {vi:.2f}\nR² {ri:.3f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(3), names, fontsize=8)
        ax.set_ylim(0, max(v) * 1.3)
        ax.set_ylabel("daily RMSE at unseen stations (% vol)")
        ax.set_title("Lag-definition mismatch", loc="left", fontsize=9.5)
        _save(fig, "fig09_exp4_lag.png")
    p = f"{TAB}/exp4_g4_series.parquet"
    if os.path.exists(p):
        S = pd.read_parquet(p)
        site = 2068 if 2068 in S.site.unique() else S.groupby("site").size().idxmax()
        g = S[S.site == site].sort_values("date")
        fig, ax = plt.subplots(figsize=(10.5, 3.4))
        ax.plot(g.date, g.sm_5cm, color=INK2, lw=1.2, label="SCAN in-situ 5 cm")
        ax.plot(g.date, g.linear, color=C1, lw=2, label="satellite-only: anchors + linear interpolation")
        ax.set_ylabel("soil moisture (% vol)")
        ax.set_title(f"Satellite-only daily reconstruction at an unseen station (SCAN {site})", loc="left", fontsize=10)
        ax.legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.1))
        _save(fig, "fig10_exp4_satellite_only_series.png")


if __name__ == "__main__":
    fig_coordinates()
    fig_coverage()
    fig_exp1()
    fig_exp2a()
    fig_exp2b()
    fig_exp2c()
    fig_exp3()
    fig_exp4()
