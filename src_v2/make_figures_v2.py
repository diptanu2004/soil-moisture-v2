"""Figures for results_v2 (every value read from results_v2/tables). Missing tables are skipped."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common_v2 import FIG, ROOT, TAB  # noqa: E402

# reference categorical palette (light mode), fixed slot order; gray reserved for baselines
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


def _t(name):
    p = os.path.join(TAB, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def _bars(ax, labels, values, colors, fmt="{:.2f}"):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62, edgecolor="white", linewidth=1)
    for yi, v in zip(y, values):
        if np.isfinite(v):
            ax.text(max(v, 0) + 0.01, yi, fmt.format(v), va="center", ha="left", color=INK, fontsize=8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#b5b4ae", lw=0.8)


# ----------------------------------------------------------------------------
def fig_before_after():
    a, b2, b3 = _t("v2_1a_before_paper_protocol.csv"), _t("v2_2_protocols.csv"), _t("v2_3_models.csv")
    if a is None or b2 is None:
        return
    rows = []
    r = a[(a.coords == "paper") & (a.features == "paper16 (+lag prevS1)") & (a.model == "Ensemble")]
    for ev, lab in (("random 20% test rows (15 stations)", "Paper protocol: random 80/20, 15 stations"),
                    ("Shagbark Hills (unseen station)", "Paper protocol: Shagbark Hills")):
        x = r[r.evaluation == ev]
        if len(x):
            rows.append((lab, x.R2.iloc[0], C[1]))
    lo = b2[b2.protocol == "leave-stations-out"]
    for feats, model, lab, col in (
            ("paper16 as published (+lag prevS1)", "Ensemble", "Paper model, unseen stations (with lag)", C[1]),
            ("baseline", "Persistence(prevS1)", "Persistence (in-situ at previous S1 date)", GRAY),
            ("paper16 as published, no lag", "Ensemble", "Paper model, unseen stations, satellite-only", C[1]),
            ("improved v2, satellite-only", "Ensemble", "Improved features + trees, satellite-only", C[0])):
        x = lo[(lo.features == feats) & (lo.model == model)]
        if len(x):
            rows.append((lab, x.R2.iloc[0], col))
    if b3 is not None:
        x = b3[(b3.protocol == "leave-stations-out") & (b3.model == "Hybrid(TabPFN+TempoSAR+Trees)")]
        if len(x):
            rows.append(("Proposed hybrid (TabPFN+TempoSAR+trees), satellite-only", x.R2.iloc[0], C[2]))
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 0.8))
    _bars(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
    ax.set_xlabel("R² (5 cm soil moisture, 2019–2023)")
    ax.set_title("Before vs after: the paper's headline vs honest evaluation", loc="left", color=INK, fontsize=10)
    _save(fig, "fig_v2_01_before_after.png")


def fig_fix_ladder():
    b = _t("v2_1b_fix_ladder.csv")
    if b is None:
        return
    steps = list(dict.fromkeys(b.step))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, proto in zip(axes, ("random80/20", "leave-one-station-out")):
        g = b[b.protocol == proto]
        for i, (var, col, lab) in enumerate((("with lag", C[1], "model with lag"),
                                             ("no lag", C[0], "model without lag (satellite-only)"))):
            v = [g[(g.step == s) & (g.variant == var)].R2.mean() for s in steps]
            ax.plot(range(len(steps)), v, "-o", color=col, lw=2, ms=5, label=lab)
        per = [g[(g.step == s) & g.variant.str.startswith("persistence")].R2.mean() for s in steps]
        ax.plot(range(len(steps)), per, "--", color=GRAY, lw=1.5, label="persistence (the lag itself)")
        ax.set_xticks(range(len(steps)), [s.split(" ", 1)[0] for s in steps])
        ax.set_title(proto, loc="left", color=INK, fontsize=10)
    axes[0].set_ylabel("R² (paper's 16 stations)")
    axes[0].set_ylim(min(0, axes[0].get_ylim()[0]), 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.text(0, -0.16, "  ".join(f"{s.split(' ', 1)[0]}: {s.split(' ', 1)[1]}" for s in steps), fontsize=7, color=INK2,
             wrap=True)
    _save(fig, "fig_v2_02_fix_ladder.png")


def fig_protocols():
    b = _t("v2_2_protocols.csv")
    if b is None:
        return
    protos = [p for p in ("random80/20", "leave-stations-out", "leave-one-region-out", "temporal(2022-23)")
              if p in set(b.protocol)]
    sets = [("paper16 as published, no lag", C[1]), ("paper16 fixed, no lag", C[3]),
            ("improved v2, satellite-only", C[0])]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.26
    for i, (s, col) in enumerate(sets):
        v = [b[(b.protocol == p) & (b.features == s) & (b.model == "Ensemble")].R2.mean() for p in protos]
        x = np.arange(len(protos)) + (i - 1) * w
        ax.bar(x, v, w * 0.92, color=col, label=s)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.01 if vi >= 0 else vi - 0.04, f"{vi:.2f}", ha="center", fontsize=7, color=INK)
    ax.axhline(0, color="#b5b4ae", lw=0.8)
    ax.set_xticks(np.arange(len(protos)), protos)
    ax.set_ylabel("R², satellite-only (no in-situ input)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Satellite-only skill by evaluation protocol (all stations)", loc="left", color=INK, fontsize=10)
    _save(fig, "fig_v2_03_protocols_satellite_only.png")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    sets = [("paper16 as published (+lag prevS1)", "Persistence(prevS1)", C[1], "paper model + lag at previous S1 date"),
            ("improved v2 (+lag 1d)", "Persistence(1d)", C[0], "improved + consistent 1-day lag")]
    for i, (s, per, col, lab) in enumerate(sets):
        v = [b[(b.protocol == p) & (b.features == s) & (b.model == "Ensemble")].R2.mean() for p in protos]
        pv = [b[(b.protocol == p) & (b.model == per)].R2.mean() for p in protos]
        x = np.arange(len(protos)) + (i - 0.5) * 0.36
        ax.bar(x, v, 0.33, color=col, label=lab)
        ax.scatter(x, pv, marker="_", s=400, color=INK, zorder=3, label="its persistence baseline" if i == 0 else None)
    ax.set_xticks(np.arange(len(protos)), protos)
    ax.set_ylim(0, 1)
    ax.set_ylabel("R² (with in-situ lag)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_title("With the in-situ lag, models barely beat persistence", loc="left", color=INK, fontsize=10)
    _save(fig, "fig_v2_04_protocols_with_lag.png")


def fig_models():
    b = _t("v2_3_models.csv")
    if b is None:
        return
    models = [m for m in ("Trees", "TempoSAR", "TabPFN", "Hybrid(TempoSAR+Trees)", "Hybrid(TabPFN+Trees)",
                          "Hybrid(TabPFN+TempoSAR+Trees)") if m in set(b.model)]
    protos = list(dict.fromkeys(b.protocol))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, met, lab in zip(axes, ("R2", "site_median_r"), ("R² (pooled)", "median per-station r")):
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            v = [b[(b.protocol == p) & (b.model == m)][met].mean() for p in protos]
            ax.bar(np.arange(len(protos)) + (i - (len(models) - 1) / 2) * w, v, w * 0.92, color=C[i], label=m)
        short = {"leave-stations-out": "unseen stations", "leave-one-region-out": "unseen region",
                 "temporal(2022-23)": "unseen years\n(2022-23)"}
        ax.set_xticks(np.arange(len(protos)), [short.get(p, p) for p in protos], fontsize=8)
        ax.set_ylabel(lab)
        ax.axhline(0, color="#b5b4ae", lw=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Satellite-only models at unseen stations / regions / years", x=0.01, ha="left", color=INK, fontsize=10)
    _save(fig, "fig_v2_05_models.png")


def fig_fewshot():
    b = _t("v2_4_fewshot.csv")
    if b is None:
        return
    lines = [("Hybrid(TabPFN+Trees)|none", GRAY, "no adaptation (hybrid)"),
             ("Trees|bias", C[3], "trees + bias correction"),
             ("FT-TempoSAR", C[1], "TempoSAR fine-tuned (L2-SP)"),
             ("TabPFN in-context", C[2], "TabPFN in-context"),
             ("Hybrid(TabPFN in-context+Trees|bias)", C[0], "TabPFN in-context + trees(bias)")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, met, lab in zip(axes, ("R2", "ubRMSE"), ("R²", "ubRMSE (% vol)")):
        for m, col, l in lines:
            g = b[b.method == m].sort_values("k")
            if len(g):
                ax.plot(g.k, g[met], "-o", color=col, lw=2, ms=5, label=l)
        ax.set_xticks(sorted(b.k.unique()))
        ax.set_xlabel("in-situ samples from the new station (k)")
        ax.set_ylabel(lab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Few-shot adaptation to a new station (evaluated on its later dates)", x=0.01, ha="left",
                 color=INK, fontsize=10)
    _save(fig, "fig_v2_06_fewshot.png")


def fig_daily():
    b = _t("v2_5_daily.csv")
    if b is None:
        return
    g = b[b.eval_set.str.startswith("common")]
    order = g.sort_values("R2", ascending=False)
    cols = [GRAY if "Persistence" in m or "+lag" in m else C[0] if "DailyLSTM" in m else C[3] if "LGBM" in m else C[2]
            for m in order.model]
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(order) + 0.8))
    _bars(ax, list(order.model), list(order.R2), cols)
    ax.set_xlabel("R², every labelled day at unseen stations (gray = uses in-situ yesterday)")
    ax.set_title("Daily reconstruction at unseen stations", loc="left", color=INK, fontsize=10)
    _save(fig, "fig_v2_07_daily_models.png")

    p = os.path.join(TAB, "v2_5_daily_preds.parquet")
    if os.path.exists(p):
        R = pd.read_parquet(p)
        sat = [c for c in R.columns if c.startswith("DailyLSTM satOnly (pretrained ERA5")]
        if sat:
            s = R.groupby("site")[ "sm_5cm"].count().sort_values().index[-1]
            x = R[(R.site == s)].sort_values("date")
            x = x[x.date.dt.year.isin([2021, 2022])]
            fig, ax = plt.subplots(figsize=(9, 2.8))
            ax.plot(x.date, x.sm_5cm, color=INK, lw=1.2, label="in-situ (SCAN, 5 cm)")
            ax.plot(x.date, x[sat[0]], color=C[0], lw=1.5, label="DailyLSTM, satellite-only (station unseen)")
            ax.set_ylabel("soil moisture (% vol)")
            ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
            ax.set_title(f"Daily series at held-out SCAN station {s} (station with the longest record)", loc="left",
                         color=INK, fontsize=10)
            _save(fig, "fig_v2_08_daily_example.png")


def fig_daily_loop():
    p = os.path.join(TAB, "v2_1c_daily_loop_series.parquet")
    if not os.path.exists(p):
        return
    S = pd.read_parquet(p)
    x = S[S.coords == "paper"]
    if x.empty:
        return
    yr = x.dropna(subset=["sm_5cm"]).date.dt.year.value_counts().idxmax()
    x = x[x.date.dt.year == yr]
    fig, ax = plt.subplots(figsize=(9, 3.0))
    base = x[x["mode"] == x["mode"].iloc[0]].sort_values("date")
    ax.plot(base.date, base.sm_5cm, color=INK, lw=1.2, label="in-situ")
    for (m, g), col in zip(x.groupby("mode"), (C[0], C[1], C[2])):
        g = g.sort_values("date")
        ax.plot(g.date, g.pred, color=col, lw=1.3, label=m)
    ax.set_ylabel("soil moisture (% vol)")
    ax.legend(fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    ax.set_title(f"Paper's daily loop at Shagbark Hills (paper coordinates, {yr})", loc="left", color=INK,
                 fontsize=10)
    _save(fig, "fig_v2_09_daily_loop_shagbark.png")


def fig_map():
    p = os.path.join(TAB, "v2_3_oof_leave-stations-out.parquet")
    st = os.path.join(ROOT, "data", "processed_v2", "stations.csv")
    if not (os.path.exists(p) and os.path.exists(st)):
        return
    import geopandas as gpd
    from common_v2 import per_site_metrics
    O = pd.read_parquet(p)
    m = "Hybrid(TabPFN+TempoSAR+Trees)" if "Hybrid(TabPFN+TempoSAR+Trees)" in O else "Trees"
    ps = per_site_metrics(O, m).merge(pd.read_csv(st, dtype={"sid": str}), left_on="site", right_on="sid")
    states = gpd.read_file(os.path.join(ROOT, "data", "raw", "states.shp")).to_crs(4326)
    states = states.cx[-125:-66, 24:50]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    states.boundary.plot(ax=ax, color="#d6d5d0", lw=0.5)
    sc = ax.scatter(ps.lon, ps.lat, c=ps.r, cmap="Blues", vmin=0, vmax=1, s=26, edgecolor="white", linewidth=0.8)
    pp = ps[ps.paper16.astype(str) == "True"]
    ax.scatter(pp.lon, pp.lat, facecolor="none", edgecolor=C[1], s=70, lw=1.5, label="paper's 16 stations")
    plt.colorbar(sc, ax=ax, shrink=0.7, label="per-station r (station unseen in training)")
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect(1.25)
    ax.grid(False)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"{len(ps)} SCAN stations, satellite-only {m}", loc="left", color=INK, fontsize=10)
    _save(fig, "fig_v2_10_station_map.png")


def fig_ablation():
    b = _t("v2_6_ablation.csv")
    if b is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, proto in zip(axes, ("leave-stations-out", "leave-one-region-out")):
        g = b[b.protocol == proto]
        cols = [GRAY if "raw" in m else C[0] if m == "all improved v2 features" else C[3] for m in g.model]
        _bars(ax, list(g.model), list(g.R2), cols)
        ax.set_title(proto, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("R², satellite-only (gray = operational product used directly)")
    fig.suptitle("Where the satellite-only skill comes from (trees; one feature group removed at a time)", x=0.01,
                 ha="left", color=INK, fontsize=10)
    _save(fig, "fig_v2_11_ablation.png")


if __name__ == "__main__":
    for f in (fig_before_after, fig_fix_ladder, fig_protocols, fig_models, fig_fewshot, fig_daily, fig_daily_loop,
              fig_map, fig_ablation):
        try:
            f()
        except Exception as e:  # keep going; report
            print("FAILED", f.__name__, repr(e))
