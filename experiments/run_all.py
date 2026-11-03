"""Fit the proposed models and produce every result table (CSV) in results/tables.

    python -m experiments.run_all [--data results/data] [--out results]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from app.dynamic.model import (DRWA, EVIDENCE_COLS, FULL_COLS, GRAPH_COLS, THETA_COLS, W0, HybridModel,  # noqa: E402
                               RuleInducer, best_threshold, cols, log_damage, metrics, save)
from app.dynamic.params import PARAM_NAMES, ParameterIdentifier  # noqa: E402
from app.dynamic.templates import TEMPLATES, TEMPLATE_IDS  # noqa: E402

AGGS = ["max", "top5", "sum"]


def load_splits(d: Path, real_only: bool = False) -> dict:
    if real_only:
        return {k: pd.read_pickle(d / f"real_{k}.pkl") for k in ["train", "val", "test"]}
    return {s: pd.read_pickle(d / f"{s}.pkl") for s in ["train", "val", "test", "test_large"] if (d / f"{s}.pkl").exists()}


def prior_score(df, ctx=True, use_neg=True):
    """Table VI weights, no learning: sum w0 * log1p(evidence)."""
    p = "ctx" if ctx else "raw"
    s = (np.log1p(df[cols(f"pos_{p}")].values) * W0).sum(1)
    if use_neg:
        s = s - (np.log1p(df[cols(f"neg_{p}")].values) * W0).sum(1)
    return s


class Method:
    """name, score fn, threshold (native binary alert or tuned on validation)."""

    def __init__(self, name, score_val, score_fn, thr=None, native=None):
        self.name, self.score_fn, self.native = name, score_fn, native
        self.thr = thr if thr is not None else best_threshold(score_val[0], score_val[1])

    def evaluate(self, df):
        s = self.score_fn(df)
        return metrics(df["dangerous"].values, s, self.thr, df["dD"].values)


def build_methods(S, hybrid, drwa, rinduce):
    v = S["val"]
    yv = v["dangerous"].values
    m = []
    m.append(Method("Existing: static rules (original project, 5 rules)", None, lambda d: d["static_alert"].astype(float).values, thr=0.5))
    m.append(Method("Existing: static rules (score, tuned thr)", (yv, v["static_score"].values), lambda d: d["static_score"].values))
    best_agg = max(AGGS, key=lambda a: metrics(yv, v[f"p1_{a}"].values, best_threshold(yv, v[f"p1_{a}"].values))["f1"])
    m.append(Method(f"Existing: Paper-1 attack-graph risk delta (agg={best_agg})", (yv, v[f"p1_{best_agg}"].values),
                    lambda d, a=best_agg: d[f"p1_{a}"].values))
    m.append(Method("Existing: Paper-2 misconfig rules M1-M7 (alert)", None, lambda d: d["p2_alert"].astype(float).values, thr=0.5))
    m.append(Method("Existing: Paper-2 rules (count, tuned thr)", (yv, v["p2_score"].values), lambda d: d["p2_score"].values))
    m.append(Method("Proposed: 54 rules, prior weights (static)", (yv, prior_score(v, False)), lambda d: prior_score(d, False)))
    m.append(Method("Proposed: + graph context", (yv, prior_score(v, True)), lambda d: prior_score(d, True)))
    m.append(Method("Proposed: + DRWA dynamic weights", (yv, drwa.predict(v)), drwa.predict))
    m.append(Method("Proposed: DRWA + GBM hybrid (full)", (yv, hybrid.proba(v)), hybrid.proba))
    m.append(Method("Proposed: induced compound rules (tree)", (yv, rinduce.score(v)), rinduce.score))
    return m


def table(methods, dfs: dict) -> pd.DataFrame:
    rows = []
    for m in methods:
        for name, df in dfs.items():
            r = {"method": m.name, "test_set": name, "n": len(df)}
            r.update(m.evaluate(df))
            rows.append(r)
    return pd.DataFrame(rows)


def fit_models(tr, va, seed=0, phi_k=10, quick=False):
    y, dD = tr["dangerous"].values, tr["dD"].values
    hy = HybridModel(seed=seed).fit(tr, y, dD)
    rel = hy.param_relevance(va, va["dangerous"].values)
    phi = sorted(rel, key=rel.get, reverse=True)[:phi_k]
    best = None
    for a in ([30.0] if quick else [10.0, 30.0, 100.0, 300.0]):
        d = DRWA(phi, a).fit(tr, dD)
        sc = d.predict(va)
        f = metrics(va["dangerous"].values, sc, best_threshold(va["dangerous"].values, sc))["f1"]
        if best is None or f > best[0]:
            best = (f, d)
    dr = best[1]
    ri = RuleInducer().fit(tr, y, THETA_COLS)
    return hy, dr, ri, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results" / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--real-only", action="store_true", help="train/val/test on real_train/real_val/real_test.pkl (split by chart)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    T = out / ("tables_real" if a.real_only else "tables")
    T.mkdir(parents=True, exist_ok=True)
    S = load_splits(Path(a.data), a.real_only)
    tr, va, te = S["train"], S["val"], S["test"]
    tl = S.get("test_large")
    print({k: (len(v), round(v['dangerous'].mean(), 3)) for k, v in S.items()})

    # dataset summary ------------------------------------------------------------------------
    summ = []
    for k, v in S.items():
        summ.append({"split": k, "episodes": len(v), "clusters": v["cluster_id"].nunique(), "dangerous_rate": v["dangerous"].mean(),
                     "pods_min": v["n_pods"].min(), "pods_median": v["n_pods"].median(), "pods_max": v["n_pods"].max(),
                     "k8s_nodes_max": v["n_nodes"].max(), **{f"sev_{s}": (v["severity"] == s).mean() for s in ["none", "low", "high", "critical"]}})
    pd.DataFrame(summ).to_csv(T / "dataset_summary.csv", index=False)

    # PIA + models -----------------------------------------------------------------------------
    pia = ParameterIdentifier().fit_reference([{p: r[f"theta::{p}"] for p in PARAM_NAMES} for _, r in tr.iterrows()])
    hy, dr, ri, rel = fit_models(tr, va, quick=a.quick)
    pia.relevance = rel
    methods = build_methods(S, hy, dr, ri)
    dfs = {"test (in-range)": te}
    if tl is not None:
        dfs["test_large (400-1500 pods)"] = tl
    real_p = Path(a.data) / "real.pkl"
    if real_p.exists() and not a.real_only:
        dfs["REAL manifests (out-of-distribution)"] = pd.read_pickle(real_p)
    main_t = table(methods, dfs)
    main_t.to_csv(T / "main_comparison.csv", index=False)
    print(main_t[["method", "test_set", "precision", "recall", "f1", "roc_auc", "fpr"]].round(3).to_string())

    # parameter relevance ----------------------------------------------------------------------
    pd.DataFrame(sorted(rel.items(), key=lambda kv: -kv[1]), columns=["parameter", "relevance_auc_drop"]).to_csv(T / "parameter_relevance.csv", index=False)

    # per-scenario recall / flag-rate -------------------------------------------------------------
    hyb = next(m for m in methods if m.name.startswith("Proposed: DRWA + GBM"))
    stat = methods[0]
    p1m = methods[2]
    p2m = methods[3]
    rows = []
    for (cat, sc), g in te.groupby(["category", "scenario"]):
        row = {"category": cat, "scenario": sc, "n": len(g), "dangerous_rate": g["dangerous"].mean()}
        for lab, m in [("static", stat), ("paper1", p1m), ("paper2", p2m), ("proposed", hyb)]:
            pred = m.score_fn(g) >= m.thr
            row[f"{lab}_flag_rate"] = pred.mean()
            pos = g["dangerous"] == 1
            row[f"{lab}_recall"] = pred[pos.values].mean() if pos.any() else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(T / "per_scenario.csv", index=False)
    cat_rows = []
    for cat, g in te.groupby("category"):
        for m in [stat, p1m, p2m, hyb]:
            r = m.evaluate(g)
            cat_rows.append({"category": cat, "method": m.name, "n": len(g), "dangerous_rate": g["dangerous"].mean(), **r})
    pd.DataFrame(cat_rows).to_csv(T / "per_category.csv", index=False)

    # per-profile -------------------------------------------------------------------------------
    pr = []
    for prof, g in te.groupby("profile"):
        for m in [stat, p1m, hyb]:
            pr.append({"profile": prof, "method": m.name, "n": len(g), **m.evaluate(g)})
    pd.DataFrame(pr).to_csv(T / "per_profile.csv", index=False)

    # node-scaling study --------------------------------------------------------------------------
    rows = []
    n_ep = min(1000, len(tr))
    rng = np.random.default_rng(0)
    for cap in [20, 40, 80, 160, 300]:
        sub_all = tr[tr["n_pods"] <= cap]
        sub = sub_all.sample(min(n_ep, len(sub_all)), random_state=0)
        h, d, r_, _ = fit_models(sub, va, quick=True)
        for nm, df in dfs.items():
            thr = best_threshold(va["dangerous"].values, h.proba(va))
            r = metrics(df["dangerous"].values, h.proba(df), thr, df["dD"].values)
            rows.append({"max_pods_in_training": cap, "train_episodes": len(sub), "train_clusters": sub["cluster_id"].nunique(),
                         "train_median_pods": sub["n_pods"].median(), "test_set": nm, **r})
    pd.DataFrame(rows).to_csv(T / "node_scaling.csv", index=False)
    print(pd.DataFrame(rows)[["max_pods_in_training", "test_set", "f1", "roc_auc"]].round(3).to_string())

    # MLP hidden-width study -------------------------------------------------------------------------
    sc = StandardScaler().fit(tr[FULL_COLS].values)
    Xtr, Xva = sc.transform(tr[FULL_COLS].values), sc.transform(va[FULL_COLS].values)
    rows = []
    for w in [2, 4, 8, 16, 32, 64, 128, 256]:
        mlp = MLPClassifier((w,), max_iter=400, early_stopping=True, random_state=0).fit(Xtr, tr["dangerous"].values)
        thr = best_threshold(va["dangerous"].values, mlp.predict_proba(Xva)[:, 1])
        for nm, df in dfs.items():
            s = mlp.predict_proba(sc.transform(df[FULL_COLS].values))[:, 1]
            rows.append({"hidden_nodes": w, "test_set": nm, **metrics(df["dangerous"].values, s, thr, df["dD"].values)})
    pd.DataFrame(rows).to_csv(T / "mlp_width.csv", index=False)

    # leave-one-profile-out -------------------------------------------------------------------------
    pool = pd.concat([tr, va])
    if pool["profile"].nunique() < 2:
        print("only one environment profile in the data -> leave-one-profile-out skipped")
    else:
        rows = []
        for prof in sorted(pool["profile"].unique()):
            trn, hold = pool[pool["profile"] != prof], pool[pool["profile"] == prof]
            t2, v2 = trn.sample(frac=0.85, random_state=0), None
            v2 = trn.drop(t2.index)
            h, d, r_, _ = fit_models(t2, v2, quick=True)
            thr = best_threshold(v2["dangerous"].values, h.proba(v2))
            fixed = prior_score
            thr_p = best_threshold(v2["dangerous"].values, prior_score(v2, False))
            thr_1 = None
            yv2 = v2["dangerous"].values
            agg = "top5"
            thr_1 = best_threshold(yv2, v2[f"p1_{agg}"].values)
            for lab, s, t in [("Proposed (hybrid)", h.proba(hold), thr), ("Static Table-VI weights", prior_score(hold, False), thr_p),
                              ("Paper-1 graph delta", hold[f"p1_{agg}"].values, thr_1), ("Existing static rules", hold["static_alert"].astype(float).values, 0.5)]:
                rows.append({"held_out_profile": prof, "method": lab, "n": len(hold), **metrics(hold["dangerous"].values, s, t, hold["dD"].values)})
        pd.DataFrame(rows).to_csv(T / "leave_one_profile_out.csv", index=False)

    # dynamic rule behaviour --------------------------------------------------------------------------
    # weights of every rule evaluated at the mean theta of each profile
    prof_theta = {p: {k: g[f"theta::{k}"].mean() for k in PARAM_NAMES} for p, g in pool.groupby("profile")}
    wrows = []
    for p, th in prof_theta.items():
        W = dr.weights(th)
        for _, r in W.iterrows():
            wrows.append({"profile": p, "rule": r["template"], "prior_w0": r["prior_w0"], "w_introduced": r["w_introduced"], "w_removed": r["w_removed"]})
    wd = pd.DataFrame(wrows)
    wd.to_csv(T / "rule_weights_by_profile.csv", index=False)
    piv = wd.pivot(index="rule", columns="profile", values="w_introduced")
    piv["spread"] = piv.max(axis=1) - piv.min(axis=1)
    piv["prior_w0"] = wd.groupby("rule")["prior_w0"].first()
    piv.sort_values("spread", ascending=False).to_csv(T / "rule_weight_spread.csv")

    # induced rules and rule counts -------------------------------------------------------------------------
    induced = ri.rules(va, va["dangerous"].values)
    pd.DataFrame(induced).to_csv(T / "induced_rules.csv", index=False)
    fam = pd.Series([t.family for t in TEMPLATES]).value_counts()
    cnt = [{"item": "Existing static rules", "count": 5}] + [{"item": f"Template rules ({k})", "count": int(v)} for k, v in fam.items()] + \
          [{"item": "Total template rules", "count": len(TEMPLATES)}, {"item": "Induced compound rules (this run)", "count": len(induced)},
           {"item": "Environment parameters tracked", "count": len(PARAM_NAMES)}]
    pd.DataFrame(cnt).to_csv(T / "rule_counts.csv", index=False)
    avg_active = float(np.mean([(np.maximum(dr.weights(th)["w_introduced"], dr.weights(th)["w_removed"]) >= 0.02).sum() for th in prof_theta.values()]))

    # bundle ------------------------------------------------------------------------------------------------------
    (ROOT / "models").mkdir(exist_ok=True)
    sv = hy.proba(va); yv_ = va["dangerous"].values
    ops = []
    for tgt in (0.7, 0.8, 0.9):
        cands = [t for t in np.unique(sv) if (sv[yv_ == 1] >= t).mean() >= tgt]
        thr_r = float(max(cands)) if cands else float(sv.min())
        for nm, df in dfs.items():
            ops.append({"operating_point": f"validation recall >= {tgt}", "threshold": thr_r, "test_set": nm, **metrics(df["dangerous"].values, hy.proba(df), thr_r, df["dD"].values)})
        if tgt == 0.8:
            thr_hi = thr_r
    ops.append({"operating_point": "max F1 (default)", "threshold": hyb.thr, "test_set": "test (in-range)", **hyb.evaluate(te)})
    pd.DataFrame(ops).to_csv(T / "operating_points.csv", index=False)
    save({"pia": pia, "hybrid": hy, "drwa": dr, "threshold": hyb.thr, "threshold_high_recall": thr_hi, "induced_rules": induced, "avg_active_rules": avg_active},
         ROOT / "models" / ("dynamic_bundle_real.pkl" if a.real_only else "dynamic_bundle.pkl"))
    print("done; tables in", T)


if __name__ == "__main__":
    main()
