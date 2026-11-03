"""Episode features: what changed between baseline B and current C, in rule-instance terms.

For every template t the drift is summarised by four numbers
   pos_raw / neg_raw : intensity of instances introduced / removed (context free)
   pos_ctx / neg_ctx : the same, weighted by (EPS + reach) of the affected pods in the attack graph
Context-free sums reproduce what a static rule library can do; context weighting is the first
dynamic element (the weight of a finding depends on how reachable it is in *this* cluster).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from app.dynamic.params import PARAM_NAMES, StateAnalysis, drift_params
from app.dynamic.templates import TEMPLATES, transform

EPS = 0.15


def _ctx(a: StateAnalysis, pods) -> float:
    if not pods:
        return 0.0
    r = a.ag.reach
    return float(r[list(pods)].max())


def template_deltas(aB: StateAnalysis, aC: StateAnalysis) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for t in TEMPLATES:
        iB, iC = aB.inst[t.id], aC.inst[t.id]
        d = {"pos_raw": 0.0, "neg_raw": 0.0, "pos_ctx": 0.0, "neg_ctx": 0.0, "n_new": 0.0, "n_gone": 0.0}
        if iB is iC or (not iB and not iC):
            out[t.id] = d
            continue
        for k in set(iB) | set(iC):
            xb = transform(t.transform, iB[k][0]) if k in iB else 0.0
            xc = transform(t.transform, iC[k][0]) if k in iC else 0.0
            diff = xc - xb
            if diff > 1e-9:
                d["pos_raw"] += diff
                d["pos_ctx"] += diff * (EPS + _ctx(aC, iC[k][1]))
                d["n_new"] += k not in iB
            elif diff < -1e-9:
                d["neg_raw"] += -diff
                d["neg_ctx"] += -diff * (EPS + _ctx(aB, iB[k][1]))
                d["n_gone"] += k not in iC
        out[t.id] = d
    return out


def episode_features(aB: StateAnalysis, aC: StateAnalysis, B: dict, C: dict) -> Dict[str, float]:
    f: Dict[str, float] = {}
    deltas = template_deltas(aB, aC)
    for tid, d in deltas.items():
        for key in ("pos_raw", "neg_raw", "pos_ctx", "neg_ctx"):
            f[f"{key}::{tid}"] = d[key]
    f["n_new_total"] = sum(d["n_new"] for d in deltas.values())
    f["n_gone_total"] = sum(d["n_gone"] for d in deltas.values())
    f["n_templates_new"] = sum(1 for d in deltas.values() if d["n_new"] > 0)
    f["n_templates_gone"] = sum(1 for d in deltas.values() if d["n_gone"] > 0)
    # attack-graph deltas: extended (proposed) and base-paper graphs
    for agg in ("max", "top5", "sum"):
        f[f"d_ext_{agg}"] = aC.ag.cluster_risk(agg) - aB.ag.cluster_risk(agg)
        f[f"d_p1_{agg}"] = aC.ag_base.cluster_risk(agg) - aB.ag_base.cluster_risk(agg)
    f["d_mean_reach"] = float(aC.ag.reach.mean() - aB.ag.reach.mean()) if aC.view.n_pods and aB.view.n_pods else 0.0
    f["d_high_reach"] = float((aC.ag.reach > 0.5).mean() - (aB.ag.reach > 0.5).mean()) if aC.view.n_pods and aB.view.n_pods else 0.0
    f["d_pods"] = float(aC.view.n_pods - aB.view.n_pods)
    # parameters theta (posture of B + drift statistics)
    theta = dict(aB.posture)
    theta.update(drift_params(B, C))
    for k in PARAM_NAMES:
        f[f"theta::{k}"] = theta[k]
    return f
