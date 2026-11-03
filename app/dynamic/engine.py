"""DynamicRiskEngine: the proposed detector as a drop-in for the original static pipeline.

    engine = DynamicRiskEngine.load("models/dynamic_bundle.pkl")
    result = engine.assess(baseline_snapshot, current_snapshot)

``result`` contains the decision, the estimated damage increase (worker-node equivalents), a severity level,
the rules that fired with their *dynamic* weights, the riskiest attack paths and the generated rule set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from app.dynamic.features import episode_features
from app.dynamic.model import DRWA, HybridModel, TAU, load
from app.dynamic.params import ParameterIdentifier, StateAnalysis
from app.dynamic.rule_generator import DynamicRuleGenerator
from app.dynamic.templates import TEMPLATES, TEMPLATE_IDS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "models" / "dynamic_bundle.pkl"


def severity_from_damage(dd: float) -> str:
    if dd < 0.05:
        return "None"
    if dd < TAU:
        return "Low"
    if dd < 1.0:
        return "High"
    return "Critical"


class DynamicRiskEngine:
    def __init__(self, bundle: dict):
        self.b = bundle
        self.pia: ParameterIdentifier = bundle["pia"]
        self.hybrid: HybridModel = bundle["hybrid"]
        self.drwa: DRWA = bundle["drwa"]
        self.thr: float = bundle["threshold"]
        self.thr_high_recall: float = bundle.get("threshold_high_recall", self.thr)
        self.generator = DynamicRuleGenerator(self.pia, self.drwa, self.thr, bundle.get("induced_rules", []))

    @classmethod
    def load(cls, path=DEFAULT_BUNDLE) -> "DynamicRiskEngine":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Train it first: python -m experiments.run_all")
        return cls(load(p))

    def assess(self, baseline: dict, current: dict, high_recall: bool = False) -> Dict:
        thr = self.thr_high_recall if high_recall else self.thr
        aB = StateAnalysis(baseline)
        aC = StateAnalysis(current, aB.vuln_thr)
        feats = episode_features(aB, aC, baseline, current)
        df = pd.DataFrame([feats])
        p = float(self.hybrid.proba(df)[0])
        g = float(self.hybrid.damage(df)[0])
        dd = float(TAU * np.expm1(abs(g)) * np.sign(g))
        ruleset = self.generator.generate(aB, baseline, current, aC)
        W = self.drwa.weights(ruleset.theta).set_index("template")
        fired: List[dict] = []
        for t in TEMPLATES:
            pos, neg = feats[f"pos_ctx::{t.id}"], feats[f"neg_ctx::{t.id}"]
            if pos > 0 or neg > 0:
                contrib = float(W.loc[t.id, "w_introduced"] * np.log1p(pos) - W.loc[t.id, "w_removed"] * np.log1p(neg))
                new = sorted(set(aC.inst[t.id]) - set(aB.inst[t.id]))[:3]
                gone = sorted(set(aB.inst[t.id]) - set(aC.inst[t.id]))[:3]
                fired.append({"rule": t.id, "tactic": t.tactic, "description": t.description, "contribution": contrib,
                              "weight_introduced": float(W.loc[t.id, "w_introduced"]), "introduced": new, "removed": gone})
        fired.sort(key=lambda r: -abs(r["contribution"]))
        return {
            "alert": p >= thr, "probability": p, "threshold": thr, "estimated_damage_increase": dd,
            "severity": severity_from_damage(dd), "rules_fired": fired, "attack_paths": aC.ag.top_paths(3),
            "active_rules": ruleset.n_active, "active_parameters": ruleset.active_params, "ruleset": ruleset,
            "theta": ruleset.theta,
        }
