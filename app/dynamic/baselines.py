"""The three systems the proposed solution is compared against.

1. ``StaticRulesBaseline``  - the ORIGINAL project pipeline, run unmodified:
   DriftDetectionEngine -> SecurityRuleEngine(rules/security_rules.yaml, 5 static rules) ->
   AttackPathAnalyzer -> calculate_risk. Native decision: any matched rule creates an alert.
2. ``Paper1Baseline``       - Blaise & Rebecchi: graph + Table VI scores + riskiest path (analysis_graph,
   extended=False). Drift score = change of the cluster path risk between B and C.
3. ``Paper2Baseline``       - Bufalino et al.: the M1-M7 misconfiguration checks; drift score = number of
   misconfiguration instances introduced by the drift (paper 2 itself assesses a single state).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from app.alerts import AlertManager  # noqa: F401  (kept to show the original alert path is unchanged)
from app.attack_path_analyzer import AttackPathAnalyzer
from app.drift_detector import DriftDetectionEngine
from app.risk_assesment import calculate_risk
from app.rule_engine import SecurityRuleEngine
from app.dynamic.params import StateAnalysis

ROOT = Path(__file__).resolve().parents[2]
PAPER2_TEMPLATES = (
    "R_M1_UNDECLARED_PORT", "R_M2_DYNAMIC_PORTS", "R_M3_DECLARED_NOT_OPEN", "R_M4A_UNIT_COLLISION",
    "R_M4B_SERVICE_COLLISION", "R_M4C_SUBSET_COLLISION", "R_M4G_GLOBAL_COLLISION", "R_M5A_SVC_UNOPENED",
    "R_M5B_SVC_UNDECLARED", "R_M5C_HEADLESS_MISMATCH", "R_M5D_SVC_NO_TARGET", "R_M6_NS_NO_NETPOL", "R_HOSTNET",
)


class StaticRulesBaseline:
    def __init__(self, rule_file: Optional[str] = None):
        self.rules = SecurityRuleEngine(rule_file or ROOT / "rules" / "security_rules.yaml")
        self.drift = DriftDetectionEngine()

    def score(self, B: dict, C: dict) -> Dict[str, float]:
        events = self.drift.detect_drift(B, C)
        an = AttackPathAnalyzer()
        an.build_graph(C)
        path_found = bool(an.find_attack_paths())
        total, matched, ids = 0, 0, []
        for e in events:
            for m in self.rules.evaluate_event(e):
                total += m["impact"]
                matched += 1
                ids.append(m["rule_id"])
        score = calculate_risk(total, path_found)["score"] if matched else 0
        return {"score": float(score), "alert": float(matched > 0), "matched": float(matched), "rule_ids": ids,
                "events": float(len(events))}


class Paper1Baseline:
    """Change of the base paper's riskiest-path score. ``agg`` is chosen on validation data."""

    AGGS = ("max", "top5", "sum")

    @staticmethod
    def score(aB: StateAnalysis, aC: StateAnalysis, agg: str = "top5") -> float:
        return aC.ag_base.cluster_risk(agg) - aB.ag_base.cluster_risk(agg)


class Paper2Baseline:
    @staticmethod
    def score(aB: StateAnalysis, aC: StateAnalysis) -> Dict[str, float]:
        new = 0
        kinds = 0
        for tid in PAPER2_TEMPLATES:
            n = len(set(aC.inst[tid]) - set(aB.inst[tid]))
            new += n
            kinds += n > 0
        return {"score": float(new), "alert": float(new > 0), "kinds": float(kinds)}
