"""Algorithm 1 - Parameter Identification (PIA).

Goal: turn *any* pair of snapshots (baseline B, current C) into a numeric parameter vector theta that
describes the environment the rules must adapt to. The rule generator (Algorithm 2) conditions rule
weights/thresholds on theta, so the rule set changes whenever the environment changes.

Stages
  1  PARSE      snapshot -> ClusterView (labels, NetworkPolicy selection, RBAC resolution)
  2  STRUCTURE  size / topology parameters (pods, nodes, namespaces, density)
  3  POSTURE    hardening rates per control (root, privileged, NP coverage, exposure, RBAC power, ...)
  4  GRAPH      attack-graph parameters (mean reach, exposed reach, baseline path risk) - paper-1 graph
  5  DRIFT      change parameters between B and C (changed resources by kind, churn)
  6  NORMALISE  z-score against a reference population (fit once on the training clusters)
  7  SELECT     relevance-ranked active parameters (relevance learned from outcomes, see model.py)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from app.dynamic.analysis_graph import AnalysisGraph
from app.dynamic.templates import evaluate, t_m4a
from app.dynamic.view import ClusterView, DANGEROUS_CAPS, SYSTEM_NAMESPACES

POSTURE_PARAMS = [
    "log_pods", "log_nodes", "log_namespaces", "log_pods_per_node", "service_ratio",
    "np_coverage", "np_allow_all_rate", "ns_without_np", "exposure_rate", "root_rate", "priv_rate",
    "privesc_rate", "hostnet_rate", "hostpath_rate", "sens_hostpath_rate", "caps_rate", "automount_rate",
    "powerful_sa_rate", "nolimits_rate", "vuln_median", "vuln_p90", "opaque_rate", "latest_rate",
    "m1_rate", "m2_rate", "m3_rate", "collision_rate", "sensitive_rate", "colocated_sensitive",
    "mean_reach", "high_reach_rate", "base_risk_top5", "p1_risk_top5",
]
DRIFT_PARAMS = ["changed_resources", "changed_pods", "changed_rbac", "changed_netpol", "changed_services",
                "added_resources", "removed_resources"]
PARAM_NAMES = POSTURE_PARAMS + DRIFT_PARAMS


def vuln_threshold(view: ClusterView) -> float:
    """Dynamic rule threshold: 'highly vulnerable' means above the cluster's own 75th percentile (>=10)."""
    vs = [p.vuln for p in view.pods if p.vuln > 0]
    return float(np.clip(np.percentile(vs, 75), 10, 200)) if vs else 10.0


class StateAnalysis:
    """Everything derived from ONE snapshot: view, template instances, extended + base graphs, posture theta."""

    def __init__(self, snapshot: dict, vuln_thr: Optional[float] = None):
        self.snapshot = snapshot
        self.view = ClusterView(snapshot)
        self.vuln_thr = vuln_threshold(self.view) if vuln_thr is None else vuln_thr
        self.prm = {"vuln_thr": self.vuln_thr}
        self.inst = evaluate(self.view, self.prm)
        self.ag = AnalysisGraph(self.view, extended=True)       # proposed: extended graph
        self.ag_base = AnalysisGraph(self.view, extended=False)  # base paper 1 graph
        self.posture = self._posture()

    def _posture(self) -> Dict[str, float]:
        v, n = self.view, max(self.view.n_pods, 1)
        pods = v.pods
        rate = lambda f: float(np.mean([bool(f(p)) for p in pods])) if pods else 0.0
        vs = np.array([p.vuln for p in pods]) if pods else np.zeros(1)
        covered = {pol["ns"] for pol in v.netpols}
        app_ns = [ns for ns in v.pods_by_ns if ns not in SYSTEM_NAMESPACES]
        n_nodes = max(len(v.nodes), 1)
        sens_by_node = [sum(1 for j in idx if v.sensitive_pod(pods[j])) for idx in v.pods_by_node.values()] or [0]
        m4a = self.inst["R_M4A_UNIT_COLLISION"]
        th = {
            "log_pods": np.log1p(v.n_pods), "log_nodes": np.log1p(n_nodes), "log_namespaces": np.log1p(len(v.namespaces)),
            "log_pods_per_node": np.log1p(v.n_pods / n_nodes),
            "service_ratio": len(v.services) / n,
            "np_coverage": float(v.isolated.mean()) if v.n_pods else 0.0,
            "np_allow_all_rate": float(v.np_allow_all_pods.mean()) if v.n_pods else 0.0,
            "ns_without_np": (sum(1 for ns in app_ns if ns not in covered) / len(app_ns)) if app_ns else 0.0,
            "exposure_rate": float((v.exposure > 0).mean()) if v.n_pods else 0.0,
            "root_rate": rate(lambda p: p.root), "priv_rate": rate(lambda p: p.privileged),
            "privesc_rate": rate(lambda p: p.priv_esc), "hostnet_rate": rate(lambda p: p.host_network),
            "hostpath_rate": rate(lambda p: bool(p.hostpaths)), "sens_hostpath_rate": rate(lambda p: p.sensitive_hostpath),
            "caps_rate": rate(lambda p: bool(p.caps_extra & DANGEROUS_CAPS)),
            "automount_rate": rate(lambda p: v.token_mounted(p)),
            "powerful_sa_rate": rate(lambda p: bool(v.sa_power(p))),
            "nolimits_rate": rate(lambda p: not p.has_limits),
            "vuln_median": float(np.log1p(np.median(vs))), "vuln_p90": float(np.log1p(np.percentile(vs, 90))),
            "opaque_rate": rate(lambda p: p.opaque), "latest_rate": rate(lambda p: p.latest_tag),
            "m1_rate": len(self.inst["R_M1_UNDECLARED_PORT"]) / n, "m2_rate": len(self.inst["R_M2_DYNAMIC_PORTS"]) / n,
            "m3_rate": len(self.inst["R_M3_DECLARED_NOT_OPEN"]) / n, "collision_rate": len(m4a) / n,
            "sensitive_rate": rate(lambda p: v.sensitive_pod(p)),
            "colocated_sensitive": float(np.mean(sens_by_node)),
            "mean_reach": float(self.ag.reach.mean()) if v.n_pods else 0.0,
            "high_reach_rate": float((self.ag.reach > 0.5).mean()) if v.n_pods else 0.0,
            "base_risk_top5": float(np.log1p(self.ag.cluster_risk("top5"))),
            "p1_risk_top5": float(np.log1p(self.ag_base.cluster_risk("top5"))),
        }
        return {k: float(x) for k, x in th.items()}


def drift_params(B: dict, C: dict) -> Dict[str, float]:
    """Stage 5: cheap structural diff (shared unchanged objects are compared by identity first)."""
    ib = {(r["kind"], r["namespace"], r["name"]): r for r in B["resources"]}
    ic = {(r["kind"], r["namespace"], r["name"]): r for r in C["resources"]}
    out = {k: 0.0 for k in DRIFT_PARAMS}
    rbac = {"Role", "ClusterRole", "RoleBinding", "ClusterRoleBinding", "ServiceAccount"}
    for k in set(ib) | set(ic):
        a, b = ib.get(k), ic.get(k)
        if a is b:
            continue
        if a is None:
            out["added_resources"] += 1
        elif b is None:
            out["removed_resources"] += 1
        elif a["data"] == b["data"]:
            continue
        else:
            pass
        out["changed_resources"] += 1
        kind = k[0]
        if kind == "Pod":
            out["changed_pods"] += 1
        elif kind in rbac:
            out["changed_rbac"] += 1
        elif kind == "NetworkPolicy":
            out["changed_netpol"] += 1
        elif kind in ("Service", "Ingress"):
            out["changed_services"] += 1
    return out


class ParameterIdentifier:
    """Algorithm 1: parameters + normalisation + relevance-based selection."""

    def __init__(self):
        self.mean: Dict[str, float] = {}
        self.std: Dict[str, float] = {}
        self.relevance: Dict[str, float] = {}

    def fit_reference(self, thetas: Iterable[Dict[str, float]]) -> "ParameterIdentifier":
        M = np.array([[t[k] for k in PARAM_NAMES] for t in thetas], dtype=float)
        mu, sd = M.mean(0), M.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        self.mean = dict(zip(PARAM_NAMES, mu))
        self.std = dict(zip(PARAM_NAMES, sd))
        return self

    def identify(self, aB: StateAnalysis, B: dict, C: dict) -> Dict[str, float]:
        theta = dict(aB.posture)
        theta.update(drift_params(B, C))
        return theta

    def normalise(self, theta: Dict[str, float]) -> Dict[str, float]:
        return {k: float(np.clip((theta[k] - self.mean.get(k, 0.0)) / self.std.get(k, 1.0), -4, 4)) for k in PARAM_NAMES}

    def active(self, theta: Dict[str, float], top_k: int = 12) -> List[str]:
        """Active parameters = most relevant (learned) plus any parameter that is anomalous (|z| > 2)."""
        z = self.normalise(theta)
        ranked = sorted(self.relevance, key=self.relevance.get, reverse=True)[:top_k] if self.relevance else []
        anomalous = [k for k, x in z.items() if abs(x) > 2.0]
        return list(dict.fromkeys(ranked + anomalous))
