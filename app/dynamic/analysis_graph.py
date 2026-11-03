"""Paper-1 style analysis graph (Blaise & Rebecchi, CLOUD 2022) + an *extended* mode.

``extended=False`` reproduces the base paper's model:
  * directed pod graph; edge weight z(i, j) in [0, 1] is the probability that an attacker in pod i
    reaches pod j: gamma_1 = 0.8 when no network policy blocks it, gamma_2 = 0.4 for pod affinity
    (co-location), gamma_3 = 1.0 when pod i's service account can exec into pod j (Table VI);
  * pod score = weighted sum of attributes with the Table VI control values (alpha_1 per vulnerability,
    alpha_2 per Linux capability, beta_1 accessible service, beta_2 root, beta_3 privilege escalation,
    beta_4 permissions, beta_5 exec, beta_6 host network, beta_7 host paths, beta_8 host volumes);
  * the riskiest path maximises prod z(e) -> shortest path with weights -log z(e) (eq. 1-2). The paper
    uses Dijkstra; here the identical optimum is obtained with vectorised max-product relaxation, which
    is faster for dense pod graphs and also yields the predecessor tree for path reconstruction.

``extended=True`` adds mechanisms the paper does not model (host-network NetworkPolicy bypass, ports that
leak through policies, label-collision interception, node blast-radius edges). The numbers used there are
analyst priors, deliberately *not* identical to the simulator's world model.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from app.dynamic.view import ClusterView, POWERFUL_PERMS, DANGEROUS_CAPS

# Table VI of the base paper (control variables) -- kept unchanged by this project
CONTROL = {
    "alpha1": 1.0,      # per high/critical vulnerability
    "alpha2": 5.0,      # per Linux capability
    "beta1": 100.0,     # accessible service
    "beta2": 100.0,     # can run as root
    "beta3": 100.0,     # can allow privilege escalation
    "beta4": 10.0,      # per permission (defence evasion / credential access / discovery / impact)
    "beta5": 100.0,     # can exec other pods
    "beta6": 100.0,     # can reach host network
    "beta7": 100.0,     # can reach host paths
    "beta8": 10.0,      # per allowed host volume type
    "gamma1": 0.8,      # no network policy
    "gamma2": 0.4,      # pod affinity
    "gamma3": 1.0,      # pod execution capacity
}
OMEGA = 0.1             # default accessibility probability of a non-exposed pod (paper example, Sec. V-E)

# analyst priors for the extended mode (NOT the simulator's constants)
EXT = {"hostnet_bypass": 0.8, "port_leak": 0.2, "mitm": 0.4, "node_blast": 0.7}


class AnalysisGraph:
    def __init__(self, view: ClusterView, control: Dict[str, float] = None, omega: float = OMEGA,
                 extended: bool = False):
        self.v = view
        self.c = dict(CONTROL if control is None else control)
        self.omega = omega
        self.extended = extended
        n = view.n_pods
        self.n = n
        self.Z = self._edges()
        self.local = self._local_scores()
        self.reach, self.pred = self._reach()

    # ----------------------------------------------------------------- edges
    def _edges(self) -> np.ndarray:
        v, n, c = self.v, self.n, self.c
        Z = np.where(v.allowed, c["gamma1"], 0.0).astype(np.float32)
        if n == 0:
            return Z
        pods = v.pods
        ns_arr = np.array([p.ns for p in pods])
        # pod execution capacity (gamma_3)
        for p in pods:
            perms = v.pod_perms(p)
            if "exec" in perms:
                sc = perms["exec"]
                mask = np.ones(n, dtype=bool) if "*" in sc else np.isin(ns_arr, list(sc))
                mask[p.idx] = False
                Z[p.idx, mask] = np.maximum(Z[p.idx, mask], c["gamma3"])
        # pod affinities (gamma_2): pod i declares affinity to pods matching a selector -> co-located
        for p in pods:
            for aff in p.affinity:
                sel = aff.get("match_labels") or {}
                tns = aff.get("ns", p.ns)
                for j in v.pods_matching(tns, sel):
                    if j != p.idx:
                        Z[p.idx, j] = max(Z[p.idx, j], c["gamma2"])
                        Z[j, p.idx] = max(Z[j, p.idx], c["gamma2"])
        if self.extended:
            hostnet = np.array([p.host_network for p in pods])
            Z[:, hostnet] = np.maximum(Z[:, hostnet], EXT["hostnet_bypass"])
            leak = np.array([bool(p.ports_open - p.ports_declared) or p.dyn_ports > 0 for p in pods])
            Z[:, leak] = np.maximum(Z[:, leak], EXT["port_leak"])
            for s in v.services:
                intended = s.get("intended_app")
                if not intended or not s["pods"]:
                    continue
                foreign = [i for i in s["pods"] if pods[i].labels.get("app") != intended]
                legit = [i for i in v.pods_by_ns.get(s["ns"], []) if pods[i].labels.get("app") == intended]
                if foreign and legit:
                    ix = np.ix_(foreign, legit)
                    Z[ix] = np.maximum(Z[ix], EXT["mitm"])
            for p in pods:
                if p.privileged or p.sensitive_hostpath or (p.host_pid and (p.root or p.caps_extra & DANGEROUS_CAPS)):
                    co = [j for j in v.pods_by_node.get(p.node, []) if j != p.idx]
                    if co:
                        Z[p.idx, co] = np.maximum(Z[p.idx, co], EXT["node_blast"])
        np.fill_diagonal(Z, 0.0)
        return Z

    # ------------------------------------------------------------ local score
    def _local_scores(self) -> Dict[str, np.ndarray]:
        v, c, n = self.v, self.c, self.n
        keys = ("alpha1", "alpha2", "beta1", "beta2", "beta3", "beta4", "beta5", "beta6", "beta7", "beta8")
        out = {k: np.zeros(n) for k in keys}
        for p in v.pods:
            perms = v.pod_perms(p) if v.token_mounted(p) else {}
            i = p.idx
            out["alpha1"][i] = c["alpha1"] * p.vuln
            out["alpha2"][i] = c["alpha2"] * p.ncaps
            out["beta1"][i] = c["beta1"] * float(v.exposure[i] > 0)
            out["beta2"][i] = c["beta2"] * float(p.root)
            out["beta3"][i] = c["beta3"] * float(p.priv_esc or p.privileged)
            out["beta4"][i] = c["beta4"] * sum(1 for f in ("secrets", "list_pods", "create_pods", "delete") if f in perms)
            out["beta5"][i] = c["beta5"] * float("exec" in perms)
            out["beta6"][i] = c["beta6"] * float(p.host_network)
            out["beta7"][i] = c["beta7"] * float(bool(p.hostpaths))
            out["beta8"][i] = c["beta8"] * len({vv.get("type") for vv in p.volumes if vv.get("type") in ("hostPath",)})
        out["global"] = sum(out[k] for k in keys)
        return out

    # ------------------------------------------------------------------ reach
    def _reach(self) -> Tuple[np.ndarray, np.ndarray]:
        """Max over paths of the product of edge weights (eq. 1), starting from accessible pods."""
        n = self.n
        init = np.where(self.v.exposure > 0, 1.0, self.omega).astype(np.float32)
        r = init.copy()
        pred = -np.ones(n, dtype=int)
        for _ in range(12):
            cand = r[:, None] * self.Z                                  # cand[i, j] = r_i * z(i, j)
            best_i = cand.argmax(axis=0)
            best = cand[best_i, np.arange(n)]
            upd = best > r + 1e-9
            if not upd.any():
                break
            r = np.where(upd, best, r)
            pred = np.where(upd, best_i, pred)
        return r, pred

    # ------------------------------------------------------------- aggregates
    def pod_risk(self) -> np.ndarray:
        """Path-aware pod score: probability of reaching the pod x its Table VI danger score."""
        return self.reach * self.local["global"]

    def cluster_risk(self, agg: str = "top5") -> float:
        s = self.pod_risk()
        if s.size == 0:
            return 0.0
        if agg == "max":
            return float(s.max())
        if agg == "top5":
            return float(np.sort(s)[-5:].mean())
        if agg == "sum":
            return float(s.sum() / 1000.0)
        raise ValueError(agg)

    def top_paths(self, k: int = 3, max_len: int = 6) -> List[dict]:
        """Riskiest attack paths (entry -> ... -> pod) with their product weight, for explanations."""
        s = self.pod_risk()
        out = []
        for j in np.argsort(-s)[:k]:
            path, cur = [int(j)], int(j)
            while self.pred[cur] >= 0 and len(path) < max_len:
                cur = int(self.pred[cur])
                path.append(cur)
            path.reverse()
            out.append({"path": [self.v.pods[i].key for i in path], "score": float(s[j]), "reach": float(self.reach[j])})
        return out
