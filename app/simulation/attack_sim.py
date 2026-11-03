"""Monte-Carlo attacker simulation = ground truth for the experiments.

An attacker campaign enters the cluster (externally exposed vulnerable pods, or a
supply-chain event), then moves laterally / escalates until nothing new is
compromised. Risk R(state) = E[ value of the most valuable asset compromised ].

IMPORTANT design rule: this module shares *parsing* (``ClusterView``) with the
analysis code but NOT its probability model. The edge probabilities below are
an independent "world model". Where paper 1 gives probabilities (no network
policy -> 0.8, pod exec -> 1) the same numbers are used; additional mechanics
absent from both papers (the "exceptional scenarios") are modelled as well:

* label-collision traffic interception (M4) -> credential theft
* dynamic / undeclared ports leaking through NetworkPolicies (M1, M2)
* NetworkPolicies that select no pod, or allow everything
* hostNetwork pods bypassing NetworkPolicies
* group-subject RBAC bindings (``system:serviceaccounts``) and cross-namespace subjects
* node blast radius: a compromised node compromises every co-located pod
* cryptojacking / economic abuse of pods without resource limits
* supply-chain compromise of a single pod
All probabilities are assumptions documented in the report.
"""
from __future__ import annotations

import zlib
from typing import Dict, Optional

import numpy as np

from app.dynamic.view import ClusterView, SYSTEM_NAMESPACES

# ----- world-model constants (documented assumptions) -----------------------------------------
P_LATERAL_OPEN = 0.8          # paper 1, Table VI gamma_1: no network policy
P_LEAK_NP = 0.03              # residual reachability through a restrictive policy
P_LEAK_UNDECLARED = 0.25      # M1 ports are not covered by policies generated from declared ports
P_LEAK_DYNAMIC = 0.30         # M2 ephemeral ports cannot be restricted
P_EXEC = 1.0                  # paper 1, Table VI gamma_3
P_TOKEN_MOUNTED, P_TOKEN_OTHER = 0.90, 0.05
P_MITM = 0.50
P_HOSTNET_NODE = 0.15
P_SUPPLY_EVENT = 0.30
ENTRY_BUDGET = 1.5
DANGER_TAU = 0.25      # a drift is 'dangerous' if it raises expected damage by >= 0.25 worker-node equivalents
MOVE_BUDGET, MOVE_S0 = 0.8, 4.0    # attacker effort per compromised pod, spread over all potential targets
VAL_FOOTHOLD, VAL_SYSTEM, VAL_DB, VAL_HOST, VAL_SECRETS_NS = 0.01, 0.7, 0.5, 0.6, 0.4
VAL_SECRETS_ALL, VAL_ADMIN, VAL_CRYPTO = 0.8, 1.0, 0.15

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _node_hash(key: str) -> np.uint64:
    a = zlib.crc32(key.encode()) & 0xFFFFFFFF
    b = zlib.crc32((key + "#").encode()) & 0xFFFFFFFF
    return np.uint64((a << 32) | b)


def _uniform(seed: int, trials: int, hashes: np.ndarray) -> np.ndarray:
    """Counter-based uniforms keyed by (seed, trial, node): common random numbers across states."""
    with np.errstate(over="ignore"):
        t = np.arange(trials, dtype=np.uint64)[:, None] * np.uint64(0xBF58476D1CE4E5B9)
        x = hashes[None, :] * np.uint64(0x9E3779B97F4A7C15) + t + np.uint64((seed * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def _neglog(p: np.ndarray) -> np.ndarray:
    return -np.log1p(-np.clip(p, 0.0, 0.999999)).astype(np.float32)


class World:
    """Attack-graph in 'log-survival' form: L[i, j] = -log(1 - P(i compromises j))."""

    def __init__(self, v: ClusterView):
        self.v = v
        pods = v.pods
        N = v.n_pods
        hosts = sorted(v.nodes)
        namespaces = sorted(set(v.pods_by_ns) | set(v.namespaces))
        H = len(hosts)
        self.N, self.H = N, H
        i_host0 = N
        i_admin = N + H
        i_secrets_all = i_admin + 1
        i_anyhost = i_secrets_all + 1
        i_ns0 = i_anyhost + 1
        M = i_ns0 + len(namespaces)
        self.M = M
        self.i_admin, self.i_secrets_all, self.i_anyhost = i_admin, i_secrets_all, i_anyhost
        self.i_host0, self.i_ns0 = i_host0, i_ns0
        host_idx = {h: i_host0 + k for k, h in enumerate(hosts)}
        ns_idx = {n: i_ns0 + k for k, n in enumerate(namespaces)}

        keys = [p.key for p in pods] + [f"host:{h}" for h in hosts] + ["jewel:admin", "jewel:secrets", "jewel:anyhost"] \
            + [f"jewel:ns:{n}" for n in namespaces]
        self.hashes = np.array([_node_hash(k) for k in keys], dtype=np.uint64)

        # values -----------------------------------------------------------------------------
        val = np.zeros(M, dtype=np.float32)
        for p in pods:
            x = VAL_FOOTHOLD
            if p.ns in SYSTEM_NAMESPACES:
                x = VAL_SYSTEM
            elif p.labels.get("tier") == "db":
                x = VAL_DB
            elif not p.has_limits:
                x = max(x, VAL_CRYPTO)
            val[p.idx] = x
        val[i_host0:i_host0 + H] = VAL_HOST
        val[i_admin] = VAL_ADMIN
        val[i_secrets_all] = VAL_SECRETS_ALL
        val[i_anyhost] = VAL_HOST
        val[i_ns0:] = VAL_SECRETS_NS
        self.value = val

        L = np.zeros((M, M), dtype=np.float32)
        if N:
            vuln = np.array([p.vuln for p in pods], dtype=np.float64)
            opaque = np.array([p.opaque for p in pods])
            exploit = np.where(opaque, 0.30, 0.03 + 0.90 * (1 - np.exp(-vuln / 80.0)))
            has_port = np.array([bool(p.ports_open) or p.dyn_ports > 0 or not any(c["known_ports"] for c in p.containers)
                                 for p in pods])
            m1 = np.array([bool(p.ports_open - p.ports_declared) and any(c["known_ports"] for c in p.containers) for p in pods])
            m2 = np.array([p.dyn_ports > 0 for p in pods])
            hostnet = np.array([p.host_network for p in pods])
            leak = 1 - (1 - P_LEAK_NP) * (1 - P_LEAK_UNDECLARED * m1) * (1 - P_LEAK_DYNAMIC * m2)
            reach = np.where(v.allowed | hostnet[None, :], P_LATERAL_OPEN, leak[None, :])
            reach = reach * has_port[None, :]
            # a rational attacker prefers attractive targets (privileged, powerful service account, sensitive)
            attract = np.array([
                1.0 + 3.0 * (p.privileged or p.sensitive_hostpath) + 1.0 * (p.root and p.priv_esc)
                + 2.0 * bool(v.sa_power(p)) + 1.0 * v.sensitive_pod(p) for p in pods])
            # Effort is spread over *potential* targets (reach-independent denominator): effort spent on a
            # target blocked by a NetworkPolicy is wasted, so hardening can never raise another pod's risk.
            n_targets = float((0.8 * has_port).sum())
            w_att = reach * attract[None, :]
            np.fill_diagonal(w_att, 0.0)
            p_move = np.minimum(1.0, MOVE_BUDGET * w_att / (n_targets + MOVE_S0)) * exploit[None, :]
            L[:N, :N] += _neglog(p_move)

            # token / RBAC-based moves ------------------------------------------------------
            ns_arr = np.array([p.ns for p in pods])
            for p in pods:
                perms = v.pod_perms(p)
                if not perms:
                    continue
                ptok = P_TOKEN_MOUNTED if v.token_mounted(p) else P_TOKEN_OTHER
                i = p.idx
                if "exec" in perms:
                    scope = perms["exec"]
                    mask = np.ones(N, dtype=bool) if "*" in scope else np.isin(ns_arr, list(scope))
                    mask[i] = False
                    L[i, :N][mask] += _neglog(np.array(ptok * P_EXEC))
                if "wildcard" in perms and "*" in perms["wildcard"]:
                    L[i, i_admin] += _neglog(np.array(ptok * 0.95))
                if "impersonate" in perms:
                    L[i, i_admin] += _neglog(np.array(ptok * (0.6 if "*" in perms["impersonate"] else 0.25)))
                if "secrets" in perms:
                    sc = perms["secrets"]
                    if "*" in sc:                          # cluster-wide read implies every namespace's secrets too
                        L[i, i_secrets_all] += _neglog(np.array(ptok * 0.9))
                        for n in ns_idx.values():
                            L[i, n] += _neglog(np.array(ptok * 0.9))
                    else:
                        for n in sc:
                            if n in ns_idx:
                                L[i, ns_idx[n]] += _neglog(np.array(ptok * 0.9))
                if "create_pods" in perms:
                    L[i, i_anyhost] += _neglog(np.array(ptok * (0.8 if "*" in perms["create_pods"] else 0.5)))

            # container escape -> host ---------------------------------------------------------
            for p in pods:
                mech = []
                caps = p.caps_extra
                if p.privileged:
                    mech.append(0.95)
                if p.sensitive_hostpath:
                    mech.append(0.90)
                if p.host_pid and (p.root or caps & {"SYS_PTRACE", "SYS_ADMIN"}):
                    mech.append(0.70)
                if caps & {"SYS_ADMIN", "SYS_MODULE", "SYS_RAWIO"}:
                    mech.append(0.45)
                if p.hostpaths and not p.sensitive_hostpath:
                    mech.append(0.05)
                if p.host_ipc:
                    mech.append(0.10)
                if p.root and p.priv_esc:
                    mech.append(0.07)
                elif p.root:
                    mech.append(0.03)
                else:
                    mech.append(0.005)
                if p.host_network:
                    mech.append(P_HOSTNET_NODE)
                surv = np.prod([1 - m for m in mech])
                L[p.idx, host_idx[p.node]] += _neglog(np.array(1 - surv))

            # label-collision traffic interception (M4) ---------------------------------------
            for s in v.services:
                intended = s.get("intended_app")
                if not intended or not s["pods"]:
                    continue
                foreign = [i for i in s["pods"] if pods[i].labels.get("app") != intended]
                if not foreign:
                    continue
                legit = [i for i in v.pods_by_ns.get(s["ns"], []) if pods[i].labels.get("app") == intended]
                if legit:
                    L[np.ix_(foreign, legit)] += _neglog(np.array(P_MITM))

        # host edges ---------------------------------------------------------------------------
        for h, hi in host_idx.items():
            on = v.pods_by_node.get(h, [])
            if on:
                L[hi, on] += 20.0                                     # node compromise -> every co-located pod
            L[hi, i_secrets_all] += _neglog(np.array(0.5))
            L[hi, i_admin] += _neglog(np.array(0.05))
        self.L = L

        # entry vector -------------------------------------------------------------------------
        entry = np.zeros(M, dtype=np.float64)
        if N:
            pe = np.where(opaque, 0.30, 0.03 + 0.90 * (1 - np.exp(-vuln / 60.0))) * (v.exposure > 0)
            tot = pe.sum()
            if tot > ENTRY_BUDGET:
                pe = pe * (ENTRY_BUDGET / tot)
            entry[:N] = pe
            w = (0.5 + (1 - np.exp(-vuln / 80.0))) * np.where([p.latest_tag for p in pods], 1.5, 1.0)
            self.supply_cum = np.cumsum(w) / w.sum()
        else:
            self.supply_cum = np.zeros(0)
        self.entry = entry


class AttackSimulator:
    """Runs the Monte-Carlo campaign. Counter-based uniforms make trial t identical across states
    (common random numbers), so per-trial B/C differences give a proper standard error for delta-D."""

    def __init__(self, trials: int = 400, rounds: int = 12, seed: int = 0):
        self.trials, self.rounds, self.seed = trials, rounds, seed

    def world(self, snapshot_or_view) -> World:
        v = snapshot_or_view if isinstance(snapshot_or_view, ClusterView) else ClusterView(snapshot_or_view)
        return World(v)

    def run(self, w: World, trials: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Return per-trial outcomes: damage (node-equivalents), max asset value and the compromised matrix summary."""
        seed = self.seed if seed is None else seed
        T, N = trials or self.trials, w.N
        if N == 0:
            z = np.zeros(T)
            return {"damage": z, "maxval": z, "admin": z, "host": z, "secrets": z, "foothold": z}
        u0 = _uniform(seed, T, w.hashes)
        S = u0 < w.entry[None, :]
        sup = _uniform(seed + 17, T, np.array([_node_hash("supply-event"), _node_hash("supply-pick")], dtype=np.uint64))
        event = sup[:, 0] < P_SUPPLY_EVENT
        pick = np.minimum(np.searchsorted(w.supply_cum, sup[:, 1]), N - 1)
        S[np.nonzero(event)[0], pick[event]] = True
        # Independent cascade == threshold model with a fixed Exp(1) resistance per (trial, node):
        # node j falls once accumulated hazard sum_i L_ij reaches theta_j. Same distribution as per-edge
        # coin flips, but monotone in L, so B/C trials are tightly coupled (low-variance delta).
        theta = -np.log1p(-_uniform(seed + 101, T, w.hashes))
        hazard = np.zeros((T, w.M), dtype=np.float32)
        F = S.copy()
        for _ in range(self.rounds):
            if not F.any():
                break
            hazard += F.astype(np.float32) @ w.L
            new = (hazard >= theta) & ~S
            S |= new
            F = new
        return {
            "damage": (S.astype(np.float32) @ w.value) / VAL_HOST,      # expected damage in worker-node equivalents
            "maxval": np.where(S, w.value[None, :], 0.0).max(axis=1),
            "admin": S[:, w.i_admin].astype(float),
            "host": (S[:, w.i_host0:w.i_host0 + w.H].any(axis=1) | S[:, w.i_anyhost]).astype(float),
            "secrets": (S[:, w.i_secrets_all] | S[:, w.i_ns0:].any(axis=1)).astype(float),
            "foothold": S[:, :N].any(axis=1).astype(float),
        }

    def simulate(self, snapshot_or_view, seed: Optional[int] = None, trials: Optional[int] = None) -> Dict[str, float]:
        out = self.run(self.world(snapshot_or_view), trials, seed)
        return {"R": float(out["maxval"].mean()), "D": float(out["damage"].mean()),
                "p_admin": float(out["admin"].mean()), "p_host": float(out["host"].mean()),
                "p_secrets": float(out["secrets"].mean()), "p_foothold": float(out["foothold"].mean())}

    def paired_delta(self, wB: World, wC: World, seed: Optional[int] = None, t0: int = 500,
                     tau: float = DANGER_TAU, t_max: int = 4000) -> Dict[str, float]:
        """delta-D = D(C) - D(B) with a paired standard error.

        Trials double until the *label* (delta-D >= tau) is statistically clear (|dD - tau| > 2.5 se),
        the se is tiny (<= 0.03), or t_max is reached. Cheap for obvious cases, thorough for borderline ones.
        """
        T = t0
        while True:
            b, c = self.run(wB, T, seed), self.run(wC, T, seed)
            d = c["damage"] - b["damage"]
            se = float(d.std(ddof=1) / np.sqrt(T))
            m = float(d.mean())
            if abs(m - tau) > 2.5 * se or se <= 0.03 or T >= t_max:
                break
            T = min(T * 2, t_max)
        return {"dD": m, "se": se, "trials": T, "D_B": float(b["damage"].mean()),
                "D_C": float(c["damage"].mean()), "dR": float(c["maxval"].mean() - b["maxval"].mean()),
                "p_admin_C": float(c["admin"].mean()), "p_host_C": float(c["host"].mean()),
                "p_secrets_C": float(c["secrets"].mean())}
