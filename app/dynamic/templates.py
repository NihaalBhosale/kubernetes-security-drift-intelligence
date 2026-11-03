"""Rule *template* library (54 templates).

A template is a parameterised, state-based security condition. For a snapshot it returns the set of
violating *instances* (``key -> (intensity, affected pod indices)``). A drift is analysed by comparing the
instance sets of the baseline B and the current state C: new instances = risk introduced by the drift,
vanished instances = risk removed. This makes every rule applicable to any snapshot pair (any dataset).

Provenance per template:
  * ``paper1``      CIS / MITRE-tactic checks of Blaise & Rebecchi with their Table VI control weight ``w0``
  * ``paper2``      network misconfigurations M1-M7 of Bufalino et al.
  * ``extension``   scenarios beyond both papers (exceptional scenarios)
``w0`` is the initial (prior) weight, i.e. Table VI control value / 100 where the paper defines one;
the dynamic rule generator adapts it to the cluster at hand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from app.dynamic.view import (ClusterView, DANGEROUS_CAPS, POWERFUL_PERMS, SYSTEM_NAMESPACES, PodRec)

Inst = Dict[str, Tuple[float, Tuple[int, ...]]]


@dataclass(frozen=True)
class Template:
    id: str
    family: str          # paper1 | paper2 | extension
    source: str          # human readable provenance (control variable / misconfiguration id)
    tactic: str          # MITRE ATT&CK for Kubernetes tactic
    w0: float            # prior weight per unit of intensity (Table VI control / 100 where defined)
    transform: str       # binary | count | log  (how intensity enters learned features)
    description: str
    fn: Callable[[ClusterView, dict], Inst]


def _wl(p: PodRec) -> Tuple[str, str]:
    return (p.ns, p.name.rsplit("-", 1)[0])


def _pods(v: ClusterView, pred, inten=lambda p: 1.0) -> Inst:
    out: Inst = {}
    for p in v.pods:
        if pred(p):
            x = inten(p)
            if x > 0:
                out[p.key] = (float(x), (p.idx,))
    return out


def _powerful(v: ClusterView, p: PodRec, perm: str) -> bool:
    return perm in v.pod_perms(p) and v.token_mounted(p)


# ------------------------------------------------------------------ paper 1 (CIS / tactics)
def t_priv(v, prm): return _pods(v, lambda p: p.privileged)
def t_root(v, prm): return _pods(v, lambda p: p.root)
def t_privesc(v, prm): return _pods(v, lambda p: p.priv_esc)
def t_caps(v, prm): return _pods(v, lambda p: bool(p.caps_extra & DANGEROUS_CAPS), lambda p: len(p.caps_extra & DANGEROUS_CAPS))
def t_hostnet(v, prm): return _pods(v, lambda p: p.host_network)
def t_hostpid(v, prm): return _pods(v, lambda p: p.host_pid)
def t_hostipc(v, prm): return _pods(v, lambda p: p.host_ipc)
def t_hostpath(v, prm): return _pods(v, lambda p: bool(p.hostpaths), lambda p: len(p.hostpaths))
def t_hostpath_sens(v, prm): return _pods(v, lambda p: p.sensitive_hostpath)
def t_vol_secret(v, prm): return _pods(v, lambda p: any(x.get("type") == "secret" for x in p.volumes))
def t_svc_lb(v, prm): return _pods(v, lambda p: v.exposure_kind.get(p.idx) == "lb")
def t_svc_np(v, prm): return _pods(v, lambda p: v.exposure_kind.get(p.idx) == "nodeport")
def t_ingress(v, prm): return _pods(v, lambda p: v.exposure_kind.get(p.idx) == "ingress")
def t_vuln(v, prm):
    thr = prm.get("vuln_thr", 10)
    return _pods(v, lambda p: p.vuln >= thr, lambda p: p.vuln)
def t_opaque(v, prm): return _pods(v, lambda p: p.opaque)
def t_latest(v, prm): return _pods(v, lambda p: p.latest_tag)
def t_sa_secrets(v, prm): return _pods(v, lambda p: _powerful(v, p, "secrets"))
def t_sa_create(v, prm): return _pods(v, lambda p: _powerful(v, p, "create_pods"))
def t_sa_exec(v, prm): return _pods(v, lambda p: _powerful(v, p, "exec"))
def t_sa_imp(v, prm): return _pods(v, lambda p: _powerful(v, p, "impersonate"))
def t_sa_wild(v, prm): return _pods(v, lambda p: _powerful(v, p, "wildcard"))
def t_sa_list(v, prm): return _pods(v, lambda p: _powerful(v, p, "list_pods"))
def t_token(v, prm): return _pods(v, lambda p: v.token_mounted(p))
def t_np_none(v, prm): return _pods(v, lambda p: not v.isolated[p.idx])
def t_np_allow_all(v, prm): return _pods(v, lambda p: bool(v.np_allow_all_pods[p.idx]))
def t_np_orphan(v, prm):
    out: Inst = {}
    for pol in v.netpols:
        if not pol["selected"]:
            out[pol["key"]] = (1.0, tuple(v.pods_by_ns.get(pol["ns"], [])))
    return out
def t_affinity(v, prm): return _pods(v, lambda p: bool(p.affinity), lambda p: len(p.affinity))
def t_nolimits(v, prm): return _pods(v, lambda p: not p.has_limits)
def t_rwfs(v, prm): return _pods(v, lambda p: not p.ro_fs)
def t_caps_default(v, prm): return _pods(v, lambda p: p.ncaps >= 14, lambda p: p.ncaps / 14.0)
def t_sys_open(v, prm): return _pods(v, lambda p: p.ns in SYSTEM_NAMESPACES and not v.isolated[p.idx])


# ------------------------------------------------------------------ paper 2 (M1-M7)
def _known(p: PodRec) -> bool:
    return any(c["known_ports"] for c in p.containers)

def t_m1(v, prm): return _pods(v, lambda p: _known(p) and bool(p.ports_open - p.ports_declared))
def t_m2(v, prm): return _pods(v, lambda p: p.dyn_ports > 0, lambda p: p.dyn_ports)
def t_m3(v, prm): return _pods(v, lambda p: _known(p) and bool(p.ports_declared - p.ports_open))

def t_m4a(v, prm):
    """Compute-unit collision: different workloads of a namespace share the same 'app' label value."""
    groups: Dict[Tuple[str, str], set] = {}
    for p in v.pods:
        a = p.labels.get("app")
        if a is not None and p.ns not in SYSTEM_NAMESPACES:
            groups.setdefault((p.ns, a), set()).add(_wl(p))
    bad = {k for k, w in groups.items() if len(w) > 1}
    return _pods(v, lambda p: (p.ns, p.labels.get("app")) in bad)

def t_m4b(v, prm):
    seen: Dict[Tuple[str, tuple], List[dict]] = {}
    for s in v.services:
        if s["selector"] and s["pods"]:
            seen.setdefault((s["ns"], tuple(sorted(s["selector"].items()))), []).append(s)
    out: Inst = {}
    for lst in seen.values():
        if len(lst) > 1:
            for s in lst:
                out[s["key"]] = (1.0, tuple(s["pods"]))
    return out

def t_m4c(v, prm):
    out: Inst = {}
    for s in v.services:
        if len({_wl(v.pods[i]) for i in s["pods"]}) > 1:
            out[s["key"]] = (1.0, tuple(s["pods"]))
    return out

def t_m4g(v, prm):
    apps: Dict[str, set] = {}
    for p in v.pods:
        a = p.labels.get("app")
        if a is not None and p.ns not in SYSTEM_NAMESPACES:
            apps.setdefault(a, set()).add(p.ns)
    bad = {a for a, ns in apps.items() if len(ns) > 1}
    out: Inst = {}
    for a in bad:
        idxs = tuple(p.idx for p in v.pods if p.labels.get("app") == a)
        out[f"app-label:{a}"] = (1.0, idxs)
    return out

def _svc_port_check(v, pred) -> Inst:
    out: Inst = {}
    for s in v.services:
        if not s["pods"]:
            continue
        for pt in s["ports"]:
            tp = pt.get("target_port")
            if not isinstance(tp, int):
                continue
            sel = [v.pods[i] for i in s["pods"]]
            if sel and any(_known(p) for p in sel) and pred(s, tp, sel):
                out[s["key"]] = (1.0, tuple(s["pods"]))
    return out

def t_m5a(v, prm):
    return _svc_port_check(v, lambda s, tp, sel: not s["headless"] and all(tp in p.ports_declared and tp not in p.ports_open for p in sel))
def t_m5b(v, prm):
    return _svc_port_check(v, lambda s, tp, sel: not s["headless"] and all(tp not in p.ports_declared for p in sel))
def t_m5c(v, prm):
    return _svc_port_check(v, lambda s, tp, sel: s["headless"] and all(tp not in p.ports_open for p in sel))
def t_m5d(v, prm):
    out: Inst = {}
    for s in v.services:
        if s["selector"] and not s["pods"] and s["ns"] not in SYSTEM_NAMESPACES:
            out[s["key"]] = (1.0, ())
    return out

def t_m6(v, prm):
    covered = {pol["ns"] for pol in v.netpols}
    return {f"ns:{ns}": (1.0, tuple(idx)) for ns, idx in v.pods_by_ns.items()
            if ns not in SYSTEM_NAMESPACES and ns not in covered}


# ------------------------------------------------------------------ RBAC bindings
def _sa_pods(v: ClusterView, targets) -> Tuple[int, ...]:
    ts = set(targets)
    return tuple(p.idx for p in v.pods if (p.ns, p.sa) in ts)

def t_bind_admin(v, prm):
    return {b["key"]: (1.0, _sa_pods(v, b["targets"])) for b in v.binding_info
            if b["binds_cluster_admin"] and b["targets"] and not b["broadcast"]}
def t_bind_broadcast(v, prm):
    return {b["key"]: (1.0, tuple(range(v.n_pods))) for b in v.binding_info
            if b["broadcast"] and any(f in POWERFUL_PERMS for f in b["flags"])}
def t_bind_cross(v, prm):
    return {b["key"]: (1.0, _sa_pods(v, b["targets"])) for b in v.binding_info
            if b["cross_ns"] and any(f in POWERFUL_PERMS for f in b["flags"])}
def t_default_sa(v, prm):
    out: Inst = {}
    for (ns, sa), perms in v.sa_perms.items():
        if sa == "default" and any(f in POWERFUL_PERMS for f in perms) and ns not in SYSTEM_NAMESPACES:
            out[f"sa:{ns}/default"] = (float(sum(f in POWERFUL_PERMS for f in perms)),
                                       tuple(p.idx for p in v.pods if p.ns == ns and p.sa == "default"))
    return out


# ------------------------------------------------------------------ extension (exceptional)
def _escape(p: PodRec) -> bool:
    return p.privileged or p.sensitive_hostpath or (p.host_pid and (p.root or bool(p.caps_extra & DANGEROUS_CAPS)))

def t_node_blast(v, prm):
    out: Inst = {}
    for p in v.pods:
        if _escape(p):
            sens = [j for j in v.pods_by_node.get(p.node, []) if j != p.idx and v.sensitive_pod(v.pods[j])]
            if sens:
                out[p.key] = (float(len(sens)), (p.idx, *sens))
    return out

def t_np_bypass(v, prm): return _pods(v, lambda p: p.host_network and bool(v.isolated[p.idx]))
def t_chain_token(v, prm):
    return _pods(v, lambda p: bool(v.sa_power(p)) and v.token_mounted(p) and not v.isolated[p.idx])
def t_chain_root(v, prm):
    thr = prm.get("vuln_thr", 10)
    return _pods(v, lambda p: v.exposure[p.idx] > 0 and p.root and p.priv_esc and p.vuln >= thr)
def t_chain_host(v, prm): return _pods(v, lambda p: v.exposure[p.idx] > 0 and (p.privileged or p.sensitive_hostpath))
def t_nolimits_exposed(v, prm): return _pods(v, lambda p: v.exposure[p.idx] > 0 and not p.has_limits)

def t_collide_power(v, prm):
    m4a = t_m4a(v, prm)
    m4c = t_m4c(v, prm)
    hit = {i for _, idxs in m4c.values() for i in idxs} | {i for _, idxs in m4a.values() for i in idxs}
    return _pods(v, lambda p: p.idx in hit and bool(v.sa_power(p)))


TEMPLATES: List[Template] = [
    # ---- paper 1: container / pod security context (CIS 5.2) ---------------------------------
    Template("R_PRIVILEGED", "paper1", "beta3 (100)", "Privilege Escalation", 1.00, "binary", "Privileged container", t_priv),
    Template("R_ROOT", "paper1", "beta2 (100)", "Execution", 1.00, "binary", "Container can run as root", t_root),
    Template("R_PRIVESC", "paper1", "beta3 (100)", "Privilege Escalation", 1.00, "binary", "allowPrivilegeEscalation enabled", t_privesc),
    Template("R_CAPS", "paper1", "alpha2 (5/cap)", "Privilege Escalation", 0.05, "count", "Dangerous Linux capabilities added", t_caps),
    Template("R_HOSTNET", "paper1+paper2", "beta6 (100) / M7", "Lateral Movement", 1.00, "binary", "Pod uses the host network (bypasses NetworkPolicy)", t_hostnet),
    Template("R_HOSTPID", "paper1", "prior", "Privilege Escalation", 0.60, "binary", "Pod shares the host PID namespace", t_hostpid),
    Template("R_HOSTIPC", "paper1", "prior", "Privilege Escalation", 0.40, "binary", "Pod shares the host IPC namespace", t_hostipc),
    Template("R_HOSTPATH", "paper1", "beta7 (100)", "Lateral Movement", 1.00, "count", "hostPath volume mounted", t_hostpath),
    Template("R_HOSTPATH_SENSITIVE", "paper1", "beta7 (100)", "Privilege Escalation", 1.00, "binary", "Sensitive hostPath (docker.sock, /etc, /proc ...)", t_hostpath_sens),
    Template("R_VOL_SECRET", "paper1", "beta8 (10/vol)", "Credential Access", 0.10, "binary", "Secret volume mounted into pod", t_vol_secret),
    Template("R_SVC_LB", "paper1", "beta1 (100)", "Initial Access", 1.00, "binary", "Pod exposed through a LoadBalancer", t_svc_lb),
    Template("R_SVC_NODEPORT", "paper1", "beta1 (100)", "Initial Access", 1.00, "binary", "Pod exposed through a NodePort", t_svc_np),
    Template("R_INGRESS", "paper1", "beta1 (100)", "Initial Access", 1.00, "binary", "Pod exposed through an Ingress", t_ingress),
    Template("R_VULN", "paper1", "alpha1 (1/vuln)", "Initial Access", 0.01, "log", "High/critical image vulnerabilities above dynamic threshold", t_vuln),
    Template("R_IMAGE_OPAQUE", "paper1", "prior", "Initial Access", 0.50, "binary", "Image cannot be scanned (opaque)", t_opaque),
    Template("R_IMAGE_LATEST", "paper1", "prior", "Initial Access", 0.20, "binary", "Mutable ':latest' image tag", t_latest),
    # ---- paper 1: access control (RBAC, service accounts) -----------------------------------
    Template("R_SA_SECRETS", "paper1", "beta4 (10/perm)", "Credential Access", 0.10, "binary", "Pod's service account can read secrets", t_sa_secrets),
    Template("R_SA_CREATE_PODS", "paper1", "beta4 (10/perm)", "Impact", 0.10, "binary", "Pod's service account can create pods", t_sa_create),
    Template("R_SA_EXEC", "paper1", "beta5 (100) / gamma3", "Lateral Movement", 1.00, "binary", "Pod's service account can exec into pods", t_sa_exec),
    Template("R_SA_IMPERSONATE", "paper1", "beta4 (10/perm)", "Privilege Escalation", 0.10, "binary", "Service account can impersonate / escalate / bind", t_sa_imp),
    Template("R_SA_WILDCARD", "paper1", "beta4 (10/perm)", "Privilege Escalation", 0.50, "binary", "Service account bound to wildcard permissions", t_sa_wild),
    Template("R_SA_LIST_PODS", "paper1", "beta4 (10/perm)", "Discovery", 0.10, "binary", "Service account can list pods", t_sa_list),
    Template("R_TOKEN_MOUNT", "paper1", "prior", "Credential Access", 0.05, "binary", "Service-account token auto-mounted", t_token),
    # ---- paper 1: network policies, affinity, hygiene ---------------------------------------
    Template("R_NP_NONE", "paper1", "gamma1 (0.8)", "Lateral Movement", 0.80, "binary", "Pod not selected by any ingress NetworkPolicy", t_np_none),
    Template("R_NP_ALLOW_ALL", "paper1", "gamma1 (0.8)", "Lateral Movement", 0.80, "binary", "NetworkPolicy that allows all ingress (ineffective)", t_np_allow_all),
    Template("R_NP_ORPHAN", "paper1", "gamma1 (0.8)", "Lateral Movement", 0.50, "binary", "NetworkPolicy selecting no pod (silently ineffective)", t_np_orphan),
    Template("R_AFFINITY", "paper1", "gamma2 (0.4)", "Lateral Movement", 0.40, "count", "Pod affinity co-locates pods on one node", t_affinity),
    Template("R_NO_LIMITS", "paper1", "prior", "Impact", 0.20, "binary", "No resource limits (cryptojacking / EDoS)", t_nolimits),
    Template("R_RW_ROOTFS", "paper1", "prior", "Persistence", 0.10, "binary", "Writable root filesystem", t_rwfs),
    Template("R_CAPS_DEFAULT", "paper1", "alpha2 (5/cap)", "Privilege Escalation", 0.05, "count", "Default Linux capabilities not dropped", t_caps_default),
    Template("R_SYSTEM_NS_OPEN", "paper1", "prior", "Lateral Movement", 0.50, "binary", "kube-system pod reachable without NetworkPolicy", t_sys_open),
    # ---- paper 2: network misconfigurations ---------------------------------------------------
    Template("R_M1_UNDECLARED_PORT", "paper2", "M1", "Discovery", 0.40, "binary", "Open port not declared in the manifest", t_m1),
    Template("R_M2_DYNAMIC_PORTS", "paper2", "M2", "Lateral Movement", 0.40, "count", "Ephemeral (dynamic) ports in use", t_m2),
    Template("R_M3_DECLARED_NOT_OPEN", "paper2", "M3", "Credential Access", 0.20, "binary", "Declared port is not open", t_m3),
    Template("R_M4A_UNIT_COLLISION", "paper2", "M4A", "Credential Access", 0.50, "binary", "Different workloads share the same app label", t_m4a),
    Template("R_M4B_SERVICE_COLLISION", "paper2", "M4B", "Credential Access", 0.40, "binary", "Several services share one selector", t_m4b),
    Template("R_M4C_SUBSET_COLLISION", "paper2", "M4C", "Credential Access", 0.40, "binary", "Service selects pods of unrelated workloads", t_m4c),
    Template("R_M4G_GLOBAL_COLLISION", "paper2", "M4*", "Credential Access", 0.30, "binary", "App label reused across namespaces", t_m4g),
    Template("R_M5A_SVC_UNOPENED", "paper2", "M5A", "Impact", 0.30, "binary", "Service targets a declared but unopened port", t_m5a),
    Template("R_M5B_SVC_UNDECLARED", "paper2", "M5B", "Defense Evasion", 0.30, "binary", "Service targets an undeclared port", t_m5b),
    Template("R_M5C_HEADLESS_MISMATCH", "paper2", "M5C", "Impact", 0.20, "binary", "Headless service port not available", t_m5c),
    Template("R_M5D_SVC_NO_TARGET", "paper2", "M5D", "Credential Access", 0.30, "binary", "Service without target pods", t_m5d),
    Template("R_M6_NS_NO_NETPOL", "paper2", "M6", "Lateral Movement", 0.80, "binary", "Namespace without any NetworkPolicy", t_m6),
    # ---- RBAC bindings (paper 1 permissions, extended to group / cross-namespace subjects) ----
    Template("R_BIND_CLUSTER_ADMIN", "paper1", "beta4 (10/perm)", "Privilege Escalation", 1.00, "binary", "Workload bound to cluster-admin", t_bind_admin),
    Template("X_BIND_GROUP_BROADCAST", "extension", "new", "Privilege Escalation", 0.80, "binary", "Powerful role bound to a group covering all service accounts", t_bind_broadcast),
    Template("X_BIND_CROSS_NS", "extension", "new", "Lateral Movement", 0.50, "binary", "Powerful role granted to a service account of another namespace", t_bind_cross),
    Template("X_DEFAULT_SA_POWER", "extension", "new", "Privilege Escalation", 0.50, "count", "Default service account holds powerful permissions", t_default_sa),
    # ---- extension: exceptional scenarios ----------------------------------------------------
    Template("X_NODE_BLAST_RADIUS", "extension", "new", "Lateral Movement", 0.70, "count", "Host-escape-capable pod co-located with sensitive pods", t_node_blast),
    Template("X_NP_BYPASS_HOSTNET", "extension", "new", "Defense Evasion", 0.80, "binary", "Host-network pod nominally covered by a NetworkPolicy", t_np_bypass),
    Template("X_CHAIN_TOKEN_OPEN", "extension", "new", "Credential Access", 0.60, "binary", "Mounted token + powerful permissions + no network isolation", t_chain_token),
    Template("X_CHAIN_EXPOSED_ROOT", "extension", "new", "Execution", 0.60, "binary", "Exposed vulnerable root pod with privilege escalation", t_chain_root),
    Template("X_CHAIN_EXPOSED_HOST", "extension", "new", "Privilege Escalation", 1.00, "binary", "Exposed pod that can escape to the host", t_chain_host),
    Template("X_NOLIMITS_EXPOSED", "extension", "new", "Impact", 0.30, "binary", "Exposed pod without resource limits", t_nolimits_exposed),
    Template("X_COLLISION_POWER", "extension", "new", "Credential Access", 0.50, "binary", "Label collision involving a pod with powerful permissions", t_collide_power),
]
TEMPLATE_IDS = [t.id for t in TEMPLATES]
TEMPLATE_INDEX = {t.id: i for i, t in enumerate(TEMPLATES)}


def evaluate(view: ClusterView, prm: dict = None) -> Dict[str, Inst]:
    prm = prm or {}
    return {t.id: t.fn(view, prm) for t in TEMPLATES}


def transform(kind: str, x: float) -> float:
    if kind == "binary":
        return 1.0
    if kind == "log":
        return math.log1p(x) / math.log(100.0)
    return float(x)
