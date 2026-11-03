"""Synthetic Kubernetes cluster generator (snapshot schema of this project).

Why a generator? The supplied dataset has no labels and only 138 resources. To
*train* a model and to test generalisation across cluster sizes / environments
we synthesise many clusters whose statistics are anchored to:

* the real snapshot in ``snapshots/current.json`` (RBAC permission prior and a
  pool of real ClusterRoles / ClusterRoleBindings used as background objects);
* the misconfiguration rates reported by paper 1 (Blaise & Rebecchi, Table IV:
  root 32.2 %, privileged 4.2 %, hostPath 1 %, host network 0.8 % ...) and
  paper 2 (Bufalino et al., Table 2: M1 65 %, M3 23 %, M6 84 %, M7 4 % ...).

Six environment *profiles* play the role of different datasets/scenarios.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PROFILES: Dict[str, Dict[str, float]] = {
    # chart-sharing style (Bitnami / Banzai): feature-rich defaults, few NPs, many undeclared ports
    "sharing": dict(np=0.12, root=0.40, priv=0.05, privesc=0.40, capadd=0.10, hostnet=0.04, hostpath=0.03,
                    undecl=0.60, dyn=0.12, notopen=0.22, collide=0.12, lb=0.18, nodeport=0.10, ingress=0.15,
                    automount=0.90, rbac=0.45, secrets=0.14, create=0.10, exec=0.06, wild=0.02, imp=0.01,
                    vuln_mu=4.1, vuln_sd=1.1, opaque=0.15, limits=0.45, affinity=0.08, dropall=0.05, rofs=0.10),
    # in-house charts (Wikimedia / EEA): tighter, NP-aware
    "internal": dict(np=0.75, root=0.12, priv=0.01, privesc=0.15, capadd=0.02, hostnet=0.01, hostpath=0.01,
                     undecl=0.25, dyn=0.05, notopen=0.10, collide=0.04, lb=0.03, nodeport=0.02, ingress=0.20,
                     automount=0.60, rbac=0.30, secrets=0.08, create=0.04, exec=0.02, wild=0.005, imp=0.005,
                     vuln_mu=3.4, vuln_sd=1.0, opaque=0.10, limits=0.75, affinity=0.05, dropall=0.25, rofs=0.30),
    # production tooling (CNCF / Prometheus): exporters on host network, hostPath
    "production": dict(np=0.45, root=0.25, priv=0.06, privesc=0.25, capadd=0.08, hostnet=0.09, hostpath=0.07,
                       undecl=0.45, dyn=0.12, notopen=0.20, collide=0.10, lb=0.10, nodeport=0.08, ingress=0.25,
                       automount=0.80, rbac=0.55, secrets=0.18, create=0.08, exec=0.05, wild=0.03, imp=0.01,
                       vuln_mu=3.9, vuln_sd=1.1, opaque=0.12, limits=0.60, affinity=0.10, dropall=0.15, rofs=0.20),
    # zero-trust, hardened posture
    "hardened": dict(np=0.97, root=0.02, priv=0.0, privesc=0.02, capadd=0.0, hostnet=0.0, hostpath=0.0,
                     undecl=0.05, dyn=0.01, notopen=0.03, collide=0.01, lb=0.03, nodeport=0.0, ingress=0.15,
                     automount=0.10, rbac=0.15, secrets=0.02, create=0.01, exec=0.0, wild=0.0, imp=0.0,
                     vuln_mu=2.5, vuln_sd=0.9, opaque=0.05, limits=0.95, affinity=0.03, dropall=0.75, rofs=0.75),
    # permissive legacy estate
    "legacy": dict(np=0.03, root=0.65, priv=0.12, privesc=0.60, capadd=0.20, hostnet=0.07, hostpath=0.10,
                   undecl=0.70, dyn=0.15, notopen=0.25, collide=0.15, lb=0.25, nodeport=0.15, ingress=0.10,
                   automount=0.95, rbac=0.70, secrets=0.25, create=0.15, exec=0.10, wild=0.08, imp=0.03,
                   vuln_mu=4.8, vuln_sd=1.2, opaque=0.20, limits=0.25, affinity=0.08, dropall=0.0, rofs=0.03),
    # many tenants, cross-namespace sharing
    "multitenant": dict(np=0.55, root=0.20, priv=0.03, privesc=0.25, capadd=0.05, hostnet=0.02, hostpath=0.02,
                        undecl=0.40, dyn=0.08, notopen=0.15, collide=0.14, lb=0.10, nodeport=0.05, ingress=0.30,
                        automount=0.70, rbac=0.50, secrets=0.12, create=0.07, exec=0.05, wild=0.02, imp=0.01,
                        vuln_mu=3.8, vuln_sd=1.1, opaque=0.12, limits=0.55, affinity=0.08, dropall=0.15, rofs=0.20),
}
PROFILE_NAMES = list(PROFILES)

TIERS = ("frontend", "backend", "worker", "db", "cache")
TIER_P = (0.22, 0.33, 0.15, 0.15, 0.15)
CALLERS = {"frontend": [], "backend": ["frontend"], "worker": ["backend"], "db": ["backend", "worker"],
           "cache": ["backend", "frontend"]}
HOSTPATHS_SENSITIVE = ["/var/run/docker.sock", "/etc", "/proc", "/var/lib/kubelet"]
HOSTPATHS_BENIGN = ["/var/log", "/data", "/mnt/disk"]
DANGEROUS = ["SYS_ADMIN", "NET_ADMIN", "SYS_PTRACE", "SYS_RESOURCE", "SYS_TIME", "NET_BROADCAST"]


@dataclass
class ClusterSpec:
    profile: str
    n_pods: int
    n_nodes: int
    n_namespaces: int
    seed: int


_REAL_POOL: Optional[dict] = None


def real_pool() -> dict:
    """RBAC objects and permission prior taken from the *real* project snapshot."""
    global _REAL_POOL
    if _REAL_POOL is not None:
        return _REAL_POOL
    path = ROOT / "snapshots" / "current.json"
    roles, bindings = [], []
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for r in data.get("resources", []):
            if r["kind"] == "ClusterRole":
                roles.append(r)
            elif r["kind"] == "ClusterRoleBinding":
                bindings.append(r)
    _REAL_POOL = {"roles": roles, "bindings": bindings}
    return _REAL_POOL


def sample_spec(rng: np.random.Generator, min_pods: int = 8, max_pods: int = 300,
                profile: Optional[str] = None, seed: Optional[int] = None) -> ClusterSpec:
    n_pods = int(round(np.exp(rng.uniform(np.log(min_pods), np.log(max_pods)))))
    n_nodes = int(np.clip(n_pods / rng.uniform(4, 25), 1, 150))
    n_ns = int(np.clip(round(rng.uniform(0.6, 1.4) * np.sqrt(n_pods) / 2), 2, 70))
    return ClusterSpec(profile or str(rng.choice(PROFILE_NAMES)), n_pods, max(1, n_nodes), n_ns,
                       int(rng.integers(0, 2**31 - 1)) if seed is None else seed)


def _rules_for(flags: List[str]) -> List[dict]:
    out = []
    if "secrets" in flags:
        out.append({"api_groups": [""], "resources": ["secrets"], "verbs": ["get", "list"]})
    if "create_pods" in flags:
        out.append({"api_groups": [""], "resources": ["pods"], "verbs": ["create"]})
    if "exec" in flags:
        out.append({"api_groups": [""], "resources": ["pods/exec"], "verbs": ["create", "get"]})
    if "wildcard" in flags:
        out.append({"api_groups": ["*"], "resources": ["*"], "verbs": ["*"]})
    if "impersonate" in flags:
        out.append({"api_groups": [""], "resources": ["users"], "verbs": ["impersonate"]})
    if not out:
        out.append({"api_groups": [""], "resources": ["configmaps"], "verbs": ["get", "list"]})
    return out


def _res(kind: str, name: str, ns: str, data: dict) -> dict:
    return {"kind": kind, "name": name, "namespace": ns, "data": data}


def refresh_aggregates(pod_data: dict) -> None:
    """Keep the legacy aggregate flags consistent (as the real collector does)."""
    cs = pod_data.get("containers") or []
    pod_data["privileged"] = any(c.get("privileged") for c in cs)
    pod_data["run_as_root"] = any(c.get("run_as_root") for c in cs)


def generate_cluster(spec: ClusterSpec) -> dict:
    rng = np.random.default_rng(spec.seed)
    P = PROFILES[spec.profile]
    resources: List[dict] = []
    node_names = [f"node-{i}" for i in range(spec.n_nodes)]
    for n in node_names:
        resources.append(_res("Node", n, "default", {"labels": {"kubernetes.io/hostname": n}}))
    ns_names = ["default"] + [f"ns{i}" for i in range(spec.n_namespaces - 1)]
    ns_has_np = {ns: bool(rng.random() < P["np"]) for ns in ns_names}
    ns_weight = rng.dirichlet(np.ones(len(ns_names)) * 1.5)

    sa_seen: set = set()
    pods_created: Dict[str, List[dict]] = {}
    apps: List[dict] = []

    def add_sa(ns, name, automount=True):
        if (ns, name) not in sa_seen:
            sa_seen.add((ns, name))
            resources.append(_res("ServiceAccount", name, ns,
                                  {"labels": {}, "automount_service_account_token": automount}))

    # --- kube-system (always present, as in every real cluster) -------------------------------
    for r in range(2):
        resources.append(_res("Pod", f"coredns-{r}", "kube-system", {
            "labels": {"k8s-app": "kube-dns", "tier": "infra", "app": "coredns"},
            "service_account_name": "coredns", "host_network": False, "host_pid": False, "host_ipc": False,
            "node_name": node_names[r % len(node_names)], "automount_sa_token": True,
            "containers": [_container(rng, P, hard=True, ports=[53], app_vuln=int(rng.lognormal(1.5, 0.8)))],
            "volumes": [{"type": "configMap"}], "affinity_pods": [], "privileged": False, "run_as_root": False}))
    add_sa("kube-system", "coredns")
    for i, n in enumerate(node_names):
        c = _container(rng, P, hard=False, ports=[10256], app_vuln=int(rng.lognormal(1.5, 0.8)))
        c.update(privileged=True, run_as_root=True)
        resources.append(_res("Pod", f"kube-proxy-{i}", "kube-system", {
            "labels": {"k8s-app": "kube-proxy", "tier": "infra", "app": "kube-proxy"},
            "service_account_name": "kube-proxy", "host_network": True, "host_pid": False, "host_ipc": False,
            "node_name": n, "automount_sa_token": True, "containers": [c],
            "volumes": [{"type": "hostPath", "path": "/run/xtables.lock"}], "affinity_pods": [],
            "privileged": True, "run_as_root": True}))
    add_sa("kube-system", "kube-proxy")
    resources.append(_res("Service", "kube-dns", "kube-system", {
        "type": "ClusterIP", "selector": {"k8s-app": "kube-dns"},
        "ports": [{"port": 53, "target_port": 53}], "labels": {}, "intended_app": "coredns"}))
    if spec.profile in ("hardened", "internal", "multitenant") and rng.random() < 0.7:
        resources.append(_res("NetworkPolicy", "allow-dns", "kube-system", {
            "pod_selector": {"match_labels": {"k8s-app": "kube-dns"}}, "policy_types": ["Ingress"],
            "ingress": [{}], "egress": [], "labels": {}}))

    # --- application workloads ------------------------------------------------------------------
    pod_count = 0
    app_idx = 0
    while pod_count < spec.n_pods:
        ns = str(rng.choice(ns_names, p=ns_weight))
        tier = str(rng.choice(TIERS, p=TIER_P))
        reps = int(min(1 + rng.geometric(0.5) - 1 + (1 if tier != "db" else 0), 8))
        reps = max(1, min(reps, spec.n_pods - pod_count + 1))
        name = f"app{app_idx}"
        app_label = name
        same_ns = [a for a in apps if a["ns"] == ns]
        collide_with = None
        if same_ns and rng.random() < P["collide"] * 0.5:            # M4A compute-unit collision
            collide_with = same_ns[int(rng.integers(len(same_ns)))]
            app_label = collide_with["app_label"]
        labels = {"app": app_label, "tier": tier, "app.kubernetes.io/part-of": f"suite{app_idx // 5}"}
        ports = [8000 + app_idx % 900] + ([9500 + app_idx % 400] if rng.random() < 0.3 else [])
        vul = int(np.clip(rng.lognormal(P["vuln_mu"], P["vuln_sd"]), 0, 3000))
        c = _container(rng, P, hard=False, ports=ports, app_vuln=vul)
        dedicated = rng.random() < max(P["rbac"], 0.2)
        sa = f"{name}-sa" if dedicated else "default"
        automount = bool(rng.random() < P["automount"])
        add_sa(ns, sa, True)
        volumes = [{"type": "emptyDir"}]
        if rng.random() < P["hostpath"]:
            path = str(rng.choice(HOSTPATHS_SENSITIVE if rng.random() < 0.5 else HOSTPATHS_BENIGN))
            volumes.append({"type": "hostPath", "path": path})
        if rng.random() < 0.25:
            volumes.append({"type": "secret"})
        host_net = bool(rng.random() < P["hostnet"])
        host_pid = bool(rng.random() < P["hostnet"] * 0.5)
        host_ipc = bool(rng.random() < P["hostnet"] * 0.3)
        aff = []
        if apps and rng.random() < P["affinity"]:
            tgt = apps[int(rng.integers(len(apps)))]
            aff = [{"match_labels": {"app": tgt["app_label"]}, "ns": tgt["ns"]}]
        app_rec = {"name": name, "ns": ns, "tier": tier, "app_label": app_label, "reps": reps, "sa": sa}
        apps.append(app_rec)
        base_node = int(rng.integers(len(node_names)))
        for r in range(reps):
            node = node_names[(base_node + r) % len(node_names)]
            if aff:
                tgt_pods = [p for p in resources if p["kind"] == "Pod" and p["namespace"] == aff[0]["ns"]
                            and p["data"]["labels"].get("app") == aff[0]["match_labels"]["app"]]
                if tgt_pods:
                    node = tgt_pods[0]["data"]["node_name"]
            pod = _res("Pod", f"{name}-{r}", ns, {
                "labels": dict(labels), "service_account_name": sa,
                "host_network": host_net, "host_pid": host_pid, "host_ipc": host_ipc,
                "node_name": node, "automount_sa_token": automount,
                "containers": [copy.deepcopy(c)], "volumes": copy.deepcopy(volumes),
                "affinity_pods": copy.deepcopy(aff), "privileged": False, "run_as_root": False})
            refresh_aggregates(pod["data"])
            resources.append(pod)
            pod_count += 1
        # service ---------------------------------------------------------------------------------
        if rng.random() < 0.92:
            stype = "ClusterIP"
            if tier == "frontend":
                u = rng.random()
                stype = "LoadBalancer" if u < P["lb"] * 2 else ("NodePort" if u < (P["lb"] + P["nodeport"]) * 2 else "ClusterIP")
            elif rng.random() < P["nodeport"] * 0.3:
                stype = "NodePort"
            selector = {"app": app_label}
            if rng.random() < P["collide"] * 0.4:                          # M4C subset collision
                selector = {"tier": tier}
            tport = ports[0]
            u = rng.random()
            if u < 0.04:                                                  # M5A target declared but not open
                tport = ports[0]
                for pod in resources[-reps:]:
                    for cc in pod["data"]["containers"]:
                        cc["ports_open"] = [p for p in cc["ports_open"] if p != tport]
            elif u < 0.11:                                                # M5B target undeclared
                tport = 7000 + app_idx % 500
            headless = bool(rng.random() < 0.03)
            if rng.random() < 0.01:                                       # M5D service without target
                selector = {"app": f"ghost{app_idx}"}
            resources.append(_res("Service", f"{name}-svc", ns, {
                "type": stype, "selector": selector, "headless": headless,
                "ports": [{"port": 80, "target_port": tport}], "labels": {}, "intended_app": app_label}))
            if rng.random() < P["collide"] * 0.3:                          # M4B service label collision
                resources.append(_res("Service", f"{name}-alt", ns, {
                    "type": "ClusterIP", "selector": dict(selector), "headless": False,
                    "ports": [{"port": 81, "target_port": tport}], "labels": {}, "intended_app": app_label}))
            if stype == "ClusterIP" and tier == "frontend" and rng.random() < P["ingress"] * 2:
                resources.append(_res("Ingress", f"{name}-ing", ns, {"rules": [{"http": {"paths": [
                    {"backend": {"service": {"name": f"{name}-svc"}}}]}}], "default_backend": None}))
        # RBAC ------------------------------------------------------------------------------------
        if dedicated and rng.random() < max(P["rbac"], 0.2):
            flags = [f for f in ("secrets", "create_pods", "exec", "wildcard", "impersonate")
                     if rng.random() < {"secrets": P["secrets"], "create_pods": P["create"],
                                        "exec": P["exec"], "wildcard": P["wild"], "impersonate": P["imp"]}[f]]
            cluster_scope = bool(flags) and rng.random() < 0.15
            if cluster_scope:
                resources.append(_res("ClusterRole", f"{name}-cr", "default", {"labels": {}, "rules": _rules_for(flags)}))
                resources.append(_res("ClusterRoleBinding", f"{name}-crb", "default", {
                    "labels": {}, "role_ref": {"kind": "ClusterRole", "name": f"{name}-cr"},
                    "subjects": [{"kind": "ServiceAccount", "name": sa, "namespace": ns}]}))
            else:
                resources.append(_res("Role", f"{name}-role", ns, {"labels": {}, "rules": _rules_for(flags)}))
                resources.append(_res("RoleBinding", f"{name}-rb", ns, {
                    "labels": {}, "role_ref": {"kind": "Role", "name": f"{name}-role"},
                    "subjects": [{"kind": "ServiceAccount", "name": sa, "namespace": ns}]}))
        app_idx += 1

    # --- NetworkPolicies --------------------------------------------------------------------------
    for ns in ns_names:
        ns_apps = [a for a in apps if a["ns"] == ns]
        if not ns_apps:
            continue
        if ns_has_np[ns]:
            allow_all = bool(rng.random() < 0.05)
            if rng.random() < 0.35:                                       # optional namespace-wide default-deny
                resources.append(_res("NetworkPolicy", "default-deny", ns, {
                    "pod_selector": {"match_labels": {}}, "policy_types": ["Ingress"],
                    "ingress": [{}] if allow_all else [], "egress": [], "labels": {}}))
            for a in ns_apps:                                             # per-app policy restricting callers
                callers = CALLERS[a["tier"]]
                if a["tier"] == "frontend":
                    ing = [{}] if allow_all or rng.random() < 0.8 else [{"_from": [{"pod_selector": {"match_labels": {"tier": "frontend"}}}]}]
                else:
                    ing = [{"_from": [{"pod_selector": {"match_labels": {"tier": t}}} for t in callers]}] if callers else []
                resources.append(_res("NetworkPolicy", f"allow-{a['name']}", ns, {
                    "pod_selector": {"match_labels": {"app": a["app_label"]}}, "policy_types": ["Ingress"],
                    "ingress": ing, "egress": [], "labels": {}}))
        elif rng.random() < 0.10:                                         # lone partial policy
            a = ns_apps[int(rng.integers(len(ns_apps)))]
            resources.append(_res("NetworkPolicy", f"lone-{a['name']}", ns, {
                "pod_selector": {"match_labels": {"app": a["app_label"]}}, "policy_types": ["Ingress"],
                "ingress": [], "egress": [], "labels": {}}))

    # --- real RBAC background pool (from the supplied dataset) ------------------------------------
    pool = real_pool()
    if pool["roles"]:
        k = int(np.clip(spec.n_pods // 10, 8, 60))
        for r in rng.choice(len(pool["roles"]), size=min(k, len(pool["roles"])), replace=False):
            resources.append(copy.deepcopy(pool["roles"][int(r)]))
        kb = int(np.clip(spec.n_pods // 12, 6, 45))
        for r in rng.choice(len(pool["bindings"]), size=min(kb, len(pool["bindings"])), replace=False):
            resources.append(copy.deepcopy(pool["bindings"][int(r)]))

    resources.sort(key=lambda r: (r["kind"], r["namespace"], r["name"]))
    return {"timestamp": "synthetic", "configuration_hash": f"synthetic-{spec.seed}", "resources": resources,
            "meta": {"profile": spec.profile, "n_pods": spec.n_pods, "n_nodes": spec.n_nodes,
                     "n_namespaces": spec.n_namespaces, "seed": spec.seed}}


def _container(rng: np.random.Generator, P: Dict[str, float], hard: bool, ports: List[int], app_vuln: int) -> dict:
    root = (not hard) and rng.random() < P["root"]
    priv = (not hard) and rng.random() < P["priv"]
    caps_add = []
    if (not hard) and rng.random() < P["capadd"]:
        caps_add = list(rng.choice(DANGEROUS, size=int(rng.integers(1, 3)), replace=False))
    opaque = rng.random() < P["opaque"]
    open_ports = [p for p in ports if rng.random() >= P["notopen"]]
    declared = list(ports)
    if rng.random() < P["undecl"] * 0.7:                                   # M1 undeclared open port
        open_ports.append(int(20000 + rng.integers(0, 9000)))
    dyn = int(rng.integers(1, 4)) if rng.random() < P["dyn"] else 0         # M2 ephemeral ports
    return {
        "name": "main", "image": f"registry.example/img{int(rng.integers(0, 10**6))}:{'latest' if rng.random() < 0.08 else '1.' + str(int(rng.integers(0, 30)))}",
        "privileged": bool(priv), "run_as_root": bool(root),
        "allow_privilege_escalation": bool((not hard) and rng.random() < P["privesc"]),
        "capabilities_add": [str(c) for c in caps_add],
        "caps_drop_all": bool(rng.random() < P["dropall"]),
        "read_only_root_fs": bool(rng.random() < P["rofs"]),
        "ports_declared": declared, "ports_open": [int(p) for p in open_ports], "ports_dynamic": dyn,
        "vuln_high_critical": -1 if opaque else int(app_vuln),
        "has_limits": bool(rng.random() < P["limits"] or hard), "seccomp": None,
    }
