"""Drift injectors and the scenario catalogue (benign, covered-risky, *exceptional*).

A scenario edits a baseline snapshot B into a drifted snapshot C and reports what
it did. Categories:

* ``benign``       harmless or risk-reducing drift (false-alarm traps included)
* ``covered``      a single risky change that the original static rules and/or the base
                   papers model (privileged, root, NP removed, LoadBalancer, host network,
                   hostPath, capabilities, RBAC grants, M1-M7 misconfigurations ...)
* ``exceptional``  scenarios *not considered* by the original project or the base papers:
                   chained drift, silently-ineffective NetworkPolicies, group/cross-namespace
                   RBAC broadcast, node blast radius, cryptojacking, slow-burn accumulation,
                   isolated-privileged false-positive traps ...
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from app.dynamic.view import ClusterView, PodRec, SYSTEM_NAMESPACES
from app.simulation.generator import (DANGEROUS, HOSTPATHS_SENSITIVE, _rules_for, refresh_aggregates)


class MutCtx:
    def __init__(self, snapshot: dict, view: ClusterView, rng: np.random.Generator):
        self.base = snapshot
        self.view = view
        self.rng = rng
        self.res: List[Optional[dict]] = list(snapshot["resources"])
        self.pos: Dict[Tuple[str, str, str], int] = {(r["kind"], r["namespace"], r["name"]): i for i, r in enumerate(self.res)}
        self.cloned: set = set()
        self.events: List[dict] = []
        self.counter = 0
        # workloads = replicas sharing a name prefix inside a namespace (not kube-system)
        self.workloads: Dict[Tuple[str, str], List[PodRec]] = {}
        for p in view.pods:
            if p.ns in SYSTEM_NAMESPACES:
                continue
            self.workloads.setdefault((p.ns, p.name.rsplit("-", 1)[0]), []).append(p)

    # ------------------------------------------------------------------ helpers
    def edit(self, kind: str, ns: str, name: str) -> Optional[dict]:
        i = self.pos.get((kind, ns, name))
        if i is None or self.res[i] is None:
            return None
        if i not in self.cloned:
            self.res[i] = copy.deepcopy(self.res[i])
            self.cloned.add(i)
        return self.res[i]

    def edit_workload(self, w: Tuple[str, str], fn: Callable[[dict], None]) -> None:
        for p in self.workloads[w]:
            r = self.edit("Pod", p.ns, p.name)
            if r is not None:
                fn(r["data"])
                refresh_aggregates(r["data"])

    def add(self, r: dict) -> None:
        k = (r["kind"], r["namespace"], r["name"])
        if k in self.pos and self.res[self.pos[k]] is not None:
            return
        self.pos[k] = len(self.res)
        self.res.append(r)
        self.cloned.add(self.pos[k])

    def remove(self, kind: str, ns: str, name: str) -> None:
        i = self.pos.get((kind, ns, name))
        if i is not None:
            self.res[i] = None

    def uid(self) -> int:
        self.counter += 1
        return int(self.rng.integers(10**6)) * 10 + self.counter

    def pick(self, pred: Optional[Callable[[PodRec], bool]] = None) -> Optional[Tuple[str, str]]:
        c = [w for w, pods in self.workloads.items() if pods and (pred is None or pred(pods[0]))]
        return c[int(self.rng.integers(len(c)))] if c else None

    def finish(self) -> dict:
        out = dict(self.base)
        out["resources"] = [r for r in self.res if r is not None]
        return out

    def log(self, name: str, category: str, targets: List[str]) -> dict:
        ev = {"mutation": name, "category": category, "targets": targets}
        self.events.append(ev)
        return ev


def _wl(w): return f"{w[0]}/{w[1]}"


def _set_c(d: dict, **kw):
    for c in d["containers"]:
        c.update(kw)


# ============================================================ benign / hardening
def m_image_bump(ctx):
    w = ctx.pick()
    if not w:
        return None
    f = float(ctx.rng.uniform(0.5, 1.1))
    def fn(d):
        for c in d["containers"]:
            c["image"] = f"registry.example/bumped{ctx.uid()}:1.{int(ctx.rng.integers(30, 60))}"
            if c.get("vuln_high_critical", 0) >= 0:
                c["vuln_high_critical"] = int(c["vuln_high_critical"] * f)
    ctx.edit_workload(w, fn)
    return ctx.log("image_bump", "routine", [_wl(w)])


def m_scale(ctx):
    w = ctx.pick()
    if not w:
        return None
    pods = ctx.workloads[w]
    base = next(r for r in ctx.res if r is not None and r["kind"] == "Pod" and r["namespace"] == w[0] and r["name"] == pods[0].name)
    nodes = sorted(ctx.view.nodes)
    for k in range(int(ctx.rng.integers(1, 4))):
        nr = copy.deepcopy(base)
        nr["name"] = f"{w[1]}-{len(pods) + k}"
        nr["data"]["node_name"] = str(ctx.rng.choice(nodes))
        ctx.add(nr)
    return ctx.log("scale_out", "routine", [_wl(w)])


def m_label_noise(ctx):
    w = ctx.pick()
    if not w:
        return None
    ctx.edit_workload(w, lambda d: d["labels"].update({"release": f"r{ctx.uid() % 100}"}))
    return ctx.log("label_noise", "routine", [_wl(w)])


def m_reschedule(ctx):
    w = ctx.pick()
    if not w:
        return None
    nodes = sorted(ctx.view.nodes)
    node = str(ctx.rng.choice(nodes))
    ctx.edit_workload(w, lambda d: d.update(node_name=node))
    return ctx.log("reschedule", "routine", [_wl(w)])


def m_limits(ctx):
    w = ctx.pick(lambda p: not p.has_limits)
    if not w:
        return None
    ctx.edit_workload(w, lambda d: _set_c(d, has_limits=True))
    return ctx.log("limits_added", "routine", [_wl(w)])


def m_drop_priv(ctx):
    w = ctx.pick(lambda p: p.privileged or p.root or p.priv_esc or bool(p.caps_extra) or p.host_network)
    if not w:
        return None
    def fn(d):
        _set_c(d, privileged=False, run_as_root=False, allow_privilege_escalation=False, capabilities_add=[])
        d["host_network"] = False
    ctx.edit_workload(w, fn)
    return ctx.log("hardening_drop_privileges", "routine", [_wl(w)])


def m_token_off(ctx):
    w = ctx.pick(lambda p: p.automount)
    if not w:
        return None
    ctx.edit_workload(w, lambda d: d.update(automount_sa_token=False))
    return ctx.log("hardening_token_off", "routine", [_wl(w)])


def m_np_add(ctx):
    nss = [n for n in ctx.view.pods_by_ns if n not in SYSTEM_NAMESPACES and not any(p["ns"] == n for p in ctx.view.netpols)]
    if not nss:
        return None
    ns = str(ctx.rng.choice(nss))
    ctx.add({"kind": "NetworkPolicy", "name": "default-deny", "namespace": ns, "data": {
        "pod_selector": {"match_labels": {}}, "policy_types": ["Ingress"], "ingress": [], "egress": [], "labels": {}}})
    seen = set()
    for i in ctx.view.pods_by_ns[ns]:
        p = ctx.view.pods[i]
        app = p.labels.get("app")
        if app in seen:
            continue
        seen.add(app)
        tier = p.labels.get("tier")
        callers = {"backend": ["frontend"], "worker": ["backend"], "db": ["backend", "worker"], "cache": ["backend", "frontend"]}.get(tier)
        ing = [{}] if tier == "frontend" else ([{"_from": [{"pod_selector": {"match_labels": {"tier": t}}} for t in callers]}] if callers else [])
        ctx.add({"kind": "NetworkPolicy", "name": f"allow-{app}", "namespace": ns, "data": {
            "pod_selector": {"match_labels": {"app": app}}, "policy_types": ["Ingress"], "ingress": ing, "egress": [], "labels": {}}})
    return ctx.log("hardening_np_added", "routine", [ns])


def m_add_app(ctx):
    w = ctx.pick()
    if not w:
        return None
    pods = ctx.workloads[w]
    src = next(r for r in ctx.res if r is not None and r["kind"] == "Pod" and r["namespace"] == w[0] and r["name"] == pods[0].name)
    n = ctx.uid()
    nr = copy.deepcopy(src)
    nr["name"] = f"newapp{n}-0"
    nr["data"]["labels"] = {"app": f"newapp{n}", "tier": "backend", "app.kubernetes.io/part-of": f"suite{n % 7}"}
    nr["data"]["service_account_name"] = "default"
    nr["data"]["host_network"] = False
    for c in nr["data"]["containers"]:
        c.update(privileged=False, run_as_root=False, allow_privilege_escalation=False, capabilities_add=[], vuln_high_critical=int(ctx.rng.integers(0, 15)), has_limits=True)
    nr["data"]["automount_sa_token"] = False
    refresh_aggregates(nr["data"])
    ctx.add(nr)
    ctx.add({"kind": "Service", "name": f"newapp{n}-svc", "namespace": w[0], "data": {
        "type": "ClusterIP", "selector": {"app": f"newapp{n}"}, "headless": False,
        "ports": [{"port": 80, "target_port": (nr["data"]["containers"][0]["ports_declared"] or [8080])[0]}],
        "labels": {}, "intended_app": f"newapp{n}"}})
    return ctx.log("add_hardened_app", "routine", [f"{w[0]}/newapp{n}"])


def m_delete_app(ctx):
    w = ctx.pick()
    if not w:
        return None
    for p in ctx.workloads[w]:
        ctx.remove("Pod", p.ns, p.name)
    return ctx.log("delete_app", "routine", [_wl(w)])


# ============================================================ covered risky (single mutations)
def m_priv_on(ctx):
    w = ctx.pick(lambda p: not p.privileged)
    if not w:
        return None
    ctx.edit_workload(w, lambda d: _set_c(d, privileged=True))
    return ctx.log("privileged_on", "covered", [_wl(w)])


def m_root_on(ctx):
    w = ctx.pick(lambda p: not p.root)
    if not w:
        return None
    ctx.edit_workload(w, lambda d: _set_c(d, run_as_root=True))
    return ctx.log("root_on", "covered", [_wl(w)])


def m_privesc_on(ctx):
    w = ctx.pick(lambda p: not p.priv_esc)
    if not w:
        return None
    ctx.edit_workload(w, lambda d: _set_c(d, allow_privilege_escalation=True))
    return ctx.log("privesc_on", "covered", [_wl(w)])


def m_cap_add(ctx):
    w = ctx.pick()
    if not w:
        return None
    caps = [str(c) for c in ctx.rng.choice(DANGEROUS, size=int(ctx.rng.integers(1, 4)), replace=False)]
    ctx.edit_workload(w, lambda d: [c.__setitem__("capabilities_add", sorted(set(c.get("capabilities_add", [])) | set(caps))) for c in d["containers"]])
    return ctx.log("capability_added", "covered", [_wl(w)])


def m_hostnet_on(ctx):
    w = ctx.pick(lambda p: not p.host_network)
    if not w:
        return None
    ctx.edit_workload(w, lambda d: d.update(host_network=True))
    return ctx.log("host_network_on", "covered", [_wl(w)])


def m_hostpath(ctx):
    w = ctx.pick(lambda p: not p.sensitive_hostpath)
    if not w:
        return None
    path = str(ctx.rng.choice(HOSTPATHS_SENSITIVE))
    ctx.edit_workload(w, lambda d: d["volumes"].append({"type": "hostPath", "path": path}))
    return ctx.log("hostpath_mounted", "covered", [_wl(w)])


def m_svc_expose(ctx):
    c = [s for s in ctx.view.services if s["type"] == "ClusterIP" and s["pods"] and s["ns"] not in SYSTEM_NAMESPACES]
    if not c:
        return None
    s = c[int(ctx.rng.integers(len(c)))]
    r = ctx.edit("Service", s["ns"], s["name"])
    r["data"]["type"] = "LoadBalancer" if ctx.rng.random() < 0.6 else "NodePort"
    return ctx.log("service_exposed_" + r["data"]["type"], "covered", [s["key"]])


def m_np_removed(ctx):
    c = [p for p in ctx.view.netpols if p["ns"] not in SYSTEM_NAMESPACES]
    if not c:
        return None
    p = c[int(ctx.rng.integers(len(c)))]
    ctx.remove("NetworkPolicy", p["ns"], p["name"])
    return ctx.log("network_policy_removed", "covered", [p["key"]])


def m_np_allow_all(ctx):
    c = [p for p in ctx.view.netpols if p["ns"] not in SYSTEM_NAMESPACES and not any(r["allow_all"] for r in p["rules"])]
    if not c:
        return None
    p = c[int(ctx.rng.integers(len(c)))]
    r = ctx.edit("NetworkPolicy", p["ns"], p["name"])
    r["data"]["ingress"] = [{}]
    return ctx.log("network_policy_allow_all", "covered", [p["key"]])


def m_vuln_up(ctx):
    w = ctx.pick()
    if not w:
        return None
    def fn(d):
        for c in d["containers"]:
            c["image"] = f"registry.example/vuln{ctx.uid()}:0.1"
            c["vuln_high_critical"] = int(max(c.get("vuln_high_critical", 0), 0) * ctx.rng.uniform(1.5, 4) + ctx.rng.integers(30, 300))
    ctx.edit_workload(w, fn)
    return ctx.log("vulnerable_image_deployed", "covered", [_wl(w)])


def _ensure_sa(ctx, w) -> Tuple[str, str]:
    ns = w[0]
    sa = f"{w[1]}-sa"
    ctx.add({"kind": "ServiceAccount", "name": sa, "namespace": ns, "data": {"labels": {}, "automount_service_account_token": True}})
    ctx.edit_workload(w, lambda d: d.update(service_account_name=sa))
    return ns, sa


def _grant(ctx, ns: str, sa: str, flags: List[str], sa_ns: Optional[str] = None, cluster: bool = False):
    n = ctx.uid()
    rules = _rules_for(flags)
    if cluster:
        ctx.add({"kind": "ClusterRole", "name": f"drift-cr-{n}", "namespace": "default", "data": {"labels": {}, "rules": rules}})
        ctx.add({"kind": "ClusterRoleBinding", "name": f"drift-crb-{n}", "namespace": "default", "data": {
            "labels": {}, "role_ref": {"kind": "ClusterRole", "name": f"drift-cr-{n}"},
            "subjects": [{"kind": "ServiceAccount", "name": sa, "namespace": sa_ns or ns}]}})
    else:
        ctx.add({"kind": "Role", "name": f"drift-role-{n}", "namespace": ns, "data": {"labels": {}, "rules": rules}})
        ctx.add({"kind": "RoleBinding", "name": f"drift-rb-{n}", "namespace": ns, "data": {
            "labels": {}, "role_ref": {"kind": "Role", "name": f"drift-role-{n}"},
            "subjects": [{"kind": "ServiceAccount", "name": sa, "namespace": sa_ns or ns}]}})


def _rbac(flag):
    def fn(ctx):
        w = ctx.pick()
        if not w:
            return None
        ns, sa = _ensure_sa(ctx, w)
        _grant(ctx, ns, sa, [flag], cluster=bool(ctx.rng.random() < 0.25))
        return ctx.log(f"rbac_grant_{flag}", "covered", [_wl(w)])
    return fn


def m_cluster_admin(ctx):
    w = ctx.pick()
    if not w:
        return None
    ns, sa = _ensure_sa(ctx, w)
    n = ctx.uid()
    ctx.add({"kind": "ClusterRoleBinding", "name": f"drift-admin-{n}", "namespace": "default", "data": {
        "labels": {}, "role_ref": {"kind": "ClusterRole", "name": "cluster-admin"},
        "subjects": [{"kind": "ServiceAccount", "name": sa, "namespace": ns}]}})
    return ctx.log("cluster_admin_bound", "covered", [_wl(w)])


def m_port_m1(ctx):
    w = ctx.pick()
    if not w:
        return None
    port = int(ctx.rng.integers(20000, 29000))
    ctx.edit_workload(w, lambda d: [c.setdefault("ports_open", []).append(port) for c in d["containers"]])
    return ctx.log("undeclared_port_M1", "covered", [_wl(w)])


def m_port_m2(ctx):
    w = ctx.pick(lambda p: p.dyn_ports == 0)
    if not w:
        return None
    ctx.edit_workload(w, lambda d: _set_c(d, ports_dynamic=int(ctx.rng.integers(1, 4))))
    return ctx.log("dynamic_ports_M2", "covered", [_wl(w)])


def m_port_m3(ctx):
    w = ctx.pick()
    if not w:
        return None
    port = int(ctx.rng.integers(30000, 31000))
    ctx.edit_workload(w, lambda d: [c.setdefault("ports_declared", []).append(port) for c in d["containers"]])
    return ctx.log("declared_not_open_M3", "covered", [_wl(w)])


def m_collide(ctx):
    ws = [w for w in ctx.workloads if ctx.workloads[w]]
    if len(ws) < 2:
        return None
    a = ws[int(ctx.rng.integers(len(ws)))]
    same = [w for w in ws if w[0] == a[0] and w != a]
    if not same:
        return None
    b = same[int(ctx.rng.integers(len(same)))]
    app = ctx.workloads[a][0].labels.get("app")
    ctx.edit_workload(b, lambda d: d["labels"].update({"app": app}))
    return ctx.log("label_collision_M4", "covered", [_wl(a), _wl(b)])


def m_svc_misref(ctx):
    c = [s for s in ctx.view.services if s["pods"] and s["ns"] not in SYSTEM_NAMESPACES]
    if not c:
        return None
    s = c[int(ctx.rng.integers(len(c)))]
    r = ctx.edit("Service", s["ns"], s["name"])
    u = ctx.rng.random()
    if u < 0.4:
        r["data"]["ports"] = [{"port": 80, "target_port": int(ctx.rng.integers(6000, 7000))}]
        nm = "M5B"
    elif u < 0.7:
        nm = "M5A"
        p0 = ctx.view.pods[s["pods"][0]]
        port = next(iter(p0.ports_declared), None)
        if port is None:
            return None
        r["data"]["ports"] = [{"port": 80, "target_port": port}]
        for i in s["pods"]:
            pp = ctx.view.pods[i]
            pr = ctx.edit("Pod", pp.ns, pp.name)
            for cc in pr["data"]["containers"]:
                cc["ports_open"] = [x for x in cc.get("ports_open", []) if x != port]
    else:
        r["data"]["selector"] = {"app": f"ghost{ctx.uid()}"}
        nm = "M5D"
    return ctx.log(f"service_misreference_{nm}", "covered", [s["key"]])


# ============================================================ exceptional scenarios
def x_chain(ctx):
    """Compound drift on ONE workload: token mounted + permissions + loosened network + new image."""
    w = ctx.pick()
    if not w:
        return None
    ns, sa = _ensure_sa(ctx, w)
    ctx.edit_workload(w, lambda d: d.update(automount_sa_token=True))
    _grant(ctx, ns, sa, [str(ctx.rng.choice(["secrets", "exec", "create_pods"]))])
    for p in ctx.view.netpols:
        if p["ns"] == ns and ctx.rng.random() < 0.7:
            ctx.remove("NetworkPolicy", p["ns"], p["name"])
    if ctx.rng.random() < 0.6:
        ctx.edit_workload(w, lambda d: [c.update(vuln_high_critical=int(max(c.get("vuln_high_critical", 0), 0) + ctx.rng.integers(40, 250))) for c in d["containers"]])
    return ctx.log("chain_token_perms_netloosen", "exceptional", [_wl(w)])


def x_np_selector_drift(ctx):
    """NetworkPolicy still exists but its selector no longer matches (silently ineffective)."""
    c = [p for p in ctx.view.netpols if p["ns"] not in SYSTEM_NAMESPACES and p["selected"] and p["applies_ingress"]]
    if not c:
        return None
    p = c[int(ctx.rng.integers(len(c)))]
    r = ctx.edit("NetworkPolicy", p["ns"], p["name"])
    sel = r["data"].get("pod_selector") or {}
    r["data"]["pod_selector"] = {"match_labels": {"app": f"renamed-{ctx.uid()}"}}
    return ctx.log("np_selector_drift", "exceptional", [p["key"]])


def x_group_binding(ctx):
    n = ctx.uid()
    flag = str(ctx.rng.choice(["secrets", "exec", "create_pods"]))
    ctx.add({"kind": "ClusterRole", "name": f"bcast-cr-{n}", "namespace": "default", "data": {"labels": {}, "rules": _rules_for([flag])}})
    ctx.add({"kind": "ClusterRoleBinding", "name": f"bcast-crb-{n}", "namespace": "default", "data": {
        "labels": {}, "role_ref": {"kind": "ClusterRole", "name": f"bcast-cr-{n}"},
        "subjects": [{"kind": "Group", "name": "system:serviceaccounts"}]}})
    return ctx.log("group_binding_broadcast", "exceptional", [f"group:system:serviceaccounts:{flag}"])


def x_cross_ns(ctx):
    ws = list(ctx.workloads)
    if len(ws) < 2:
        return None
    a = ws[int(ctx.rng.integers(len(ws)))]
    others = [w for w in ws if w[0] != a[0]]
    if not others:
        return None
    victim = others[int(ctx.rng.integers(len(others)))]
    ns, sa = _ensure_sa(ctx, a)
    flag = str(ctx.rng.choice(["secrets", "exec"]))
    _grant(ctx, victim[0], sa, [flag], sa_ns=ns)                # RoleBinding in victim ns, subject SA from ns
    return ctx.log("cross_namespace_subject", "exceptional", [_wl(a), victim[0]])


def x_node_colocate(ctx):
    """Privileged / hostPath workload rescheduled onto the node that hosts the most sensitive pods."""
    best, best_n = None, -1
    for node, idxs in ctx.view.pods_by_node.items():
        n = sum(1 for i in idxs if ctx.view.sensitive_pod(ctx.view.pods[i]))
        if n > best_n:
            best, best_n = node, n
    w = ctx.pick(lambda p: not p.privileged and p.ns not in SYSTEM_NAMESPACES)
    if not w or best is None:
        return None
    def fn(d):
        d["node_name"] = best
        if ctx.rng.random() < 0.5:
            _set_c(d, privileged=True)
        else:
            d["volumes"].append({"type": "hostPath", "path": "/var/run/docker.sock"})
    ctx.edit_workload(w, fn)
    return ctx.log("node_colocation_blast_radius", "exceptional", [_wl(w), best])


def x_cryptojack(ctx):
    w = ctx.pick(lambda p: p.has_limits and ctx.view.exposure[p.idx] > 0)
    if not w:
        w = ctx.pick(lambda p: p.has_limits)
    if not w:
        return None
    ctx.edit_workload(w, lambda d: _set_c(d, has_limits=False))
    return ctx.log("limits_removed_cryptojack", "exceptional", [_wl(w)])


def x_isolated_priv(ctx):
    """False-positive trap: privileged monitoring pod inside a deny-all namespace, no token, clean image."""
    ns = f"monitor{ctx.uid() % 50}"
    ctx.add({"kind": "NetworkPolicy", "name": "default-deny", "namespace": ns, "data": {
        "pod_selector": {"match_labels": {}}, "policy_types": ["Ingress"], "ingress": [], "egress": [], "labels": {}}})
    src = next((r for r in ctx.res if r is not None and r["kind"] == "Pod" and r["namespace"] not in SYSTEM_NAMESPACES), None)
    if src is None:
        return None
    nr = copy.deepcopy(src)
    nr["namespace"] = ns
    nr["name"] = f"agent{ctx.uid()}-0"
    nr["data"]["labels"] = {"app": nr["name"], "tier": "worker"}
    nr["data"]["automount_sa_token"] = False
    nr["data"]["service_account_name"] = "default"
    nr["data"]["affinity_pods"] = []
    for c in nr["data"]["containers"]:
        c.update(privileged=True, vuln_high_critical=int(ctx.rng.integers(0, 8)), ports_open=[], ports_declared=[], ports_dynamic=0, has_limits=True)
    refresh_aggregates(nr["data"])
    ctx.add(nr)
    return ctx.log("isolated_privileged_agent", "exceptional", [f"{ns}/{nr['name']}"])


def x_default_sa(ctx):
    """Powerful role bound to a namespace's *default* ServiceAccount -> every pod using it inherits."""
    ns_c = [n for n in ctx.view.pods_by_ns if n not in SYSTEM_NAMESPACES and any(
        ctx.view.pods[i].sa == "default" for i in ctx.view.pods_by_ns[n])]
    if not ns_c:
        return None
    ns = str(ctx.rng.choice(ns_c))
    _grant(ctx, ns, "default", [str(ctx.rng.choice(["secrets", "exec", "create_pods"]))])
    return ctx.log("default_sa_granted", "exceptional", [ns])


def x_slow_burn(ctx):
    """Many individually mild drifts accumulating (no single change crosses a threshold)."""
    k = int(ctx.rng.integers(4, 9))
    fns = [m_root_on, m_privesc_on, m_cap_add, m_vuln_up, m_port_m1, m_limits_off_mild, m_token_on_mild]
    n_ok = 0
    for _ in range(k):
        f = fns[int(ctx.rng.integers(len(fns)))]
        if f(ctx):
            n_ok += 1
    if not n_ok:
        return None
    return ctx.log("slow_burn_accumulation", "exceptional", [f"{n_ok} mild changes"])


def m_limits_off_mild(ctx):
    w = ctx.pick(lambda p: p.has_limits)
    if not w:
        return None
    ctx.edit_workload(w, lambda d: _set_c(d, has_limits=False))
    return ctx.log("limits_removed", "covered", [_wl(w)])


def m_token_on_mild(ctx):
    w = ctx.pick(lambda p: not p.automount)
    if not w:
        return None
    ctx.edit_workload(w, lambda d: d.update(automount_sa_token=True))
    return ctx.log("token_enabled", "covered", [_wl(w)])


# ============================================================ catalogue
# (name, category, weight, function)
SCENARIOS: List[Tuple[str, str, float, Callable]] = [
    ("image_bump", "routine", 3, m_image_bump), ("scale_out", "routine", 3, m_scale),
    ("label_noise", "routine", 3, m_label_noise), ("reschedule", "routine", 2, m_reschedule),
    ("limits_added", "routine", 1, m_limits), ("hardening_drop_privileges", "routine", 2, m_drop_priv),
    ("hardening_token_off", "routine", 1, m_token_off), ("hardening_np_added", "routine", 2, m_np_add),
    ("add_hardened_app", "routine", 2, m_add_app), ("delete_app", "routine", 2, m_delete_app),
    ("privileged_on", "covered", 3, m_priv_on), ("root_on", "covered", 2, m_root_on),
    ("privesc_on", "covered", 2, m_privesc_on), ("capability_added", "covered", 2, m_cap_add),
    ("host_network_on", "covered", 2, m_hostnet_on), ("hostpath_mounted", "covered", 2, m_hostpath),
    ("service_exposed", "covered", 3, m_svc_expose), ("network_policy_removed", "covered", 3, m_np_removed),
    ("network_policy_allow_all", "covered", 1, m_np_allow_all), ("vulnerable_image_deployed", "covered", 3, m_vuln_up),
    ("rbac_grant_secrets", "covered", 1.5, _rbac("secrets")), ("rbac_grant_create_pods", "covered", 1.5, _rbac("create_pods")),
    ("rbac_grant_exec", "covered", 1.5, _rbac("exec")), ("cluster_admin_bound", "covered", 1, m_cluster_admin),
    ("undeclared_port_M1", "covered", 1.5, m_port_m1), ("dynamic_ports_M2", "covered", 1, m_port_m2),
    ("declared_not_open_M3", "covered", 1, m_port_m3), ("label_collision_M4", "covered", 1.5, m_collide),
    ("service_misreference_M5", "covered", 1.5, m_svc_misref),
    ("chain_token_perms_netloosen", "exceptional", 3, x_chain), ("np_selector_drift", "exceptional", 2.5, x_np_selector_drift),
    ("group_binding_broadcast", "exceptional", 2, x_group_binding), ("cross_namespace_subject", "exceptional", 2, x_cross_ns),
    ("node_colocation_blast_radius", "exceptional", 2.5, x_node_colocate), ("limits_removed_cryptojack", "exceptional", 1.5, x_cryptojack),
    ("isolated_privileged_agent", "exceptional", 2, x_isolated_priv), ("default_sa_granted", "exceptional", 2, x_default_sa),
    ("slow_burn_accumulation", "exceptional", 2.5, x_slow_burn),
]
CATEGORY_WEIGHT = {"routine": 0.30, "covered": 0.38, "exceptional": 0.32}
NOISE = [m_label_noise, m_image_bump, m_reschedule]


def sample_scenario(rng: np.random.Generator, category: Optional[str] = None, name: Optional[str] = None):
    pool = [s for s in SCENARIOS if (name is None or s[0] == name)]
    if name is None:
        cat = category or str(rng.choice(list(CATEGORY_WEIGHT), p=list(CATEGORY_WEIGHT.values())))
        pool = [s for s in pool if s[1] == cat]
    w = np.array([s[2] for s in pool], dtype=float)
    return pool[int(rng.choice(len(pool), p=w / w.sum()))]


def make_drift(snapshot: dict, view: ClusterView, rng: np.random.Generator,
               category: Optional[str] = None, name: Optional[str] = None, noise: Optional[int] = None):
    """Return (drifted_snapshot, info) or None when no applicable scenario was found."""
    for _ in range(12):
        sc = sample_scenario(rng, category, name)
        ctx = MutCtx(snapshot, view, rng)
        ok = sc[3](ctx)
        if not ok:
            continue
        n_noise = int(rng.choice([0, 1, 2], p=[0.55, 0.30, 0.15])) if noise is None else noise
        for _n in range(n_noise):
            NOISE[int(rng.integers(len(NOISE)))](ctx)
        out = ctx.finish()
        return out, {"scenario": sc[0], "category": sc[1], "events": list(ctx.events), "n_noise": n_noise}
    return None
