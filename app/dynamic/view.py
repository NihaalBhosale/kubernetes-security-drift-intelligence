"""ClusterView: a read-only, indexed interpretation of a configuration snapshot.

The view is *parsing only* (labels, NetworkPolicy selection, RBAC resolution,
service->pod mapping). It deliberately contains no risk model, so it can be
shared by the rule templates, the paper-1 style analysis graph and the
attack simulator without leaking one component's assumptions into another.

It accepts both the project's legacy snapshot rows (only ``privileged``,
``run_as_root``, ``host_network``) and the enriched rows produced by the
extended collector / synthetic generator. Missing information is treated as
*neutral* (never as a risk) so that sparse real snapshots do not raise alarms.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

DEFAULT_CAPS = 14  # Linux capabilities Docker/containerd grant by default (paper 1, Sec. V-D)
DANGEROUS_CAPS = {
    "SYS_ADMIN", "NET_ADMIN", "SYS_PTRACE", "SYS_MODULE", "SYS_RAWIO", "SYS_TIME",
    "SYS_RESOURCE", "NET_BROADCAST", "DAC_READ_SEARCH", "ALL",
}
SENSITIVE_HOST_PATHS = (
    "/", "/etc", "/proc", "/sys", "/root", "/var/run/docker.sock", "/run/containerd",
    "/var/lib/kubelet", "/var/lib/docker",
)
PERMS = ("wildcard", "secrets", "create_pods", "exec", "impersonate", "list_pods", "delete")
POWERFUL_PERMS = ("wildcard", "secrets", "create_pods", "exec", "impersonate")
SYSTEM_NAMESPACES = ("kube-system",)
BROADCAST_GROUPS = ("system:serviceaccounts", "system:authenticated", "system:unauthenticated")


def key_of(r: dict) -> str:
    return f'{r.get("kind")}/{r.get("namespace") or "default"}/{r.get("name")}'


def _labels_of(d: dict, r: dict) -> dict:
    return dict(d.get("labels") or r.get("labels") or {})


def _sel(sel: Any) -> Dict[str, str]:
    """Normalise a label selector (camelCase or snake_case, or a plain dict)."""
    if not sel:
        return {}
    if isinstance(sel, dict):
        if "match_labels" in sel or "matchLabels" in sel:
            return dict(sel.get("match_labels") or sel.get("matchLabels") or {})
        # plain mapping (services use this form)
        return {k: v for k, v in sel.items() if isinstance(v, (str, int, float, bool))}
    return {}


def rule_flags(rules: Iterable[dict]) -> Set[str]:
    """Translate RBAC PolicyRules into the coarse permission classes used here."""
    flags: Set[str] = set()
    for ru in rules or []:
        verbs = set(ru.get("verbs") or [])
        res = set(ru.get("resources") or [])
        any_verb, any_res = "*" in verbs, "*" in res
        if any_verb and any_res:
            flags.update(PERMS)
            continue
        if (any_res or "secrets" in res) and (any_verb or verbs & {"get", "list", "watch"}):
            flags.add("secrets")
        if (any_res or res & {"pods", "deployments", "daemonsets", "jobs", "cronjobs", "statefulsets"}) \
                and (any_verb or "create" in verbs):
            flags.add("create_pods")
        if (any_res or res & {"pods/exec", "pods/attach"}) and (any_verb or verbs & {"create", "get"}):
            flags.add("exec")
        if verbs & {"impersonate", "escalate", "bind"}:
            flags.add("impersonate")
        if (any_res or "pods" in res) and (any_verb or verbs & {"get", "list"}):
            flags.add("list_pods")
        if verbs & {"delete", "deletecollection"} or any_verb:
            flags.add("delete")
    return flags


class PodRec:
    __slots__ = (
        "idx", "key", "ns", "name", "labels", "sa", "host_network", "host_pid", "host_ipc",
        "node", "containers", "volumes", "automount", "affinity", "privileged", "root",
        "priv_esc", "caps_extra", "ncaps", "hostpaths", "sensitive_hostpath", "vuln", "opaque",
        "latest_tag", "ports_declared", "ports_open", "dyn_ports", "has_limits", "ro_fs",
        "label_items",
    )


class ClusterView:
    def __init__(self, snapshot: Any):
        resources = snapshot.get("resources", []) if isinstance(snapshot, dict) else list(snapshot)
        self.resources = resources
        self.pods: List[PodRec] = []
        self.services: List[dict] = []
        self.ingresses: List[dict] = []
        self.netpols: List[dict] = []
        self.namespaces: Set[str] = set()
        self.nodes: Set[str] = set()
        self._sa_res: Dict[Tuple[str, str], dict] = {}
        self._roles: Dict[Tuple[str, str], dict] = {}
        self._croles: Dict[str, dict] = {}
        self._bindings: List[dict] = []
        self.counts: Dict[str, int] = {}
        self._parse(resources)
        self._index_pods()
        self._map_services()
        self._select_netpols()
        self._resolve_rbac()
        self._exposure()

    # ---------------------------------------------------------------- parse
    def _parse(self, resources: List[dict]) -> None:
        for r in resources:
            kind = r.get("kind")
            ns = r.get("namespace") or "default"
            d = r.get("data") or {}
            self.counts[kind] = self.counts.get(kind, 0) + 1
            if kind not in ("ClusterRole", "ClusterRoleBinding", "Node"):
                self.namespaces.add(ns)
            if kind == "Pod":
                self.pods.append(self._pod(r, d, ns))
            elif kind == "Service":
                self.services.append({
                    "key": key_of(r), "ns": ns, "name": r.get("name"),
                    "type": d.get("type") or "ClusterIP",
                    "selector": _sel(d.get("selector")),
                    "ports": list(d.get("ports") or []),
                    "headless": bool(d.get("headless")),
                    "intended_app": d.get("intended_app"),
                    "pods": [],
                })
            elif kind == "Ingress":
                backends = []
                for rule in d.get("rules") or []:
                    for p in ((rule.get("http") or {}).get("paths") or []):
                        nm = ((p.get("backend") or {}).get("service") or {}).get("name")
                        if nm:
                            backends.append(nm)
                self.ingresses.append({"key": key_of(r), "ns": ns, "name": r.get("name"), "backends": backends})
            elif kind == "NetworkPolicy":
                self.netpols.append(self._netpol(r, d, ns))
            elif kind == "ServiceAccount":
                self._sa_res[(ns, r.get("name"))] = d
            elif kind == "Role":
                self._roles[(ns, r.get("name"))] = d
            elif kind == "ClusterRole":
                self._croles[r.get("name")] = d
            elif kind in ("RoleBinding", "ClusterRoleBinding"):
                ref = d.get("role_ref") or d.get("roleRef") or {}
                self._bindings.append({
                    "key": key_of(r), "kind": kind, "ns": ns if kind == "RoleBinding" else "*",
                    "role_kind": ref.get("kind"), "role": ref.get("name"),
                    "subjects": list(d.get("subjects") or []),
                })
            elif kind == "Node":
                self.nodes.add(r.get("name"))

    @staticmethod
    def _container(c: dict) -> dict:
        caps = [str(x).upper() for x in (c.get("capabilities_add") or [])]
        vuln = c.get("vuln_high_critical")
        image = str(c.get("image") or "")
        return {
            "name": c.get("name", ""),
            "privileged": bool(c.get("privileged")),
            "run_as_root": bool(c.get("run_as_root")),
            "allow_priv_esc": bool(c.get("allow_privilege_escalation", False)),
            "caps_add": caps,
            "caps_drop_all": bool(c.get("caps_drop_all", False)),
            "ro_fs": bool(c.get("read_only_root_fs", False)),
            "ports_declared": set(int(p) for p in (c.get("ports_declared") or [])),
            "ports_open": set(int(p) for p in (c.get("ports_open") or [])),
            "dyn_ports": int(c.get("ports_dynamic") or 0),
            "vuln": None if vuln is None else int(vuln),
            "has_limits": True if c.get("has_limits") is None else bool(c.get("has_limits")),
            "image": image,
            "latest": image.endswith(":latest") or (":" not in image and image != ""),
            "seccomp": c.get("seccomp"),
            "known_ports": ("ports_declared" in c) or ("ports_open" in c),
        }

    def _pod(self, r: dict, d: dict, ns: str) -> PodRec:
        p = PodRec()
        p.key = key_of(r)
        p.ns, p.name = ns, r.get("name")
        p.labels = _labels_of(d, r)
        p.label_items = frozenset((str(k), str(v)) for k, v in p.labels.items())
        p.sa = d.get("service_account_name") or "default"
        p.host_network = bool(d.get("host_network"))
        p.host_pid = bool(d.get("host_pid"))
        p.host_ipc = bool(d.get("host_ipc"))
        p.node = d.get("node_name") or "node-0"
        raw = d.get("containers")
        if not raw:  # legacy row: synthesise one container from aggregate flags
            raw = [{"name": "main", "privileged": d.get("privileged"), "run_as_root": d.get("run_as_root")}]
        p.containers = [self._container(c) for c in raw]
        p.volumes = list(d.get("volumes") or [])
        p.automount = d.get("automount_sa_token", True) is not False
        p.affinity = list(d.get("affinity_pods") or [])
        cs = p.containers
        p.privileged = any(c["privileged"] for c in cs)
        p.root = any(c["run_as_root"] for c in cs)
        p.priv_esc = any(c["allow_priv_esc"] for c in cs)
        extra: Set[str] = set()
        for c in cs:
            extra.update(c["caps_add"])
        p.caps_extra = extra
        p.ncaps = max(
            (0 if c["caps_drop_all"] else DEFAULT_CAPS) + len(c["caps_add"]) for c in cs
        )
        p.hostpaths = [v.get("path", "") for v in p.volumes if v.get("type") == "hostPath"]
        p.sensitive_hostpath = any(
            hp == s or (s != "/" and hp.startswith(s + "/")) for hp in p.hostpaths for s in SENSITIVE_HOST_PATHS
        )
        vs = [c["vuln"] for c in cs if c["vuln"] is not None and c["vuln"] >= 0]
        p.vuln = int(sum(vs))
        p.opaque = any(c["vuln"] is not None and c["vuln"] < 0 for c in cs)
        p.latest_tag = any(c["latest"] for c in cs)
        p.ports_declared = set().union(*[c["ports_declared"] for c in cs])
        p.ports_open = set().union(*[c["ports_open"] for c in cs])
        p.dyn_ports = sum(c["dyn_ports"] for c in cs)
        p.has_limits = all(c["has_limits"] for c in cs)
        p.ro_fs = all(c["ro_fs"] for c in cs)
        self.nodes.add(p.node)
        return p

    @staticmethod
    def _netpol(r: dict, d: dict, ns: str) -> dict:
        types = list(d.get("policy_types") or d.get("policyTypes") or [])
        ingress = d.get("ingress")
        rules = []
        for rule in ingress or []:
            frm = rule.get("_from") or rule.get("from_") or rule.get("from")
            if not frm:
                rules.append({"allow_all": True, "sources": []})
                continue
            srcs = []
            for peer in frm:
                ps = peer.get("pod_selector") if "pod_selector" in peer else peer.get("podSelector")
                nss = peer.get("namespace_selector") if "namespace_selector" in peer else peer.get("namespaceSelector")
                srcs.append((None if ps is None else _sel(ps), None if nss is None else _sel(nss)))
            rules.append({"allow_all": False, "sources": srcs})
        applies_ingress = ("Ingress" in types) or (not types)
        return {
            "key": key_of(r), "ns": ns, "name": r.get("name"),
            "selector": _sel(d.get("pod_selector") or d.get("podSelector")),
            "applies_ingress": applies_ingress,
            "rules": rules, "selected": None,
        }

    # -------------------------------------------------------------- indexes
    def _index_pods(self) -> None:
        self.pods.sort(key=lambda p: p.key)
        self.pod_index: Dict[str, int] = {}
        self.pods_by_ns: Dict[str, List[int]] = {}
        self.pods_by_node: Dict[str, List[int]] = {}
        for i, p in enumerate(self.pods):
            p.idx = i
            self.pod_index[p.key] = i
            self.pods_by_ns.setdefault(p.ns, []).append(i)
            self.pods_by_node.setdefault(p.node, []).append(i)
        self.n_pods = len(self.pods)

    @staticmethod
    def _matches(label_items: frozenset, selector: Dict[str, str]) -> bool:
        return all((str(k), str(v)) in label_items for k, v in selector.items())

    def pods_matching(self, ns: str, selector: Dict[str, str]) -> List[int]:
        return [i for i in self.pods_by_ns.get(ns, []) if self._matches(self.pods[i].label_items, selector)]

    def _map_services(self) -> None:
        for s in self.services:
            s["pods"] = self.pods_matching(s["ns"], s["selector"]) if s["selector"] else []

    def _select_netpols(self) -> None:
        n = self.n_pods
        self.isolated = np.zeros(n, dtype=bool)          # selected by an ingress policy
        # allowed[i, j]: may pod i open connections to pod j given policies (True when j is not isolated)
        self.allowed = np.ones((n, n), dtype=bool)
        self.np_effective: Dict[str, bool] = {}
        for pol in self.netpols:
            sel = self.pods_matching(pol["ns"], pol["selector"])
            pol["selected"] = sel
            self.np_effective[pol["key"]] = len(sel) > 0
            if not pol["applies_ingress"] or not sel:
                continue
            for j in sel:
                if not self.isolated[j]:
                    self.isolated[j] = True
                    self.allowed[:, j] = False
            src_mask = np.zeros(n, dtype=bool)
            for rule in pol["rules"]:
                if rule["allow_all"]:
                    src_mask[:] = True
                    continue
                for ps, nss in rule["sources"]:
                    for i, q in enumerate(self.pods):
                        if nss is not None and not self._ns_labels_match(q.ns, nss):
                            continue
                        if nss is None and q.ns != pol["ns"]:
                            continue
                        if ps is None or self._matches(q.label_items, ps):
                            src_mask[i] = True
            if src_mask.any():
                idx = np.array(sel)
                self.allowed[np.ix_(np.nonzero(src_mask)[0], idx)] = True
        self.np_allow_all_pods = np.zeros(n, dtype=bool)
        for pol in self.netpols:
            if pol["selected"] and pol["applies_ingress"] and any(r["allow_all"] for r in pol["rules"]):
                self.np_allow_all_pods[pol["selected"]] = True

    @staticmethod
    def _ns_labels_match(ns: str, selector: Dict[str, str]) -> bool:
        if not selector:
            return True
        return selector.get("kubernetes.io/metadata.name") == ns or selector.get("name") == ns

    # ----------------------------------------------------------------- RBAC
    def _role_flags(self, binding: dict) -> Set[str]:
        if binding["role_kind"] == "Role":
            role = self._roles.get((binding["ns"], binding["role"]))
        else:
            role = self._croles.get(binding["role"])
        if role is None:
            return set()
        flags = rule_flags(role.get("rules") or [])
        if binding["role"] == "cluster-admin":
            flags.update(PERMS)
        return flags

    def _resolve_rbac(self) -> None:
        sa_keys: Set[Tuple[str, str]] = set(self._sa_res) | {(p.ns, p.sa) for p in self.pods}
        self.sa_perms: Dict[Tuple[str, str], Dict[str, Set[str]]] = {k: {} for k in sa_keys}
        self.sa_group_granted: Dict[Tuple[str, str], Set[str]] = {k: set() for k in sa_keys}
        self.sa_cross_ns: Dict[Tuple[str, str], Set[str]] = {k: set() for k in sa_keys}
        self.binding_info: List[dict] = []
        for b in self._bindings:
            flags = self._role_flags(b)
            if not flags:
                continue
            scope = "*" if b["kind"] == "ClusterRoleBinding" else b["ns"]
            targets: List[Tuple[str, str]] = []
            broadcast = False
            cross = False
            for s in b["subjects"]:
                sk = s.get("kind")
                nm = s.get("name") or ""
                if sk == "ServiceAccount":
                    sns = s.get("namespace") or (b["ns"] if b["ns"] != "*" else "default")
                    if (sns, nm) in self.sa_perms:
                        targets.append((sns, nm))
                        if b["ns"] != "*" and sns != b["ns"]:
                            cross = True
                elif sk == "Group":
                    if nm in BROADCAST_GROUPS:
                        broadcast = True
                        targets.extend(self.sa_perms.keys())
                    elif nm.startswith("system:serviceaccounts:"):
                        gns = nm.split(":", 2)[2]
                        targets.extend(k for k in self.sa_perms if k[0] == gns)
            self.binding_info.append({
                "key": b["key"], "flags": flags, "scope": scope, "role": b["role"],
                "broadcast": broadcast, "cross_ns": cross, "targets": list(dict.fromkeys(targets)),
                "binds_cluster_admin": b["role"] == "cluster-admin",
                "subjects": b["subjects"], "kind": b["kind"],
            })
            for t in set(targets):
                for f in flags:
                    self.sa_perms[t].setdefault(f, set()).add(scope)
                if broadcast:
                    self.sa_group_granted[t].update(f for f in flags if f in POWERFUL_PERMS)
                if cross:
                    self.sa_cross_ns[t].update(f for f in flags if f in POWERFUL_PERMS)

    def pod_perms(self, p: PodRec) -> Dict[str, Set[str]]:
        return self.sa_perms.get((p.ns, p.sa), {})

    def token_mounted(self, p: PodRec) -> bool:
        sa = self._sa_res.get((p.ns, p.sa), {})
        return p.automount and sa.get("automount_service_account_token", True) is not False

    # ------------------------------------------------------------- exposure
    def _exposure(self) -> None:
        """exposure[i] in {0, 1}; 1 if reachable from outside via LB/NodePort/Ingress."""
        self.exposure = np.zeros(self.n_pods, dtype=np.float32)
        self.exposure_kind: Dict[int, str] = {}
        ing_targets = {(g["ns"], b) for g in self.ingresses for b in g["backends"]}
        for s in self.services:
            kind = None
            if s["type"] == "LoadBalancer":
                kind = "lb"
            elif s["type"] == "NodePort":
                kind = "nodeport"
            elif (s["ns"], s["name"]) in ing_targets:
                kind = "ingress"
            if kind:
                for i in s["pods"]:
                    self.exposure[i] = 1.0
                    self.exposure_kind.setdefault(i, kind)

    # ------------------------------------------------------------- helpers
    def is_system(self, p: PodRec) -> bool:
        return p.ns in SYSTEM_NAMESPACES

    def sensitive_pod(self, p: PodRec) -> bool:
        return p.ns in SYSTEM_NAMESPACES or p.labels.get("tier") == "db"

    def sa_power(self, p: PodRec) -> Set[str]:
        return {f for f in self.pod_perms(p) if f in POWERFUL_PERMS}
