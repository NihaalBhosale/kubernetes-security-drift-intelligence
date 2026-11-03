"""Heuristic attack-path candidates from Kubernetes snapshots (not exploitability proof)."""
from __future__ import annotations
import networkx as nx

def _data(r):
    data = dict(r.get("data") or {})
    # Compatibility with newer collector schema.
    if "spec" in r and "spec" not in data: data["spec"] = r["spec"]
    if "labels" in r and "labels" not in data: data["labels"] = r["labels"]
    return data

class AttackPathAnalyzer:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, snapshot):
        self.graph = nx.DiGraph()
        resources = snapshot.get("resources", []) if isinstance(snapshot, dict) else snapshot
        pods = [r for r in resources if r.get("kind") == "Pod"]
        services = [r for r in resources if r.get("kind") == "Service"]
        ingresses = [r for r in resources if r.get("kind") == "Ingress"]
        for r in resources:
            rid = f'{r.get("kind")}/{r.get("namespace") or "default"}/{r.get("name")}'
            self.graph.add_node(rid, kind=r.get("kind"), namespace=r.get("namespace") or "default", name=r.get("name"))
        for svc in services:
            sd = _data(svc); sid = f'Service/{svc.get("namespace") or "default"}/{svc.get("name")}'
            selector = sd.get("selector") or (sd.get("spec") or {}).get("selector") or {}
            if sd.get("type") == "LoadBalancer":
                self.graph.add_edge("EXTERNAL", sid, relation="externally-exposed")
            for pod in pods:
                pd = _data(pod); labels = pd.get("labels") or (pd.get("spec") or {}).get("labels") or {}
                if pod.get("namespace", "default") == svc.get("namespace", "default") and selector and all(labels.get(k) == v for k, v in selector.items()):
                    self.graph.add_edge(sid, f'Pod/{pod.get("namespace") or "default"}/{pod.get("name")}', relation="selects")
        for ing in ingresses:
            iid = f'Ingress/{ing.get("namespace") or "default"}/{ing.get("name")}'
            self.graph.add_edge("EXTERNAL", iid, relation="ingress-entry")
            idata = _data(ing)
            rules = idata.get("rules") or (idata.get("spec") or {}).get("rules") or []
            for rule in rules:
                http = rule.get("http") or {}
                for path in http.get("paths") or []:
                    backend = path.get("backend") or {}
                    service = backend.get("service") or {}
                    name = service.get("name")
                    if name:
                        target = f'Service/{ing.get("namespace") or "default"}/{name}'
                        if target in self.graph:
                            self.graph.add_edge(iid, target, relation="routes-to")
        for pod in pods:
            pd = _data(pod); pid = f'Pod/{pod.get("namespace") or "default"}/{pod.get("name")}'
            sa = pd.get("service_account_name", "default")
            sid = f'ServiceAccount/{pod.get("namespace") or "default"}/{sa}'
            if sid in self.graph:
                self.graph.add_edge(pid, sid, relation="uses")
            if pd.get("privileged") or pd.get("run_as_root") or pd.get("host_network"):
                self.graph.add_node("SENSITIVE: elevated-container", kind="SensitiveTarget")
                self.graph.add_edge(pid, "SENSITIVE: elevated-container", relation="elevated-container-risk")
        # RBAC bindings: binding -> role -> service account subject.
        role_nodes = {f'{r["kind"]}/{r.get("namespace") or "default"}/{r["name"]}' for r in resources if r.get("kind") in ("Role", "ClusterRole")}
        for binding in [r for r in resources if r.get("kind") in ("RoleBinding", "ClusterRoleBinding")]:
            bd = _data(binding); ref = bd.get("role_ref") or bd.get("roleRef") or {}
            role_kind, role_name = ref.get("kind"), ref.get("name")
            role_ns = binding.get("namespace") or "default"
            role_id = f'{role_kind}/{role_ns}/{role_name}'
            if role_id not in self.graph and role_kind == "ClusterRole":
                role_id = f'ClusterRole/default/{role_name}'
            if role_id in self.graph:
                for subject in bd.get("subjects") or []:
                    if subject.get("kind") == "ServiceAccount":
                        sa_ns = subject.get("namespace") or role_ns
                        sa_id = f'ServiceAccount/{sa_ns}/{subject.get("name")}'
                        if sa_id in self.graph:
                            self.graph.add_edge(sa_id, role_id, relation="bound-to-role")
                rules = next((rdata.get("data", {}).get("rules", []) for rdata in resources if f'{rdata["kind"]}/{rdata.get("namespace") or "default"}/{rdata["name"]}' == role_id), [])
                if any("*" in (rule.get("verbs") or []) or "*" in (rule.get("resources") or []) for rule in rules):
                    self.graph.add_node("SENSITIVE: broad RBAC", kind="SensitiveTarget")
                    self.graph.add_edge(role_id, "SENSITIVE: broad RBAC", relation="broad-rbac")
        return self.graph

    def find_attack_paths(self, max_paths=20):
        if "EXTERNAL" not in self.graph:
            return []
        targets = [n for n, d in self.graph.nodes(data=True) if d.get("kind") == "SensitiveTarget"]
        found = []
        for target in targets:
            try:
                paths = nx.all_simple_paths(self.graph, "EXTERNAL", target, cutoff=8)
                for path in paths:
                    found.append(path)
                    if len(found) >= max_paths:
                        return found
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        return found


    def build_focus_graph(self, max_paths=5):
        """Return a compact graph containing candidate paths, or connected nodes if none exist."""
        paths = self.find_attack_paths(max_paths=max_paths)
        if paths:
            nodes = {node for path in paths for node in path}
            return self.graph.subgraph(nodes).copy()
        nodes = [node for node, degree in self.graph.degree() if degree > 0]
        return self.graph.subgraph(nodes).copy()
