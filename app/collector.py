"""Collect Kubernetes resources into the project's normalized snapshot schema."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

RESOURCE_TYPES = ("Pod", "Service", "NetworkPolicy", "ServiceAccount", "Role",
                  "ClusterRole", "RoleBinding", "ClusterRoleBinding", "Ingress")

def _plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value

def _meta(obj):
    return getattr(obj, "metadata", None)

def _base(kind, obj):
    meta = _meta(obj)
    return {
        "kind": kind,
        "name": getattr(meta, "name", "") or "",
        "namespace": getattr(meta, "namespace", None) or "default",
        "labels": dict(getattr(meta, "labels", None) or {}),
        "data": {},
    }

def collect_resources(namespace: str | None = None, kubeconfig: str | None = None) -> list[dict]:
    """Load local kubeconfig (or in-cluster config) and collect supported resources."""
    try:
        if kubeconfig:
            config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_kube_config()
            except ConfigException:
                config.load_incluster_config()
    except Exception as exc:
        raise RuntimeError(
            "Could not load Kubernetes credentials. Start your cluster and verify `kubectl get nodes`."
        ) from exc

    v1, net, rbac, networking = client.CoreV1Api(), client.NetworkingV1Api(), client.RbacAuthorizationV1Api(), client.NetworkingV1Api()
    out = []
    def add(kind, items, transform):
        for obj in items:
            row = _base(kind, obj)
            row["data"].update(transform(obj))
            out.append(row)

    pods = v1.list_namespaced_pod(namespace).items if namespace else v1.list_pod_for_all_namespaces().items
    def _vuln_cache():
        # optional Trivy results produced by experiments/scan_images.py
        try:
            return json.loads((Path(__file__).resolve().parent.parent / "data" / "real" / "vuln_counts.json").read_text())
        except Exception:
            return {}
    vcache = _vuln_cache()

    def pod_data(p):
        containers = []
        psc = getattr(p.spec, "security_context", None)
        for c in (p.spec.containers or []):
            sc = getattr(c, "security_context", None)
            caps = getattr(sc, "capabilities", None)
            uid = getattr(sc, "run_as_user", None)
            if uid is None and psc is not None:
                uid = getattr(psc, "run_as_user", None)
            non_root = getattr(sc, "run_as_non_root", None)
            if non_root is None and psc is not None:
                non_root = getattr(psc, "run_as_non_root", None)
            ports = [int(x.container_port) for x in (c.ports or []) if getattr(x, "container_port", None)]
            lim = getattr(getattr(c, "resources", None), "limits", None)
            seccomp = getattr(getattr(sc, "seccomp_profile", None), "type", None) or getattr(getattr(psc, "seccomp_profile", None), "type", None)
            row = {
                "name": c.name,
                "privileged": bool(getattr(sc, "privileged", False) or False),
                # root unless the container (or pod) says otherwise
                "run_as_root": uid == 0 or (uid is None and non_root is not True),
                "image": c.image,
                "allow_privilege_escalation": bool(getattr(sc, "allow_privilege_escalation", True) is not False),
                "capabilities_add": list(getattr(caps, "add", None) or []),
                "caps_drop_all": "ALL" in [str(x).upper() for x in (getattr(caps, "drop", None) or [])],
                "read_only_root_fs": bool(getattr(sc, "read_only_root_filesystem", False) or False),
                "ports_declared": ports, "ports_open": ports,
                "has_limits": bool(lim), "seccomp": seccomp,
            }
            if c.image in vcache:
                row["vuln_high_critical"] = int(vcache[c.image])
            containers.append(row)
        vols = []
        for v in (p.spec.volumes or []):
            if getattr(v, "host_path", None):
                vols.append({"type": "hostPath", "path": v.host_path.path})
            elif getattr(v, "secret", None):
                vols.append({"type": "secret"})
            else:
                vols.append({"type": "other"})
        return {
            "labels": dict(p.metadata.labels or {}),
            "service_account_name": p.spec.service_account_name or "default",
            "host_network": bool(p.spec.host_network),
            "host_pid": bool(p.spec.host_pid), "host_ipc": bool(p.spec.host_ipc),
            "node_name": p.spec.node_name,
            "automount_sa_token": p.spec.automount_service_account_token is not False,
            "volumes": vols,
            "containers": containers,
            "privileged": any(c["privileged"] for c in containers),
            "run_as_root": any(c["run_as_root"] for c in containers),
        }
    add("Pod", pods, pod_data)

    services = v1.list_namespaced_service(namespace).items if namespace else v1.list_service_for_all_namespaces().items
    add("Service", services, lambda s: {"type": s.spec.type, "headless": s.spec.cluster_ip == "None", "selector": dict(s.spec.selector or {}), "ports": [_plain(p) for p in (s.spec.ports or [])]})

    policies = net.list_namespaced_network_policy(namespace).items if namespace else net.list_network_policy_for_all_namespaces().items
    add("NetworkPolicy", policies, lambda p: {"pod_selector": _plain(p.spec.pod_selector), "policy_types": list(p.spec.policy_types or []), "ingress": _plain(p.spec.ingress or []), "egress": _plain(p.spec.egress or [])})

    sas = v1.list_namespaced_service_account(namespace).items if namespace else v1.list_service_account_for_all_namespaces().items
    add("ServiceAccount", sas, lambda x: {"automount_service_account_token": x.automount_service_account_token is not False})
    roles = rbac.list_namespaced_role(namespace).items if namespace else rbac.list_role_for_all_namespaces().items
    add("Role", roles, lambda x: {"rules": _plain(x.rules or [])})
    add("ClusterRole", rbac.list_cluster_role().items, lambda x: {"rules": _plain(x.rules or [])})
    rbs = rbac.list_namespaced_role_binding(namespace).items if namespace else rbac.list_role_binding_for_all_namespaces().items
    add("RoleBinding", rbs, lambda x: {"role_ref": _plain(x.role_ref), "subjects": _plain(x.subjects or [])})
    add("ClusterRoleBinding", rbac.list_cluster_role_binding().items, lambda x: {"role_ref": _plain(x.role_ref), "subjects": _plain(x.subjects or [])})
    ingresses = networking.list_namespaced_ingress(namespace).items if namespace else networking.list_ingress_for_all_namespaces().items
    add("Ingress", ingresses, lambda x: {"rules": _plain(x.spec.rules or []), "default_backend": _plain(x.spec.default_backend)})
    add("Node", v1.list_node().items, lambda n: {})
    return sorted(out, key=lambda r: (r["kind"], r["namespace"], r["name"]))

if __name__ == "__main__":
    resources = collect_resources()
    print(f"Collected {len(resources)} resources.")
    print("Kinds:", ", ".join(sorted({r["kind"] for r in resources})))
