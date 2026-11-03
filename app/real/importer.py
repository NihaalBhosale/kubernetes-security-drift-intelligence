"""Import real Kubernetes manifests (plain YAML, or the output of `helm template`) into the project's snapshot format.

    docs = load_yaml_files([...])                     # list of manifest dicts
    resources = manifests_to_resources(docs, rng)     # snapshot 'resources' (Pod/Service/NetworkPolicy/RBAC/...)

Image vulnerability counts come from Trivy scans (experiments/scan_images.py -> data/real/vuln_counts.json). Images that were not
scanned fall back to a seeded lognormal draw (repeatable).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import yaml

WORKLOADS = {"Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "Job", "ReplicationController"}


def load_yaml_files(paths: Iterable[Path]) -> List[dict]:
    docs: List[dict] = []
    for p in paths:
        try:
            for d in yaml.safe_load_all(Path(p).read_text(errors="ignore")):
                if isinstance(d, dict) and d.get("kind") and isinstance(d.get("metadata"), dict):
                    docs.append(d)
        except Exception:
            continue
    return docs


_CACHE = None


def _scan_cache() -> dict:
    """Trivy results from experiments/scan_images.py (data/real/vuln_counts.json), if present."""
    global _CACHE
    if _CACHE is None:
        import json, os
        p = Path(os.environ.get("VULN_CACHE", Path(__file__).resolve().parents[2] / "data" / "real" / "vuln_counts.json"))
        try:
            _CACHE = json.loads(p.read_text()) if p.exists() else {}
        except Exception:
            _CACHE = {}
    return _CACHE


def scan_coverage(images: Iterable[str]) -> float:
    c = _scan_cache()
    imgs = list(images)
    return sum(1 for i in imgs if i in c) / max(len(imgs), 1)


def _image_vuln(image: str, seed: int) -> int:
    c = _scan_cache()
    if image in c and int(c[image]) >= 0:   # real Trivy result; failed/old -1 entries are ignored -> simulated fallback
        return int(c[image])
    h = int(hashlib.sha1(f"{seed}:{image}".encode()).hexdigest()[:8], 16)
    return int(np.random.default_rng(h).lognormal(2.3, 1.0))


def _container(c: dict, pod_sc: dict, seed: int) -> dict:
    sc = c.get("securityContext") or {}
    caps = sc.get("capabilities") or {}
    run_non_root = sc.get("runAsNonRoot", pod_sc.get("runAsNonRoot"))
    uid = sc.get("runAsUser", pod_sc.get("runAsUser"))
    root = not (run_non_root is True or (uid is not None and int(uid) != 0))
    ports = [int(p["containerPort"]) for p in c.get("ports") or [] if "containerPort" in p]
    lim = (c.get("resources") or {}).get("limits")
    img = str(c.get("image") or "")
    return {"name": c.get("name", "c"), "image": img, "privileged": bool(sc.get("privileged", False)), "run_as_root": root,
            "allow_privilege_escalation": bool(sc.get("allowPrivilegeEscalation", True)),
            "capabilities_add": list(caps.get("add") or []), "caps_drop_all": "ALL" in [str(x).upper() for x in caps.get("drop") or []],
            "read_only_root_fs": bool(sc.get("readOnlyRootFilesystem", False)),
            "ports_declared": ports, "ports_open": ports, "ports_dynamic": 0,
            "vuln_high_critical": _image_vuln(img, seed), "has_limits": bool(lim),
            "seccomp": (sc.get("seccompProfile") or pod_sc.get("seccompProfile") or {}).get("type")}


def _volumes(spec: dict) -> List[dict]:
    out = []
    for v in spec.get("volumes") or []:
        if "hostPath" in v:
            out.append({"type": "hostPath", "path": v["hostPath"].get("path", "")})
        elif "secret" in v:
            out.append({"type": "secret"})
        elif "emptyDir" in v:
            out.append({"type": "emptyDir"})
        else:
            out.append({"type": "other"})
    return out


def _pod_data(labels: dict, spec: dict, node: str, seed: int) -> dict:
    sc = spec.get("securityContext") or {}
    cs = [_container(c, sc, seed) for c in (spec.get("containers") or []) + (spec.get("initContainers") or [])]
    return {"labels": dict(labels or {}), "service_account_name": spec.get("serviceAccountName") or "default",
            "host_network": bool(spec.get("hostNetwork")), "host_pid": bool(spec.get("hostPID")), "host_ipc": bool(spec.get("hostIPC")),
            "node_name": node, "automount_sa_token": spec.get("automountServiceAccountToken", True) is not False,
            "containers": cs, "volumes": _volumes(spec), "affinity_pods": [],
            "privileged": any(c["privileged"] for c in cs), "run_as_root": any(c["run_as_root"] for c in cs)}


def _rules(rules) -> List[dict]:
    return [{"api_groups": r.get("apiGroups") or [], "resources": r.get("resources") or [], "verbs": r.get("verbs") or [],
             "resource_names": r.get("resourceNames"), "non_resource_urls": r.get("nonResourceURLs")} for r in rules or []]


def manifests_to_resources(docs: List[dict], rng: np.random.Generator, namespace: str = "default", nodes: Optional[List[str]] = None,
                           max_replicas: int = 3, seed: int = 0) -> List[dict]:
    nodes = nodes or ["node-0"]
    res: List[dict] = []
    for d in docs:
        kind, md, spec = d["kind"], d["metadata"], d.get("spec") or {}
        name = md.get("name") or "unnamed"
        ns = md.get("namespace") or namespace
        if kind in WORKLOADS or kind == "CronJob" or kind == "Pod":
            if kind == "Pod":
                tmpl, labels, reps = spec, md.get("labels"), 1
            elif kind == "CronJob":
                t = ((spec.get("jobTemplate") or {}).get("spec") or {}).get("template") or {}
                tmpl, labels, reps = t.get("spec") or {}, (t.get("metadata") or {}).get("labels") or md.get("labels"), 1
            else:
                t = spec.get("template") or {}
                tmpl, labels = t.get("spec") or {}, (t.get("metadata") or {}).get("labels") or md.get("labels")
                reps = 1 if kind == "Job" else int(spec.get("replicas", 1) or 1)
            if kind == "DaemonSet":
                reps = len(nodes)
            for i in range(max(1, min(reps, max_replicas if kind != "DaemonSet" else len(nodes)))):
                node = nodes[i % len(nodes)] if kind == "DaemonSet" else nodes[int(rng.integers(len(nodes)))]
                res.append({"kind": "Pod", "name": f"{name}-{i}", "namespace": ns, "data": _pod_data(labels or {}, tmpl, node, seed)})
        elif kind == "Service":
            ports = [{"port": p.get("port"), "target_port": p.get("targetPort", p.get("port"))} for p in spec.get("ports") or []]
            res.append({"kind": "Service", "name": name, "namespace": ns, "data": {
                "type": spec.get("type", "ClusterIP"), "selector": spec.get("selector") or {}, "headless": spec.get("clusterIP") == "None",
                "ports": ports, "labels": md.get("labels") or {}}})
        elif kind == "Ingress":
            res.append({"kind": "Ingress", "name": name, "namespace": ns, "data": {"rules": spec.get("rules") or [], "default_backend": spec.get("defaultBackend")}})
        elif kind == "NetworkPolicy":
            res.append({"kind": "NetworkPolicy", "name": name, "namespace": ns, "data": {
                "pod_selector": spec.get("podSelector") or {}, "policy_types": spec.get("policyTypes") or [],
                "ingress": [{"from": r.get("from")} if r.get("from") else {} for r in spec.get("ingress") or []], "egress": []}})
        elif kind == "ServiceAccount":
            res.append({"kind": "ServiceAccount", "name": name, "namespace": ns,
                        "data": {"labels": md.get("labels") or {}, "automount_service_account_token": d.get("automountServiceAccountToken", True) is not False}})
        elif kind in ("Role", "ClusterRole"):
            res.append({"kind": kind, "name": name, "namespace": ns if kind == "Role" else "default", "data": {"labels": md.get("labels") or {}, "rules": _rules(d.get("rules"))}})
        elif kind in ("RoleBinding", "ClusterRoleBinding"):
            ref = d.get("roleRef") or {}
            subs = [{"kind": s.get("kind"), "name": s.get("name"), "namespace": s.get("namespace") or ns, "api_group": s.get("apiGroup")} for s in d.get("subjects") or []]
            res.append({"kind": kind, "name": name, "namespace": ns if kind == "RoleBinding" else "default",
                        "data": {"labels": md.get("labels") or {}, "role_ref": {"kind": ref.get("kind"), "name": ref.get("name"), "api_group": ref.get("apiGroup")}, "subjects": subs}})
    return res
