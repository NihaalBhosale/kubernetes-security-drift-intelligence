"""Normalized JSON snapshots with stable SHA-256 hashes."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _jsonable(value: Any):
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def normalize_resource(resource: dict) -> dict:
    """Accept both collector dictionaries and legacy {kind,name,namespace,data} rows."""
    if not isinstance(resource, dict):
        if all(hasattr(resource, a) for a in ("kind", "name", "namespace")):
            resource = {"kind": resource.kind, "name": resource.name,
                        "namespace": resource.namespace, "data": getattr(resource, "data", {})}
        else:
            raise TypeError(f"Unsupported resource representation: {type(resource).__name__}")
    kind = resource.get("kind") or resource.get("resource_kind") or "Unknown"
    name = resource.get("name") or resource.get("resource_name") or ""
    namespace = resource.get("namespace") or "default"
    data = dict(_jsonable(resource.get("data") or {}))
    # Older collector versions put security flags at top-level.
    for key in ("privileged", "run_as_root", "labels", "spec", "selector", "type"):
        if key in resource and key not in data:
            data[key] = _jsonable(resource[key])
    if "labels" in resource and "labels" not in data:
        data["labels"] = _jsonable(resource["labels"])
    return {"kind": kind, "name": name, "namespace": namespace, "data": data}

class ConfigurationSnapshot:
    def __init__(self, resources):
        self.resources = [normalize_resource(r) for r in resources]
        self.timestamp = datetime.now(timezone.utc).isoformat()
        canonical = json.dumps(self.resources, sort_keys=True, separators=(",", ":"))
        self.configuration_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self):
        return {"timestamp": self.timestamp, "configuration_hash": self.configuration_hash,
                "resources": self.resources}

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        instance = cls(payload.get("resources", []))
        instance.timestamp = payload.get("timestamp", instance.timestamp)
        instance.configuration_hash = payload.get("configuration_hash", instance.configuration_hash)
        return instance
