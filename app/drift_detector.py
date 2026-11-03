"""Field-level baseline comparison; supports the legacy event fields used by rules."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib, json
from typing import Any

@dataclass
class DriftEvent:
    resource_kind: str
    resource_name: str
    namespace: str
    field: str
    change_type: str
    old_value: Any
    new_value: Any
    severity: str = "Medium"
    timestamp: str = ""
    drift_id: str = ""
    resource_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.resource_id:
            self.resource_id = f"{self.resource_kind}/{self.namespace}/{self.resource_name}"
        if not self.drift_id:
            # Stable across scans for the same change; timestamp intentionally excluded.
            raw = json.dumps([self.resource_id, self.field, self.change_type, self.old_value, self.new_value], sort_keys=True, default=str)
            self.drift_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self):
        return asdict(self)

def _resources(payload):
    if isinstance(payload, dict):
        return payload.get("resources", [])
    if isinstance(payload, list):
        return payload
    return []

def _flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], path))
    elif isinstance(value, list):
        # Treat lists as a single value so order/content changes are detectable.
        result[prefix] = value
    else:
        result[prefix] = value
    return result

def _index(payload):
    indexed = {}
    for raw in _resources(payload):
        kind, ns, name = raw.get("kind", "Unknown"), raw.get("namespace") or "default", raw.get("name", "")
        data = dict(raw.get("data") or {})
        # Newer collector rows may put data in spec/top-level fields.
        for key in ("privileged", "run_as_root", "labels", "spec", "selector", "type"):
            if key in raw and key not in data:
                data[key] = raw[key]
        indexed[(kind, ns, name)] = data
    return indexed

class DriftDetectionEngine:
    def __init__(self, baseline_path=None):
        self.baseline_path = baseline_path

    def detect_drift(self, baseline, current=None):
        """Accept dictionaries or paths; when current is omitted, baseline_path is used."""
        if isinstance(baseline, str):
            baseline = json.loads(open(baseline, encoding="utf-8").read())
        if current is None:
            if not self.baseline_path:
                raise ValueError("Provide baseline and current snapshots")
            current = json.loads(open(self.baseline_path, encoding="utf-8").read())
        elif isinstance(current, str):
            current = json.loads(open(current, encoding="utf-8").read())
        old, new = _index(baseline), _index(current)
        events = []
        for key in sorted(set(old) | set(new)):
            kind, ns, name = key
            if key not in old:
                events.append(DriftEvent(kind, name, ns, "__resource__", "added", None, new[key], "Medium"))
                continue
            if key not in new:
                events.append(DriftEvent(kind, name, ns, "__resource__", "removed", old[key], None, "High"))
                continue
            old_fields, new_fields = _flatten(old[key]), _flatten(new[key])
            for field in sorted(set(old_fields) | set(new_fields)):
                ov, nv = old_fields.get(field), new_fields.get(field)
                if field not in old_fields or field not in new_fields or ov != nv:
                    events.append(DriftEvent(kind, name, ns, field, "modified", ov, nv,
                                             "High" if field in ("privileged", "run_as_root", "host_network") else "Medium"))
        return events

    compare = detect_drift
