"""Small JSON-backed alert store with stable-drift deduplication."""
import json
from datetime import datetime, timezone
from pathlib import Path

class AlertManager:
    def __init__(self, storage_path="data/alerts.json", threshold=30):
        self.storage_path = Path(storage_path)
        self.threshold = threshold
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if not self.storage_path.exists():
            return []
        try:
            return json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def create_alert(self, event, rule, risk=None):
        impact = int(rule.get("impact", rule.get("impact_score", 0)))
        if impact < self.threshold and not (risk and risk.get("score", 0) >= self.threshold):
            return None
        event = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        alerts = self._load()
        rule_id = rule.get("rule_id", rule.get("id", "UNNAMED_RULE"))
        drift_id = event.get("drift_id") or f'{event.get("resource_kind")}/{event.get("namespace")}/{event.get("resource_name")}/{event.get("field")}'
        for existing in alerts:
            if existing.get("drift_id") == drift_id and existing.get("rule_id") == rule_id and existing.get("status") == "Open":
                return existing
        alert = {
            "alert_id": f'{drift_id}:{rule_id}',
            "drift_id": drift_id, "rule_id": rule_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "Open",
            "resource": rule.get("resource", f'{event.get("resource_kind")}/{event.get("namespace")}/{event.get("resource_name")}'),
            "field": event.get("field"),
            "description": rule.get("description", "Security drift detected"),
            "impact": impact, "risk": risk or {"score": impact, "level": "Low"},
            "old_value": event.get("old_value"), "new_value": event.get("new_value"),
        }
        alerts.append(alert)
        self.storage_path.write_text(json.dumps(alerts, indent=2, default=str), encoding="utf-8")
        return alert

    def process(self, events, rule_engine, risk_assessment=None, paths_found=False):
        created = []
        for event in events:
            for rule in rule_engine.evaluate_event(event):
                risk = risk_assessment(event, rule, paths_found) if callable(risk_assessment) else None
                alert = self.create_alert(event, rule, risk)
                if alert:
                    created.append(alert)
        return created
