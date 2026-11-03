"""YAML rule evaluation compatible with the project's security_rules.yaml."""
from pathlib import Path
import yaml

class SecurityRuleEngine:
    def __init__(self, rule_file="rules/security_rules.yaml"):
        self.rule_file = Path(rule_file)
        if not self.rule_file.exists():
            raise FileNotFoundError(f"Rule file not found: {self.rule_file}")
        loaded = yaml.safe_load(self.rule_file.read_text(encoding="utf-8")) or {}
        self.rules = loaded.get("rules", loaded if isinstance(loaded, list) else [])

    @staticmethod
    def _get(event, key, default=None):
        return event.get(key, default) if isinstance(event, dict) else getattr(event, key, default)

    def evaluate_event(self, event):
        matches = []
        for rule in self.rules:
            if rule.get("resource_kind") and rule["resource_kind"] != self._get(event, "resource_kind"):
                continue
            if rule.get("field") and rule["field"] != self._get(event, "field"):
                continue
            if rule.get("change_type") and rule["change_type"] != self._get(event, "change_type"):
                continue
            if "old_value" in rule and rule["old_value"] != self._get(event, "old_value"):
                continue
            if "new_value" in rule and rule["new_value"] != self._get(event, "new_value"):
                continue
            matches.append({
                "rule_id": rule.get("id", rule.get("rule_id", "UNNAMED_RULE")),
                "description": rule.get("description", "Security rule matched"),
                "impact": int(rule.get("impact", rule.get("impact_score", 10))),
                "resource": f'{self._get(event, "resource_kind")}/{self._get(event, "namespace")}/{self._get(event, "resource_name")}',
                "field": self._get(event, "field"),
            })
        return matches

    def evaluate(self, events):
        return [{"event": e, "matches": self.evaluate_event(e)} for e in events]

    def calculate_impact(self, event):
        return sum(match["impact"] for match in self.evaluate_event(event))
