import unittest
from app.drift_detector import DriftDetectionEngine
from app.rule_engine import SecurityRuleEngine

BASELINE = {"resources": [{"kind":"Pod","name":"demo-pod","namespace":"default","data":{"privileged":False,"run_as_root":False}}]}
CURRENT = {"resources": [{"kind":"Pod","name":"demo-pod","namespace":"default","data":{"privileged":True,"run_as_root":False}}]}

class DriftDetectionTests(unittest.TestCase):
    def test_privileged_drift_matches_rule(self):
        events = DriftDetectionEngine().detect_drift(BASELINE, CURRENT)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.field, "privileged")
        self.assertEqual(event.change_type, "modified")
        matches = SecurityRuleEngine("rules/security_rules.yaml").evaluate_event(event)
        self.assertEqual(matches[0]["rule_id"], "PRIVILEGED_CONTAINER")
        self.assertEqual(matches[0]["impact"], 30)

    def test_removed_network_policy(self):
        old = {"resources":[{"kind":"NetworkPolicy","name":"p","namespace":"default","data":{}}]}
        new = {"resources":[]}
        event = DriftDetectionEngine().detect_drift(old, new)[0]
        self.assertEqual(event.change_type, "removed")
        self.assertEqual(event.field, "__resource__")
        self.assertEqual(SecurityRuleEngine("rules/security_rules.yaml").evaluate_event(event)[0]["rule_id"], "NETWORK_POLICY_REMOVED")

if __name__ == "__main__":
    unittest.main()
