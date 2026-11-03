"""Run the Kubernetes Security Drift Intelligence pipeline."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .snapshot import ConfigurationSnapshot
from .drift_detector import DriftDetectionEngine
from .rule_engine import SecurityRuleEngine
from .attack_path_analyzer import AttackPathAnalyzer
from .risk_assesment import calculate_risk
from .alerts import AlertManager

ROOT = Path(__file__).resolve().parent.parent

def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def run(demo=True, namespace=None, baseline_path=None, current_path=None, dynamic=False):
    baseline_path = Path(baseline_path or ROOT / "snapshots/baseline.json")
    current_path = Path(current_path or ROOT / "snapshots/current.json")
    if demo:
        baseline, current = _read(baseline_path), _read(current_path)
        print("MODE: saved-snapshot demo")
    else:
        baseline = _read(baseline_path)
        from .collector import collect_resources  # Import only for live mode; snapshot demo needs no cluster client.
        resources = collect_resources(namespace=namespace)
        current = ConfigurationSnapshot(resources).to_dict()
        current_path.parent.mkdir(parents=True, exist_ok=True)
        current_path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        print(f"MODE: live Kubernetes cluster (namespace={namespace or 'all'})")
    events = DriftDetectionEngine().detect_drift(baseline, current)
    rules = SecurityRuleEngine(ROOT / "rules/security_rules.yaml")
    analyzer = AttackPathAnalyzer()
    graph = analyzer.build_graph(current)
    paths = analyzer.find_attack_paths()
    path_found = bool(paths)
    print(f"Baseline resources: {len(baseline.get('resources', []))}")
    print(f"Current resources:  {len(current.get('resources', []))}")
    print(f"Drift events:       {len(events)}")
    print(f"Attack-path candidates: {len(paths)}")
    print("\nDRIFT FINDINGS")
    if not events:
        print("No drift detected.")
    total_impact = 0
    matched = 0
    alert_manager = AlertManager(ROOT / "data/alerts.json", threshold=1)
    for event in events:
        matches = rules.evaluate_event(event)
        if not matches:
            print(f"- {event.resource_kind}/{event.namespace}/{event.resource_name}: {event.field} changed ({event.old_value!r} -> {event.new_value!r}); no rule matched")
            continue
        for rule in matches:
            matched += 1
            total_impact += rule["impact"]
            risk = calculate_risk(rule["impact"], path_found)
            alert = alert_manager.create_alert(event, rule, risk)
            print(f"- [{risk['level']} risk {risk['score']}/100] {event.resource_kind}/{event.namespace}/{event.resource_name}.{event.field}: {event.old_value!r} -> {event.new_value!r}")
            print(f"  Rule: {rule['rule_id']} | impact={rule['impact']} | alert={'stored' if alert else 'not stored'}")
    print(f"\nRules matched: {matched}")
    print(f"Combined rule impact (uncapped): {total_impact}")
    if paths:
        print("\nATTACK-PATH CANDIDATES (heuristic, validate manually)")
        for p in paths[:10]:
            print("  " + " -> ".join(p))
    print(f"\nAlerts file: {ROOT / 'data/alerts.json'}")
    result = {"events": events, "paths": paths, "matched_rules": matched}
    if dynamic:
        result["dynamic"] = run_dynamic(baseline, current)
    return result


def run_dynamic(baseline, current):
    """Proposed dynamic, parameter-driven engine (needs models/dynamic_bundle.pkl: python -m experiments.run_all)."""
    from .dynamic.engine import DynamicRiskEngine
    r = DynamicRiskEngine.load().assess(baseline, current)
    print("\n=== DYNAMIC ENGINE (proposed) ===")
    print(f"Alert: {r['alert']}  (probability {r['probability']:.3f}, threshold {r['threshold']:.3f})")
    print(f"Estimated damage increase: {r['estimated_damage_increase']:.2f} worker-node equivalents | severity: {r['severity']}")
    print(f"Active rules for this cluster: {r['active_rules']} of 54 templates | key parameters: {', '.join(r['active_parameters'][:6])}")
    for f in r["rules_fired"][:10]:
        print(f"- {f['rule']} (dynamic weight {f['weight_introduced']:.2f}, contribution {f['contribution']:+.2f}): {f['description']}")
        if f["introduced"]:
            print(f"    new: {', '.join(map(str, f['introduced']))}")
    for pth in r["attack_paths"][:3]:
        nodes = pth["path"] if isinstance(pth, dict) else pth
        print(f"  path (score {pth.get('score', 0):.2f}): " + " -> ".join(nodes) if isinstance(pth, dict) else "  path: " + " -> ".join(nodes))
    return r

def main():
    parser = argparse.ArgumentParser(description="Kubernetes security drift demo")
    parser.add_argument("--live", action="store_true", help="collect current resources from a running Kubernetes cluster")
    parser.add_argument("--namespace", help="limit live collection to one namespace")
    parser.add_argument("--baseline", help="baseline JSON path")
    parser.add_argument("--current", help="current JSON path/output path")
    parser.add_argument("--dynamic", action="store_true", help="also run the proposed dynamic engine")
    args = parser.parse_args()
    run(demo=not args.live, namespace=args.namespace, baseline_path=args.baseline, current_path=args.current, dynamic=args.dynamic)

if __name__ == "__main__":
    main()
