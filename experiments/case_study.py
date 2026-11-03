"""Real-snapshot case study (the project's own baseline.json -> current.json) + rule-count table."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from app.dynamic.engine import DynamicRiskEngine
from app.dynamic.baselines import StaticRulesBaseline, Paper1Baseline, Paper2Baseline
from app.dynamic.params import StateAnalysis
from app.dynamic.templates import TEMPLATES

B = json.load(open(ROOT / "snapshots/baseline.json")); C = json.load(open(ROOT / "snapshots/current.json"))
eng = DynamicRiskEngine.load()
r = eng.assess(B, C)
aB = StateAnalysis(B); aC = StateAnalysis(C, aB.vuln_thr)
st = StaticRulesBaseline().score(B, C)
T = ROOT / "results/tables"
rows = [{"method": "Existing: static rules", "alert": bool(st["alert"]), "score": st["score"], "detail": ",".join(sorted(set(st["rule_ids"])))},
        {"method": "Existing: Paper-1 graph delta (sum)", "alert": "", "score": Paper1Baseline.score(aB, aC, "sum"), "detail": ""},
        {"method": "Existing: Paper-2 rules", "alert": bool(Paper2Baseline.score(aB, aC)["alert"]), "score": Paper2Baseline.score(aB, aC)["score"], "detail": ""},
        {"method": "Proposed: dynamic engine", "alert": r["alert"], "score": r["probability"],
         "detail": f"est. damage increase {r['estimated_damage_increase']:.2f} node-eq, severity {r['severity']}, {r['active_rules']} active rules"}]
pd.DataFrame(rows).to_csv(T / "case_study.csv", index=False)
pd.DataFrame([{k: v for k, v in f.items() if k in ("rule", "tactic", "contribution", "weight_introduced")} | {"introduced": "; ".join(f["introduced"]), "removed": "; ".join(f["removed"])}
              for f in r["rules_fired"]]).to_csv(T / "case_study_rules.csv", index=False)
(T / "generated_ruleset.yaml").write_text(r["ruleset"].to_yaml())
fam = pd.Series([t.family for t in TEMPLATES]).value_counts()
pd.DataFrame([{"item": "Existing static rules (original project)", "count": 5}, {"item": "Paper-1 derived templates", "count": int(fam.get("paper1", 0) + fam.get("paper1+paper2", 0))},
              {"item": "Paper-2 derived templates", "count": int(fam.get("paper2", 0))}, {"item": "New extension templates (exceptional scenarios)", "count": int(fam.get("extension", 0))},
              {"item": "Total template rules", "count": len(TEMPLATES)}]).to_csv(T / "rule_counts.csv", index=False)
print(pd.DataFrame(rows).to_string()); print(len(r["rules_fired"]), "rules fired;", len(r["ruleset"].generic), "generic rules")
