"""Algorithm 2 - Dynamic Rule Generation.

Input : environment parameters theta (Algorithm 1), learned DRWA weights, the template library,
        optional induced compound rules, the pair of snapshots (B, C).
Output: a RuleSet for THIS cluster - which rules are active, their weights and thresholds - rendered in the
        same YAML shape as ``rules/security_rules.yaml`` so it can replace the static file.

  1  identify theta and the active parameter set P*                (Algorithm 1)
  2  for every template t:  w_t(theta) = lambda*w0_t + delta_t + sum_j kappa_tj * phi_j(theta)   (DRWA)
  3  threshold adaptation:  vuln_thr = clip(p75(vulns of B), 10, 200);  alert threshold from validation
  4  gating: a rule is ACTIVE if it has instances in B or C, or its weight is non-negligible; otherwise it is
     DORMANT (kept in the library but not evaluated) - the active set therefore changes with the environment
  5  schema induction: unseen risky fields (any kind/field matching the risk lexicon) yield GENERIC rules
  6  append compound rules induced from outcome data (RuleInducer)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import yaml

from app.drift_detector import _flatten, _index
from app.dynamic.model import DRWA, HybridModel
from app.dynamic.params import ParameterIdentifier, StateAnalysis
from app.dynamic.templates import TEMPLATES

# risk lexicon for generic (schema-driven) rules: substring -> (direction of worsening, prior weight)
BAD_WHEN_TRUE = {"privileged": 1.0, "host_network": 1.0, "hostnetwork": 1.0, "host_pid": 0.6, "host_ipc": 0.4,
                 "allow_privilege_escalation": 0.8, "run_as_root": 0.7, "automount": 0.3, "anonymous": 0.8,
                 "insecure": 0.8, "debug": 0.4, "unsafe": 0.7, "public": 0.6, "wildcard": 0.7, "hostpath": 1.0,
                 "disable_auth": 0.9, "skip_tls": 0.8}
BAD_WHEN_FALSE = {"read_only": 0.4, "readonly": 0.4, "tls": 0.6, "encrypt": 0.6, "enforce": 0.6, "has_limits": 0.2,
                  "require_auth": 0.9, "authentication": 0.8, "run_as_non_root": 0.7}
EXPOSING_VALUES = {"LoadBalancer": 0.8, "NodePort": 0.7}


@dataclass
class DynamicRule:
    id: str
    template: str
    family: str
    source: str
    tactic: str
    description: str
    prior_w0: float
    weight_introduced: float
    weight_removed: float
    active: bool
    support: int
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class RuleSet:
    theta: Dict[str, float]
    active_params: List[str]
    rules: List[DynamicRule]
    generic: List[dict]
    induced: List[dict]
    alert_threshold: float

    @property
    def n_active(self) -> int:
        return sum(r.active for r in self.rules) + len(self.generic) + len(self.induced)

    def to_yaml(self) -> str:
        doc = {"rules": [
            {"id": r.id, "description": r.description, "template": r.template, "tactic": r.tactic, "source": r.source,
             "active": r.active, "weight_introduced": round(r.weight_introduced, 4),
             "weight_removed": round(r.weight_removed, 4), "prior_w0": r.prior_w0, "support": r.support,
             **({"params": r.params} if r.params else {})} for r in self.rules],
            "generic_rules": self.generic, "induced_rules": self.induced,
            "alert_threshold": self.alert_threshold, "active_parameters": self.active_params}
        return yaml.safe_dump(doc, sort_keys=False)


def induce_generic(B: dict, C: dict) -> List[dict]:
    """Schema-driven rules: any security-relevant field that moved in the worsening direction."""
    ib, ic = _index(B), _index(C)
    found: Dict[str, dict] = {}
    for key in set(ib) & set(ic):
        kind = key[0]
        fb, fc = _flatten(ib[key]), _flatten(ic[key])
        for fld in set(fb) | set(fc):
            ov, nv = fb.get(fld), fc.get(fld)
            if ov == nv:
                continue
            low = fld.lower()
            w = None
            if isinstance(nv, bool) and nv and not ov:
                w = next((x for k, x in BAD_WHEN_TRUE.items() if k in low), None)
            elif isinstance(nv, bool) and not nv and ov:
                w = next((x for k, x in BAD_WHEN_FALSE.items() if k in low), None)
            elif isinstance(nv, str) and nv in EXPOSING_VALUES and ov != nv:
                w = EXPOSING_VALUES[nv]
            elif isinstance(nv, list) and "*" in [str(x) for x in nv] and not (isinstance(ov, list) and "*" in [str(x) for x in ov or []]):
                w = 0.7
            if w is not None:
                rid = f"GEN::{kind}::{fld.split('.')[-1]}"
                e = found.setdefault(rid, {"id": rid, "kind": kind, "field": fld, "prior_weight": w, "instances": 0})
                e["instances"] += 1
    return sorted(found.values(), key=lambda e: -e["prior_weight"])


class DynamicRuleGenerator:
    def __init__(self, pia: ParameterIdentifier, drwa: DRWA, alert_threshold: float,
                 induced: Optional[List[dict]] = None, min_weight: float = 0.02):
        self.pia, self.drwa, self.alert_threshold = pia, drwa, alert_threshold
        self.induced = induced or []
        self.min_weight = min_weight

    def generate(self, aB: StateAnalysis, B: dict, C: Optional[dict] = None, aC: Optional[StateAnalysis] = None) -> RuleSet:
        C = C or B
        theta = self.pia.identify(aB, B, C)
        W = self.drwa.weights(theta).set_index("template")
        rules = []
        for t in TEMPLATES:
            n_b = len(aB.inst[t.id])
            n_c = len(aC.inst[t.id]) if aC is not None else 0
            wi, wr = float(W.loc[t.id, "w_introduced"]), float(W.loc[t.id, "w_removed"])
            active = (n_b + n_c > 0) or max(wi, wr) >= self.min_weight
            params = {"vuln_thr": round(aB.vuln_thr, 1)} if t.id in ("R_VULN", "X_CHAIN_EXPOSED_ROOT") else {}
            rules.append(DynamicRule(t.id, t.id, t.family, t.source, t.tactic, t.description, t.w0, wi, wr, active,
                                     n_b + n_c, params))
        return RuleSet(theta, self.pia.active(theta), rules, induce_generic(B, C), self.induced, self.alert_threshold)
