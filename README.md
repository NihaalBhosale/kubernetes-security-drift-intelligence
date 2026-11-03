# Kubernetes Security Drift Intelligence

Detect **security-relevant configuration drift** in Kubernetes clusters and estimate how much it increases the risk of an attacker spreading through the cluster.

The project has two layers:

| Layer | What it does | ML? |
|---|---|---|
| **Static pipeline** (`app/*.py`) | Diffs a known-good baseline snapshot against a current snapshot, matches the changes against 5 YAML rules, builds a heuristic attack graph, scores risk and stores alerts. | No |
| **Dynamic engine** (`app/dynamic/`, `app/simulation/`) | Re-weights a library of 54 rule templates for each cluster using 40 environment parameters, then combines them with gradient boosting to decide whether a drift is *dangerous*. Trained and evaluated against a Monte-Carlo attacker simulator. | Yes |

The static pipeline is the original prototype and is left intact. The dynamic engine is a drop-in extension built on two base papers:

1. *Stay at the Helm* – attack-graph risk model (Paper 1; its Table VI control values are used as rule priors)
2. *Defending Kubernetes Clusters Against Network Misconfigurations* – misconfiguration rules M1–M7 (Paper 2)

---

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Results](#results)
- [Project layout](#project-layout)
- [Running with Docker](#running-with-docker)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Live-cluster mode](#live-cluster-mode)
- [Dashboard](#dashboard)
- [Using the engine from Python](#using-the-engine-from-python)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## Quick start

**Requirements:** Python 3.10+ (the Docker image uses 3.11). `kubectl` and a cluster are only needed for live mode.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the static pipeline on the bundled snapshots (no cluster needed):

```bash
python -m app.main
```

Run it together with the dynamic engine (uses the pre-trained `models/dynamic_bundle.pkl`):

```bash
python -m app.main --dynamic
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

> **Heads-up about the bundled snapshots.** `snapshots/baseline.json` holds 3 resources and `snapshots/current.json` holds 138 (a capture of a real local cluster, including its default RBAC roles). Running the demo therefore reports ~140 "added resource" drift events, most of which match no static rule. That is expected. To see a clean, rule-triggering example, supply your own pair with `--baseline` / `--current` (for instance one that flips `privileged: false` → `true`, which fires `PRIVILEGED_CONTAINER`, impact 30).

---

## How it works

### Static pipeline

```
collect/load snapshots → normalise + hash → field-level drift diff
        → match rules/security_rules.yaml → heuristic attack paths
        → risk score (impact, boosted if a path exists) → data/alerts.json
```

| Module | Role |
|---|---|
| `collector.py` | Reads resources from the Kubernetes API (live mode) |
| `snapshot.py` | Normalises resources into a stable JSON shape and hashes it |
| `drift_detector.py` | Produces added / removed / modified field-level events |
| `rule_engine.py` + `rules/security_rules.yaml` | 5 rules: privileged container, root container, NetworkPolicy removed, LoadBalancer service, host network |
| `attack_path_analyzer.py` | Exposure/RBAC graph and candidate paths (NetworkX) |
| `risk_assesment.py` | Impact + path → score and level (filename spelling kept for compatibility) |
| `alerts.py` | Persists alerts to `data/alerts.json` |
| `dashboard.py` | Streamlit UI |

### Dynamic engine

```
baseline, current
   │
   ├─ Algorithm 1 – Parameter Identification (params.py)
   │     40 parameters: size/topology, hardening rates, graph reach, drift size…
   │
   ├─ Algorithm 2 – Dynamic Rule Weight Adaptation (model.py, rule_generator.py)
   │     54 rule templates, each weight = prior + learned adjustment × environment parameters
   │     + generic rules for unseen risky fields + induced compound rules (decision tree)
   │
   └─ Hybrid model: DRWA rule evidence + graph deltas + parameters → gradient boosting
         → alert probability, estimated damage increase, severity, rules fired, attack paths
```

Key ideas:

- **Weights depend on the cluster.** A rule such as "exposed root chain" matters a lot in a hardened cluster and little in a legacy one, so its weight is a function of the identified parameters rather than a constant.
- **Rule library (54 templates):** 31 from Paper 1, 12 from Paper 2, 1 shared, 10 new ones for exceptional scenarios (`app/dynamic/templates.py`).
- **Ground truth from simulation.** `app/simulation/attack_sim.py` runs a Monte-Carlo attacker (paired common random numbers). A drift is labelled *dangerous* if it raises expected damage by at least **0.25 worker-node equivalents**.
- **Synthetic clusters.** `app/simulation/generator.py` creates clusters of 8–1,500 pods across 6 environment profiles (sharing, internal, production, hardened, legacy, multi-tenant), seeded from `snapshots/current.json`. `mutations.py` generates drift scenarios, including 9 *exceptional* ones: chained small changes, group-wide role grants, node co-location, removed resource limits (cryptojacking), slow accumulation, and a harmless privileged agent (a false-alarm trap).

Output severity levels: `None` (< 0.05), `Low` (< 0.25), `High` (< 1.0), `Critical` (≥ 1.0) worker-node equivalents.

---

## Results

Held-out synthetic test set (1,512 drift episodes), from `results/tables/main_comparison.csv`:

| Method | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Original 5 static rules | 0.385 | 0.354 | 0.369 | 0.587 |
| Paper 1 attack-graph risk delta | 0.430 | 0.813 | 0.562 | 0.783 |
| Paper 2 rules (M1–M7) | 0.131 | 0.104 | 0.116 | 0.442 |
| **Proposed (DRWA + GBM hybrid)** | **0.701** | **0.868** | **0.775** | **0.942** |

Other test sets (F1, proposed vs. Paper 1 vs. static rules):

| Test set | Proposed | Paper 1 | Static |
|---|---|---|---|
| Large clusters, 400–1,500 pods (384 episodes) | 0.749 | 0.569 | 0.291 |
| Real manifests, out-of-distribution (597 episodes) | 0.704 | 0.535 | 0.186 |

Ablation (in-range test F1): static prior weights 0.619 → + graph context 0.588 → + dynamic weights 0.734 → full hybrid 0.775. The tables folder also contains per-scenario, per-profile, per-category, node-scaling, leave-one-profile-out, rule-weight and parameter-relevance breakdowns.

> These numbers measure agreement with **our simulator's** definition of "dangerous". See [Limitations](#limitations) before quoting them.

---

## Project layout

```
.
├── app/
│   ├── main.py, collector.py, snapshot.py, drift_detector.py      static pipeline
│   ├── rule_engine.py, attack_path_analyzer.py, risk_assesment.py
│   ├── alerts.py, baseline.py, dashboard.py
│   ├── dynamic/        params, templates, analysis_graph, model, rule_generator,
│   │                   features, baselines, view, engine
│   ├── simulation/     generator, mutations, attack_sim
│   └── real/importer.py   load real YAML manifests as cluster state
├── experiments/        build_dataset, build_real_dataset, run_all, case_study,
│                       scan_images (Trivy), fetch_helm_charts.sh
├── rules/security_rules.yaml      the 5 static rules
├── snapshots/          baseline.json, current.json
├── models/             dynamic_bundle.pkl, dynamic_bundle_real.pkl (pre-trained)
├── results/tables/     CSV result tables + generated_ruleset.yaml
├── data/real/apps/     real manifests (argo-cd, metrics-server, metallb,
│                       prometheus-operator, ingress-nginx, kube-state-metrics)
├── tests/              test_drift_detection.py, test_dynamic.py
├── pod.yaml, service.yaml, network-policy.yaml    sample resources for live demo
├── Dockerfile, docker-compose.yml
└── EXPLAINER.md        short non-technical summary
```

---

## Running with Docker

The image bundles Python 3.11, Helm and Trivy.

```bash
docker compose run --rm demo          # static pipeline
docker compose run --rm dynamic       # case study with the dynamic engine
docker compose run --rm tests         # unit tests
docker compose up dashboard           # Streamlit at http://localhost:8501
docker compose run --rm experiments   # rebuild datasets + retrain + all tables (~30–60 min)
docker compose run --rm realcharts    # fetch Helm charts → Trivy scan → train/test on real charts
docker compose run --rm scan          # Trivy step only (resumable)
docker compose run --rm realtrain     # retrain on already fetched/scanned charts
```

---

## Reproducing the experiments

The generated datasets (`results/data/*.pkl`) are **not** shipped; the pre-trained models and result tables are. To regenerate everything:

```bash
python -m experiments.build_dataset --split train      --clusters 720 --episodes 6 --max-pods 300 --seed 101
python -m experiments.build_dataset --split val        --clusters 150 --episodes 6 --seed 202
python -m experiments.build_dataset --split test       --clusters 252 --episodes 6 --seed 303
python -m experiments.build_dataset --split test_large --clusters 96  --episodes 4 --min-pods 400 --max-pods 1500 --seed 404
python -m experiments.build_real_dataset --clusters 150 --episodes 4 --split real   # optional: real-manifest row-set
python -m experiments.run_all          # trains models/dynamic_bundle.pkl, writes results/tables/*.csv
python -m experiments.case_study       # baseline.json → current.json case study
```

Seeds are fixed, so reruns should reproduce the tables (up to library-version differences).

### Training and testing on real Helm charts

The base papers use public Helm charts. To split **by chart** (60/15/25 by hash of chart name, so test charts are never seen in training):

```bash
bash experiments/fetch_helm_charts.sh 60   # needs helm + network → data/real/apps/<chart>/
python -m experiments.scan_images --workers 4   # optional: real Trivy vulnerability counts
python -m experiments.build_real_dataset --part train --clusters 700 --episodes 6 --split real_train --seed 1
python -m experiments.build_real_dataset --part val   --clusters 150 --episodes 6 --split real_val   --seed 2
python -m experiments.build_real_dataset --part test  --clusters 250 --episodes 6 --split real_test  --seed 3
python -m experiments.run_all --real-only
```

This writes `models/dynamic_bundle_real.pkl`. Without a Trivy scan, image vulnerability counts are drawn per image (seeded) rather than measured.

---

## Live-cluster mode

> Live collection works for the static pipeline. The extended collector (container security fields, volumes, host flags, node names, Trivy counts) has **not been tested against a real cluster**.

```bash
kubectl get nodes
kubectl apply -f pod.yaml -f service.yaml -f network-policy.yaml

# Save the approved baseline from the live cluster
python -c "from app.collector import collect_resources; from app.snapshot import ConfigurationSnapshot; ConfigurationSnapshot(collect_resources(namespace='default')).save('snapshots/baseline.json')"

# Compare live state against it (add --dynamic for the proposed engine)
python -m app.main --live --namespace default
```

To create drift: set `privileged: true` in `pod.yaml` and re-apply, or `kubectl delete -f network-policy.yaml`. Some clusters reject privileged containers through admission control, in which case use the NetworkPolicy example instead. The baseline stays fixed until you recreate it.

Note: `--live` overwrites `snapshots/current.json`, and `data/alerts.json` is appended to on each run (both are git-ignored).

---

## Dashboard

```bash
streamlit run app/dashboard.py        # http://localhost:8501
```

Shows resource counts, drift events, rule matches, risk scores, stored alerts, a simplified attack graph and a **Dynamic engine** section. It reads `snapshots/baseline.json`, `snapshots/current.json` and `data/alerts.json`; it does not talk to the cluster itself, so run `python -m app.main --live` first and refresh.

---

## Using the engine from Python

```python
import json
from app.dynamic.engine import DynamicRiskEngine

baseline = json.load(open("snapshots/baseline.json"))
current  = json.load(open("snapshots/current.json"))

engine = DynamicRiskEngine.load()                       # models/dynamic_bundle.pkl
r = engine.assess(baseline, current)                    # high_recall=True for a lower alert threshold

r["alert"], r["probability"], r["severity"]
r["estimated_damage_increase"]                          # worker-node equivalents
r["rules_fired"]                                        # rule, dynamic weight, contribution, new/removed instances
r["attack_paths"]                                       # top-3 riskiest paths
r["ruleset"]                                            # generated per-cluster rule set (YAML via .to_yaml())
```

If the bundle is missing, `load()` tells you to run `python -m experiments.run_all`.

---

## Limitations

- **Labels come from our simulator.** Neither base paper publishes attack labels, so "dangerous" is defined by a world model whose probabilities are our assumptions. Reported scores show how well each method agrees with that model, not with real-world incident data.
- **Exceptional scenarios are self-defined.** The 9 exceptional scenarios and 10 extension rules are ours.
- **Small real-manifest sample.** Only a handful of real applications are bundled; the papers' full chart sets need `fetch_helm_charts.sh`.
- **Simulated vulnerability counts** unless you run the Trivy scan.
- **Collector untested on a live cluster** beyond the original static fields.
- **Routine (harmless) changes remain hard:** F1 ≈ 0.42 on that category, so false alarms on benign drift are still a weakness.
- **Attack paths are heuristic candidates**, not proof of exploitability. NetworkPolicy enforcement, RBAC verbs, admission policy and runtime state need fuller analysis for production use.
- Pre-trained models are pickles (`.pkl`); only load bundles you trust.

---

## Roadmap

1. Train and test on the papers' full Helm chart sets (split by chart).
2. Add Trivy scans for real vulnerability counts.
3. Extend and test `collector.py` for the new fields on a real cluster.
4. Reduce false alarms on routine changes.
5. Final report and slides.