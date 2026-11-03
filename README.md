# Kubernetes Security Drift Intelligence

A runnable, rule-based prototype that compares a known-good Kubernetes configuration snapshot with a current snapshot, identifies field-level drift, evaluates YAML security rules, estimates risk, builds heuristic attack-path candidates, and stores alerts. It does **not** use ML or require a training dataset.

## 1. Requirements
- Python 3.10+
- For live mode: a working Kubernetes cluster and `kubectl` configured for it (Docker Desktop Kubernetes, Minikube, etc.).

## 2. Install
Run these commands from this project folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run the deterministic demo (no cluster needed)
```bash
python -m app.main
```
The included snapshots intentionally represent a change from `privileged: false` to `privileged: true` and removal of a NetworkPolicy. Expected rule matches include `PRIVILEGED_CONTAINER` (impact 30) and `NETWORK_POLICY_REMOVED` (impact 25).

## 4. Run tests
```bash
python -m unittest discover -s tests -v
```

## 5. Live Kubernetes demo
First confirm your cluster is running:
```bash
kubectl get nodes
```
Apply the safe sample resources:
```bash
kubectl apply -f pod.yaml
kubectl apply -f service.yaml
kubectl apply -f network-policy.yaml
```
Create a baseline from the current live cluster:
```bash
python -c "from app.collector import collect_resources; from app.snapshot import ConfigurationSnapshot; ConfigurationSnapshot(collect_resources(namespace='default')).save('snapshots/baseline.json')"
```
Run live comparison:
```bash
python -m app.main --live --namespace default
```
To demonstrate drift, edit `pod.yaml` and change `privileged: false` to `privileged: true`, then apply it:
```bash
kubectl apply -f pod.yaml
python -m app.main --live --namespace default
```
**Important:** Some clusters enforce Pod security policies/admission controls and may reject privileged containers. If so, keep the deterministic snapshot demo, or demonstrate a different rule that the cluster allows. Revert the YAML to `privileged: false` after the demo.

To demonstrate NetworkPolicy removal:
```bash
kubectl delete -f network-policy.yaml
python -m app.main --live --namespace default
```
The live baseline remains unchanged unless you explicitly recreate it.

## 6. Pipeline
1. Collect resources from Kubernetes API (live mode).
2. Normalize resources to a stable JSON shape and hash.
3. Compare baseline with current snapshot.
4. Match field-level drift against `rules/security_rules.yaml`.
5. Build heuristic exposure/RBAC graph and identify candidate paths.
6. Calculate impact/risk and persist alerts to `data/alerts.json`.

## Notes / limitations
- `snapshots/baseline.json` is the approved reference state for comparison; it is not a training dataset.
- Attack paths are heuristic candidates, not proof that an attack is exploitable. Kubernetes NetworkPolicy enforcement, RBAC verbs/resources, admission policy, and runtime state need fuller analysis for production use.
- The demo uses fixed snapshots so it runs even without a Kubernetes cluster. Live mode needs Kubernetes access.
- `app/risk_assesment.py` retains the original project filename spelling for compatibility.


## 7. Dashboard and attack graph

Install the updated dependencies (`pip install -r requirements.txt`), then launch the dashboard from the project root:

```powershell
streamlit run app/dashboard.py
```

Streamlit will print a local URL, normally `http://localhost:8501`. Open it in your browser. The dashboard shows resource counts, detected drift, rule matches, risk scores, stored alerts, and a simplified attack graph. The graph hides isolated built-in cluster resources and prioritizes candidate paths so the visualization remains readable.

For live data, first run this in another terminal from the project root:

```powershell
python -m app.main --live --namespace default
```

Then refresh the dashboard page. The dashboard reads `snapshots/baseline.json`, `snapshots/current.json`, and `data/alerts.json`; it does not collect cluster resources on its own. Keep the terminal running while you use the dashboard.


---
## Dynamic, parameter-driven extension (this revision)

What was added on top of the original static pipeline (original files are unchanged):

| Area | Where |
|---|---|
| Parameter identification (Algorithm 1) + dynamic rule weight adaptation (Algorithm 2) | `app/dynamic/params.py`, `model.py`, `rule_generator.py` |
| 54 rule templates (31 Paper-1, 12 Paper-2, 1 shared, 10 new for exceptional scenarios), Table VI control values as priors | `app/dynamic/templates.py`, `analysis_graph.py` |
| Simulation-based attack ground truth (Monte-Carlo, paired common random numbers) | `app/simulation/attack_sim.py` |
| Cluster generator (6 environment profiles, 8 to 1500 pods) seeded from `snapshots/current.json`; drift scenarios incl. 9 exceptional ones | `app/simulation/generator.py`, `mutations.py` |
| Drop-in engine | `app/dynamic/engine.py` (`DynamicRiskEngine.assess(baseline, current)`) |

Reproduce:
```
python -m experiments.build_dataset --split train --clusters 720 --episodes 6 --max-pods 300 --seed 101
python -m experiments.build_dataset --split val --clusters 150 --episodes 6 --seed 202
python -m experiments.build_dataset --split test --clusters 252 --episodes 6 --seed 303
python -m experiments.build_dataset --split test_large --clusters 96 --episodes 4 --min-pods 400 --max-pods 1500 --seed 404
python -m experiments.run_all && python -m experiments.case_study
python -m unittest discover -s tests
```
Result tables are in `results/tables/*.csv`.

Caveats: labels come from a simulator whose world model is our assumption (the supplied zip has no labelled attack data); the
exceptional scenarios are defined by us; the live collector has not been extended or tested against a real cluster.

### Real manifests / the papers' Helm charts
The papers' datasets are public Helm charts (static, unlabelled). To use them:
```
bash experiments/fetch_helm_charts.sh 60            # needs helm + network; renders charts into data/real/apps/<chart>/
python -m experiments.build_real_dataset --clusters 600 --episodes 6 --split real
python -m experiments.run_all                       # adds a "REAL manifests" row-set to results/tables/main_comparison.csv
```
Attack/drift labels still come from the simulator (the papers publish no attack labels). Image vulnerability counts are drawn per image
(seeded) unless you add a Trivy scan.

### Train AND test on the papers' real charts (split by chart)
`docker compose run --rm realcharts` (or run the four commands in docker-compose.yml by hand). Charts are split 60/15/25 by hash of the chart
name into train/val/test, so test charts are never seen in training. Results: `results/tables_real/main_comparison.csv`.

### Final feature list (this revision)
- `python -m app.main --dynamic` runs the proposed engine after the static demo; the Streamlit dashboard has a "Dynamic engine" section.
- `collector.py` now also collects container security fields, volumes, host flags, node names and (if `data/real/vuln_counts.json` exists) Trivy counts. **Not tested on a live cluster.**
- `engine.assess(baseline, current, high_recall=True)` uses a higher-recall alert threshold (see `tables*/operating_points.csv`).
- Trivy: `docker compose run --rm scan`, then `docker compose run --rm realtrain`.
- More chart repositories (Wikimedia, Argo, cert-manager, Fluent, Jaeger, ingress-nginx, Grafana, Cilium); unreachable ones are skipped and listed.
