# Project summary (for team / teacher)

## The problem
Kubernetes clusters "drift": their security settings change over time. The base papers (Stay at the Helm; Defending Kubernetes
Clusters Against Network Misconfigurations) give fixed rules and a fixed attack-graph score to judge risk. Our original project
used 5 fixed rules.

## What was weak
1. Only 5 static rules, same weights for every cluster.
2. Edge cases (e.g. several small changes that are dangerous together) were not handled.
3. No data or training, so no fair comparison.

## What we did
1. **54 rules** (31 from Paper 1, 12 from Paper 2, 1 shared, 10 new for exceptional scenarios). Paper 1's Table VI values are the starting weights.
2. **Dynamic rules.** Algorithm 1 reads 40 parameters from a cluster (size, how many NetworkPolicies, privileged pods, RBAC power, drift size...). Algorithm 2 re-weights every rule from those parameters, so a rule's importance changes per cluster (e.g. "exposed root chain" is high in hardened clusters, low in legacy ones). Also: generic rules for unseen risky fields, and compound rules learned from data.
3. **Simulation-based attack.** A Monte-Carlo simulator lets an attacker spread through the cluster. A drift is "dangerous" if it raises expected damage by at least 0.25 worker-node equivalents. This gives labels to train and test on.
4. **Exceptional scenarios** (9): chained changes, group-wide role grants, node co-location, removed limits (cryptojacking), slow accumulation, a harmless privileged agent (false-alarm trap), etc.
5. **More nodes.** Clusters from 8 to 1,500 pods (up to 150 worker nodes), 6 environment types (sharing, internal, production, hardened, legacy, multi-tenant).
6. **Model:** dynamic-weight rules + gradient boosting. Compared against: original 5 rules, Paper 1 graph score, Paper 2 rules.

## Results (held-out test, 1,512 episodes)
| Method | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Original 5 static rules | 0.385 | 0.354 | 0.369 | 0.587 |
| Paper 1 graph risk | 0.430 | 0.813 | 0.562 | 0.783 |
| Paper 2 rules | 0.131 | 0.104 | 0.116 | 0.442 |
| **Proposed** | **0.701** | **0.868** | **0.775** | **0.942** |

On real manifests (6 real apps, 597 episodes): proposed F1 0.704 vs Paper 1 0.535 vs static 0.186.
Large clusters (400-1500 pods): proposed F1 0.749 vs Paper 1 0.569.

## Limitations (say these honestly)
- The papers' datasets are Helm charts with no attack labels, so "dangerous" comes from our simulator; its assumptions are ours.
- Only 6 real applications were tested here; the papers' 592/287 charts need Helm (script provided: experiments/fetch_helm_charts.sh).
- Image vulnerability counts are simulated (no Trivy scan).
- Live-cluster collector not extended or tested.
- Routine/harmless-change detection is still weak (F1 about 0.42).

## Still to do
1. Run fetch_helm_charts.sh and train/test on the papers' real charts (split by chart).
2. Add Trivy scans for real vulnerability counts.
3. Extend collector.py for the new fields and test on a real cluster.
4. Reduce false alarms on routine changes.
5. Write the final report / slides.
