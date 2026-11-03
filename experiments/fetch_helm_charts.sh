#!/usr/bin/env bash
# Render the public Helm charts used by the base papers into data/real/apps/<chart>/rendered.yaml.
# Needs: helm (>=3.10) and network access.   Usage: bash experiments/fetch_helm_charts.sh [max_charts_per_repo]
# Paper 1 (Stay at the helm) repos: prometheus-community, kube-state-metrics, renovate, artefact, hashicorp, enix, klustair,
# camunda, hpe-storage, nutanix, bitnami, halkeye, ibm-charts, helm-charts.   Paper 2: Bitnami, Banzai Cloud, Wikimedia, EEA, CNCF, Prometheus Community.
# Edit REPOS to add/fix URLs (some of the paper's repo URLs have moved).
set -u
MAX=${1:-60}
OUT="$(dirname "$0")/../data/real/apps"; mkdir -p "$OUT"
declare -A REPOS=(
  [bitnami]="https://charts.bitnami.com/bitnami"
  [prometheus-community]="https://prometheus-community.github.io/helm-charts"
  [hashicorp]="https://helm.releases.hashicorp.com"
  [camunda]="https://helm.camunda.io"
  [ibm-charts]="https://raw.githubusercontent.com/IBM/charts/master/repo/stable"
  [enix]="https://charts.enix.io"
  [hpe-storage]="https://hpe-storage.github.io/co-deployments"
  [nutanix]="https://nutanix.github.io/helm"
  [renovate]="https://docs.renovatebot.com/helm-charts"
  [halkeye]="https://halkeye.github.io/helm-charts"
  [eea]="https://eea.github.io/helm-charts"
  # Paper 2 'internal' (Wikimedia) and 'production' (CNCF projects). Unreachable URLs are skipped automatically.
  [wikimedia]="https://helm-charts.wikimedia.org/stable/"
  [argo]="https://argoproj.github.io/argo-helm"
  [jetstack]="https://charts.jetstack.io"
  [fluent]="https://fluent.github.io/helm-charts"
  [jaegertracing]="https://jaegertracing.github.io/helm-charts"
  [ingress-nginx]="https://kubernetes.github.io/ingress-nginx"
  [grafana]="https://grafana.github.io/helm-charts"
  [cilium]="https://helm.cilium.io"
)
LOG="$OUT/../render_errors.log"; : > "$LOG"
for repo in "${!REPOS[@]}"; do
  helm repo add "$repo" "${REPOS[$repo]}" >/dev/null 2>&1 || { echo "skip $repo (cannot add repo)"; continue; }
  helm repo update "$repo" >/dev/null 2>&1
  charts=$(helm search repo "$repo/" -o json 2>/dev/null | python3 -c "import sys,json; print('\n'.join(c['name'] for c in json.load(sys.stdin)[:$MAX]))")
  ok=0; tot=0
  for chart in $charts; do
    tot=$((tot+1)); name="${chart//\//__}"; mkdir -p "$OUT/$name"
    # --kube-version: many charts refuse to render under helm's default (old) Kubernetes version
    if helm template rel "$chart" --namespace default --kube-version 1.30.0 > "$OUT/$name/rendered.yaml" 2>>"$LOG.tmp"; then
      ok=$((ok+1))
    else
      echo "== $chart" >> "$LOG"; tail -n 3 "$LOG.tmp" >> "$LOG"; rm -rf "$OUT/$name"
    fi
    : > "$LOG.tmp"
  done
  echo "$repo: rendered $ok / $tot charts"
done

# ---- charts that live in plain GitHub repositories (no helm repo index): clone, then render every Chart.yaml found
declare -A GIT_REPOS=(
  [klustair]="https://github.com/klustair/klustair-helm.git"
  [banzaicloud]="https://github.com/banzaicloud/banzai-charts.git"
)
GITTMP="$(mktemp -d)"
for repo in "${!GIT_REPOS[@]}"; do
  git clone -q --depth 1 "${GIT_REPOS[$repo]}" "$GITTMP/$repo" 2>/dev/null || { echo "skip $repo (git clone failed)"; continue; }
  ok=0; tot=0
  while read -r cy; do
    dir=$(dirname "$cy"); name="${repo}-$(basename "$dir")"
    tot=$((tot+1)); mkdir -p "$OUT/$name"
    helm dependency build "$dir" >/dev/null 2>&1 || true      # pulls sub-charts when reachable; ignored otherwise
    if helm template rel "$dir" --namespace default --kube-version 1.30.0 > "$OUT/$name/rendered.yaml" 2>>"$LOG.tmp"; then
      ok=$((ok+1))
    else
      echo "== $repo/$(basename "$dir")" >> "$LOG"; tail -n 3 "$LOG.tmp" >> "$LOG"; rm -rf "$OUT/$name"
    fi
    : > "$LOG.tmp"
  done < <(find "$GITTMP/$repo" -maxdepth 2 -name Chart.yaml | head -n "$MAX")
  echo "$repo (git): rendered $ok / $tot charts"
done
rm -rf "$GITTMP"
rm -f "$LOG.tmp"
echo "charts rendered in total: $(ls "$OUT" | wc -l)   (failures and reasons: data/real/render_errors.log)"
