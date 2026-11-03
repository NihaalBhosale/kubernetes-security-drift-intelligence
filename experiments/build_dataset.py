"""Build the labelled drift-episode dataset (simulation-based ground truth).

One row = one drift episode (baseline cluster B -> drifted cluster C):
  label       : dD = expected-damage increase from the Monte-Carlo attack simulation (+ standard error);
                dangerous = dD >= 0.25 worker-node equivalents
  baselines   : outputs of the existing static rules / paper-1 / paper-2 pipelines
  features    : template deltas, graph deltas and environment parameters theta (proposed method)

Usage:  python -m experiments.build_dataset --split train --clusters 700 --episodes 6 --min-pods 8 --max-pods 300
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.dynamic.baselines import Paper1Baseline, Paper2Baseline, StaticRulesBaseline  # noqa: E402
from app.dynamic.features import episode_features  # noqa: E402
from app.dynamic.params import StateAnalysis  # noqa: E402
from app.simulation.attack_sim import AttackSimulator, DANGER_TAU  # noqa: E402
from app.simulation.generator import PROFILE_NAMES, ClusterSpec, generate_cluster  # noqa: E402
from app.simulation.mutations import make_drift  # noqa: E402

_STATIC = None
_SIM = None


def _init():
    global _STATIC, _SIM
    _STATIC = StaticRulesBaseline()
    _SIM = AttackSimulator()


def severity(dd: float) -> str:
    if dd < 0.05:
        return "none"
    if dd < DANGER_TAU:
        return "low"
    if dd < 1.0:
        return "high"
    return "critical"


def cluster_job(job):
    """Generate one cluster and `episodes` drift episodes from it."""
    split, cid, spec, episodes, seed = job
    rng = np.random.default_rng(seed)
    B = spec if isinstance(spec, dict) else generate_cluster(spec)
    profile = B.get('meta', {}).get('profile', 'real') if isinstance(spec, dict) else spec.profile
    seed_i = B['meta'].get('seed', seed) if isinstance(spec, dict) else spec.seed
    aB = StateAnalysis(B)
    wB = _SIM.world(aB.view)
    rows = []
    attempts = 0
    while len(rows) < episodes and attempts < episodes * 4:
        attempts += 1
        out = make_drift(B, aB.view, rng)
        if out is None:
            continue
        C, info = out
        aC = StateAnalysis(C, aB.vuln_thr)
        wC = _SIM.world(aC.view)
        lab = _SIM.paired_delta(wB, wC, seed=int(seed_i) % 100000)
        feats = episode_features(aB, aC, B, C)
        st = _STATIC.score(B, C)
        p2 = Paper2Baseline.score(aB, aC)
        row = {
            "split": split, "cluster_id": cid, "episode_id": f"{cid}-{len(rows)}", "profile": profile,
            "n_pods": aB.view.n_pods, "n_nodes": len(aB.view.nodes), "n_namespaces": len(aB.view.namespaces),
            "n_resources": len(B["resources"]),
            "scenario": info["scenario"], "category": info["category"], "n_noise": info["n_noise"],
            "mutations": "|".join(e["mutation"] for e in info["events"]),
            "dD": lab["dD"], "se": lab["se"], "trials": lab["trials"], "D_B": lab["D_B"], "D_C": lab["D_C"], "dR": lab["dR"],
            "dangerous": int(lab["dD"] >= DANGER_TAU), "severity": severity(lab["dD"]),
            "static_score": st["score"], "static_alert": st["alert"], "static_matched": st["matched"],
            "static_rules": ",".join(sorted(set(st["rule_ids"]))),
            "p2_score": p2["score"], "p2_alert": p2["alert"],
        }
        for agg in Paper1Baseline.AGGS:
            row[f"p1_{agg}"] = Paper1Baseline.score(aB, aC, agg)
        row.update(feats)
        rows.append(row)
    return rows


def build(split: str, n_clusters: int, episodes: int, min_pods: int, max_pods: int, seed: int,
          profiles=None, workers: int = 2) -> pd.DataFrame:
    profiles = profiles or PROFILE_NAMES
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n_clusters):
        n_pods = int(round(np.exp(rng.uniform(np.log(min_pods), np.log(max_pods)))))
        n_nodes = int(np.clip(n_pods / rng.uniform(4, 25), 1, 150))
        n_ns = int(np.clip(round(rng.uniform(0.6, 1.4) * np.sqrt(n_pods) / 2), 2, 70))
        spec = ClusterSpec(profiles[i % len(profiles)], n_pods, max(1, n_nodes), n_ns, int(rng.integers(0, 2**31 - 1)))
        jobs.append((split, f"{split}{i:04d}", spec, episodes, int(rng.integers(0, 2**31 - 1))))
    rows = []
    t0 = time.time()
    with mp.Pool(workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(cluster_job, jobs, chunksize=1), 1):
            rows.extend(r)
            if k % 25 == 0 or k == len(jobs):
                print(f"[{split}] {k}/{len(jobs)} clusters, {len(rows)} episodes, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--clusters", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--min-pods", type=int, default=8)
    ap.add_argument("--max-pods", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "data"))
    a = ap.parse_args()
    df = build(a.split, a.clusters, a.episodes, a.min_pods, a.max_pods, a.seed, workers=a.workers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out / f"{a.split}.pkl")
    print(f"saved {len(df)} rows -> {out / (a.split + '.pkl')}; dangerous rate {df['dangerous'].mean():.3f}")


if __name__ == "__main__":
    main()
