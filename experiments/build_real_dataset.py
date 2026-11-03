"""Build clusters from REAL manifests (Helm charts rendered with `helm template`, or plain YAML) and label drift episodes.

Layout expected:   data/real/apps/<app-name>/*.yaml      (one directory per chart/application)
Produce it with:   bash experiments/fetch_helm_charts.sh   (needs `helm` + `git`; see the script)

Each cluster = k randomly chosen real applications deployed side by side (as in Paper 2's 'production' datasets), each in its own
namespace, on a few worker nodes with kube-system infrastructure. Drift episodes + attack-simulation labels then work as before.
"""
from __future__ import annotations
import argparse, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from app.real.importer import load_yaml_files, manifests_to_resources  # noqa: E402
from app.simulation.generator import ClusterSpec, generate_cluster  # noqa: E402
from experiments import build_dataset as bd  # noqa: E402


def load_apps(root: Path) -> dict:
    apps = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        docs = load_yaml_files(sorted(d.rglob("*.yaml")) + sorted(d.rglob("*.yml")))
        if any(x["kind"] in ("Deployment", "StatefulSet", "DaemonSet", "Pod") for x in docs):
            apps[d.name] = docs
    return apps


def make_cluster(apps: dict, names: list, n_nodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    base = generate_cluster(ClusterSpec("production", 1, n_nodes, 1, seed))
    keep = [r for r in base["resources"] if r["kind"] == "Node" or r["namespace"] == "kube-system" or r["kind"] in ("ClusterRole", "ClusterRoleBinding")]
    nodes = [r["name"] for r in base["resources"] if r["kind"] == "Node"]
    res = list(keep)
    for i, n in enumerate(names):
        res += manifests_to_resources(apps[n], rng, namespace=f"app-{n}"[:50], nodes=nodes, seed=seed)
    # real manifests may reference namespaces that do not exist as ServiceAccounts: add default SAs
    have = {(r["namespace"], r["name"]) for r in res if r["kind"] == "ServiceAccount"}
    for r in list(res):
        if r["kind"] == "Pod":
            k = (r["namespace"], r["data"]["service_account_name"])
            if k not in have:
                have.add(k); res.append({"kind": "ServiceAccount", "name": k[1], "namespace": k[0], "data": {"labels": {}, "automount_service_account_token": True}})
    return {"timestamp": base["timestamp"], "configuration_hash": f"real-{seed}", "resources": res, "meta": {"profile": "real", "seed": seed, "apps": names}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps-dir", default=str(ROOT / "data/real/apps")); ap.add_argument("--clusters", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=6); ap.add_argument("--min-apps", type=int, default=1); ap.add_argument("--max-apps", type=int, default=8)
    ap.add_argument("--split", default="real"); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--part", choices=["all", "train", "val", "test"], default="all",
                    help="use only the charts whose hash falls in this part (60/15/25 split BY CHART, so test charts are never seen in training)")
    ap.add_argument("--out", default=str(ROOT / "results/data"))
    a = ap.parse_args()
    apps = load_apps(Path(a.apps_dir))
    if a.part != "all":
        import hashlib
        def bucket(n):
            h = int(hashlib.md5(n.encode()).hexdigest()[:6], 16) % 100
            return "train" if h < 60 else "val" if h < 75 else "test"
        apps = {k: v for k, v in apps.items() if bucket(k) == a.part}
    names = sorted(apps)
    from app.real.importer import _scan_cache
    from experiments.scan_images import collect_images
    _imgs = collect_images(Path(a.apps_dir)); _c = _scan_cache()
    _cov = sum(1 for i in _imgs if _c.get(i, -1) >= 0) / max(len(_imgs), 1)
    print(f"Trivy coverage: {_cov:.0%} of {len(_imgs)} images have REAL counts (rest simulated)" + ("  <-- LOW: rerun 'docker compose run --rm scan'" if _cov < 0.8 else ""))
    print(f"{len(names)} real applications loaded")
    rng = np.random.default_rng(a.seed); jobs = []
    for i in range(a.clusters):
        k = int(rng.integers(a.min_apps, min(a.max_apps, len(names)) + 1))
        pick = list(rng.choice(names, size=k, replace=False))
        sd = int(rng.integers(0, 2**31 - 1))
        B = make_cluster(apps, pick, max(1, int(rng.integers(2, 12))), sd)
        jobs.append((a.split, f"{a.split}{i:04d}", B, a.episodes, int(rng.integers(0, 2**31 - 1))))
    rows, t0 = [], time.time()
    with mp.Pool(a.workers, initializer=bd._init) as pool:
        for k, r in enumerate(pool.imap_unordered(bd.cluster_job, jobs), 1):
            rows += r
            if k % 25 == 0: print(f"{k}/{len(jobs)} clusters {len(rows)} episodes {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); Path(a.out).mkdir(parents=True, exist_ok=True); df.to_pickle(Path(a.out) / f"{a.split}.pkl")
    print(f"saved {len(df)} rows; dangerous rate {df['dangerous'].mean():.3f}; pods median {df['n_pods'].median()}")


if __name__ == "__main__":
    main()
