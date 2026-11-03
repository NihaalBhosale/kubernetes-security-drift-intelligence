"""Scan every container image found in data/real/apps with Trivy and cache HIGH+CRITICAL vulnerability counts.

    python -m experiments.scan_images [--workers 4] [--max-images 0] [--timeout 300]

Writes data/real/vuln_counts.json   {image: n_high_critical, or (only successful scans are stored; failures are logged to data/real/scan_errors.log and retried on the next run)}.
Resumable: images already in the cache are skipped, the cache is saved after every image.
app/real/importer.py then uses these counts instead of simulated ones (images not in the cache fall back to simulated values;
-1 is treated as 'opaque image', exactly like the base paper's unscannable images).
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from app.real.importer import load_yaml_files  # noqa: E402

CACHE = ROOT / "data" / "real" / "vuln_counts.json"


def collect_images(apps_dir: Path) -> list:
    imgs = set()
    for d in sorted(p for p in apps_dir.iterdir() if p.is_dir()):
        for doc in load_yaml_files(sorted(d.rglob("*.yaml"))):
            spec = doc.get("spec") or {}
            for tmpl in (spec, spec.get("template", {}).get("spec") or {}, ((spec.get("jobTemplate") or {}).get("spec") or {}).get("template", {}).get("spec") or {}):
                for c in (tmpl.get("containers") or []) + (tmpl.get("initContainers") or []):
                    im = str(c.get("image") or "").strip()
                    if im and "{{" not in im and " " not in im:
                        imgs.add(im)
    return sorted(imgs)


_tls = threading.local()
BASE_CACHE = Path(os.environ.get("TRIVY_CACHE_DIR", Path.home() / ".cache" / "trivy"))
LOGF = ROOT / "data" / "real" / "scan_errors.log"


def preflight() -> None:
    """Download the vulnerability DB once and prove Trivy works on a tiny public image. Stops early with a clear error."""
    r = subprocess.run(["trivy", "image", "--download-db-only", "--no-progress"], capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        sys.exit("Trivy could not download its vulnerability database (needs internet access to ghcr.io / mirror.gcr.io):\n" + r.stderr[-1500:])
    r = subprocess.run(["trivy", "image", "--quiet", "--format", "json", "--scanners", "vuln", "--skip-db-update", "--skip-java-db-update",
                        "alpine:3.19"], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        sys.exit("Trivy cannot pull/scan a test image (alpine:3.19). Check internet/Docker Hub access:\n" + r.stderr[-1500:])
    print("preflight OK (database downloaded, test scan works)", flush=True)


def _worker_cache(wid: int) -> str:
    """Each worker gets its own copy of the DB: parallel Trivy processes sharing one cache lock each other out."""
    d = Path(tempfile.gettempdir()) / f"trivy-w{wid}"
    if not (d / "db").exists():
        d.mkdir(parents=True, exist_ok=True)
        shutil.copytree(BASE_CACHE / "db", d / "db")
    return str(d)


def scan(image: str, timeout: int, wid: int):
    """returns (count, error_text). count = -1 on failure."""
    try:
        r = subprocess.run(["trivy", "image", "--quiet", "--format", "json", "--scanners", "vuln", "--severity", "HIGH,CRITICAL",
                            "--cache-dir", _worker_cache(wid), "--skip-db-update", "--skip-java-db-update", "--no-progress",
                            "--timeout", f"{timeout}s", image], capture_output=True, text=True, timeout=timeout + 60)
        if r.returncode != 0 or not r.stdout.strip():
            return -1, (r.stderr or "empty output").strip()[-400:]
        data = json.loads(r.stdout)
        return sum(len(res.get("Vulnerabilities") or []) for res in data.get("Results") or []), ""
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps-dir", default=str(ROOT / "data/real/apps")); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all"); ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    if not shutil.which("trivy"):
        sys.exit("trivy not found on PATH (it is installed in the Docker image; locally see https://trivy.dev)")
    preflight()
    imgs = collect_images(Path(a.apps_dir))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    cache = {k: v for k, v in cache.items() if v >= 0}          # failed scans are never kept: they are retried on every run
    todo = [i for i in imgs if i not in cache]
    if a.max_images:
        todo = todo[:a.max_images]
    print(f"{len(imgs)} unique images, {len(cache)} already scanned, scanning {len(todo)}", flush=True)
    LOGF.parent.mkdir(parents=True, exist_ok=True); LOGF.write_text("")
    lock = threading.Lock(); done = fails = 0; counter = iter(range(10**6))

    def job(img):
        if not hasattr(_tls, "wid"):
            with lock:
                _tls.wid = next(counter) % a.workers
        return scan(img, a.timeout, _tls.wid)

    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(job, i): i for i in todo}
        for f in as_completed(futs):
            n, err = f.result()
            with lock:
                done += 1
                if n >= 0:
                    cache[futs[f]] = n
                    CACHE.parent.mkdir(parents=True, exist_ok=True); CACHE.write_text(json.dumps(cache, indent=0))
                else:
                    fails += 1
                    with open(LOGF, "a") as fh:
                        fh.write(f"== {futs[f]}\n{err}\n")
                if done % 10 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)} done, {fails} failed (reasons: data/real/scan_errors.log)", flush=True)
    cov = sum(1 for i in imgs if i in cache) / max(len(imgs), 1)
    print(f"done: {len(cache)} images with real Trivy counts = {cov:.0%} of {len(imgs)} unique images. Unscanned images keep SIMULATED counts.")


if __name__ == "__main__":
    main()
