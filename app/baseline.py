"""Baseline snapshot helpers."""
import json
from pathlib import Path
from .snapshot import ConfigurationSnapshot

class BaselineManager:
    def __init__(self, path="snapshots/baseline.json"):
        self.path = Path(path)

    def create_baseline(self, resources):
        return ConfigurationSnapshot(resources).save(self.path)

    def load_baseline(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def set_from_snapshot(self, snapshot_path):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(Path(snapshot_path).read_text(encoding="utf-8"), encoding="utf-8")
        return self.path
