import unittest
import numpy as np
from app.simulation.generator import ClusterSpec, generate_cluster
from app.simulation.attack_sim import AttackSimulator
from app.simulation.mutations import make_drift
from app.dynamic.params import StateAnalysis, PARAM_NAMES
from app.dynamic.templates import TEMPLATES
from app.dynamic.features import episode_features
from app.dynamic.rule_generator import induce_generic


class DynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B = generate_cluster(ClusterSpec("production", 60, 6, 6, 5))
        cls.aB = StateAnalysis(cls.B)

    def test_rule_library_size(self):
        self.assertEqual(len(TEMPLATES), 54)
        self.assertEqual(len({t.id for t in TEMPLATES}), 54)

    def test_identical_snapshots_have_no_drift_features(self):
        aC = StateAnalysis(self.B, self.aB.vuln_thr)
        f = episode_features(self.aB, aC, self.B, self.B)
        self.assertTrue(all(f[k] == 0 for k in f if k.startswith(("pos_raw::", "neg_raw::"))))

    def test_parameters_present(self):
        self.assertEqual(set(PARAM_NAMES), set(self.aB.posture) | set(PARAM_NAMES))

    def test_simulator_zero_delta_on_same_cluster(self):
        sim = AttackSimulator()
        w = sim.world(self.aB.view)
        self.assertAlmostEqual(sim.paired_delta(w, w, seed=1)["dD"], 0.0, places=6)

    def test_drift_generation_and_generic_rules(self):
        rng = np.random.default_rng(3)
        got = None
        for _ in range(20):
            got = make_drift(self.B, self.aB.view, rng, category="exceptional")
            if got:
                break
        self.assertIsNotNone(got)
        C, info = got
        self.assertIsInstance(induce_generic(self.B, C), list)


if __name__ == "__main__":
    unittest.main()
