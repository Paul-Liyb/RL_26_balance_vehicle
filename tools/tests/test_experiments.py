import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from rl_balance.experiments import build_model, make_env


class ExperimentCoreTests(unittest.TestCase):
    def test_sac_batch128_profile_changes_batch_size(self) -> None:
        env = make_env(seed=0)
        model = build_model("sac", env, seed=0, sac_profile="batch128")
        self.assertEqual(model.batch_size, 128)
        env.close()

    def test_sac_batch128_lr1e4_profile_changes_learning_rate(self) -> None:
        env = make_env(seed=0)
        model = build_model("sac", env, seed=0, sac_profile="batch128_lr1e4")
        self.assertEqual(model.batch_size, 128)
        self.assertAlmostEqual(model.lr_schedule(1.0), 1e-4)
        env.close()

    def test_sac_batch128_gamma998_profile_changes_gamma(self) -> None:
        env = make_env(seed=0)
        model = build_model("sac", env, seed=0, sac_profile="batch128_gamma998")
        self.assertEqual(model.batch_size, 128)
        self.assertAlmostEqual(model.gamma, 0.998)
        env.close()

    def test_make_env_supports_residual_lqr_action_mode(self) -> None:
        env = make_env(seed=0, action_mode="residual_lqr", residual_scale=0.15)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (8,))
        _, _, _, _, info = env.step([0.0, 0.0])
        self.assertEqual(info["action_mode"], "residual_lqr")
        self.assertAlmostEqual(info["residual_scale"], 0.15)
        self.assertIn("teacher_action", info)
        self.assertIn("residual_action", info)
        env.close()


if __name__ == "__main__":
    unittest.main()
