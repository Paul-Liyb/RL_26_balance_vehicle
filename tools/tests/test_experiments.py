import sys
import subprocess
import unittest
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_real_log_model
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

    def test_make_env_passes_model_profile_to_base_env(self) -> None:
        env = make_env(seed=0, model_profile="measured_estimate")
        self.assertEqual(env.model_profile, "measured_estimate")
        env.close()

    def test_real_log_fit_profile_uses_fitted_transition(self) -> None:
        fit_path = fit_real_log_model.default_output_dir() / "real_log_fit_parameters.npz"
        if not fit_path.exists():
            result = subprocess.run(
                ["python3", str(TOOLS_DIR / "fit_real_log_model.py")],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

        env = make_env(seed=0, model_profile="real_log_fit")
        env.reset(seed=0)
        env.state = np.array([0.1, -0.1, 0.02, 0.06, 0.3, -0.2, 0.05, -0.04], dtype=np.float64)
        action = np.array([0.01, -0.02], dtype=np.float32)
        physical_action = env.action_scale * action
        data = np.load(fit_path, allow_pickle=False)
        features = np.concatenate([env.state, physical_action, np.ones(1, dtype=np.float64)])
        expected_state = features @ data["xu_to_x_next_coefficients"]

        _, _, _, _, info = env.step(action)
        np.testing.assert_allclose(info["raw_obs"], expected_state.astype(np.float32), atol=1e-6)
        env.close()

    def test_measured_delta_fit_profile_adds_fitted_residual(self) -> None:
        fit_path = fit_real_log_model.default_output_dir() / "real_log_fit_parameters.npz"
        if not fit_path.exists():
            result = subprocess.run(
                ["python3", str(TOOLS_DIR / "fit_real_log_model.py")],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

        env = make_env(seed=0, model_profile="measured_delta_fit")
        env.reset(seed=0)
        env.state = np.array([0.1, -0.1, 0.02, 0.06, 0.3, -0.2, 0.05, -0.04], dtype=np.float64)
        action = np.array([0.01, -0.02], dtype=np.float32)
        physical_action = env.action_scale * action
        data = np.load(fit_path, allow_pickle=False)
        features = np.concatenate([env.state, physical_action, np.ones(1, dtype=np.float64)])
        expected_state = (
            env.unwrapped.empirical_model.base_G @ env.state
            + env.unwrapped.empirical_model.base_H @ physical_action
            + features @ data["measured_delta_coefficients"]
        )

        _, _, _, _, info = env.step(action)
        np.testing.assert_allclose(info["raw_obs"], expected_state.astype(np.float32), atol=1e-6)
        env.close()


if __name__ == "__main__":
    unittest.main()
