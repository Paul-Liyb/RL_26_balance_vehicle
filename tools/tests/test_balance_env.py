import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from rl_balance.env import BalanceStandEnv
from rl_balance.policies import LqrPolicy
from rl_balance.experiments import make_env


class BalanceStandEnvTests(unittest.TestCase):
    def test_reset_returns_expected_shape_and_dtype(self) -> None:
        env = BalanceStandEnv(seed=123)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (8,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertIn("raw_obs", info)
        np.testing.assert_allclose(obs, env.normalize_obs(info["raw_obs"]))

    def test_measured_profile_changes_dynamics(self) -> None:
        vendor_env = BalanceStandEnv(seed=123, model_profile="vendor_matlab")
        measured_env = BalanceStandEnv(seed=123, model_profile="measured_estimate")
        self.assertEqual(measured_env.model_profile, "measured_estimate")
        self.assertFalse(np.allclose(vendor_env.G, measured_env.G, atol=1e-9, rtol=0.0))
        self.assertFalse(np.allclose(vendor_env.H, measured_env.H, atol=1e-9, rtol=0.0))

    def test_seeded_rollout_is_reproducible(self) -> None:
        env1 = BalanceStandEnv(seed=7)
        env2 = BalanceStandEnv(seed=7)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        np.testing.assert_allclose(obs1, obs2)
        for _ in range(10):
            action = np.array([0.1, -0.2], dtype=np.float32)
            step1 = env1.step(action)
            step2 = env2.step(action)
            np.testing.assert_allclose(step1[0], step2[0], atol=1e-8)
            self.assertAlmostEqual(step1[1], step2[1], places=8)
            self.assertEqual(step1[2], step2[2])
            self.assertEqual(step1[3], step2[3])
            self.assertEqual(step1[4]["termination_reason"], step2[4]["termination_reason"])
            np.testing.assert_allclose(step1[4]["raw_obs"], step2[4]["raw_obs"], atol=1e-8)
            np.testing.assert_allclose(step1[4]["physical_action"], step2[4]["physical_action"], atol=1e-8)

    def test_termination_on_large_body_angle(self) -> None:
        env = BalanceStandEnv(seed=0)
        env.reset()
        env.state = np.array([0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertLess(reward, -20.0)
        self.assertEqual(info["termination_reason"], "body_angle_limit")

    def test_custom_reset_profile_stays_within_profile_bounds(self) -> None:
        env = BalanceStandEnv(seed=0, reset_profile="narrow")
        for episode_seed in range(20):
            _, info = env.reset(seed=episode_seed)
            raw_obs = info["raw_obs"]
            np.testing.assert_array_less(np.abs(raw_obs[:2]), np.array([0.030001, 0.030001], dtype=np.float32))
            np.testing.assert_array_less(np.abs(raw_obs[2:4]), np.array([0.060001, 0.060001], dtype=np.float32))
            np.testing.assert_array_less(np.abs(raw_obs[4:]), np.array([0.300001] * 4, dtype=np.float32))

    def test_posture_focus_reset_profile_only_narrows_posture_terms(self) -> None:
        env = BalanceStandEnv(seed=0, reset_profile="posture_focus")
        for episode_seed in range(20):
            _, info = env.reset(seed=episode_seed)
            raw_obs = info["raw_obs"]
            np.testing.assert_array_less(np.abs(raw_obs[:2]), np.array([0.050001, 0.050001], dtype=np.float32))
            np.testing.assert_array_less(np.abs(raw_obs[2:4]), np.array([0.060001, 0.060001], dtype=np.float32))
            np.testing.assert_array_less(np.abs(raw_obs[4:6]), np.array([0.500001, 0.500001], dtype=np.float32))
            np.testing.assert_array_less(np.abs(raw_obs[6:]), np.array([0.300001, 0.300001], dtype=np.float32))

    def test_posture_focus_reward_profile_penalizes_posture_error_more(self) -> None:
        default_env = BalanceStandEnv(seed=0)
        posture_env = BalanceStandEnv(seed=0, reward_profile="posture_focus")
        state = np.array([0.01, -0.01, 0.08, -0.08, 0.05, -0.05, 0.10, -0.10], dtype=np.float64)
        action = np.zeros(2, dtype=np.float32)
        default_reward = default_env._reward(state, action, False)
        posture_reward = posture_env._reward(state, action, False)
        self.assertLess(posture_reward, default_reward)

    def test_lqr_policy_stabilizes_small_disturbances(self) -> None:
        env = BalanceStandEnv(seed=0)
        policy = LqrPolicy()
        successes = 0
        episodes = 20
        for episode_seed in range(episodes):
            obs, _ = env.reset(seed=episode_seed)
            env.state = np.array(
                [
                    env.np_random.uniform(-0.02, 0.02),
                    env.np_random.uniform(-0.02, 0.02),
                    env.np_random.uniform(-0.04, 0.04),
                    env.np_random.uniform(-0.04, 0.04),
                    env.np_random.uniform(-0.2, 0.2),
                    env.np_random.uniform(-0.2, 0.2),
                    env.np_random.uniform(-0.2, 0.2),
                    env.np_random.uniform(-0.2, 0.2),
                ],
                dtype=np.float64,
            )
            obs = env.normalize_obs(env.state)
            done = False
            while not done:
                action = policy.predict(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            if env.steps >= env.max_steps and not env.failed:
                successes += 1
        self.assertGreaterEqual(successes / episodes, 0.95)

    def test_residual_lqr_mode_with_zero_residual_matches_teacher_action(self) -> None:
        base_env = BalanceStandEnv(seed=11)
        residual_env = make_env(seed=11, action_mode="residual_lqr", residual_scale=0.15)
        teacher = LqrPolicy()

        base_obs, _ = base_env.reset(seed=11)
        residual_obs, _ = residual_env.reset(seed=11)
        np.testing.assert_allclose(base_obs, residual_obs)

        teacher_action = teacher.predict(base_obs)
        base_step = base_env.step(teacher_action)
        residual_step = residual_env.step(np.zeros(2, dtype=np.float32))

        np.testing.assert_allclose(base_step[0], residual_step[0], atol=1e-8)
        self.assertAlmostEqual(base_step[1], residual_step[1], places=8)
        self.assertEqual(base_step[2], residual_step[2])
        self.assertEqual(base_step[3], residual_step[3])
        np.testing.assert_allclose(base_step[4]["raw_obs"], residual_step[4]["raw_obs"], atol=1e-8)
        np.testing.assert_allclose(base_step[4]["physical_action"], residual_step[4]["physical_action"], atol=1e-8)
        np.testing.assert_allclose(residual_step[4]["teacher_action"], teacher_action, atol=1e-8)
        np.testing.assert_allclose(residual_step[4]["residual_action"], np.zeros(2, dtype=np.float32), atol=1e-8)

    def test_residual_lqr_mode_uses_wrapped_env_model_profile(self) -> None:
        base_env = BalanceStandEnv(seed=12, model_profile="measured_estimate")
        residual_env = make_env(
            seed=12,
            action_mode="residual_lqr",
            residual_scale=0.15,
            model_profile="measured_estimate",
        )
        teacher = LqrPolicy(model_profile="measured_estimate")

        base_obs, _ = base_env.reset(seed=12)
        residual_obs, _ = residual_env.reset(seed=12)
        np.testing.assert_allclose(base_obs, residual_obs)

        teacher_action = teacher.predict(base_obs)
        base_step = base_env.step(teacher_action)
        residual_step = residual_env.step(np.zeros(2, dtype=np.float32))

        np.testing.assert_allclose(base_step[0], residual_step[0], atol=1e-8)
        np.testing.assert_allclose(base_step[4]["physical_action"], residual_step[4]["physical_action"], atol=1e-8)
        np.testing.assert_allclose(residual_step[4]["teacher_action"], teacher_action, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
