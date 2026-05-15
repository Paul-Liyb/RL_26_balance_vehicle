import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import lqr_from_matlab as lqr


class LqrFromMatlabTests(unittest.TestCase):
    def test_gain_matches_reference(self) -> None:
        result = lqr.solve_lqr_from_matlab()
        reference = lqr.firmware_gain_reference()
        max_abs_diff = np.max(np.abs(result.K - reference))
        self.assertLessEqual(max_abs_diff, 5e-4)

    def test_measured_profile_uses_estimated_robot_dimensions(self) -> None:
        params = lqr.get_physical_params("measured_estimate")
        self.assertAlmostEqual(params.m_1, 1.0)
        self.assertAlmostEqual(params.m_2, 0.1)
        self.assertAlmostEqual(params.r, 0.0325)
        self.assertAlmostEqual(params.L_1, 0.119)
        self.assertAlmostEqual(params.L_2, 0.390)
        self.assertAlmostEqual(params.l_1, 0.055)
        self.assertAlmostEqual(params.I_1, 0.001532, places=6)

    def test_measured_profile_gain_differs_from_vendor_profile(self) -> None:
        vendor = lqr.solve_lqr_from_matlab(model_profile="vendor_matlab")
        measured = lqr.solve_lqr_from_matlab(model_profile="measured_estimate")
        self.assertFalse(np.allclose(vendor.K, measured.K, atol=1e-3, rtol=0.0))

    def test_control_and_wifi_files_match(self) -> None:
        project_root = TOOLS_DIR.parents[0]
        control_path = lqr.default_control_path(project_root)
        wifi_path = lqr.default_wifi_path(project_root)
        if not control_path.exists() or not wifi_path.exists():
            self.skipTest("Full vendor firmware/WiFi reference files are not included in this checkout")
        control_gain = lqr.parse_gain_matrix_from_file(control_path)
        wifi_gain = lqr.parse_gain_matrix_from_file(wifi_path)
        self.assertTrue(np.allclose(control_gain, wifi_gain, atol=1e-9, rtol=0.0))

    def test_system_is_controllable(self) -> None:
        result = lqr.solve_lqr_from_matlab()
        tc = lqr.controllability_matrix(result.G, result.H)
        self.assertEqual(np.linalg.matrix_rank(tc), 8)

    def test_measured_profile_system_is_controllable(self) -> None:
        result = lqr.solve_lqr_from_matlab(model_profile="measured_estimate")
        tc = lqr.controllability_matrix(result.G, result.H)
        self.assertEqual(np.linalg.matrix_rank(tc), 8)


if __name__ == "__main__":
    unittest.main()
