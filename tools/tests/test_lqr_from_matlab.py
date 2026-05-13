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


if __name__ == "__main__":
    unittest.main()
