import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from rl_balance.live_control import ActionPacketCodec, StateFrameCodec


class LiveProtocolTests(unittest.TestCase):
    def test_action_packet_round_trip(self) -> None:
        packet = ActionPacketCodec.encode(12.5, -7.25)
        u_l, u_r = ActionPacketCodec.decode(packet)
        self.assertAlmostEqual(u_l, 12.5, places=5)
        self.assertAlmostEqual(u_r, -7.25, places=5)

    def test_state_frame_parses_expected_fields(self) -> None:
        frame = "{0.1:0.2:0.3:0.4:0.5:0.6:0.7:0.8:0:0:0:0:0}"
        state = StateFrameCodec.parse_frame(frame)
        self.assertEqual(state.shape, (13,))
        np.testing.assert_allclose(
            state,
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
