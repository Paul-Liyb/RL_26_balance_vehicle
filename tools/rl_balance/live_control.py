"""Live WiFi control utilities and CLI."""

from __future__ import annotations

import argparse
import socket
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DEFAULT_MODEL_PROFILE, MODEL_PROFILES
from .policies import LqrPolicy, SB3Policy


class ActionPacketCodec:
    PACKET_HEADER = 0xAA
    PACKET_SIZE = 10

    @staticmethod
    def encode(u_l: float, u_r: float) -> bytes:
        packet = struct.pack("<Bff", ActionPacketCodec.PACKET_HEADER, float(u_l), float(u_r))
        checksum = sum(packet) & 0xFF
        return packet + bytes([checksum])

    @staticmethod
    def decode(packet: bytes) -> tuple[float, float]:
        if len(packet) != ActionPacketCodec.PACKET_SIZE:
            raise ValueError(f"Expected {ActionPacketCodec.PACKET_SIZE} bytes, got {len(packet)}")
        body = packet[:-1]
        checksum = sum(body) & 0xFF
        if checksum != packet[-1]:
            raise ValueError("Checksum mismatch")
        header, u_l, u_r = struct.unpack("<Bff", body)
        if header != ActionPacketCodec.PACKET_HEADER:
            raise ValueError("Invalid packet header")
        return float(u_l), float(u_r)


class StateFrameCodec:
    EXPECTED_VALUES = 13

    @staticmethod
    def parse_frame(frame: str) -> np.ndarray:
        frame = frame.strip()
        if frame.startswith("{") and frame.endswith("}"):
            frame = frame[1:-1]
        values = np.array([float(token) for token in frame.split(":")], dtype=np.float32)
        if values.shape != (StateFrameCodec.EXPECTED_VALUES,):
            raise ValueError(f"Expected {StateFrameCodec.EXPECTED_VALUES} floats, got {values.shape[0]}")
        return values

    @staticmethod
    def extract_state_for_policy(frame: str) -> np.ndarray:
        values = StateFrameCodec.parse_frame(frame)
        raw_obs = np.array(
            [values[4], values[5], values[0], values[2], values[6], values[7], values[1], values[3]],
            dtype=np.float32,
        )
        return raw_obs


@dataclass
class SocketController:
    host: str
    port: int
    policy_type: str
    model_path: Path | None = None
    algorithm: str | None = None
    model_profile: str = DEFAULT_MODEL_PROFILE

    def __post_init__(self) -> None:
        if self.policy_type == "lqr":
            self.policy = LqrPolicy(model_profile=self.model_profile)
        else:
            if self.model_path is None or self.algorithm is None:
                raise ValueError("Model path and algorithm are required for RL live control")
            self.policy = SB3Policy.load(self.algorithm, self.model_path)

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        buffer = ""
        try:
            while True:
                data = sock.recv(128).decode()
                if not data:
                    break
                buffer += data
                start = buffer.find("{")
                end = buffer.find("}", start)
                if start == -1 or end == -1:
                    continue
                frame = buffer[start : end + 1]
                buffer = buffer[end + 1 :]
                raw_obs = StateFrameCodec.extract_state_for_policy(frame)
                normalized_obs = raw_obs / np.array([0.5, 0.5, 0.2, 0.25, 12.0, 12.0, 12.0, 12.0], dtype=np.float32)
                action = np.asarray(self.policy.predict(normalized_obs), dtype=np.float32)
                u = 6000.0 * np.clip(action, -1.0, 1.0)
                sock.send(ActionPacketCodec.encode(float(u[0]), float(u[1])))
        finally:
            sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Live TCP controller for the balance car.")
    parser.add_argument("--host", default="192.168.4.1")
    parser.add_argument("--port", type=int, default=6390)
    parser.add_argument("--policy", choices=["lqr", "rl"], default="lqr")
    parser.add_argument("--algo", choices=["sac", "td3", "ppo"], help="Algorithm used by the RL checkpoint.")
    parser.add_argument("--model-path", type=Path, help="Path to the saved RL checkpoint.")
    parser.add_argument("--model-profile", choices=MODEL_PROFILES, default=DEFAULT_MODEL_PROFILE)
    args = parser.parse_args()

    controller = SocketController(
        host=args.host,
        port=args.port,
        policy_type=args.policy,
        model_path=args.model_path,
        algorithm=args.algo,
        model_profile=args.model_profile,
    )
    controller.run()
    return 0
