import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = TOOLS_DIR / "train.py"
EVAL_SCRIPT = TOOLS_DIR / "evaluate.py"
PLOT_SCRIPT = TOOLS_DIR / "plot_results.py"
VIDEO_SCRIPT = TOOLS_DIR / "render_rollout_video.py"
MUJOCO_VIDEO_SCRIPT = TOOLS_DIR / "render_mujoco_rollout.py"
TEAM_PIPELINE_SCRIPT = TOOLS_DIR / "run_team_pipeline_comparison.py"


class ExperimentCliTests(unittest.TestCase):
    def test_train_cli_creates_checkpoint_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "runs"
            cmd = [
                "python3",
                str(TRAIN_SCRIPT),
                "--algo",
                "sac",
                "--timesteps",
                "128",
                "--seeds",
                "0",
                "--eval-freq",
                "64",
                "--eval-episodes",
                "2",
                "--output-dir",
                str(out_dir),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((out_dir / "sac" / "seed_0" / "best_model.zip").exists())
            self.assertTrue((out_dir / "sac" / "seed_0" / "metrics.csv").exists())

    def test_train_cli_accepts_and_records_train_reset_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "runs"
            cmd = [
                "python3",
                str(TRAIN_SCRIPT),
                "--algo",
                "sac",
                "--timesteps",
                "128",
                "--seeds",
                "0",
                "--eval-freq",
                "64",
                "--eval-episodes",
                "2",
                "--train-reset-profile",
                "narrow",
                "--output-dir",
                str(out_dir),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            metadata_path = out_dir / "sac" / "seed_0" / "run_metadata.json"
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["train_reset_profile"], "narrow")

    def test_train_cli_accepts_and_records_train_reward_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "runs"
            cmd = [
                "python3",
                str(TRAIN_SCRIPT),
                "--algo",
                "sac",
                "--timesteps",
                "128",
                "--seeds",
                "0",
                "--eval-freq",
                "64",
                "--eval-episodes",
                "2",
                "--train-reward-profile",
                "posture_focus",
                "--output-dir",
                str(out_dir),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            metadata_path = out_dir / "sac" / "seed_0" / "run_metadata.json"
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["train_reward_profile"], "posture_focus")

    def test_train_cli_accepts_and_records_sac_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "runs"
            cmd = [
                "python3",
                str(TRAIN_SCRIPT),
                "--algo",
                "sac",
                "--timesteps",
                "128",
                "--seeds",
                "0",
                "--eval-freq",
                "64",
                "--eval-episodes",
                "2",
                "--sac-profile",
                "batch128",
                "--output-dir",
                str(out_dir),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            metadata_path = out_dir / "sac" / "seed_0" / "run_metadata.json"
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["sac_profile"], "batch128")

    def test_train_cli_accepts_and_records_residual_action_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "runs"
            cmd = [
                "python3",
                str(TRAIN_SCRIPT),
                "--algo",
                "sac",
                "--timesteps",
                "128",
                "--seeds",
                "0",
                "--eval-freq",
                "64",
                "--eval-episodes",
                "2",
                "--action-mode",
                "residual_lqr",
                "--residual-scale",
                "0.15",
                "--output-dir",
                str(out_dir),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            metadata_path = out_dir / "sac" / "seed_0" / "run_metadata.json"
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["action_mode"], "residual_lqr")
            self.assertAlmostEqual(float(metadata["residual_scale"]), 0.15)

    def test_train_cli_accepts_and_records_model_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "runs"
            cmd = [
                "python3",
                str(TRAIN_SCRIPT),
                "--algo",
                "sac",
                "--timesteps",
                "128",
                "--seeds",
                "0",
                "--eval-freq",
                "64",
                "--eval-episodes",
                "2",
                "--model-profile",
                "measured_estimate",
                "--output-dir",
                str(out_dir),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            metadata_path = out_dir / "sac" / "seed_0" / "run_metadata.json"
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["model_profile"], "measured_estimate")

    def test_train_cli_runs_dqn_with_discrete_action_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "runs"
            cmd = [
                "python3",
                str(TRAIN_SCRIPT),
                "--algo",
                "dqn",
                "--timesteps",
                "128",
                "--seeds",
                "0",
                "--eval-freq",
                "64",
                "--eval-episodes",
                "2",
                "--model-profile",
                "measured_estimate",
                "--output-dir",
                str(out_dir),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            metadata_path = out_dir / "dqn" / "seed_0" / "run_metadata.json"
            self.assertTrue((out_dir / "dqn" / "seed_0" / "best_model.zip").exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["algo"], "dqn")
            self.assertEqual(metadata["action_mode"], "discrete_direct")
            self.assertEqual(metadata["model_profile"], "measured_estimate")

    def test_evaluate_and_plot_scripts_generate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "runs"
            summary_dir = Path(temp_dir) / "summary"
            plot_dir = Path(temp_dir) / "plots"
            train_cmd = [
                "python3",
                str(TRAIN_SCRIPT),
                "--algo",
                "ppo",
                "--timesteps",
                "256",
                "--seeds",
                "0",
                "--eval-freq",
                "128",
                "--eval-episodes",
                "2",
                "--output-dir",
                str(out_dir),
            ]
            train_result = subprocess.run(train_cmd, check=False, capture_output=True, text=True)
            self.assertEqual(train_result.returncode, 0, msg=train_result.stderr)

            eval_cmd = [
                "python3",
                str(EVAL_SCRIPT),
                "--input-dir",
                str(out_dir),
                "--output-dir",
                str(summary_dir),
                "--episodes",
                "4",
            ]
            eval_result = subprocess.run(eval_cmd, check=False, capture_output=True, text=True)
            self.assertEqual(eval_result.returncode, 0, msg=eval_result.stderr)
            summary_csv = summary_dir / "summary.csv"
            self.assertTrue(summary_csv.exists())
            with summary_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 2)

            plot_cmd = [
                "python3",
                str(PLOT_SCRIPT),
                "--input-dir",
                str(summary_dir),
                "--output-dir",
                str(plot_dir),
            ]
            plot_result = subprocess.run(plot_cmd, check=False, capture_output=True, text=True)
            self.assertEqual(plot_result.returncode, 0, msg=plot_result.stderr)
            for file_name in [
                "training_return_curve.png",
                "success_rate_bar.png",
                "control_energy_bar.png",
                "rollout_timeseries.png",
            ]:
                self.assertTrue((plot_dir / file_name).exists(), msg=file_name)

    def test_render_rollout_video_cli_generates_gif(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rollout.gif"
            cmd = [
                "python3",
                str(VIDEO_SCRIPT),
                "--policy",
                "lqr",
                "--view",
                "3d",
                "--model-profile",
                "measured_estimate",
                "--steps",
                "3",
                "--fps",
                "2",
                "--output",
                str(output_path),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_render_mujoco_rollout_cli_generates_gif(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mujoco_rollout.gif"
            cmd = [
                "python3",
                str(MUJOCO_VIDEO_SCRIPT),
                "--policy",
                "lqr",
                "--model-profile",
                "measured_estimate",
                "--steps",
                "3",
                "--fps",
                "2",
                "--width",
                "320",
                "--height",
                "180",
                "--output",
                str(output_path),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_mujoco_model_loads(self) -> None:
        cmd = [
            "python3",
            "-c",
            (
                "import mujoco; "
                f"mujoco.MjModel.from_xml_path({str(TOOLS_DIR / 'models' / 'wheeltec_balance_vehicle.xml')!r})"
            ),
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_team_pipeline_comparison_reuses_checkpoints_and_generates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "team"
            for algo in ["sac", "td3", "dqn"]:
                train_cmd = [
                    "python3",
                    str(TRAIN_SCRIPT),
                    "--algo",
                    algo,
                    "--timesteps",
                    "128",
                    "--seeds",
                    "0",
                    "--eval-freq",
                    "64",
                    "--eval-episodes",
                    "2",
                    "--model-profile",
                    "measured_estimate",
                    "--output-dir",
                    str(out_dir),
                ]
                train_result = subprocess.run(train_cmd, check=False, capture_output=True, text=True)
                self.assertEqual(train_result.returncode, 0, msg=train_result.stderr)

            cmd = [
                "python3",
                str(TEAM_PIPELINE_SCRIPT),
                "--skip-train",
                "--output-dir",
                str(out_dir),
                "--model-profile",
                "measured_estimate",
                "--eval-episodes",
                "2",
                "--render-steps",
                "3",
                "--fps",
                "2",
                "--render-backend",
                "mujoco",
                "--render-width",
                "320",
                "--render-height",
                "180",
                "--seeds",
                "0",
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((out_dir / "summary" / "algorithm_summary.csv").exists())
            self.assertTrue((out_dir / "plots" / "success_rate_bar.png").exists())
            for file_name in ["lqr_3d_mujoco.gif", "sac_3d_mujoco.gif", "td3_3d_mujoco.gif", "dqn_3d_mujoco.gif"]:
                path = out_dir / "videos" / file_name
                self.assertTrue(path.exists(), msg=file_name)
                self.assertGreater(path.stat().st_size, 0, msg=file_name)


if __name__ == "__main__":
    unittest.main()
