# 平衡车强化学习仿真项目

本仓库用于小组协作开发平衡车站立控制的强化学习仿真实验。代码封装了 Gymnasium 环境，并提供 `LQR`、`SAC`、`TD3`、`PPO`、`DQN` 的训练、评估和画图流程。

## 目录说明

- `tools/rl_balance/`：环境、策略、训练与评估核心代码
- `tools/train.py`：训练入口
- `tools/evaluate.py`：评估入口
- `tools/plot_results.py`：结果画图入口
- `tools/render_rollout_video.py`：把仿真 rollout 渲染成 GIF/MP4
- `tools/render_mujoco_rollout.py`：用 MuJoCo 模型渲染 rollout 视频
- `tools/lqr_from_matlab.py`：复现 MATLAB/LQR 模型和增益
- `tools/tests/`：测试代码
- `matlab_reference/`：关键 MATLAB 参考脚本
- `项目梳理/`：交接文档、实验日志、控制链路说明

## 环境安装

建议使用 Python 3.10+。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果需要 GPU 训练，请按本机 CUDA/驱动版本安装合适的 PyTorch。

## 快速验证

```bash
python -m unittest discover -v tools/tests
python tools/lqr_from_matlab.py
python tools/lqr_from_matlab.py --model-profile measured_estimate
```

## 常用命令

小规模训练示例：

```bash
python tools/train.py --algo sac --device cpu --output-dir tools/artifacts/smoke --seeds 0 --timesteps 10000 --eval-freq 5000 --eval-episodes 5
python tools/train.py --algo sac --model-profile measured_estimate --device cpu --output-dir tools/artifacts/measured_smoke --seeds 0 --timesteps 10000 --eval-freq 5000 --eval-episodes 5
python tools/train.py --algo dqn --model-profile measured_estimate --device cpu --output-dir tools/artifacts/dqn_smoke --seeds 0 --timesteps 10000 --eval-freq 5000 --eval-episodes 5
```

评估：

```bash
python tools/evaluate.py --input-dir tools/artifacts/smoke --output-dir tools/artifacts/smoke/summary --episodes 20
```

画图：

```bash
python tools/plot_results.py --input-dir tools/artifacts/smoke/summary --output-dir tools/artifacts/smoke/plots
```

实车日志拟合：

```bash
python tools/fit_real_log_model.py
```

生成仿真动画：

```bash
cd tools
python render_mujoco_rollout.py --policy lqr --model-profile measured_estimate --steps 300 --fps 20 --output artifacts/videos/lqr_mujoco.gif
python render_rollout_video.py --policy lqr --view 3d --model-profile measured_estimate --steps 160 --fps 20 --output artifacts/videos/lqr_measured_3d.gif
python render_rollout_video.py --policy lqr --view 2d --model-profile measured_estimate --steps 160 --fps 20 --output artifacts/videos/lqr_measured_2d.gif
```

如果要渲染训练好的模型：

```bash
cd tools
python render_rollout_video.py --policy rl --algo sac --view 3d --model-path artifacts/measured_smoke/sac/seed_0/best_model.zip --model-profile measured_estimate --output artifacts/videos/sac_measured_3d.gif
python render_rollout_video.py --policy rl --algo dqn --view 3d --model-path artifacts/dqn_smoke/dqn/seed_0/best_model.zip --model-profile measured_estimate --output artifacts/videos/dqn_measured_3d.gif
```

使用实车日志拟合环境训练：

```bash
python tools/train.py --algo sac --model-profile real_log_fit --device cpu --output-dir tools/artifacts/real_log_fit_smoke --seeds 0 --timesteps 10000 --eval-freq 5000 --eval-episodes 5
```

完整流程：

```bash
python tools/run_full_experiment.py
```

小组汇报对比流程（SAC/TD3/DQN + 指标图 + PPT风格3D GIF）：

```bash
python tools/run_team_pipeline_comparison.py --model-profile measured_estimate --device cpu --output-dir tools/artifacts/team_pipeline --timesteps 10000 --eval-freq 5000 --eval-episodes 5 --seeds 0 --render-backend mujoco
```

## 当前状态

当前主要任务是站立平衡仿真。已有结论见：

- `项目梳理/当前实验交接.md`
- `项目梳理/自动优化研究日志.md`
- `项目梳理/控制链路说明.md`

简要结论：

- 默认 `vendor_matlab` 保留厂商 MATLAB 参数。
- `measured_estimate` 使用当前实车估计参数：整车约 `1.1kg`，上杆 `0.1kg`，尺寸 `184*65*541.5mm(含摆杆)`，轮半径约 `0.0325m`，上杆 `0.39m`。
- `LQR` 是当前最强基线。
- `SAC` 是当前最值得继续优化的 RL 方法。
- `DQN` 已接入离散动作测试路径，默认使用 9 个左右轮动作组合。
- `PPO` 和 `TD3` 当前表现较弱。
- 后续上车前需要核对实车参数、状态顺序、动作含义和通信链路。

## 注意事项

仓库未包含虚拟环境、训练产物、模型 checkpoint、完整厂商资料包、视频、APK、PDF 等大文件。生成的新实验结果默认放在 `tools/artifacts*` 下，不建议提交到 Git。
