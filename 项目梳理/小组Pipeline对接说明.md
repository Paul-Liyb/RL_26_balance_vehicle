# 小组 Pipeline 对接说明

## 目标

把当前仓库里的 `SAC`、`TD3`、`DQN` 方法放进同一套小组汇报产物流程里：

- 同一环境接口：8 维状态、2 维左右轮动作
- 同一评估指标：成功率、平均回合长度、平均回报、姿态 RMS、控制能量
- 同一可视化风格：PPT 风格 3D 双轮平衡车 GIF
- 同一输出目录：summary、plots、videos

## 和 PPT 的关系

PPT 里提到 MuJoCo/IsaacLab，但当前共享的 `RLPycode.zip` 里没有 MuJoCo 或 IsaacLab 可运行工程，实际包含的是简化 Gym 环境、行为克隆、PPO 和 STM32 权重导出流程。

因此当前对接方式是：

- 保留我们已经验证过的 Gymnasium 训练与评估环境。
- 默认用 MuJoCo 模型渲染 rollout GIF，画面尽量对齐 PPT 里的深色地面、双轮车体和上杆展示方式。
- 当前 MuJoCo 层是可视化 backend：策略和状态转移仍来自已验证的 Gymnasium pipeline，再把状态写入 MuJoCo 模型进行渲染。

如果后续组员提供可校准的 MuJoCo 动力学 XML / IsaacLab 环境，可以把同样的算法 checkpoint 接到那个环境里重新评估。

## 一键运行

在仓库根目录执行：

```bash
./start_rl_env.sh python3 run_team_pipeline_comparison.py \
  --model-profile measured_estimate \
  --device cpu \
  --output-dir artifacts/team_pipeline \
  --timesteps 10000 \
  --eval-freq 5000 \
  --eval-episodes 5 \
  --seeds 0
```

输出位置：

```text
tools/artifacts/team_pipeline/
├── dqn/seed_0/
├── sac/seed_0/
├── td3/seed_0/
├── summary/
│   ├── summary.csv
│   ├── algorithm_summary.csv
│   ├── training_curves.csv
│   └── rollout_trace.csv
├── plots/
│   ├── training_return_curve.png
│   ├── success_rate_bar.png
│   ├── control_energy_bar.png
│   └── rollout_timeseries.png
└── videos/
    ├── lqr_3d.gif
    ├── sac_3d.gif
    ├── td3_3d.gif
    └── dqn_3d.gif
```

默认 MuJoCo 文件名会带 backend 后缀：

```text
videos/lqr_3d_mujoco.gif
videos/sac_3d_mujoco.gif
videos/td3_3d_mujoco.gif
videos/dqn_3d_mujoco.gif
```

## 快速 smoke 示例

如果只是确认流程能跑：

```bash
./start_rl_env.sh python3 run_team_pipeline_comparison.py \
  --model-profile measured_estimate \
  --device cpu \
  --output-dir artifacts/team_pipeline_smoke \
  --timesteps 512 \
  --eval-freq 256 \
  --eval-episodes 3 \
  --seeds 0 \
  --render-steps 80 \
  --fps 10
```

这个配置训练很短，只用于检查 pipeline，不代表算法最终性能。

## 当前接口口径

真实数据接口按 CSV 使用：

```text
state = [
  theta_L,
  theta_R,
  theta_1,
  theta_2,
  theta_L_dot,
  theta_R_dot,
  theta_dot_1,
  theta_dot_2
]

action = [u_L, u_R]
```

旧文档里的 `4 state + action_u` 不建议继续作为新代码接口。

## 下一步可扩展

1. 加入行为克隆 `BC` baseline，使用真实 CSV 训练 `8 -> 16 -> 16 -> 2` 小网络。
2. 把 `BC` 和 `PPO+BC` 也接入 `run_team_pipeline_comparison.py`。
3. 导出小网络权重到 C 数组，对齐 PPT 的 STM32 部署叙事。
4. 继续校准 MuJoCo 动力学，让 MuJoCo 不只是渲染 backend，而是训练和评估 backend。
