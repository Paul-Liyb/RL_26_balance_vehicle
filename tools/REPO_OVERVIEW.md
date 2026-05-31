# 仓库介绍

这个仓库是一个面向“双轮平衡/立稳控制”场景的强化学习实验工具集。它把一个从 MATLAB/LQR 推导来的离散线性模型封装成 Gymnasium 环境，再用 Stable-Baselines3 训练和比较 `SAC`、`TD3`、`PPO`、`DQN` 策略，并保留一个 `LQR` 基线做对照。除了离线训练，它还提供了一个简单的 TCP 实时控制接口，可以把 LQR 或训练好的 RL 模型接到平衡车侧的状态流上。

更准确地说，这不是一个通用 RL 框架，也不是完整的机器人软件栈。它更像“实验流水线 + 控制验证工具”：先复现控制模型，再训练策略，再做评估和画图，最后把策略接到实时链路里验证。

## 仓库主要在做什么

仓库围绕一条很明确的实验链路组织：

1. `lqr_from_matlab.py`
   从原始 MATLAB/LQR 推导复现连续系统、离散化系统和 LQR 增益，并校验与固件里的增益是否一致。
2. `rl_balance/env.py`
   把离散线性系统包装成一个 8 维观测、2 维动作的 Gymnasium 环境，定义 reset 范围、终止条件和奖励函数。
3. `train.py` 与 `train_{sac,td3,ppo,dqn}.py`
   调 Stable-Baselines3 训练 RL 策略，按算法和随机种子输出模型、评估日志和元数据。
4. `evaluate.py`
   统一评估 LQR 基线和已保存的 RL 模型，生成逐次运行和按算法聚合的汇总表。
5. `plot_results.py`
   读取汇总 CSV，画训练回报曲线、成功率柱状图、控制能量柱状图和单次 rollout 时序图。
6. `rl_balance/live_control.py` 与顶层 `live_control.py`
   通过 TCP 接收设备侧状态帧，解析成策略输入，再把双电机控制量编码后发回去。

## 你可以把它理解成什么

如果从用途看，这个仓库同时承担了 3 件事：

- 控制模型复现：确认 Python 侧离散模型和 LQR 增益与原始控制逻辑一致。
- 强化学习实验：在统一环境下比较不同算法相对 LQR 基线的表现。
- 实时部署桥接：把 LQR 或 RL 策略挂到一个简化的网络控制回路里。

## 目录和文件分工

### 核心包

- `rl_balance/config.py`
  集中管理观测缩放、奖励权重、动作缩放、reset 配置、默认训练步数和种子。
- `rl_balance/env.py`
  定义 `BalanceStandEnv`。环境状态是 8 维，动作是左右轮两个归一化控制输入。
- `rl_balance/policies.py`
  定义 `LqrPolicy` 和 `SB3Policy`，把 LQR 和 RL 模型统一成同一套 `predict()` 接口。
- `rl_balance/experiments.py`
  放训练、评估、汇总、回调存盘等实验主逻辑。
- `rl_balance/live_control.py`
  放实时 TCP 协议、状态帧解析、动作包编码和在线控制循环。

### 顶层脚本

- `train.py`
  通用训练入口，可通过 `--algo` 指定算法。
- `train_sac.py`、`train_td3.py`、`train_ppo.py`、`train_dqn.py`
  对 `train.py` 的薄封装，方便分别启动不同算法。
- `evaluate.py`
  评估现有 checkpoint，生成 `summary.csv`、`algorithm_summary.csv`、`training_curves.csv`、`rollout_trace.csv`。
- `plot_results.py`
  从 summary 目录生成 PNG 图表。
- `run_full_experiment.py`
  一键串起训练、评估、绘图全流程。
- `live_control.py`
  实时控制入口，底层调用包内实现。
- `README_rl_pipeline.md`
  偏运行说明，适合快速查命令。

### 产物与测试

- `artifacts_full_cuda/`
  已提交的一组完整实验结果，包含不同算法、不同随机种子的模型和汇总图表。
- `artifacts_research/`
  另一组研究型实验产物，看起来是更窄 reset 条件下的一部分结果。
- `tests/`
  测环境可复现性、LQR 基线稳定性、实时协议编解码、训练/评估/绘图 CLI 是否能产生产物。

## 运行流程

最常见的使用路径是：

1. 安装依赖
   `python3 -m pip install --user --break-system-packages -r tools/requirements-wsl.txt`
2. 训练单个算法
   `python3 tools/train.py --algo sac`
3. 评估已有模型
   `python3 tools/evaluate.py --input-dir tools/artifacts --output-dir tools/artifacts/summary`
4. 生成图表
   `python3 tools/plot_results.py --input-dir tools/artifacts/summary --output-dir tools/artifacts/plots`
5. 一键跑全流程
   `python3 tools/run_full_experiment.py`
6. 实时控制
   `python3 tools/live_control.py --policy lqr`
   或
   `python3 tools/live_control.py --policy rl --algo ppo --model-path .../best_model.zip`

## 这个环境怎么定义

从代码看，这个任务是一个离散线性控制问题：

- 观测 8 维，内部状态直接参与奖励和终止判断。
- 连续控制动作 2 维，对应左右轮控制输入，先限制在 `[-1, 1]`，再乘以 `6000` 转成物理量。DQN 使用 9 个离散动作编号，内部映射成左右轮归一化动作组合。
- 奖励由“状态偏离惩罚 + 动作惩罚 + 失败额外惩罚”组成。
- 终止条件主要包括车身角度过大、摆杆角度过大、轮速过大，或者达到时间上限。
- reset 支持 `default` 和 `narrow` 两种扰动范围，便于控制训练难度。

这意味着仓库当前更偏向“在线性化模型上的控制实验”，而不是带复杂接触、噪声、传感器延迟的高保真仿真。

## 当前仓库里已经能看到什么结果

从已提交的 `artifacts_full_cuda/summary/algorithm_summary.csv` 看：

- `LQR` 基线成功率约 `0.69`，平均回报约 `614`。
- `SAC` 是三种 RL 算法里表现最好的，成功率约 `0.47`，平均回报约 `461`。
- `PPO` 和 `TD3` 在这组结果里基本没有成功站稳，成功率为 `0.0`。

这组结果说明了两件事：第一，这个仓库已经不仅是“能跑”，而是带着一套完整实验产物；第二，在当前环境和参数下，LQR 仍然是一个很强的基线，RL 更多是在做对比和探索，不像已经替代了经典控制。

## 如果你第一次接手这个仓库

建议按这个顺序读：

1. 先看 `README_rl_pipeline.md`，知道怎么跑。
2. 再看 `rl_balance/env.py` 和 `rl_balance/config.py`，理解任务定义。
3. 接着看 `rl_balance/experiments.py`，理解训练和评估逻辑。
4. 最后看 `rl_balance/live_control.py`，理解它如何接外部设备。

如果你的目标是改进算法表现，重点应该放在环境建模、奖励设计、reset 分布和超参数上；如果你的目标是上车联调，重点应该放在 `live_control` 的状态映射、协议兼容和模型鲁棒性上。
