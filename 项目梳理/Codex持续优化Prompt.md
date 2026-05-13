# 发给另一个 Codex 的 Prompt

你现在接手的是一个“强化学习平衡车”算法实验项目。你的目标不是一次性写一堆代码，而是按固定的 auto-research 流程，持续优化现有 RL 站立实验，直到 RL 在当前评估协议下稳定追平并超过 LQR，或者给出清晰可信的失败诊断。

## 你的工作环境和范围

- 工作目录：`D:\code_wf`
- 项目根目录：`D:\code_wf\Y1S1\强化学习`
- 主要实验目录：`D:\code_wf\Y1S1\强化学习\tools`
- 主运行环境：WSL `Ubuntu-24.04`
- 优先使用：`python3` + CUDA
- 当前已知 CUDA 环境正常，显卡是 `NVIDIA GeForce RTX 5080`

本轮只做“平衡站立”算法实验，不做：

- STM32 固件改造
- 板端 RL 推理部署
- WiFi 协议改造
- 移动任务扩展

## 先读这些文件

请按下面顺序先读文件，再开始动手：

1. `D:\code_wf\Y1S1\强化学习\项目梳理\当前实验交接.md`
2. `D:\code_wf\Y1S1\强化学习\项目梳理\Agent优化Pipeline.md`
3. `D:\code_wf\Y1S1\强化学习\项目梳理\控制链路说明.md`
4. `D:\code_wf\Y1S1\强化学习\tools\README_rl_pipeline.md`
5. `D:\code_wf\Y1S1\强化学习\tools\lqr_from_matlab.py`
6. `D:\code_wf\Y1S1\强化学习\tools\rl_balance\env.py`
7. `D:\code_wf\Y1S1\强化学习\tools\rl_balance\config.py`
8. `D:\code_wf\Y1S1\强化学习\tools\rl_balance\experiments.py`
9. `D:\code_wf\Y1S1\强化学习\tools\artifacts_full_cuda\summary\algorithm_summary.csv`
10. `D:\code_wf\Y1S1\强化学习\tools\artifacts_full_cuda\summary\summary.csv`

## 当前已知基线

当前完整实验的聚合结果大致为：

- `LQR`: `success_rate = 0.69`
- `SAC`: `success_rate = 0.47`
- `PPO`: `success_rate = 0.00`
- `TD3`: `success_rate = 0.00`

当前判断：

- 训练管线已经跑通
- 只有 `SAC` 值得优先继续优化
- 当前更可能是实验设定不够友好，而不是代码本身不可用

## 你的硬约束

你必须遵守：

- 使用 WSL 作为主要执行环境
- 优先用 CUDA
- 保持观测顺序不变：
  - `theta_L, theta_R, theta_1, theta_2, theta_L_dot, theta_R_dot, theta_dot_1, theta_dot_2`
- 保持动作语义不变：
  - 2 维连续动作，映射为 `u_L/u_R`
- 第一阶段不要优先修改：
  - `wifi3.0.py`
  - 固件协议
  - `live_control.py` 协议层
  - 板端控制逻辑
- 第一阶段不要继续浪费时间在 `PPO/TD3`
- 每一轮只允许一个主假设，不要同时乱改多个核心变量

## 你的目标

你的主目标是：

- 让至少一种 RL 方法在当前评估协议下稳定超过当前 LQR

判断“超过 LQR”时，不只看 `success_rate`，还要防止 reward hacking。请同时关注：

- `success_rate`
- `mean_return`
- `mean_episode_length`
- `rms_theta_1`
- `rms_theta_2`
- `mean_control_energy`

除非有充分理由，不接受这种“提升”：

- 成功率略升，但控制能量爆炸
- 某个单一 seed 偶然成功
- 改了评估环境却不重新计算 LQR

## 你必须遵循的研究循环

每轮必须执行以下步骤：

### 1. 基线核对

先跑基础验证：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 -m unittest discover -v tests"
```

如有必要，重新生成 summary：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 evaluate.py --input-dir artifacts_full_cuda --output-dir artifacts_full_cuda/summary --episodes 100"
```

### 2. 先写本轮假设

在改代码前，先明确写出：

- 本轮唯一主假设
- 为什么你认为它最值得试
- 你准备改哪些文件
- 什么结果算成功

### 3. 先做 screening run，再决定要不要 full run

优先只跑 `SAC` 的低成本筛选实验。建议类似：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 train.py --algo sac --device cuda --output-dir artifacts_research/iter_<tag>/screen --seeds 0 1 --timesteps 120000 --eval-freq 5000 --eval-episodes 20"
```

然后评估：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 evaluate.py --input-dir artifacts_research/iter_<tag>/screen --output-dir artifacts_research/iter_<tag>/screen/summary --episodes 100"
```

如果 screening 没提升，不要直接烧完整 5 seed 长训练。

### 4. screening 有价值，再升格 full run

只有当 screening 看起来比当前 best SAC 更好，才做完整确认：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 train.py --algo sac --device cuda --output-dir artifacts_research/iter_<tag>/full --seeds 0 1 2 3 4 --timesteps 300000 --eval-freq 10000 --eval-episodes 20"
```

然后：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 evaluate.py --input-dir artifacts_research/iter_<tag>/full --output-dir artifacts_research/iter_<tag>/full/summary --episodes 100"
```

以及：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 plot_results.py --input-dir artifacts_research/iter_<tag>/full/summary --output-dir artifacts_research/iter_<tag>/full/plots"
```

### 5. 每轮必须更新交接信息

每轮结束后，你必须更新以下两份文档：

- `D:\code_wf\Y1S1\强化学习\项目梳理\当前实验交接.md`
- 基于 `D:\code_wf\Y1S1\强化学习\项目梳理\自动优化研究日志模板.md` 追加本轮记录

### 6. 连续优化直到满足停止条件

你应持续自主推进，不要每做一小步就停下来问人。只有在以下情况才停下：

- 缺依赖且自己无法解决
- 训练脚本或环境出现明确 blocker
- 连续多轮无提升，需要输出诊断报告
- 已经稳定超过 LQR，需要进入结果固化和真车前准备

## 你的优先优化顺序

请按这个顺序尝试：

1. `env.py` 的 reset 分布
2. `env.py` 的 reward 权重和失败惩罚
3. `env.py` 的终止条件和 episode 难度
4. `config.py` 的 `SAC` 超参数
5. `experiments.py` 的研究便利性改进，例如更好的 summary 或实验参数入口

不建议你一开始优先做：

- PPO 调参
- TD3 调参
- 固件改造
- 通信改造
- 改状态定义
- 改动作语义

## 对结果的判断规则

只有在以下条件同时满足时，才可宣称出现“当前最好 RL 结果”：

- 对比当前 best SAC 有提升
- 对比 LQR 更接近或已经超过
- 控制能量没有显著恶化
- 结果不是某个 seed 的偶然值

如果你修改了 reset、termination、episode horizon 或任何会影响评估难度的设定，你必须重新跑 LQR，并基于新的 LQR 基线判断是否提升。

## 本次接手后的第一轮建议

请不要从 PPO/TD3 开始。第一轮优先做下面二选一，但一轮只能选一个主方向：

- 方向 A：把 reset 分布参数化，并先缩小训练初始扰动，观察 SAC 是否能更稳定学会站立
- 方向 B：保留 reset 分布，先重新设计 reward 权重和 action penalty，让策略更聚焦于 `theta_1/theta_2` 稳定而不是大动作补偿

推荐先做方向 A。

## 你每轮结束时必须汇报

请输出一个简明但具体的回合报告，至少包括：

- 本轮假设
- 改了哪些文件
- 跑了哪些命令
- screening 结果
- 是否做了 full run
- 与当前 best SAC 和 LQR 的对比
- 本轮是否成为新 best
- 下一轮最值得尝试什么

直接开始工作，不要先泛泛讲思路。先读文件，确认基线，然后进入第一轮优化。
