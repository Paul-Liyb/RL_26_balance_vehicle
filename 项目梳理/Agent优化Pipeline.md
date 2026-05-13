# Codex Auto-Research 优化 Pipeline

## 1. 目标

这套 pipeline 的目标不是一次性“调几个参数”，而是让另一个可执行命令、可改代码的 Codex 在当前工程上持续做小步、可验证、可回滚的算法优化，直到 RL 在现有站立任务上明显优于当前基线，或者形成清晰的失败诊断。

本轮只做“平衡站立”任务，不扩展移动任务，不动 STM32 固件协议，不做 MCU 侧部署。

## 2. 当前已知基线

- 工作目录：`D:\code_wf`
- 项目根目录：`D:\code_wf\Y1S1\强化学习`
- 实验代码主目录：`D:\code_wf\Y1S1\强化学习\tools`
- 当前最佳完整实验输出目录：`D:\code_wf\Y1S1\强化学习\tools\artifacts_full_cuda`
- 当前控制链路说明：`D:\code_wf\Y1S1\强化学习\项目梳理\控制链路说明.md`
- MATLAB 到 Python 的 LQR 复现脚本：`D:\code_wf\Y1S1\强化学习\tools\lqr_from_matlab.py`

当前已验证结果：

- WSL 环境：`Ubuntu-24.04`
- Python 训练环境：`python3`
- PyTorch：`2.11.0+cu130`
- CUDA 可用：`True`
- 显卡：`NVIDIA GeForce RTX 5080`
- 单测：`python3 -m unittest discover -v tests`，`11/11` 通过

当前完整实验聚合结果：

- `lqr`: `success_rate = 0.69`
- `sac`: `success_rate = 0.47`
- `ppo`: `success_rate = 0.00`
- `td3`: `success_rate = 0.00`

初步结论：

- 现有实验骨架已经跑通。
- `SAC` 是唯一值得优先继续优化的 RL 算法。
- `PPO` 和 `TD3` 当前不值得优先投入更多算力。
- 当前更像是“实验设定还不够友好”，而不是“训练管线坏了”。

## 3. 成功标准

优化过程采用分阶段目标，不以单次好运行为准。

### 阶段 A：追平并超过当前 LQR

在相同评估协议下，至少一个 RL 方法满足：

- `success_rate > LQR success_rate`
- 且不是靠异常大动作硬顶出来的
- 且 rollout 轨迹没有明显发散

建议同时要求：

- `mean_control_energy <= 1.5 * LQR mean_control_energy`
- `rms(theta_1)` 和 `rms(theta_2)` 不显著恶化

### 阶段 B：形成可信提升

在 5 个 seed、100 个最终评估 episode 下，最佳 RL 方法稳定优于 LQR，建议达到：

- `success_rate >= max(LQR success_rate + 0.05, 0.75)`

如果环境定义被改动到会影响评估难度，例如重置分布、终止条件、时长、动力学相关设定，则必须重新计算新的 LQR 基线，再做比较。

## 4. 不可破坏的约束

另一个 Codex 在优化过程中必须遵守这些约束：

- 主要工作环境使用 WSL，不切回 Windows Python。
- 优先使用 CUDA 训练。
- 保持观测顺序不变：
  - `theta_L, theta_R, theta_1, theta_2, theta_L_dot, theta_R_dot, theta_dot_1, theta_dot_2`
- 保持动作语义不变：
  - 连续 2 维动作，经 `u = 6000 * action` 映射到 `u_L/u_R`
- 第一阶段不要修改：
  - `wifi3.0.py`
  - 固件通信协议
  - `live_control.py` 的协议层
  - 板端控制逻辑
- 不要一开始就重新引入新的 RL 算法库。
- 不要同时改多个主要变量，否则无法判断提升来自哪里。

## 5. 优先阅读顺序

另一个 Codex 接手时，建议按下面顺序读文件：

1. `D:\code_wf\Y1S1\强化学习\项目梳理\当前实验交接.md`
2. `D:\code_wf\Y1S1\强化学习\项目梳理\控制链路说明.md`
3. `D:\code_wf\Y1S1\强化学习\tools\README_rl_pipeline.md`
4. `D:\code_wf\Y1S1\强化学习\tools\lqr_from_matlab.py`
5. `D:\code_wf\Y1S1\强化学习\tools\rl_balance\env.py`
6. `D:\code_wf\Y1S1\强化学习\tools\rl_balance\config.py`
7. `D:\code_wf\Y1S1\强化学习\tools\rl_balance\experiments.py`
8. `D:\code_wf\Y1S1\强化学习\tools\artifacts_full_cuda\summary\algorithm_summary.csv`
9. `D:\code_wf\Y1S1\强化学习\tools\artifacts_full_cuda\summary\summary.csv`

## 6. Auto-Research 循环

每一轮优化必须遵循同一个闭环。

### Step 0: 读取当前上下文

- 先读本目录下的：
  - `当前实验交接.md`
  - `自动优化研究日志模板.md`
- 然后读取当前最优实验输出目录里的 summary 和 plot。

### Step 1: 复现当前可运行基线

在任何新改动之前，至少执行：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 -m unittest discover -v tests"
```

如果需要重新核对现有结果，再执行：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 evaluate.py --input-dir artifacts_full_cuda --output-dir artifacts_full_cuda/summary --episodes 100"
```

### Step 2: 只提出一个主假设

每轮只允许有一个主优化假设，格式固定为：

- 假设：
- 为什么它可能有效：
- 预计改哪些文件：
- 用什么指标判断它是否有效：

不允许一轮同时做“奖励 + 重置分布 + 超参数 + 网络结构”四件事。

### Step 3: 先做低成本筛选

优先只跑 `SAC`，先做低成本 screening run。

推荐命令：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 train.py --algo sac --device cuda --output-dir artifacts_research/iter_<tag>/screen --seeds 0 1 --timesteps 120000 --eval-freq 5000 --eval-episodes 20"
```

然后评估：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 evaluate.py --input-dir artifacts_research/iter_<tag>/screen --output-dir artifacts_research/iter_<tag>/screen/summary --episodes 100"
```

screening run 通过的建议条件：

- `success_rate` 比当前 best SAC 有提升
- 或 `mean_return` 有明确提升且控制能量未明显爆炸

### Step 4: 通过后再做完整确认

只有 screening run 看起来有价值，才升格到完整确认实验：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 train.py --algo sac --device cuda --output-dir artifacts_research/iter_<tag>/full --seeds 0 1 2 3 4 --timesteps 300000 --eval-freq 10000 --eval-episodes 20"
```

然后执行：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 evaluate.py --input-dir artifacts_research/iter_<tag>/full --output-dir artifacts_research/iter_<tag>/full/summary --episodes 100"
```

以及：

```bash
wsl -d Ubuntu-24.04 bash -lc "cd '/mnt/d/code_wf/Y1S1/强化学习/tools' && python3 plot_results.py --input-dir artifacts_research/iter_<tag>/full/summary --output-dir artifacts_research/iter_<tag>/full/plots"
```

### Step 5: 明确是否“晋级为当前最优”

新的实验结果只有在下列条件同时满足时，才能视为新的 best：

- 比当前 best RL 有稳定提升
- 对比当前 LQR 更接近或已经超过
- 没有明显的控制能量爆炸
- 不是单一 seed 偶然成功

如果修改了评估相关设定，必须顺带重跑 LQR 并更新对比。

### Step 6: 记录并继续

每一轮结束后，必须产出：

- 改动摘要
- 实际运行命令
- 结果摘要
- 与上一 best 的对比
- 是否晋级
- 下一轮候选假设

并写入新的日志条目。

## 7. 优先优化方向

按下面顺序尝试，不要乱跳：

1. `env.py` 中的重置分布
2. `env.py` 中的奖励权重和失败惩罚
3. `env.py` 中的终止阈值和 episode 难度
4. `config.py` 中的 `SAC` 超参数
5. `experiments.py` 中的评估协议和训练调度支持
6. 必要时再考虑课程式训练或多阶段训练

明确不推荐一开始优先做：

- 继续投入 `PPO`
- 继续投入 `TD3`
- 改通信协议
- 改固件
- 改动作定义
- 改状态顺序

## 8. 允许的“第一批有效改动”

下面这些改动方向是优先鼓励的：

- 把 reset 分布参数化，支持更窄扰动开始训练
- 把 reward 参数外提，便于快速扫参数
- 给 `train.py` 增加更适合研究迭代的参数入口
- 增加“screening run” 与 “full run” 的目录规范
- 给 summary 增加对 best RL 与 LQR 的自动对比

## 9. 结果记录规范

建议每一轮使用一个独立目录：

- `D:\code_wf\Y1S1\强化学习\tools\artifacts_research\iter_YYYYMMDD_HHMM_<slug>`

每轮至少保留：

- `screen/`
- `screen/summary/`
- `full/`
- `full/summary/`
- `full/plots/`

同时在文档目录持续维护：

- `D:\code_wf\Y1S1\强化学习\项目梳理\当前实验交接.md`

## 10. 停止条件

出现以下情况之一时，应停止盲目迭代并输出诊断总结：

- 连续 3 轮 screening 都没有优于当前 best SAC
- 连续 6 轮仍无法超过当前 LQR
- 发现问题根源不是调参，而是环境建模本身与真实目标错位

诊断总结至少要回答：

- 当前瓶颈更像是 reset、reward、termination、train budget 还是模型表达能力
- 为什么 SAC 还没追上 LQR
- 下一轮最可能有效的 2 个方向是什么

## 11. 最终交付要求

另一个 Codex 每次完成一轮后，必须给出一段简明汇报：

- 看了哪些关键文件
- 本轮假设是什么
- 改了哪些文件
- 跑了哪些命令
- 哪些结果有提升，哪些没有
- 当前 best 是什么
- 下一轮应该做什么
