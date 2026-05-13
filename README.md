# 平衡车强化学习仿真项目

这个仓库是一个面向 WHEELTEC B585 DP2 二阶平衡车的 Python 仿真实验项目。当前内容主要用于算法验证和小组协作，不是实车采集数据集，也不能直接证明策略已经能上车稳定运行。

核心思路是：先用 Python 复现厂商 MATLAB 脚本中的线性动力学模型和 LQR 求解，再把离散线性模型封装成 Gymnasium 环境，最后用 Stable-Baselines3 训练和评估 `SAC`、`TD3`、`PPO` 等强化学习策略，并与 `LQR` 基线对比。

## 仓库内容

- `tools/lqr_from_matlab.py`：复现 MATLAB 中的连续模型、离散化和 LQR 增益计算。
- `tools/rl_balance/`：Gymnasium 环境、策略封装、训练/评估工具和实时控制辅助代码。
- `tools/train.py`、`tools/evaluate.py`、`tools/plot_results.py`：训练、评估、画图入口。
- `tools/tests/`：模型、环境、CLI 和通信协议相关测试。
- `matlab_reference/`：精简保留的关键 MATLAB 建模脚本。
- `项目梳理/`：中文交接文档、控制链路说明和实验日志。

没有上传完整厂商资料包、视频、APK、PDF、虚拟环境、模型 checkpoint 和训练产物。这些文件体积大，而且大多不是复现实验必须的代码。

## 模型来源和边界

当前仿真模型来自厂商 MATLAB 脚本中的标称参数和 LQR 设计：

- `m_1 = 0.9`
- `m_2 = 0.1`
- `r = 0.0335`
- `L_1 = 0.126`
- `L_2 = 0.390`
- `Ts = 0.01`

Python 复现出的 LQR 增益与现有补充固件 `control.c`、上位机示例 `wifi3.0.py` 中的 LQR 系数一致。这只能说明 Python 代码和厂商/固件参考模型一致，不能说明这些参数一定等于我们手上实际小车的真实物理参数。

如果要上车验证，需要先核对实车尺寸、质量、质心、传感器零点、状态顺序和闭环响应。

## 安装环境

建议使用 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果要用 GPU 训练，请根据自己电脑的 CUDA 和显卡驱动安装合适的 PyTorch 版本。长时间训练建议使用 CUDA。

也可以使用 editable install：

```bash
python -m pip install -e .
```

## 快速验证

运行测试：

```bash
python -m unittest discover -v tools/tests
```

验证 MATLAB/LQR 复现：

```bash
python tools/lqr_from_matlab.py
```

如果本地没有完整厂商固件目录，脚本会跳过 `control.c` 和 `wifi3.0.py` 的文件比对，但仍会输出 Python 复现的 LQR 增益。

## 常用命令

训练一个小规模 SAC 示例：

```bash
python tools/train.py --algo sac --device cpu --output-dir tools/artifacts/smoke --seeds 0 --timesteps 10000 --eval-freq 5000 --eval-episodes 5
```

评估训练结果：

```bash
python tools/evaluate.py --input-dir tools/artifacts/smoke --output-dir tools/artifacts/smoke/summary --episodes 20
```

生成图表：

```bash
python tools/plot_results.py --input-dir tools/artifacts/smoke/summary --output-dir tools/artifacts/smoke/plots
```

一键运行完整实验：

```bash
python tools/run_full_experiment.py
```

完整实验耗时较长，建议先跑小规模 smoke test。

## 当前研究状态

最新交接信息在：

- `项目梳理/当前实验交接.md`
- `项目梳理/自动优化研究日志.md`
- `项目梳理/控制链路说明.md`

目前结论简述：

- `LQR` 仍是最强基线。
- `SAC` 是当前最值得继续优化的 RL 算法。
- `PPO` 和 `TD3` 在当前设定下表现较差。
- 当前 RL 结果都是线性仿真环境中的结果，不是实车验证结果。
- 后续如果继续做 residual LQR + RL，需要优先控制动作能量，不能只看成功率。

## 组员协作建议

控制/建模方向：

确认 MATLAB 参数、状态顺序、动作含义、LQR 增益和实车固件是否一致。

强化学习方向：

主要看 `tools/rl_balance/config.py`、`tools/rl_balance/env.py`、`tools/rl_balance/experiments.py`，继续做 reward、SAC profile、residual scale 等小步实验。

嵌入式/实车方向：

先看 `项目梳理/控制链路说明.md`，确认 PC 端动作 `u_L/u_R` 的含义是左右轮角加速度，不是直接 PWM。

文档/汇报方向：

基于 `项目梳理/当前实验交接.md` 和 `项目梳理/自动优化研究日志.md` 整理阶段报告。

## 不建议上传到 Git 的内容

- `.venv/`
- `.omx/`
- `.cp-images/`
- `tools/artifacts_full_cuda/`
- `tools/artifacts_research/`
- `tools/logs/`
- `*.zip`
- 原始完整厂商资料包、PDF、视频、APK、大型固件资料目录

如果小组确实需要完整厂商资料，建议单独用网盘或 GitHub Release 分发，不要直接放进主仓库。
