# RL Balance Experiment Pipeline

## Environment

- Main runtime: WSL `Ubuntu-24.04`
- Install dependencies:

```bash
python3 -m pip install --user --break-system-packages -r tools/requirements-wsl.txt
```

## Training

```bash
python3 tools/train.py --algo sac
python3 tools/train_td3.py
python3 tools/train_ppo.py
```

## Evaluation And Plots

```bash
python3 tools/evaluate.py --input-dir tools/artifacts --output-dir tools/artifacts/summary
python3 tools/plot_results.py --input-dir tools/artifacts/summary --output-dir tools/artifacts/plots
```

## One-Click Run

```bash
python3 tools/run_full_experiment.py
```

## Live Control

LQR baseline:

```bash
python3 tools/live_control.py --policy lqr
```

RL checkpoint:

```bash
python3 tools/live_control.py --policy rl --algo ppo --model-path tools/artifacts/ppo/seed_0/best_model.zip
```
