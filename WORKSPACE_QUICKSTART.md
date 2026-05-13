# RL Workspace Quickstart

## Workspace

- Project root: `/mnt/d/code_wf/Y1S1/强化学习`
- Tools dir: `/mnt/d/code_wf/Y1S1/强化学习/tools`
- Python env: `/mnt/d/code_wf/Y1S1/强化学习/tools/.venv`
- Startup script: `/mnt/d/code_wf/Y1S1/强化学习/start_rl_env.sh`

## What The Script Does

`start_rl_env.sh` will:

- activate `tools/.venv`
- switch into `tools/`
- export:
  - `RL_PROJECT_ROOT`
  - `RL_TOOLS_DIR`
  - `RL_VENV_DIR`

## Common Usage

Enter an interactive shell with the environment activated:

```bash
/mnt/d/code_wf/Y1S1/强化学习/start_rl_env.sh
```

Run tests directly:

```bash
/mnt/d/code_wf/Y1S1/强化学习/start_rl_env.sh python3 -m unittest discover -v tests
```

Train with GPU:

```bash
/mnt/d/code_wf/Y1S1/强化学习/start_rl_env.sh python3 train.py --algo sac --device cuda
```

Evaluate:

```bash
/mnt/d/code_wf/Y1S1/强化学习/start_rl_env.sh python3 evaluate.py --input-dir artifacts_full_cuda --output-dir artifacts_full_cuda/summary
```

Plot:

```bash
/mnt/d/code_wf/Y1S1/强化学习/start_rl_env.sh python3 plot_results.py --input-dir artifacts_full_cuda/summary --output-dir artifacts_full_cuda/plots
```

## If You Prefer `source`

You can also load the environment into the current shell:

```bash
source /mnt/d/code_wf/Y1S1/强化学习/start_rl_env.sh
```

## From Windows PowerShell

Interactive shell:

```powershell
wsl -d Ubuntu-24.04 bash -lc "/mnt/d/code_wf/Y1S1/强化学习/start_rl_env.sh"
```

Run a command:

```powershell
wsl -d Ubuntu-24.04 bash -lc "/mnt/d/code_wf/Y1S1/强化学习/start_rl_env.sh python3 train.py --algo sac --device cuda"
```

## Verified State

- `torch==2.11.0+cu130`
- `torch.cuda.is_available() == True`
- detected GPU: `NVIDIA GeForce RTX 5080`
- `python3 -m unittest discover -v tests`: `20/20` passed when run through this environment
