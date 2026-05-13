#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${SCRIPT_DIR}/tools"
VENV_DIR="${TOOLS_DIR}/.venv"
ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"

if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
  echo "Missing virtual environment: ${ACTIVATE_SCRIPT}" >&2
  echo "Expected the project environment at ${VENV_DIR}" >&2
  exit 1
fi

usage() {
  cat <<EOF
Usage:
  $(basename "$0")
  $(basename "$0") <command> [args...]

Examples:
  $(basename "$0")
  $(basename "$0") python3 -m unittest discover -v tests
  $(basename "$0") python3 train.py --algo sac --device cuda
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  # shellcheck disable=SC1090
  source "${ACTIVATE_SCRIPT}"
  cd "${TOOLS_DIR}"
  export RL_PROJECT_ROOT="${SCRIPT_DIR}"
  export RL_TOOLS_DIR="${TOOLS_DIR}"
  export RL_VENV_DIR="${VENV_DIR}"
  echo "Activated RL environment in ${TOOLS_DIR}"
  return 0
fi

if (( $# > 0 )); then
  # shellcheck disable=SC1090
  source "${ACTIVATE_SCRIPT}"
  cd "${TOOLS_DIR}"
  export RL_PROJECT_ROOT="${SCRIPT_DIR}"
  export RL_TOOLS_DIR="${TOOLS_DIR}"
  export RL_VENV_DIR="${VENV_DIR}"
  exec "$@"
fi

INIT_FILE="$(mktemp)"
trap 'rm -f "${INIT_FILE}"' EXIT

cat >"${INIT_FILE}" <<EOF
source "${ACTIVATE_SCRIPT}"
cd "${TOOLS_DIR}"
export RL_PROJECT_ROOT="${SCRIPT_DIR}"
export RL_TOOLS_DIR="${TOOLS_DIR}"
export RL_VENV_DIR="${VENV_DIR}"
echo "RL environment ready."
echo "Project root: ${SCRIPT_DIR}"
echo "Tools dir: ${TOOLS_DIR}"
echo "Python: \$(command -v python3)"
EOF

exec "${SHELL:-/bin/bash}" --rcfile "${INIT_FILE}" -i
