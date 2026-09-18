#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${PROJECT_ROOT}/.venv-qwen3"
BASE_PYTHON="${YIELDMIND_BASE_PYTHON:-/opt/anaconda3/envs/amla/bin/python}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv "${ENV_DIR}"
fi

"${ENV_DIR}/bin/python" -m pip install --disable-pip-version-check \
  -r "${PROJECT_ROOT}/requirements-qwen3-embedding.txt"
"${ENV_DIR}/bin/python" -m pip check
"${ENV_DIR}/bin/python" -c \
  "import torch, transformers, sentence_transformers; print({'torch': torch.__version__, 'transformers': transformers.__version__, 'sentence_transformers': sentence_transformers.__version__})"
