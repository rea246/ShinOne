#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements-py312-cu12.txt"
PYTHON_BIN="${PYTHON_BIN:-python}"
WHEEL_DIR="${1:-}"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [wheel-directory]" >&2
    exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"Python 3.12 venv가 필요합니다. 현재 버전: {sys.version.split()[0]}"
    )
if sys.prefix == sys.base_prefix:
    raise SystemExit(
        "활성화된 venv가 아닙니다. Python 3.12 venv를 activate한 뒤 다시 실행하세요."
    )
if platform.system() != "Linux":
    raise SystemExit(f"Linux 환경이 필요합니다. 현재 OS: {platform.system()}")
if platform.machine() not in {"x86_64", "aarch64"}:
    raise SystemExit(f"지원하지 않는 아키텍처입니다: {platform.machine()}")
PY

"${PYTHON_BIN}" -m pip --version

if [[ -n "${WHEEL_DIR}" ]]; then
    if [[ ! -d "${WHEEL_DIR}" ]]; then
        echo "wheel directory가 없습니다: ${WHEEL_DIR}" >&2
        exit 2
    fi
    echo "[install] offline wheel directory: ${WHEEL_DIR}"
    "${PYTHON_BIN}" -m pip install \
        --no-index \
        --find-links="${WHEEL_DIR}" \
        --only-binary=:all: \
        -r "${REQUIREMENTS_FILE}"
else
    echo "[install] configured pip index 사용"
    "${PYTHON_BIN}" -m pip install \
        --only-binary=:all: \
        -r "${REQUIREMENTS_FILE}"
fi

"${PYTHON_BIN}" - <<'PY'
import importlib

module_names = (
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
    "klayout.db",
    "torch",
    "torch_geometric",
    "hdbscan",
    "cupy",
    "cudf",
    "cugraph",
    "igraph",
    "leidenalg",
)
for module_name in module_names:
    importlib.import_module(module_name)

import cupy
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "패키지는 설치됐지만 torch CUDA를 사용할 수 없습니다. "
        "CUDA 12 지원 torch wheel과 NVIDIA driver를 확인하세요."
    )
device_count = int(cupy.cuda.runtime.getDeviceCount())
if device_count < 1:
    raise SystemExit("패키지는 설치됐지만 CuPy가 CUDA GPU를 찾지 못했습니다.")

print(f"[verify] torch={torch.__version__}, gpu={torch.cuda.get_device_name(0)}")
print(f"[verify] cupy={cupy.__version__}, cuda_devices={device_count}")
print("[verify] ShinOne Python 3.12 / CUDA 12 dependencies OK")
PY
