#!/usr/bin/env bash
# Recreate the pinned Linux/WSL2 environment without training or publishing data.
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-cpu}"
[[ "$mode" == cpu || "$mode" == cuda ]] || { echo 'Usage: bash scripts/setup.sh [cpu|cuda]' >&2; exit 2; }
command -v uv >/dev/null || { echo 'Install uv, then rerun. See README prerequisites.' >&2; exit 1; }
command -v node >/dev/null
command -v npm >/dev/null
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export npm_config_cache="${npm_config_cache:-$PWD/.cache/npm}"
mkdir -p .cache vendor
commit=0a00a759c9a4382a2877088d828302ec294a05a5
if [[ ! -d vendor/metamon/.git ]]; then
  [[ ! -e vendor/metamon ]] || { echo 'Refusing to replace existing vendor/metamon.' >&2; exit 1; }
  git clone --no-checkout https://github.com/UT-Austin-RPL/metamon.git vendor/metamon
  git -C vendor/metamon checkout --detach "$commit"
fi
[[ "$(git -C vendor/metamon rev-parse HEAD)" == "$commit" ]] || { echo 'Metamon revision differs from pins.json; preserve it and use a fresh checkout.' >&2; exit 1; }
git -C vendor/metamon submodule update --init --recursive
uv venv --python 3.12.14 --allow-existing .venv
# The base lock excludes torch so CPU and CUDA do not silently replace each other.
uv pip install --python .venv/bin/python -r "requirements-${mode}.lock" --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python --no-deps -e vendor/metamon
if [[ "$mode" == cuda ]]; then
  .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check NVIDIA driver and WSL GPU access"; assert torch.cuda.get_device_capability()[0] >= 8, "Kakuna bf16 FlashAttention route requires Ampere-or-newer NVIDIA GPU"'
  command -v nvcc >/dev/null || { echo 'CUDA toolkit (nvcc) required for the flash-attn build. See docs/local-run.md.' >&2; exit 1; }
  uv pip install --python .venv/bin/python 'ninja==1.13.0' 'wheel==0.45.1'
  MAX_JOBS="${MAX_JOBS:-2}" uv pip install --python .venv/bin/python --no-build-isolation 'flash-attn==2.7.4.post1'
fi
(cd vendor/metamon/server/pokemon-showdown; npm ci --ignore-scripts; npm run build)
.venv/bin/python scripts/configure_server.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
echo 'Setup complete. CPU mode is preparation only; exact Kakuna inference requires CUDA mode.'
