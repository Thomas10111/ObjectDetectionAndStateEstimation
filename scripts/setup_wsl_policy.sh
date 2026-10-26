#!/usr/bin/env bash
# Python environment for training the driving policy, in WSL. No sudo needed:
#
#   bash /mnt/c/Users/fisch/MyGit/ObjectDetectionAndStateEstimation/scripts/setup_wsl_policy.sh
#   source ~/.venvs/odse_policy/bin/activate
#
# Python 3.12 like the Orin (Ubuntu 26.04 itself ships a newer one). The
# environment lives in the Linux file system, because installing PyTorch onto
# /mnt/c is very slow. Safe to run again.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
VENV=~/.venvs/odse_policy

# uv: fetches Python 3.12 and installs packages, into ~/.local/bin
if ! command -v uv > /dev/null && [[ ! -x ~/.local/bin/uv ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12
[[ -d $VENV ]] || uv venv --python 3.12 "$VENV"

# CPU-only PyTorch: the laptop has no NVIDIA GPU
uv pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$VENV/bin/python" -r "$here/../policy/requirements.txt"

"$VENV/bin/python" - <<'EOF'
import sys, torch, gymnasium, stable_baselines3, onnx, onnxruntime
print(f"Python {sys.version.split()[0]}")
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, threads: {torch.get_num_threads()}")
print(f"gymnasium {gymnasium.__version__}, stable-baselines3 {stable_baselines3.__version__}")
print(f"onnx {onnx.__version__}, onnxruntime {onnxruntime.__version__}")
x = torch.randn(1000, 1000)
print(f"matrix multiply check: {float((x @ x).sum()):.1f}")
EOF
