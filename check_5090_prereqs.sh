#!/usr/bin/env bash
#
# check_5090_prereqs.sh - quick local environment check for RTX 5090 vLLM runs.

set -euo pipefail

echo "=== Ornith RTX 5090 prerequisite check ==="

if ! command -v python >/dev/null 2>&1; then
  echo "missing: python" >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "missing: vllm. Activate your ornith venv first." >&2
  exit 1
fi

python - <<'PY'
import shutil
import subprocess
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"missing or broken torch import: {exc}")

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("gpu count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to torch")
if torch.cuda.device_count() < 1:
    raise SystemExit("No CUDA GPUs visible")

if shutil.which("nvidia-smi"):
    subprocess.run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader",
    ], check=False)
PY

echo "vLLM $(vllm --version)"
echo "OK"
