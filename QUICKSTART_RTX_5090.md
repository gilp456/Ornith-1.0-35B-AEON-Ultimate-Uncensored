# RTX 5090 QuickStart - Ornith-1.0-35B AEON Ultimate Uncensored

This guide is for local RTX 5090 systems using vLLM directly, especially WSL2
Ubuntu on Windows. It is intentionally separate from the DGX Spark guide.

The recommended model for 5090s is the NVFP4 build:

```text
AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4
```

Use this instead of the BF16 build. The BF16 build is about 66 GB before KV
cache and is not a practical all-GPU fit for 32 GB consumer cards.

## Profiles

| Profile | GPUs | Context | Tensor parallel | VRAM posture |
|---|---:|---:|---:|---|
| `single` | 1x RTX 5090 32 GB | 32K | 1 | Conservative all-GPU profile |
| `dual` | 2x RTX 5090 32 GB | 64K | 2 | Tested local profile with headroom |

Both profiles keep the model fully GPU-resident. No CPU offload is used.

## Validated Dual 5090 Snapshot

This profile has been validated on a local WSL2 Ubuntu setup with:

| Component | Value |
|---|---|
| GPUs | 2x NVIDIA GeForce RTX 5090, 32 GB each |
| vLLM | 0.23.0 |
| PyTorch | 2.11.0+cu130 |
| Model | `AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4` |
| Served name | `ornith` |
| Context | 65,536 tokens |
| Tensor parallel | 2 |

Current observed smoke-test data:

| Measurement | Value |
|---|---:|
| GPU 0 memory used after warmup | ~27.2 GB / 32.6 GB |
| GPU 1 memory used after warmup | ~26.0 GB / 32.6 GB |
| `/v1/models` reported context | 65,536 |
| Short chat HTTP end-to-end throughput | ~52-54 completion tok/s |

These are local validation numbers, not a formal benchmark. Frontends may report
higher decode-only token rates than end-to-end HTTP timing, especially when
reasoning tokens are separated from visible answer text.

## Install

Use a Python 3.12 virtual environment with a CUDA-enabled vLLM build that
supports `qwen3_5_moe` and compressed-tensors NVFP4.

Example:

```bash
python3 -m venv ~/venvs/ornith-vllm
source ~/venvs/ornith-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade vllm
```

Log in to Hugging Face if the model is not already cached:

```bash
hf auth login
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("gpu count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

## Serve

From this repository:

```bash
chmod +x ./serve_ornith_5090.sh
./serve_ornith_5090.sh dual
```

For a single 5090:

```bash
./serve_ornith_5090.sh single
```

The server exposes an OpenAI-compatible endpoint:

```text
http://127.0.0.1:8000/v1
```

The served model name is:

```text
ornith
```

## Smoke Test

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "ornith",
    "messages": [{"role": "user", "content": "Write a one-line Python hello world print statement."}],
    "max_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 20
  }' | python3 -m json.tool
```

This is a thinking model. With `--reasoning-parser qwen3`, vLLM may return
reasoning separately from `message.content`.

## Why These Defaults

### Dual 5090

The dual profile uses:

```text
--tensor-parallel-size 2
--max-model-len 65536
--gpu-memory-utilization 0.78
--max-num-seqs 8
--max-num-batched-tokens 16384
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

This is the recommended baseline for a 2x 5090 machine because it leaves VRAM
headroom for Windows/WSL display usage, CUDA graphs, compile buffers, KV cache,
and normal desktop activity.

### Single 5090

The single profile uses:

```text
--tensor-parallel-size 1
--max-model-len 32768
--gpu-memory-utilization 0.88
--max-num-seqs 4
--max-num-batched-tokens 8192
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

This is meant to keep the model all-GPU on a 32 GB card. If startup fails due
to VRAM pressure, lower `MAX_LEN` before changing anything else.

## DFlash

Do not enable DFlash by default on RTX 5090 local vLLM. The DFlash material in
the upstream repo is tuned for the AEON DGX Spark container path. Treat DFlash
on 5090s as a separate experiment, not the baseline.

## Useful Overrides

All launcher defaults can be overridden with environment variables:

```bash
MAX_LEN=49152 GPU_MEM=0.82 ./serve_ornith_5090.sh dual
PORT=8001 SERVED_NAME=ornith-32k ./serve_ornith_5090.sh single
```

Important override variables:

| Variable | Meaning |
|---|---|
| `MODEL` | Hugging Face model id or local model path |
| `SERVED_NAME` | OpenAI API model alias |
| `PORT` | HTTP port |
| `MAX_LEN` | vLLM `--max-model-len` |
| `GPU_MEM` | vLLM `--gpu-memory-utilization` |
| `MAX_NUM_SEQS` | vLLM `--max-num-seqs` |
| `MAX_BATCHED_TOKENS` | vLLM `--max-num-batched-tokens` |
| `CUDA_VISIBLE_DEVICES` | GPU selection |

## Expected Warnings

RTX 5090 may show warnings that native FP4 compute is not available and that
Marlin is used for weight-only FP4. That is expected for this local profile.

WSL may also show:

```text
Using 'pin_memory=False' as WSL is detected.
```

That warning is expected and does not mean tensor parallelism is broken.
