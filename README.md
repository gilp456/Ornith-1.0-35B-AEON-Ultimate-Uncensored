# Ornith-1.0-35B AEON Ultimate Uncensored - RTX 5090 Local Fork

This fork is tailored for running
[`AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4`](https://huggingface.co/AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4)
on single or dual NVIDIA GeForce RTX 5090 systems with local vLLM.

The upstream project includes DGX Spark and DFlash material. This fork keeps
that reference material, but the default path here is the consumer RTX 5090
setup: NVFP4, fully GPU-resident, no CPU offload, no DFlash by default.

![AEON Ornith cover art](cartridge.jpg)

## What This Fork Adds

- Single RTX 5090 profile: 32K context, tensor parallel size 1.
- Dual RTX 5090 profile: 64K context, tensor parallel size 2.
- Local vLLM launcher: [`serve_ornith_5090.sh`](serve_ornith_5090.sh).
- Environment checker: [`check_5090_prereqs.sh`](check_5090_prereqs.sh).
- Dedicated setup notes: [`QUICKSTART_RTX_5090.md`](QUICKSTART_RTX_5090.md).

## Recommended 5090 Model

| Build | Use on RTX 5090? | Notes |
|---|---:|---|
| [`NVFP4`](https://huggingface.co/AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4) | Yes | ~23.7 GB weight-only compressed-tensors build. Recommended for 32 GB cards. |
| [`BF16`](https://huggingface.co/AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16) | Not for 5090 all-GPU | ~66 GB before KV cache. Use larger-memory systems. |

## Validated Local Setup

The dual-GPU profile below was validated on:

| Component | Value |
|---|---|
| GPUs | 2x NVIDIA GeForce RTX 5090, 32 GB each |
| OS path | WSL2 Ubuntu on Windows |
| vLLM | 0.23.0 |
| PyTorch | 2.11.0+cu130 |
| CUDA visible devices | `0,1` |
| Served model name | `ornith` |
| Endpoint | `http://127.0.0.1:8000/v1` |
| Context | 65,536 tokens |

Runtime flags:

```bash
--tensor-parallel-size 2
--quantization compressed-tensors
--max-model-len 65536
--gpu-memory-utilization 0.78
--max-num-seqs 8
--max-num-batched-tokens 16384
--mamba-cache-dtype float32
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--enable-prefix-caching
--trust-remote-code
```

Observed local snapshot after warmup:

| Measurement | Value |
|---|---:|
| GPU 0 memory used | ~27.2 GB / 32.6 GB |
| GPU 1 memory used | ~26.0 GB / 32.6 GB |
| `/v1/models` max context | 65,536 |
| Short chat HTTP end-to-end throughput | ~52-54 completion tok/s |

These are local smoke-test numbers, not a full benchmark suite. Throughput will
vary with prompt length, reasoning length, sampling, cache state, and whether a
frontend reports raw decode speed or end-to-end request speed.

## Quick Start

Activate your vLLM environment, then run:

```bash
./check_5090_prereqs.sh
./serve_ornith_5090.sh dual
```

For one RTX 5090:

```bash
./serve_ornith_5090.sh single
```

Smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Chat test:

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

## Profile Defaults

| Profile | GPUs | Context | TP | GPU memory util | Max seqs | Max batched tokens |
|---|---:|---:|---:|---:|---:|---:|
| `single` | 1 | 32K | 1 | 0.88 | 4 | 8192 |
| `dual` | 2 | 64K | 2 | 0.78 | 8 | 16384 |

Use environment variables to override defaults:

```bash
MAX_LEN=49152 GPU_MEM=0.82 ./serve_ornith_5090.sh dual
PORT=8001 SERVED_NAME=ornith-32k ./serve_ornith_5090.sh single
```

## Why No DFlash By Default?

The upstream DFlash path is tuned for the AEON DGX Spark container workflow.
This 5090 fork intentionally keeps the baseline simple and stable:

- local vLLM
- NVFP4 compressed-tensors
- tensor parallel across RTX 5090s
- no CPU offload
- no DFlash unless tested separately
- OpenAI-compatible tool calling enabled for Hermes-style agents

Treat DFlash on RTX 5090 as an experiment, not the default deployment path.

## Expected Warnings

RTX 5090 runs may show warnings that native FP4 compute is not available and
that Marlin is used for weight-only FP4. That is expected for this profile.

WSL may also show:

```text
Using 'pin_memory=False' as WSL is detected.
```

That warning is expected and does not mean tensor parallelism is broken.

## Upstream Project Notes

The original upstream release is an uncensored / abliterated build of
[`deepreinforce-ai/Ornith-1.0-35B`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B).
It reports:

- 0 / 80 refusals on its harmful-prompt validation set.
- Agentic/coding pass@1 of 0.833, matching the base model in its probe.
- Hybrid `qwen3_5_moe` architecture with GatedDeltaNet, MoE, vision, and long context.

DGX Spark-specific documentation is retained here for reference:

- [`QUICKSTART_DGX_SPARK.md`](QUICKSTART_DGX_SPARK.md)
- [`hotfixes/ornith_dflash_kvfix/`](hotfixes/ornith_dflash_kvfix/)
- [`benchmarks/`](benchmarks/)

Those files describe the upstream Spark/DFlash path and should not be treated
as the default RTX 5090 deployment recipe.

## Responsibility

Safety refusals are removed. You are responsible for how you use the model and
for complying with applicable law. No warranty.

## Provenance

Cover art by [@newjordan](https://github.com/newjordan), used with permission.
Base model by DeepReinforce. Uncensored build and upstream release by AEON-7.

RTX 5090 deployment notes and scripts were adapted with assistance from AI
development tools and validated on local consumer Blackwell hardware.
