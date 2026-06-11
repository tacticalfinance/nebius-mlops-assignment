#!/usr/bin/env bash
#
# Start vLLM with the baseline (Phase 1) configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# This is the *initial* config, chosen for the workload shape
# (1.5-3K-token prompts, short structured outputs, 2-3 dependent calls
# per agent run). It will be iterated on in Phase 6 against the SLO.
# Rationale for every flag lives in REPORT.md (Phase 1 section).

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 2048 \
    --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5, "prompt_lookup_min": 2}' \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --disable-log-requests
