#!/usr/bin/env bash
#
# Start vLLM with the final tuned configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Phase 1 baseline flags + the one serving-side change that survived the
# Phase 6 SLO iterations: n-gram speculative decoding (SQL output largely
# copies schema/question tokens already in the prompt, so prompt-lookup
# drafts verify cheaply and losslessly). Rationale for every flag and the
# full iteration log live in REPORT.md (sections 1 and 6).

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5, "prompt_lookup_min": 2}' \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --disable-log-requests
