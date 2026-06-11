#!/usr/bin/env bash
#
# Start the agent server with the final (Phase 6) configuration.
#
# --workers 8 is load-bearing, not cosmetic: /answer is a sync endpoint, so
# each in-flight request holds one anyio threadpool thread (~40/worker) for
# its whole multi-second graph run. At 10 RPS x ~5s that needs 50+ concurrent
# slots; a single worker queues requests internally and was the cause of the
# 55s p95 baseline (REPORT.md, Phase 6 iteration 1).

set -euo pipefail

exec uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001 --workers 8
